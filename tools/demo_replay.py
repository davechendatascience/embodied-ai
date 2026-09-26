#!/usr/bin/env python
"""LIBERO's recorded motion replayed open loop through the Panda VLA's decode: the gate of
BRN-vla-decodes-twists-exactly (CTR-demo-replay-succeeds).

  demo_replay.py --suite libero_spatial [--tasks 0 1] [--demos 50] --trials $OUT

Each demonstration is placed from its first recorded state, with the poses of the fixtures the reset
re-samples written from its own recorded model (AXM-libero-demos-record-a-panda, AXM-libero-resamples-fixtures).
It is then driven by exactly what the VLA is taught (BRN-vla-learns-the-recorded-motion): at each step the
recorded motion's twist and the recorded gripper command, executed by the VLA's decode (VLA_EXECUTION), and after
the last recorded step a hold for SETTLE_ALLOWANCE steps. The replay succeeds if LIBERO's predicate accepts a
state it reaches. A replay that fails is motion the decode does not reproduce: the joint controller's lag, and
contact the humans' operational-space controller made differently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CPUS = "5,6,7,8,9,15,16,17"
SETTLE_ALLOWANCE = 20     # control steps (1 s) the last recorded pose is held for LIBERO's predicate to hold
SCENE_SEED = 0            # the reset's own draws are all overwritten (fixtures from the demo, state from the demo)
# the code a replay's outcome depends on: its revision keys the ledger's slices
REPLAY_CODE = ("tools/demo_replay.py", "screwhead/sim/sim_arm.py", "screwhead/sim/servo.py",
               "screwhead/sim/joint_ramp.py", "screwhead/sim/task_env.py", "screwhead/geometry/kin_np.py",
               "screwhead/student/libero_data.py", "screwhead/student/qwen_vla.py")


def replay_revision() -> str:
    h = hashlib.sha1()
    for f in REPLAY_CODE:
        h.update((ROOT / f).read_bytes())
    return f"demo_replay:{h.hexdigest()[:10]}"


def _write_fixtures(m, xml: str) -> None:
    """The fixtures LIBERO's reset re-samples -- each a scene object's root body (<object>_main) fixed to the
    world -- at the poses the demonstration's model gives them. Only those: a recorded model can describe the
    rest of the scene in other body frames. libero_10 6's table is at the origin here and at (-0.25, 0.25),
    turned 90 deg, in its demonstrations' models, with its region sites expressed in each frame; written from
    the recorded model, the table carried our sites 0.6 m off and no demonstration met the predicate (0 of 50)."""
    for b in range(1, m.nbody):
        name = m.body(b).name
        if int(m.body_parentid[b]) != 0 or int(m.body_jntnum[b]) != 0 or not name.endswith("_main"):
            continue
        tag = re.search(rf'<body[^>]*name="{re.escape(name)}"[^>]*>', xml)
        if tag is None:
            continue
        pos = re.search(r'\bpos="([^"]+)"', tag.group(0))
        quat = re.search(r'\bquat="([^"]+)"', tag.group(0))
        if pos:
            m.body_pos[b] = [float(v) for v in pos.group(1).split()]
        if quat:
            m.body_quat[b] = [float(v) for v in quat.group(1).split()]


def _place(env, state: np.ndarray, xml: str) -> None:
    """The demonstration's first state in the demonstration's scene, execution memory anchored there."""
    from screwhead.sim.task_env import POSTURE_GAIN
    env._reset_scene(SCENE_SEED)
    _write_fixtures(env.env.sim.model._model, xml)
    env.env.set_init_state(state)
    env._anchor()
    env.servo.reset(np.asarray(env.observe()["robot0_joint_pos"]))
    if env.execution.posture_start:
        env.servo.posture, env.servo.posture_gain = env.servo.ref.copy(), POSTURE_GAIN
    env.t = 0
    env.raw = env.observe()


def replay_demo(env, states: np.ndarray, actions: np.ndarray, xml: str) -> dict:
    """One demonstration's recorded motion through the decode, open loop."""
    from screwhead.geometry import kin_np
    from screwhead.student.libero_data import ARM_NQ, recorded_twists
    _place(env, states[0], xml)
    T = len(states)
    twists = recorded_twists(env.servo._np, states)                        # (T-1, 6), in the action's units
    poses = kin_np.fk(env.servo._np, np.asarray(states[:, 1:1 + ARM_NQ], float))
    sv = env.servo
    ev0 = (sv.acc_limited, sv.scaled, sv.iter_cap, sv.reanchors)
    track, first = [], None
    for t in range(T - 1 + SETTLE_ALLOWANCE):
        v = twists[t] if t < T - 1 else np.zeros(6)
        _raw, _r, _done, info = env.step(np.r_[np.clip(v, -1.0, 1.0), float(actions[min(t, T - 1), -1])])
        k = min(t + 1, T - 1)
        track.append(float(np.linalg.norm(env.tool_state()["p_tool"] - poses[k][:3, 3])))
        if info["success"]:
            first = t + 1
            break
    ev = [n - n0 for n, n0 in zip((sv.acc_limited, sv.scaled, sv.iter_cap, sv.reanchors), ev0, strict=True)]
    return dict(success=first is not None, first_success_step=first if first is not None else -1, demo_steps=T,
                max_track_mm=round(1000 * max(track), 1), mean_track_mm=round(1000 * float(np.mean(track)), 1),
                acc_limited_steps=ev[0], scaled_steps=ev[1], iter_cap_steps=ev[2], reanchored_steps=ev[3])


def _pin(cpus) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.sched_setaffinity(0, {cpus.get()})


def _task(job: tuple) -> list[dict]:
    suite, task, demos, rev = job
    import h5py
    import torch
    torch.set_num_threads(1)
    from screwhead.sim.sim_arm import Execution
    from screwhead.sim.task_env import TaskEnv
    from screwhead.student.libero_data import task_file
    from screwhead.student.qwen_vla import VLA_EXECUTION
    env = TaskEnv(suite, task, horizon=10**6, seed=SCENE_SEED, render=False, execution=Execution(**VLA_EXECUTION))
    rows = []
    with h5py.File(task_file(suite, env.task_spec.name)) as f:
        keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))[:demos]
        for key in keys:
            e = f["data"][key]
            xml = e.attrs["model_file"]
            xml = xml.decode() if isinstance(xml, bytes) else xml
            try:
                m = replay_demo(env, e["states"][()], e["actions"][()], xml)
            except Exception as err:  # noqa: BLE001  a worker boundary: a crash is a failed, recorded replay
                m = dict(success=False, first_success_step=-1, demo_steps=int(len(e["states"])),
                         max_track_mm=-1.0, mean_track_mm=-1.0, error=f"{type(err).__name__}: {err}")
            demo = int(key.split("_")[1])
            rows.append({"metrics": {k: v for k, v in m.items() if k != "error"},
                         "conditions": {"task_suite": suite, "task": task, "demo": demo,
                                        **({"error": m["error"]} if "error" in m else {})},
                         "repro": {"task_suite": suite, "task": task, "demo": demo, "replay_revision": rev,
                                   "execution": json.dumps(VLA_EXECUTION, default=list)}})
    env.close()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", type=int, nargs="*", default=None)
    ap.add_argument("--demos", type=int, default=50)
    ap.add_argument("--cpus", default=CPUS)
    ap.add_argument("--trials", required=True)
    args = ap.parse_args()
    tasks = args.tasks if args.tasks is not None else list(range(90 if args.suite == "libero_90" else 10))
    rev = replay_revision()
    cpus = [int(c) for c in args.cpus.split(",")]
    free = multiprocessing.Manager().Queue()
    for c in cpus:
        free.put(c)
    jobs = [(args.suite, t, args.demos, rev) for t in tasks]
    with ProcessPoolExecutor(len(cpus), initializer=_pin, initargs=(free,),
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        trials = [r for rows in pool.map(_task, jobs) for r in rows]
    out = Path(args.trials)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"trials": trials}, indent=1))
    print(f"{args.suite} ({rev})")
    for t in tasks:
        rs = [r for r in trials if r["conditions"]["task"] == t]
        ok = [r for r in rs if r["metrics"]["success"]]
        track = [r["metrics"]["max_track_mm"] for r in rs if r["metrics"]["max_track_mm"] >= 0]
        errors = sum("error" in r["conditions"] for r in rs)
        print(f"  task {t}: {len(ok)}/{len(rs)} replays succeed; tool off the recorded path at most "
              f"{np.median(track) if track else float('nan'):.1f} mm (median over replays)"
              + (f"; {errors} crashed" if errors else ""))
    print(f"  all: {sum(r['metrics']['success'] for r in trials)}/{len(trials)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
