#!/usr/bin/env python
"""LIBERO's human demonstrations replayed through the evaluation's execution path: the Panda VLA's
training pairs (BRN-vla-trains-on-demos-replayed-through-the-servo).

  demo_replay.py --suite libero_spatial [--tasks 0 1] [--demos 50] --trials $OUT [--render 256 --out cache/vla_replay]

Each demonstration is placed from its first recorded state, with the fixture poses its own recorded
model ran with: the reset re-samples them and writing the state restores neither
(AXM-libero-demos-record-a-panda, AXM-libero-resamples-fixtures). At each control step t the arm's
label is the body twist that carries the servo's pose reference, as it stands at t, to the tool pose
the demonstration recorded at t+1 in one control period, normalised and clipped to the action range;
the gripper channel is the demonstration's own command at t, +1 closed and -1 open
(AXM-libero-demos-record-their-actions). A label computed from the reference rather than from the
previous recorded pose is what keeps a twist the servo limited from becoming an offset in every label
after it. After its last recorded pose the replay holds that pose and command for SETTLE_ALLOWANCE
steps. A replay is kept only if LIBERO's predicate accepts some state it reaches, and its pairs end
at the first such state, where LIBERO's evaluation ends an episode.

This is the first gate (CTR-demo-replay-succeeds): the replay success rate per task, with how far the
tool strayed from the demonstration's path. With --render, the kept replays are also written, one file per
task (<out>/<suite>/task<NN>.hdf5, a group per demonstration): both cameras as the evaluation renders them
(uint8, as robosuite returns them, stored losslessly), the evaluation's proprioception and the actions.
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

DATASETS = ROOT / "third_party/LIBERO/libero/datasets"
CPUS = "5,6,7,8,9,15,16,17"
SETTLE_ALLOWANCE = 20     # control steps (1 s) the last recorded pose is held for LIBERO's predicate to hold
ARM_NQ = 7                # the Panda's arm joints, first in the recorded state after its time entry
SCENE_SEED = 0            # the reset's own draws are all overwritten (fixtures from the demo, state from the demo)
# The VLA's execution path (AXM-one-execution-path: replay, DAgger and evaluation alike): the skill teacher's, except
# that the servo keeps its free-motion acceleration bound while the jaws hold something. The holding bound (0.5
# m/s^2) was set for the teacher's rim pinch of the rack's bottle; the humans carry at up to about 1 m/s^2 (p95 0.7-1.0
# on libero_spatial 0), and under it the reference fell 60-90 mm behind a carried bowl and released it short: 2 of 5
# of that task's first demonstrations replayed, against 5 of 5 (tool within 23 mm of the path) with it off.
VLA_EXECUTION = dict(max_lin_acc_holding=None)
# the code a replay's outcome depends on: its revision keys the ledger's slices
REPLAY_CODE = ("tools/demo_replay.py", "screwhead/sim/sim_arm.py", "screwhead/sim/servo.py",
               "screwhead/sim/gripper_servo.py", "screwhead/sim/joint_ramp.py", "screwhead/sim/task_env.py",
               "screwhead/geometry/kin_np.py")


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
    env._reset_scene(SCENE_SEED)
    _write_fixtures(env.env.sim.model._model, xml)
    env.env.set_init_state(state)
    env._anchor()
    env.servo.reset(np.asarray(env.observe()["robot0_joint_pos"]))
    env.t = 0
    env.raw = env.observe()


def replay_demo(env, states: np.ndarray, actions: np.ndarray, xml: str, pairs: list | None = None) -> dict:
    """One demonstration through the servo. `pairs`, when given, receives (observation, action) up to the
    first step LIBERO's predicate accepts."""
    from screwhead.geometry import kin_np
    _place(env, states[0], xml)
    T = len(states)
    targets = kin_np.fk(env.servo._np, np.asarray(states[:, 1:1 + ARM_NQ], float))    # recorded tool poses
    dt = env.spec.dt
    track, first = [], None
    for t in range(T - 1 + SETTLE_ALLOWANCE):
        k = min(t + 1, T - 1)
        V = kin_np.log_se3(kin_np.inverse(env.servo.T_ref)[None] @ targets[k][None])[0] / dt
        a = np.empty(7)
        a[:6] = np.clip(V / env.scale, -1.0, 1.0)
        a[6] = float(actions[min(t, T - 1), -1])
        obs = env.raw
        _raw, _r, _done, info = env.step(a)
        if pairs is not None:
            pairs.append((obs, a))
        track.append(float(np.linalg.norm(env.tool_state()["p_tool"] - targets[k][:3, 3])))
        if info["success"]:
            first = t + 1
            break
    ok = first is not None
    if pairs is not None and not ok:
        pairs.clear()
    return dict(success=ok, first_success_step=first if ok else -1, demo_steps=T,
                max_track_mm=round(1000 * max(track), 1), mean_track_mm=round(1000 * float(np.mean(track)), 1))


def _write(g, pairs: list, m: dict, env) -> None:
    """One kept replay: what the policy observed at each step and the action it was labelled with."""
    obs = [o for o, _a in pairs]
    tool = [env.tool_state(o) for o in obs]
    pose = np.zeros((len(obs), 4, 4), np.float32)
    pose[:, :3, :3] = [t["R_tool"] for t in tool]
    pose[:, :3, 3] = [t["p_tool"] for t in tool]
    pose[:, 3, 3] = 1.0
    for cam, key in (("agentview", "agentview_image"), ("wrist", "robot0_eye_in_hand_image")):
        img = np.stack([o[key] for o in obs]).astype(np.uint8)
        g.create_dataset(cam, data=img, chunks=(1, *img.shape[1:]), compression="lzf")
    g.create_dataset("joint_pos", data=np.stack([o["robot0_joint_pos"] for o in obs]).astype(np.float32))
    g.create_dataset("gripper_qpos", data=np.stack([o["robot0_gripper_qpos"] for o in obs]).astype(np.float32))
    g.create_dataset("tool_pose", data=pose)
    g.create_dataset("actions", data=np.stack([a for _o, a in pairs]).astype(np.float32))
    for k in ("first_success_step", "demo_steps"):
        g.attrs[k] = m[k]


def _pin(cpus) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.sched_setaffinity(0, {cpus.get()})


def _task(job: tuple) -> list[dict]:
    suite, task, demos, rev, px, out = job
    import h5py
    import torch
    torch.set_num_threads(1)
    from screwhead.sim.sim_arm import Execution
    from screwhead.sim.task_env import TaskEnv
    env = TaskEnv(suite, task, horizon=10**6, seed=SCENE_SEED, render=px or False, execution=Execution(**VLA_EXECUTION))
    path = DATASETS / suite / f"{env.task_spec.name}_demo.hdf5"
    rows = []
    data = None
    if px:
        dst = Path(out) / suite / f"task{task:02d}.hdf5"
        dst.parent.mkdir(parents=True, exist_ok=True)
        data = h5py.File(dst, "w")
        data.attrs.update(task_suite=suite, task=task, language=env.language, replay_revision=rev,
                          execution=json.dumps(VLA_EXECUTION), camera_px=px, source=str(path.relative_to(ROOT)))
    with h5py.File(path) as f:
        keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))[:demos]
        for key in keys:
            e = f["data"][key]
            xml = e.attrs["model_file"]
            xml = xml.decode() if isinstance(xml, bytes) else xml
            pairs: list | None = [] if data is not None else None
            try:
                m = replay_demo(env, e["states"][()], e["actions"][()], xml, pairs)
                if pairs:
                    _write(data.create_group(key), pairs, m, env)
            except Exception as err:  # noqa: BLE001  a worker boundary: a crash is a failed, recorded replay
                m = dict(success=False, first_success_step=-1, demo_steps=int(len(e["states"])),
                         max_track_mm=-1.0, mean_track_mm=-1.0, error=f"{type(err).__name__}: {err}")
            demo = int(key.split("_")[1])
            rows.append({"metrics": {k: v for k, v in m.items() if k != "error"},
                         "conditions": {"task_suite": suite, "task": task, "demo": demo,
                                        **({"error": m["error"]} if "error" in m else {})},
                         "repro": {"task_suite": suite, "task": task, "demo": demo, "replay_revision": rev,
                                   "execution": json.dumps(VLA_EXECUTION)}})
    if data is not None:
        data.close()
    env.close()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", type=int, nargs="*", default=None)
    ap.add_argument("--demos", type=int, default=50)
    ap.add_argument("--cpus", default=CPUS)
    ap.add_argument("--trials", required=True)
    ap.add_argument("--render", type=int, default=0, help="camera pixels; 0 runs the gate without images")
    ap.add_argument("--out", default=str(ROOT / "cache/vla_replay"))
    args = ap.parse_args()
    tasks = args.tasks if args.tasks is not None else list(range(90 if args.suite == "libero_90" else 10))
    rev = replay_revision()
    cpus = [int(c) for c in args.cpus.split(",")]
    free = multiprocessing.Manager().Queue()
    for c in cpus:
        free.put(c)
    jobs = [(args.suite, t, args.demos, rev, args.render, args.out) for t in tasks]
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
        print(f"  task {t}: {len(ok)}/{len(rs)} replays succeed; tool off the demo's path at most "
              f"{np.median(track) if track else float('nan'):.1f} mm (median over replays)"
              + (f"; {errors} crashed" if errors else ""))
    n_ok = sum(r["metrics"]["success"] for r in trials)
    print(f"  all: {n_ok}/{len(trials)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
