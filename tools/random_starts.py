#!/usr/bin/env python
"""The randomized test set for libero_spatial (BRN-random-starts-test-set): starts none of which is one of LIBERO's
test starts or its demonstrations' starts, each solved by the skill teacher, stored with the references an
environment must match before the set is placed in it.

  random_starts.py [--per-task 50] [--max-draws 1000] [--seed 2601] --out runs/evidence/random_starts/libero_spatial.npz
  random_starts.py --scale 1.5 --exclude runs/evidence/random_starts/libero_spatial.npz --seed 2603 --out ...

Per task: the scene randomizer draws from LIBERO's initial states (objects moved in rigid groups within 8 cm and
turned, the instruction's relation kept; the tool's start moved 10 cm across, 5 cm up or down, 30 deg in yaw, 10 deg
in tilt, the redundant joint 0.3 rad), settled. A draw is kept if some free object lies at least NOVELTY_M from where
it lies in each of the task's LIBERO initial states and demonstrations' first states, and if the skill teacher, run
from the stored state placed as an evaluation places it, meets LIBERO's predicate within the suite's step limit.
With --vla-execution the teacher runs through the VLA's execution (VLA_EXECUTION: the gripper by +1/-1 command, its
target channel mapped by squeeze_command) -- the execution a VLA's training episodes are recorded in.
With --scale every randomizer bound is multiplied by it; with --exclude a kept draw must also lie NOVELTY_M from each
start of that set's task (training starts drawn around a test set, none of them its starts). Discards are counted by
reason. The file holds the kept states, their fixture poses, and per task the references:
the task file's digest, the simulator release, the model's fingerprint and the states reached from the first kept
start after 100 and 500 physics steps under fixed actuator inputs.
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SUITE = "libero_spatial"
STEP_LIMIT = 220                  # LIBERO's published step limit for libero_spatial (tools/eval_vla.py)
NOVELTY_M = 0.02
RANDOMIZER = dict(layout_radius=0.08, start_xy_m=0.10, start_z_m=0.05, start_yaw_deg=30.0, start_tilt_deg=10.0,
                  start_null_rad=0.3)   # this project's declared randomized evaluation
CPUS = "5,6,7,8,9,15,16,17"


def _pin(cpus) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.sched_setaffinity(0, {cpus.get()})


def _free_positions(m, states: np.ndarray) -> np.ndarray:
    """(N, objects, 3): every free object's position in flattened states (time first, then qpos)."""
    adr = [int(m.jnt_qposadr[j]) for j in range(m.njnt) if int(m.jnt_type[j]) == 0
           and not m.joint(j).name.startswith(("robot", "gripper"))]
    states = np.atleast_2d(states)
    return np.stack([states[:, 1 + a:1 + a + 3] for a in adr], 1)


def _teacher_solves(solve, state, fixtures, vla_execution: bool) -> bool:
    """The skill teacher from the stored start placed as an evaluation places it, through the action step; its gripper
    channel mapped to the two-valued command under the VLA's execution."""
    from screwhead.sim.gripper_servo import squeeze_command
    from screwhead.teacher.skill_teacher import SkillTeacher
    solve.place_stored(state, fixtures)
    teacher, solved = SkillTeacher(solve), False
    while solve.t < STEP_LIMIT and not solved:
        a = teacher.act(solve.snapshot())
        if vla_execution:
            a = np.concatenate([a[:6], [squeeze_command(a[6])]])
        _r, _x, _d, info = solve.step(a)
        solved = bool(info["success"])
    return solved


def _task(job: tuple) -> dict:
    task, per_task, max_draws, seed, scale, exclude, vla_execution = job
    import h5py
    import torch
    torch.set_num_threads(1)
    import mujoco
    from libero.libero import benchmark, get_libero_path

    from screwhead.sim.task_env import TaskEnv
    from screwhead.sim.task_env_place import STEP_CHECK, read_fixtures, stepping_digests
    from screwhead.sim.teacher_env import PrivilegedEnv
    from screwhead.student.libero_data import task_file
    from screwhead.student.qwen_vla import file_digest
    spec = benchmark.get_benchmark_dict()[SUITE]().get_task(task)
    draw = PrivilegedEnv(task, suite=SUITE, horizon=STEP_LIMIT, seed=seed * 100 + task,
                         **{k: v * scale for k, v in RANDOMIZER.items()})
    from screwhead.sim.sim_arm import Execution
    from screwhead.student.qwen_vla import VLA_EXECUTION
    solve = TaskEnv(SUITE, task, horizon=STEP_LIMIT, seed=0, render=False,
                    **({"execution": Execution(**VLA_EXECUTION)} if vla_execution else {}))
    m = solve.env.sim.model._model
    with h5py.File(task_file(SUITE, spec.name)) as f:
        demo_starts = np.stack([f["data"][k]["states"][0] for k in f["data"]])
    refs = np.concatenate([_free_positions(m, np.stack([np.asarray(s, float).ravel() for s in draw.init_states])),
                           _free_positions(m, demo_starts)])
    if exclude:
        from screwhead.sim.task_env_place import load_starts
        excluded = load_starts(exclude).get(task)
        if excluded is not None:
            refs = np.concatenate([refs, _free_positions(m, np.asarray(excluded["states"], float))])
    kept, counts, draws = [], collections.Counter(), 0
    while len(kept) < per_task and draws < max_draws:
        draws += 1
        try:
            draw.reset()
        except RuntimeError:
            counts["no valid layout"] += 1
            continue
        state = np.asarray(draw.env.sim.get_state().flatten(), float).copy()
        fixtures = read_fixtures(draw.env.sim.model._model)
        gaps = np.linalg.norm(_free_positions(m, state)[0][None] - refs, axis=-1).max(1)   # per reference state
        if gaps.min() < NOVELTY_M:
            counts["not novel"] += 1
            continue
        if not _teacher_solves(solve, state, fixtures, vla_execution):
            counts["teacher did not solve"] += 1
            print(f"task {task}: draw {draws}, kept {len(kept)}, teacher failed {counts['teacher did not solve']}", flush=True)
            continue
        kept.append((state, fixtures, float(gaps.min())))
        print(f"task {task}: draw {draws}, kept {len(kept)}, teacher failed {counts['teacher did not solve']}", flush=True)
    layout_reasons = dict(draw.layout.reasons) if draw.layout is not None else {}
    out = dict(task=task, draws=draws, kept=len(kept), discarded=dict(counts), randomizer_rejections=layout_reasons,
               states=[k[0] for k in kept], fixtures=[{n: (p.tolist(), q.tolist()) for n, (p, q) in k[1].items()} for k in kept],
               novelty_m=[k[2] for k in kept])
    if kept:
        solve.place_stored(kept[0][0], kept[0][1])
        bddl = os.path.join(get_libero_path("bddl_files"), spec.problem_folder, spec.bddl_file)
        out["references"] = dict(task_file_digest=file_digest(bddl), simulator=mujoco.__version__,
                                 model_fingerprint=solve.model_fingerprint(),
                                 stepping=stepping_digests(solve, kept[0][0], kept[0][1]), stepping_steps=list(STEP_CHECK))
    draw.close() if hasattr(draw, "close") else None
    solve.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-task", type=int, default=50)
    ap.add_argument("--max-draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2601, help="used for nothing else")
    ap.add_argument("--tasks", type=int, nargs="*", default=list(range(10)))
    ap.add_argument("--scale", type=float, default=1.0, help="every randomizer bound times this")
    ap.add_argument("--exclude", default=None, help="a set whose starts no kept draw may be near (a test set)")
    ap.add_argument("--vla-execution", action="store_true", help="the teacher through VLA_EXECUTION")
    ap.add_argument("--cpus", default=CPUS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cpus = [int(c) for c in args.cpus.split(",")]
    free = multiprocessing.Manager().Queue()
    for c in cpus:
        free.put(c)
    jobs = [(t, args.per_task, args.max_draws, args.seed, args.scale, args.exclude, args.vla_execution)
            for t in args.tasks]
    with ProcessPoolExecutor(len(cpus), initializer=_pin, initargs=(free,),
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(_task, jobs))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    states = np.array([s for r in results for s in r["states"]])
    tasks = np.array([r["task"] for r in results for _ in r["states"]])
    meta = [{k: v for k, v in r.items() if k not in ("states",)} for r in results]
    np.savez_compressed(args.out, states=states, tasks=tasks, meta=json.dumps(
        {"suite": SUITE, "seed": args.seed, "randomizer": {k: v * args.scale for k, v in RANDOMIZER.items()},
         "scale": args.scale, "excluded": args.exclude, "execution": "vla" if args.vla_execution else "teacher",
         "novelty_m": NOVELTY_M, "step_limit": STEP_LIMIT,
         "tasks": meta}))
    for r in results:
        failed = r["discarded"].get("teacher did not solve", 0)
        print(f"task {r['task']}: kept {r['kept']} of {r['draws']} draws; teacher solved {r['kept']}/{r['kept'] + failed}; "
              f"discarded {r['discarded']}; "
              f"randomizer rejections {r['randomizer_rejections']}; nearest LIBERO/demo start "
              f"{min(r['novelty_m']) * 100 if r['novelty_m'] else float('nan'):.1f} cm")
    print(f"{len(states)} starts -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
