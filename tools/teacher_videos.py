#!/usr/bin/env python
"""Record the skill teacher on every task: one episode per task, success or failure.

  teacher_videos.py                                   # all 130 tasks -> videos/teacher_skill/
  teacher_videos.py --suites libero_goal --tasks 3 7 --episode 2
  teacher_videos.py --init-order --items libero_10:6:7 libero_90:5:0   # LIBERO's protocol: initial state 7, 0

tools/skill_eval.py keeps only failures; this keeps every episode, named by its outcome, so
the folder shows what the teacher does on each task. The episode is the one skill_eval runs
under the same seed (seed x 100 + task, episode index), with the same runner, so a video is
the trajectory the sweeps scored.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

PERF_CORES = "5,6,7,8,9,15,16,17,18,19"
TASKS = {"libero_spatial": 10, "libero_object": 10, "libero_goal": 10, "libero_10": 10, "libero_90": 90}
HORIZON = {"libero_10": 800}          # the sweeps' horizons: 800 on libero_10, 600 elsewhere


def _record(item: tuple) -> str:
    suite, task, episode, seed, px, out, init_order, cpu = item
    os.sched_setaffinity(0, {cpu})
    import imageio.v2 as imageio

    from skill_eval import VIDEO_FPS, Job, _run_episode
    from screwhead.sim.task_env import TaskEnv
    from screwhead.teacher.skill_teacher import SkillTeacher
    job = Job(suite, task, 1, seed * 100 + task, cpu, episode, HORIZON.get(suite, 600), False, {},
              str(out), 1, px, init_order=init_order)
    env = TaskEnv(suite, task, horizon=job.horizon, seed=job.seed, render=True)
    try:
        teacher = SkillTeacher(env)
        for _ in range(0 if init_order else episode):     # the stream's earlier episodes, unsimulated
            env.skip_episode()
        row, frames = _run_episode(env, teacher, job, 0, record=True)
    except Exception as e:  # noqa: BLE001  one task's crash is reported, not the batch's
        return f"{suite}[{task}] crash: {type(e).__name__}: {e}"
    finally:
        env.close()
    tag = "ok" if row["success"] else "fail"
    name = f"i{episode}" if init_order else f"ep{episode}"
    path = out / suite / f"{suite}_t{task}_{name}_{tag}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=VIDEO_FPS, macro_block_size=1)
    return f"{suite}[{task}] {tag:4s} {row['steps']:3d} steps  {path.relative_to(ROOT)}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", nargs="*", default=list(TASKS))
    ap.add_argument("--tasks", type=int, nargs="*", default=None, help="default: every task of each suite")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--seed", type=int, default=555)
    ap.add_argument("--px", type=int, default=256, help="each camera's side, agentview beside the wrist")
    ap.add_argument("--out", default="videos/teacher_skill")
    ap.add_argument("--cpus", default=PERF_CORES)
    ap.add_argument("--init-order", action="store_true",
                    help="LIBERO's protocol, as skill_eval.py --init-order: episode e starts from initial state e")
    ap.add_argument("--items", nargs="*", default=None, metavar="SUITE:TASK:EPISODE",
                    help="these episodes instead of --suites/--tasks/--episode")
    args = ap.parse_args()
    cpus = [int(c) for c in args.cpus.split(",")]
    out = (ROOT / args.out).resolve()
    picked = ([(s, int(t), int(e)) for s, t, e in (i.split(":") for i in args.items)] if args.items is not None
              else [(s, t, args.episode) for s in args.suites
                    for t in (args.tasks if args.tasks is not None else range(TASKS[s]))])
    items = [(s, t, e, args.seed, args.px, out, args.init_order) for s, t, e in picked]
    # one LIBERO process per core, each pinned to its own (maxtasksperchild: a fresh process per task)
    cores = mp.Manager().Queue()
    for c in cpus:
        cores.put(c)
    with mp.get_context("spawn").Pool(len(cpus), maxtasksperchild=1) as pool:
        jobs = [pool.apply_async(_pinned, (it, cores)) for it in items]
        for j in jobs:
            print(j.get(), flush=True)
    return 0


def _pinned(item: tuple, cores) -> str:
    cpu = cores.get()
    try:
        return _record((*item, cpu))
    finally:
        cores.put(cpu)


if __name__ == "__main__":
    raise SystemExit(main())
