#!/usr/bin/env python
"""Run the geometry-driven teacher (screwhead/skill_teacher.py) on LIBERO tasks.

  skill_eval.py --suite libero_object --episodes 5
  skill_eval.py --suite libero_object --tasks 0 1 --episodes 3 --video videos/skill

Per-task success and the phase each failure ended in, so a missing skill is visible as a
phase rather than a number. Trials go to --trials for the ledger (CTR-skill-teacher-solves-unseen).
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _worker(remote, suite, task, episodes, seed, cpu, kw, video_dir, max_videos):
    os.sched_setaffinity(0, {cpu})
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT))
    from screwhead.skill_teacher import SkillTeacher
    from screwhead.task_env import TaskEnv
    env = TaskEnv(suite, task, seed=seed, render=True, **kw)
    teacher = SkillTeacher(env)
    rows, videos = [], 0
    for ep in range(episodes):
        env.reset()
        frames, done, info = [], False, {}
        phases = collections.Counter()
        missing = ""
        while not done:
            s = env.snapshot()
            try:
                a = teacher.act(s)
            except NotImplementedError as e:      # a skill this suite needs and we lack:
                missing = str(e).split("(")[0].strip()   # report it, do not kill the worker
                teacher.phase = f"unimplemented:{missing}"
                a = np.zeros(7)
            phases[teacher.phase] += 1
            if video_dir and videos < max_videos:
                ag, wr = env.images()
                frames.append(np.concatenate([ag[::-1], wr[::-1]], axis=1))
            _, _, done, info = env.step(a)
        rows.append(dict(task=task, episode=ep, success=bool(info["success"]), steps=env.t,
                         last_phase=teacher.phase, step_index=teacher.step_index,
                         phases=dict(phases), language=env.language))
        if video_dir and frames and videos < max_videos and not info["success"]:
            import imageio.v2 as imageio
            Path(video_dir).mkdir(parents=True, exist_ok=True)
            imageio.mimsave(Path(video_dir) / f"{suite}_t{task}_ep{ep}_fail.mp4", frames, fps=20,
                            macro_block_size=1)
            videos += 1
        remote.send(rows[-1])
    remote.send(None)
    env.close()
    remote.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--tasks", type=int, nargs="*", default=None)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=600)
    ap.add_argument("--start-xy", type=float, default=0.0)
    ap.add_argument("--start-z", type=float, default=0.0)
    ap.add_argument("--start-yaw", type=float, default=0.0)
    ap.add_argument("--start-tilt", type=float, default=0.0)
    ap.add_argument("--start-null", type=float, default=0.0)
    ap.add_argument("--cpus", default="5,6,7,8,9,15,16,17,18,19")
    ap.add_argument("--seed", type=int, default=555)
    ap.add_argument("--video", default="")
    ap.add_argument("--max-videos", type=int, default=1)
    ap.add_argument("--trials", default="")
    args = ap.parse_args()

    tasks = args.tasks if args.tasks is not None else list(range(10))
    cpus = [int(c) for c in args.cpus.split(",")]
    kw = dict(horizon=args.horizon, gripper_mode="target", start_xy_m=args.start_xy,
              start_z_m=args.start_z, start_yaw_deg=args.start_yaw, start_tilt_deg=args.start_tilt,
              start_null_rad=args.start_null)
    ctx = mp.get_context("spawn")
    procs, remotes = [], []
    for i, t in enumerate(tasks):
        a, b = ctx.Pipe()
        p = ctx.Process(target=_worker, args=(b, args.suite, t, args.episodes, args.seed * 100 + t,
                                              cpus[i % len(cpus)], kw, args.video, args.max_videos),
                        daemon=True)
        p.start(); b.close(); procs.append(p); remotes.append(a)
    rows, open_pipes, t0 = [], set(range(len(remotes))), time.time()
    while open_pipes:
        for i in list(open_pipes):
            try:
                r = remotes[i].recv()
            except EOFError:
                open_pipes.discard(i); continue
            if r is None:
                open_pipes.discard(i); continue
            rows.append(r)
            print(f"  {args.suite} task {r['task']} ep {r['episode']}: "
                  f"{'ok' if r['success'] else 'fail'} steps {r['steps']} last {r['last_phase']}", flush=True)
    for p in procs:
        p.join(timeout=10)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["task"]].append(r)
    print()
    for t in sorted(by):
        ok = sum(r["success"] for r in by[t])
        fails = collections.Counter(r["last_phase"] for r in by[t] if not r["success"])
        print(f"  task {t}: {ok}/{len(by[t])}  {by[t][0]['language'][:54]!r}"
              + (f"  failures: {dict(fails)}" if fails else ""))
    ok = sum(r["success"] for r in rows)
    print(f"{args.suite}: {ok}/{len(rows)} = {ok / max(len(rows), 1):.2f}   ({time.time() - t0:.0f}s)")
    if args.trials:
        Path(args.trials).parent.mkdir(parents=True, exist_ok=True)
        Path(args.trials).write_text(json.dumps({"trials": [
            {"metrics": {"success": r["success"]},
             "conditions": {"task": r["task"], "suite": args.suite, "last_phase": r["last_phase"]},
             "repro": {"seed": args.seed * 100 + r["task"], "task": r["task"], "task_suite": args.suite,
                       "teacher_revision": "skill_teacher"}} for r in rows]}, indent=1))
        print("->", args.trials)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
