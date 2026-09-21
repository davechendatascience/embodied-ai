#!/usr/bin/env python
"""Evaluate the scripted teacher across tasks under randomisation.

Per task: success rate, steps to success, and for every failure the phase the
program was in when the episode ran out and the highest progress stage reached --
so a miss names the step of the program that has to change. Optionally saves a
video of the first failures per task.
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def run(a):
    task, cpu, episodes, seed, env_kw, video_dir, max_videos = a
    os.sched_setaffinity(0, {cpu}); os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT))
    from screwhead.sim.teacher_env import PrivilegedEnv
    from screwhead.scripted.scripted_teacher import ScriptedTeacher
    env = PrivilegedEnv(task, seed=seed, render=bool(video_dir), **env_kw)
    teacher = ScriptedTeacher(env)
    if env.layout_radius > 0:
        env.layout_check = teacher.layout_feasible
    rows, videos = [], 0
    for ep in range(episodes):
        env.reset()
        frames, done, info, peak, phases = [], False, {}, 0, collections.Counter()
        while not done:
            s = dict(env.snapshot(), **env.ref)
            act = teacher.act(s)
            phases[teacher.phase] += 1
            peak = max(peak, env.progress(s)[1])
            _, _, done, info = env.step(act)
            if video_dir:
                a_img, w_img = env.images()
                frames.append(np.concatenate([a_img[::-1], w_img[::-1]], axis=1))
        rows.append(dict(task=task, success=bool(info["success"]), steps=env.t, last_phase=teacher.phase,
                         peak_stage=int(peak), start=None if env.start_offset is None else
                         [list(np.round(env.start_offset[0], 1)), round(float(env.start_offset[1]), 1),
                          round(float(env.start_offset[2]), 1)],
                         placement_mm=float(np.linalg.norm(env.placement[:2]) * 1000), phases=dict(phases)))
        if video_dir and not info["success"] and videos < max_videos:
            import imageio.v2 as imageio
            Path(video_dir).mkdir(parents=True, exist_ok=True)
            imageio.mimsave(Path(video_dir) / f"task{task}_ep{ep}_{teacher.phase}.mp4", frames, fps=20, macro_block_size=1)
            videos += 1
    env.close()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--cpus", default="5,6,7,8,9,15,16,17,18,19")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--radius", type=float, default=0.0, help="bowl placement disc, m")
    ap.add_argument("--layout-radius", type=float, default=0.0,
                    help="object-layout randomisation (relation-preserving), m")
    ap.add_argument("--start-xy", type=float, default=0.10)
    ap.add_argument("--start-z", type=float, default=0.05)
    ap.add_argument("--start-yaw", type=float, default=30.0)
    ap.add_argument("--start-tilt", type=float, default=10.0)
    ap.add_argument("--start-null", type=float, default=0.3)
    ap.add_argument("--video", default="")
    ap.add_argument("--max-videos", type=int, default=2)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    env_kw = dict(horizon=args.horizon, radius_m=args.radius, layout_radius=args.layout_radius,
                  start_xy_m=args.start_xy, start_z_m=args.start_z,
                  start_yaw_deg=args.start_yaw, start_tilt_deg=args.start_tilt, start_null_rad=args.start_null)
    cpus = [int(c) for c in args.cpus.split(",")]
    jobs = [(t, cpus[i % len(cpus)], args.episodes, args.seed * 100 + t, env_kw, args.video, args.max_videos)
            for i, t in enumerate(args.tasks)]
    with mp.get_context("spawn").Pool(len(cpus)) as pool:
        rows = [r for chunk in pool.map(run, jobs) for r in chunk]
    total = []
    for t in args.tasks:
        R = [r for r in rows if r["task"] == t]
        wins = [r for r in R if r["success"]]
        fails = collections.Counter((r["last_phase"], r["peak_stage"]) for r in R if not r["success"])
        total += [r["success"] for r in R]
        print(f"  task {t}: {len(wins)}/{len(R)}  steps median {np.median([r['steps'] for r in wins]) if wins else float('nan'):.0f}"
              + (f"   failures (phase, peak stage): {dict(fails)}" if fails else ""), flush=True)
    print(f"ALL: {sum(total)}/{len(total)} = {np.mean(total):.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
