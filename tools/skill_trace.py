#!/usr/bin/env python
"""Trace one episode of the geometry-driven teacher, for diagnosing a failure.

  skill_trace.py --suite libero_goal --task 0 --every 20
  skill_trace.py --suite libero_spatial --task 4 --episodes 3 --changes

Prints what the teacher is deciding and what the simulator says about it: the phase, the
tool pose, the gripper, each goal conjunct's truth, and -- for an articulated goal -- the
joint's own coordinate against the threshold LIBERO scores. `--changes` prints only the
steps where the phase changes, which is usually where the failure is.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--seed", type=int, default=555)
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--changes", action="store_true")
    ap.add_argument("--cpu", type=int, default=5)
    ap.add_argument("--video", default="")
    args = ap.parse_args()
    os.sched_setaffinity(0, {args.cpu})

    from screwhead.teacher.skill_teacher import SkillTeacher
    from screwhead.sim.task_env import TaskEnv
    env = TaskEnv(args.suite, args.task, horizon=args.horizon,
                  seed=args.seed * 100 + args.task, render=True)
    teacher = SkillTeacher(env)
    sc = env.scene
    print(f"{args.suite}[{args.task}] {env.language!r}")
    print("  goals", env.task_spec.goals)
    print("  plan ", [(s.skill, s.obj or s.region, s.mode or "") for s in env.task_spec.plan])

    for ep in range(args.episodes):
        env.reset()
        done, info, last, frames = False, {}, None, []
        while not done:
            s = env.snapshot()
            a = teacher.act(s)
            show = (env.t % args.every == 0) if not args.changes else (teacher.phase != last)
            if show:
                bits = []
                for g in env.task_spec.goals:
                    ok = teacher.satisfied(g)
                    if g[0].lower() in ("open", "close", "turnon", "turnoff"):
                        art = sc.articulation(g[1])
                        bits.append(f"{g[0]}({g[1][:18]})={ok} q={art['qpos']:+.3f}"
                                    f" th={art['thresholds'].get(g[0].lower().replace('turn', ''), '')}")
                    else:
                        q = sc.body_pose(g[1])[1]
                        bits.append(f"{g[0]}({g[1][:14]})={ok} p={np.round(q, 3)}")
                print(f"  t{env.t:4d} {teacher.phase:22s} tool {np.round(s['p_tool'], 3)} "
                      f"ap {s['aperture'] * 1000:5.1f} | " + " ".join(bits))
            last = teacher.phase
            if args.video:
                ag, wr = env.images()
                frames.append(np.concatenate([ag[::-1], wr[::-1]], axis=1))
            _, _, done, info = env.step(a)
        print(f"  ep {ep}: success {info['success']} steps {env.t} phase {teacher.phase}")
        if args.video:
            import imageio.v2 as imageio
            Path(args.video).parent.mkdir(parents=True, exist_ok=True)
            out = f"{args.video}_{args.suite}_t{args.task}_ep{ep}.mp4"
            imageio.mimsave(out, frames, fps=20, macro_block_size=1)
            print("  ->", out)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
