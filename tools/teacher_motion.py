#!/usr/bin/env python
"""How gently the teacher sets objects down and how smoothly it moves. component-belief's
TST-teacher-motion.

  teacher_motion.py --out $OUT [--episodes 5]

One trial per episode, over tasks that carry and place (a bowl on a plate, the bowl into a
drawer):
  release_gap_mm   largest gap between an object's bottom and the surface under it when
                   the teacher let go on purpose (a release phase within RELEASE_WINDOW steps);
                   -1 if it never did (component-belief's rules cannot compare a missing value)
  drop_count       times the grip let an object go anywhere else, more than DROP_GAP up
  acc_p95, jerk_p95  tool acceleration (m/s^2) and jerk (m/s^3) at PHYSICS-SUBSTEP resolution
                   (the grip site, every 2 ms). At 20 Hz the same episode read p95 3.9 against
                   12.1 at substeps: the joint target stepped every 50 ms and the arm lunged
                   and coasted inside each period, which a 20 Hz difference averages away
  reanchors        times the servo abandoned its pose reference this episode
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

CASES = [("libero_spatial", 0), ("libero_spatial", 6), ("libero_spatial", 8), ("libero_goal", 3)]
HORIZON = {"libero_goal": 800}      # goal 3 needs four skills in sequence
DEFAULT_HORIZON = 500
RELEASE_WINDOW = 10                 # steps: a let-go this soon after a release phase was meant
DROP_GAP = 0.010                    # m: an unmeant let-go below this is a set-down, not a drop
SEED = 555


def _support_gap(env, planner, obj: str) -> tuple[float, float]:
    """(gap from the object's bottom to the surface straight below it, object centre z)."""
    box = env.scene.object_box(obj)
    ext = np.abs(box.R) @ box.half
    c = box.world_centre
    bottom = c[2] - ext[2]
    _g, dist = planner._ray(np.array([c[0], c[1], bottom - 0.001]), np.array([0.0, 0.0, -1.0]),
                            env.scene.body_id(obj))
    return (dist + 0.001 if dist >= 0 else float("nan")), float(c[2])


def _record_substeps(env) -> list:
    """Grip-site positions after every physics substep, from here on."""
    m, d = env.scene.m, env.scene.d
    sim = env.env.env.sim if hasattr(env.env, "env") else env.env.sim
    site, track, step = m.site_name2id("gripper0_grip_site"), [], sim.step

    def recorded(*a, **k):
        r = step(*a, **k)
        track.append(d.site_xpos[site].copy())
        return r
    sim.step = recorded
    return track


def episode(env, teacher, objs: list[str], track: list) -> dict:
    sk = teacher.skills
    env.reset()
    track.clear()
    r0, done, info = env.servo.reanchors, False, {}
    held, last_release, gaps, drops = dict.fromkeys(objs, False), -10**6, [], 0
    while not done:
        s = env.snapshot()
        a = teacher.act(s)
        if teacher.phase.endswith("release"):
            last_release = env.t
        for o in objs:
            h = sk.held(o)
            if held[o] and not h:
                gap, _z = _support_gap(env, sk.planner, o)
                if env.t - last_release <= RELEASE_WINDOW:
                    gaps.append(gap)
                elif gap > DROP_GAP:
                    drops += 1
            held[o] = h
        _, _, done, info = env.step(a)
    dt = float(env.scene.m.opt.timestep)
    v = np.diff(np.array(track), axis=0) / dt
    acc = np.diff(v, axis=0) / dt
    jerk = np.linalg.norm(np.diff(acc, axis=0) / dt, axis=1)
    return dict(success=bool(info["success"]), steps=env.t,
                release_gap_mm=round(1000 * max(gaps), 1) if gaps else -1.0, releases=len(gaps),
                drop_count=drops, acc_p95=round(float(np.percentile(np.linalg.norm(acc, axis=1), 95)), 2),
                jerk_p95=round(float(np.percentile(jerk, 95)), 1), reanchors=env.servo.reanchors - r0)


def main() -> int:
    from screwhead.skill_teacher import SkillTeacher
    from screwhead.task_env import StartNoise, TaskEnv
    from teacher_report import teacher_revision
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=5)
    args = ap.parse_args()
    rev, trials = teacher_revision(), []
    for suite, task in CASES:
        env = TaskEnv(suite, task, horizon=HORIZON.get(suite, DEFAULT_HORIZON), seed=SEED * 100 + task,
                      render=False, start=StartNoise())
        teacher = SkillTeacher(env)
        objs = sorted({st.obj for st in teacher.plan if st.obj})
        track = _record_substeps(env)
        for ep in range(args.episodes):
            m = episode(env, teacher, objs, track)
            print(f"{suite} task {task} ep {ep}: {m}", flush=True)
            trials.append({"metrics": {k: m[k] for k in ("release_gap_mm", "drop_count", "acc_p95",
                                                          "jerk_p95", "reanchors", "success")},
                           "conditions": {"suite": suite, "task": task, "episode": ep},
                           "repro": {"teacher_revision": rev, "task_suite": suite, "task": task,
                                     "episode": ep, "horizon": env.horizon}})
        env.close()
    Path(args.out).write_text(json.dumps({"trials": trials}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
