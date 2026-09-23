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
SEED = 555

from screwhead.sim.contacts import robot_in_contact  # noqa: E402
from screwhead.teacher.grip_watch import GripWatch  # noqa: E402  release_gap_mm, drop_count


def _record_substeps(env) -> tuple[list, list]:
    """Grip-site positions after every physics substep, from here on, and whether the robot was
    touching anything at that substep -- DEF-smooth-motion (b) is a bar on free motion, and an arm
    pressing on a support accelerates because the world stops it."""
    m, d = env.scene.m, env.scene.d
    sim = env.env.env.sim if hasattr(env.env, "env") else env.env.sim
    site, track, contact, step = m.site_name2id("gripper0_grip_site"), [], [], sim.step

    def recorded(*a, **k):
        r = step(*a, **k)
        track.append(d.site_xpos[site].copy())
        contact.append(bool(robot_in_contact(m, d)))
        return r
    sim.step = recorded
    return track, contact


def episode(env, teacher, watch: GripWatch, track: list, contact: list) -> dict:
    env.reset()
    watch.reset()
    track.clear()
    contact.clear()
    r0, done, info = env.servo.reanchors, False, {}
    while not done:
        s = env.snapshot()
        a = teacher.act(s)
        watch.step()
        _, _, done, info = env.step(a)
    dt = float(env.scene.m.opt.timestep)
    v = np.diff(np.array(track), axis=0) / dt
    acc = np.linalg.norm(np.diff(v, axis=0) / dt, axis=1)
    jerk = np.linalg.norm(np.diff(np.diff(v, axis=0) / dt, axis=0) / dt, axis=1)
    free = ~np.array(contact[2:], dtype=bool)
    return dict(success=bool(info["success"]), steps=env.t, **watch.metrics(),
                acc_p95=round(float(np.percentile(acc, 95)), 2),
                acc_p95_free=round(float(np.percentile(acc[free], 95)), 2) if free.any() else -1.0,
                free_fraction=round(float(free.mean()), 3),
                jerk_p95=round(float(np.percentile(jerk, 1 * 95)), 1), reanchors=env.servo.reanchors - r0)


def main() -> int:
    from screwhead.sim.task_env import StartNoise, TaskEnv
    from screwhead.teacher.skill_teacher import SkillTeacher
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
        watch = GripWatch(env, teacher)
        track, contact = _record_substeps(env)
        for ep in range(args.episodes):
            m = episode(env, teacher, watch, track, contact)
            print(f"{suite} task {task} ep {ep}: {m}", flush=True)
            trials.append({"metrics": {k: m[k] for k in ("release_gap_mm", "drop_count", "acc_p95",
                                                          "acc_p95_free", "free_fraction",
                                                          "jerk_p95", "reanchors", "success")},
                           "conditions": {"suite": suite, "task": task, "episode": ep},
                           "repro": {"teacher_revision": rev, "task_suite": suite, "task": task,
                                     "episode": ep, "horizon": env.horizon}})
        env.close()
    Path(args.out).write_text(json.dumps({"trials": trials}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
