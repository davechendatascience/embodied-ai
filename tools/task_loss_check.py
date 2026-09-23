#!/usr/bin/env python
"""Check screwhead/teacher/task_loss.py against LIBERO's scorer on teacher episodes.

  task_loss_check.py --episodes 4 --out runs/task_loss_check.json

Drives the skill teacher and evaluates every goal conjunct's term at every step:
  - agreement: steps where LIBERO accepts a conjunct but its distance is positive, and
    steps where it rejects one whose distance is zero (the geometry has drifted from the
    scorer's where either is common);
  - the zero set: after the last step, zero loss iff LIBERO scored the episode a success;
  - the landscape: the loss and the reach at the start, the end and at their lowest, and
    how often the dense potential (distance + reach) rises, per episode.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from multiprocessing.connection import wait
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PERF_CORES = (5, 6, 7, 8, 9, 15, 16, 17, 18, 19)
SUITES = ("libero_object", "libero_spatial", "libero_goal")
RISE = 0.01             # m: a rise of the dense potential this large counts as a setback


def _episode(env, teacher, loss) -> dict:
    env.reset()
    rows, done, info = [], False, {}
    while not done:
        terms = loss.terms()
        rows.append([(t.satisfied, t.distance, t.reach, t.held) for t in terms])
        _, _, done, info = env.step(teacher.act(env.snapshot()))
    final = loss.terms()
    rows.append([(t.satisfied, t.distance, t.reach, t.held) for t in final])
    sat = np.array([[c[0] for c in r] for r in rows])
    dist = np.array([[c[1] for c in r] for r in rows])
    reach = np.array([[c[2] for c in r] for r in rows])
    held = np.array([[c[3] for c in r] for r in rows])
    gated = np.where(sat, 0.0, dist).sum(1)
    potential = gated + reach.sum(1)
    rises = np.diff(potential)
    return dict(success=bool(info["success"]), steps=env.t,
                final_zero=bool(sum(t.loss for t in final) == 0.0),
                sat_but_far=int((sat & (dist > 0)).sum()), unsat_but_zero=int((~sat & (dist == 0)).sum()),
                sat_steps=int(sat.sum()), conjunct_steps=int(sat.size),
                sat_far_max=float(dist[sat].max()) if sat.any() else 0.0,
                loss_start=float(gated[0]), loss_end=float(gated[-1]), loss_min=float(gated.min()),
                reach_start=float(reach.sum(1)[0]), reach_end=float(reach.sum(1)[-1]),
                held_end=bool(held[-1].all()), setbacks=int((rises > RISE).sum()),
                setback_total=float(rises[rises > RISE].sum()), phase=teacher.phase)


def _worker(remote, suite: str, task: int, episodes: int, seed: int, cpu: int) -> None:
    os.sched_setaffinity(0, {cpu})
    os.environ["OMP_NUM_THREADS"] = "1"
    sys.path.insert(0, str(ROOT))
    import torch
    torch.set_num_threads(1)
    from screwhead.sim.task_env import TaskEnv
    from screwhead.teacher.skill_teacher import SkillTeacher
    from screwhead.teacher.task_loss import TaskLoss
    env = TaskEnv(suite, task, horizon=500, seed=seed, render=False)
    teacher, loss = SkillTeacher(env), TaskLoss(env)
    for ep in range(episodes):
        try:
            r = _episode(env, teacher, loss)
        except Exception as e:  # noqa: BLE001  a worker boundary: report, keep going
            r = dict(error=f"{type(e).__name__}: {e}")
        remote.send(dict(suite=suite, task=task, episode=ep, goals=[list(g) for g in loss.goals], **r))
    remote.send(None)
    env.close()


def trial_row(r: dict) -> dict:
    """One episode as component-belief reads it: the agreement counts the contract accepts on,
    and the conditions it is sliced by."""
    return {
        "metrics": {
            "zero_iff_success": bool(r["final_zero"] == r["success"]),
            "sat_but_far": int(r["sat_but_far"]),
            "unsat_but_zero": int(r["unsat_but_zero"]),
            "sat_far_max": float(r["sat_far_max"]),
        },
        "conditions": {"task_suite": r["suite"], "task": r["task"], "episode": r["episode"],
                       "success": bool(r["success"])},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", nargs="*", default=list(SUITES))
    ap.add_argument("--tasks", type=int, nargs="*", default=list(range(10)))
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=555)
    ap.add_argument("--out", default="runs/task_loss_check.json")
    ap.add_argument("--trials", default="", help="write component-belief trial rows here (one per episode)")
    args = ap.parse_args()
    ctx = mp.get_context("spawn")
    jobs = [(s, t) for s in args.suites for t in args.tasks]
    rows, pending = [], list(jobs)
    running: dict = {}
    while pending or running:
        while pending and len(running) < len(PERF_CORES):
            s, t = pending.pop(0)
            cpu = next(c for c in PERF_CORES if c not in {v[1] for v in running.values()})
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, s, t, args.episodes, args.seed * 100 + t, cpu), daemon=True)
            p.start()
            b.close()
            running[a] = (p, cpu)
        for conn in wait(list(running)):
            try:
                r = conn.recv()
            except EOFError:
                r = None
            if r is None:
                running.pop(conn)[0].join(timeout=10)
                continue
            rows.append(r)
            print(json.dumps({k: r[k] for k in r if k != "goals"}), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=1))
    if args.trials:
        Path(args.trials).write_text(json.dumps([trial_row(r) for r in rows], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
