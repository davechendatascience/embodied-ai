#!/usr/bin/env python
"""Q1-Q5 of docs/design_optimization_teacher.md: run the search alone (no policy) on one task and
init, and report whether it reaches a settled success, where it stalls, and what it costs.

The teacher's initialization is randomized and its actions are executed exactly: --init -1 draws
a LIBERO init state at random and the --start-* flags add start-pose noise (the student's
pipeline uses xy 0.10, z 0.05, yaw 30, tilt 10, null 0.3). Layout randomization is not available
here: screwhead/scripted/layouts.py keeps each libero_spatial instruction true by name and has no
counterpart for the other suites.

  PYTHONPATH=third_party/LIBERO:. tools/search_probe.py libero_goal 8 --init 0 --steps 200

Each control step logs the search's key, the terminal order, how many sampled plans violated, and
the wall time; a step whose best plan neither settles nor improves the terminal order is a stall,
and the stall's phase (approach, touch, hold, carry, release) is logged with it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from screwhead.sim.sim_arm import Execution  # noqa: E402
from screwhead.sim.task_env import StartNoise, TaskEnv  # noqa: E402
from screwhead.teacher.search import Search, Settings  # noqa: E402
from screwhead.teacher.state import TeacherState  # noqa: E402
from screwhead.teacher.task_loss import TaskLoss  # noqa: E402
from screwhead.teacher.verdicts import Verdicts  # noqa: E402
def phase(v: Verdicts, watch) -> str:
    """Where the tool is in the task, from the state alone."""
    held = [n for n in v.moved if v.loss.terms()[0].held] if v.moved else []
    touching = [n for n in v.moved if watch.touching.get(n)]
    if v.env.success():
        return "placed"
    if held:
        return "carry"
    if touching:
        return "touch"
    return "approach"


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("suite")
    ap.add_argument("task", type=int)
    ap.add_argument("--init", type=int, default=0, help="LIBERO init state; -1 draws one at random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start-xy", type=float, default=0.0, help="start-pose noise, metres (LIBERO's own start: 0)")
    ap.add_argument("--start-z", type=float, default=0.0)
    ap.add_argument("--start-yaw", type=float, default=0.0, help="degrees")
    ap.add_argument("--start-tilt", type=float, default=0.0, help="degrees")
    ap.add_argument("--start-null", type=float, default=0.0, help="rad of IK seed noise (elbow)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=Settings.horizon)
    ap.add_argument("--segments", type=int, default=Settings.segments)
    ap.add_argument("--samples", type=int, default=Settings.samples)
    ap.add_argument("--iters", type=int, default=Settings.iters)
    ap.add_argument("--spread", type=float, default=Settings.spread)
    ap.add_argument("--key-order", default="plain", choices=["plain", "stability", "tiebreak"])
    ap.add_argument("--out", default="")
    ap.add_argument("--record", default="", help="npz to write the search's (state, action) labels to")
    ap.add_argument("--trials", default="", help="append a component-belief trial row for this episode")
    return ap


def main() -> int:
    args = _parser().parse_args()

    settings = Settings(horizon=args.horizon, segments=args.segments, samples=args.samples,
                        iters=args.iters, spread=args.spread, key_order=args.key_order)
    noise = StartNoise(xy_m=args.start_xy, z_m=args.start_z, yaw_deg=args.start_yaw, tilt_deg=args.start_tilt,
                       null_rad=args.start_null)
    te = TaskEnv(args.suite, args.task, seed=args.seed, render=False, start=noise,
                 execution=Execution(lean=True, anchor=True, scale_lead=True))
    init = None if args.init < 0 else args.init
    te.reset(init)
    loss = TaskLoss(te)
    v = Verdicts(te, loss)
    search = Search(te, v, loss, settings)
    watch, start_ref = v.watch(), v.reference()
    recorder = Recorder(args.record, te, v) if args.record else None
    previous_ref = dict(start_ref)
    rows, outcome, t0 = [], "timeout", time.perf_counter()
    for t in range(args.steps):
        tick = time.perf_counter()
        if recorder is not None:
            recorder.before(watch, start_ref)
        action, report = search.act(watch, start_ref, previous_ref)
        if recorder is not None:
            recorder.after(action, report)
        te.execute(action, substep=watch.substep)
        success = te.success()
        watch.period_end(success)
        violation = watch.pending() or v.disturbed(start_ref, previous_ref) or v.lost()
        rows.append(dict(step=t, key=[float(x) for x in report.key], terminal=round(report.terminal, 4),
                         settled_in=report.settled_in, violations=report.violations,
                         unsettled=report.unsettled, foresaw=report.foresaw[:40], foresaw_in=report.foresaw_in,
                         phase=phase(v, watch), success=success, seconds=round(time.perf_counter() - tick, 2)))
        if violation:
            outcome, rows[-1]["violation"] = f"violation: {violation}", violation
            rows[-1]["foreseen"] = report.foresaw_in == 0 and report.foresaw.split(":")[0] == violation.split(":")[0]
            break
        if success and v.settled(watch):
            outcome = "settled"
            break
        previous_ref = v.reference()
    final = watch.finish()
    summary = dict(suite=args.suite, task=args.task, init=te.init_index, seed=args.seed,
                   start_noise=noise.as_dict(), outcome=outcome, steps=len(rows),
                   final_watch=final, seconds=round(time.perf_counter() - t0, 1),
                   seconds_per_step=round((time.perf_counter() - t0) / max(len(rows), 1), 2),
                   terminal_first=rows[0]["terminal"] if rows else None,
                   terminal_last=rows[-1]["terminal"] if rows else None,
                   terminal_best=min((r["terminal"] for r in rows), default=None),
                   foreseen=rows[-1].get("foreseen"),
                   steps_expecting_a_violation=sum(r["foresaw_in"] is not None for r in rows),
                   settings=vars(settings))
    print(json.dumps(summary))
    if args.trials:
        write_trial(args.trials, summary)
    if args.out:
        Path(args.out).write_text("\n".join(json.dumps(r) for r in [summary, *rows]))
    if recorder is not None:
        recorder.save(args, noise, settings, outcome)
    return 0


def write_trial(path: str, summary: dict) -> None:
    """One episode as component-belief reads it: did the search reach a settled success, in how
    many control steps, and under which task and initialization. Rows accumulate, so a sweep
    writing to one path leaves the trials of every episode it ran."""
    rows = json.loads(Path(path).read_text()) if Path(path).exists() else []
    rows.append({
        "metrics": {"settled": summary["outcome"] == "settled", "steps": summary["steps"],
                    "seconds_per_step": summary["seconds_per_step"],
                    "terminal_end": summary["terminal_last"]},
        "conditions": {"task_suite": summary["suite"], "task": summary["task"],
                       "init": summary["init"], "start_noise": summary["start_noise"]["xy_m"] > 0,
                       "outcome": summary["outcome"].split(":")[0]},
    })
    Path(path).write_text(json.dumps(rows, indent=1))


class Recorder:
    """The search's solution as pi_theta's training signal: the state it saw and the plan it chose
    at every control step (BRN-teacher-policy)."""

    def __init__(self, path: str, te, verdicts):
        self.path, self.te = path, te
        self.state = TeacherState(te, verdicts)
        self.states, self.actions, self.plans, self.levels = [], [], [], []

    def before(self, watch, start_reference) -> None:
        self.states.append(self.state.vector(watch, start_reference))

    def after(self, action, report) -> None:
        self.actions.append(action.astype(np.float32))
        self.plans.append(report.best.twist.astype(np.float32))
        self.levels.append(report.best.level.astype(np.int64))

    def save(self, args, noise, settings, outcome: str) -> None:
        np.savez_compressed(self.path, state=np.asarray(self.states), action=np.asarray(self.actions),
                            plan=np.asarray(self.plans), level=np.asarray(self.levels),
                            settled=outcome == "settled", suite=args.suite, task=args.task,
                            init=self.te.init_index, seed=args.seed, start_noise=json.dumps(noise.as_dict()),
                            settings=json.dumps(vars(settings)))
        print(json.dumps(dict(recorded=self.path, steps=len(self.states), state_dim=int(self.state.dim),
                              settled=outcome == "settled")))


if __name__ == "__main__":
    sys.exit(main())
