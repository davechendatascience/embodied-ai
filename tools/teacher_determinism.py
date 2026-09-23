#!/usr/bin/env python
"""Does the same seed give the same teacher episode? component-belief's TST-teacher-deterministic.

  teacher_determinism.py --out $OUT

Runs a fixed set of seeded episodes twice through tools/skill_eval.py -- once in one process per
task, once with every episode first in a process of its own (--interleave) -- and compares the
per-episode phase timelines. One trial per episode, metric `reproducible`. Run twice in the same
order, an episode that depends on what its process ran before reproduces anyway: robosuite's
finger target carried over between episodes, and 2 of 50 libero_goal 3 episodes changed with the
order they ran in while this test passed. The set spans the teacher's paths: a face grasp, a rim grasp beside
fixtures, a drawer, the stove knob and a carry to a plate. Any difference means a
comparison between two teacher versions would be measuring noise (it was, until LIBERO's
fixture re-sampling was seeded).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from teacher_report import teacher_revision  # noqa: E402

CASES = [("libero_object", "0"), ("libero_spatial", "6"), ("libero_goal", "0 7")]
EPISODES = 4          # 4 tasks x 4 episodes = CTR-teacher-deterministic's n_min of 16
HORIZON = 500         # the reliability report's horizon: reproducible over the whole episode it scores
CPUS = "5,6,7,8,9,15,16,17,18,19"


def run_once(suite: str, tasks: str, out: Path, interleave: int = 1) -> dict:
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONPATH="third_party/LIBERO:.", MUJOCO_GL="egl")
    cmd = [".venv-libero/bin/python", "tools/skill_eval.py", "--suite", suite, "--tasks", *tasks.split(),
           "--episodes", str(EPISODES), "--horizon", str(HORIZON), "--cpus", CPUS, "--trials", str(out),
           "--interleave", str(interleave)]
    subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, check=False)
    trials = json.loads(out.read_text())["trials"] if out.exists() else []
    return {(str(t["conditions"]["task"]), int(t["detail"]["episode"])): t for t in trials}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rev = teacher_revision()
    trials = []
    with tempfile.TemporaryDirectory() as tmp:
        for suite, tasks in CASES:
            a = run_once(suite, tasks, Path(tmp) / f"{suite}_a.json")
            b = run_once(suite, tasks, Path(tmp) / f"{suite}_b.json", interleave=EPISODES)
            # every expected episode is a trial: one a crashed run lost is not reproducible
            for key in [(task, ep) for task in tasks.split() for ep in range(EPISODES)]:
                ta, tb = a.get(key), b.get(key)
                same = (ta is not None and tb is not None
                        and ta["detail"]["timeline"] == tb["detail"]["timeline"]
                        and ta["detail"]["steps"] == tb["detail"]["steps"])
                print(f"{suite} task {key[0]} ep {key[1]}: {'same' if same else 'DIFFERENT'}")
                trials.append({"metrics": {"reproducible": bool(same)},
                               "conditions": {"suite": suite, "task": key[0], "episode": key[1]},
                               "repro": {"teacher_revision": rev, "task_suite": suite, "task": key[0]}})
    Path(args.out).write_text(json.dumps({"trials": trials}))
    print(f"{sum(t['metrics']['reproducible'] for t in trials)}/{len(trials)} reproducible -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
