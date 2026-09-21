#!/usr/bin/env python
"""Is the teacher reliable enough to collect data with, and where is it not?

  teacher_report.py --suites libero_object libero_spatial libero_goal --episodes 20
  teacher_report.py --from runs/evidence            # re-read trials already collected
  teacher_report.py --episodes 20 --ingest          # and record them in component-belief

A sweep prints successes per task, which at five episodes is a coin-flip away from
anything. This prints, per task, the success rate with a 94% interval and the MECHANISM
of its failures, and classifies the task by the lower bound -- because what matters for
collecting demonstrations is not the point estimate, it is whether the task is reliable
enough that the data is not mostly failures.

  reliable   lo >= 0.90     collect from it
  marginal   lo >= 0.50     collect, but the yield is poor and the successes may be biased
  broken     otherwise      fix before collecting

A perfect record needs ~32 episodes before its lower bound clears 0.90 (20/20 is only
[0.85, 1.00]), so use --episodes 20 to find what is broken and 40 to certify.

Mechanisms come from tools/skill_eval.py: never-reached-grasp, blocked-reaching,
reached-but-no-grip, held-but-not-delivered, delivered-but-unscored, unimplemented.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PERF = "5,6,7,8,9,15,16,17,18,19"
BELIEF_REPO = Path.home() / "Documents/GitHub/Theoretically_Driven_LLM_Planning"
# the code whose behaviour a reliability number describes: change any of it and old
# trials stop being comparable (the ledger's compatibility key reads teacher_revision)
TEACHER_CODE = ["screwhead/skills.py", "screwhead/skill_teacher.py", "screwhead/scene.py",
                "screwhead/task_env.py", "screwhead/task_spec.py", "screwhead/servo.py",
                "screwhead/gripper_servo.py"]


def teacher_revision() -> str:
    import hashlib
    h = hashlib.sha1()
    for f in TEACHER_CODE:
        fp = ROOT / f
        h.update(fp.read_bytes() if fp.exists() else b"")
    return f"skill_teacher:{h.hexdigest()[:10]}"


def ingest(trials: list[dict], artifact: str) -> str:
    """Hand the trials to component-belief, run from ITS environment, not this one."""
    import tempfile
    recs = [{"contract_id": "CTR-teacher-reliable", "test_id": "TST-teacher-reliability",
             "outcome": "pass", "metrics": {"success": bool(t["metrics"]["success"])},
             "conditions": t["conditions"],
             "repro": t["repro"]} for t in trials]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(recs, f)
    code = ("import json,sys; sys.path.insert(0,'src'); from component_belief import server; "
            "fn=getattr(server.ingest,'fn',server.ingest); "
            f"print(fn(records=json.load(open({f.name!r})), source='tools/teacher_report.py', "
            f"artifact_uri={artifact!r}))")
    env = {"BELIEF_PROJECT_ROOT": str(ROOT), "BELIEF_ACTOR": "agent", "PATH": "/usr/bin:/bin:"
           + str(Path.home() / ".local/bin"), "HOME": str(Path.home()), "PYTHONDONTWRITEBYTECODE": "1"}
    out = subprocess.run(["uv", "run", "python", "-c", code], cwd=BELIEF_REPO, env=env,
                         capture_output=True, text=True)
    return (out.stdout + out.stderr).strip()


SUITE_TASKS = {"libero_object": 10, "libero_spatial": 10, "libero_goal": 10,
               "libero_10": 10, "libero_90": 90}


def interval(k: int, n: int, z: float = 1.881) -> tuple[float, float]:
    """Wilson interval, 94% by default -- the width the ledger's contracts are written to."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - r) / d), min(1.0, (c + r) / d)


def grade(lo: float) -> str:
    return "reliable" if lo >= 0.90 else ("marginal" if lo >= 0.50 else "broken")


def collect(suite: str, episodes: int, horizon: int, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [".venv-libero/bin/python", "tools/skill_eval.py", "--suite", suite,
           "--episodes", str(episodes), "--horizon", str(horizon), "--cpus", PERF,
           "--trials", str(out)]
    env = {"HF_HUB_OFFLINE": "1", "PYTHONPATH": "third_party/LIBERO:.", "MUJOCO_GL": "egl",
           "PATH": "/usr/bin:/bin", "HOME": str(Path.home())}
    log = out.with_suffix(".log")
    rev = teacher_revision()                   # the code that RAN, stamped before it runs
    with log.open("w") as f:
        subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, check=False)
    if out.exists():
        data = json.loads(out.read_text())
        for t in data["trials"]:
            t["repro"]["teacher_revision"] = rev
        out.write_text(json.dumps(data, indent=1))
    return out


def report(trials: list[dict]) -> list[dict]:
    by = collections.defaultdict(list)
    for t in trials:
        c = t["conditions"]
        by[(c["suite"], c["task"])].append(t)
    rows = []
    for (suite, task), ts in sorted(by.items()):
        k = sum(bool(t["metrics"]["success"]) for t in ts)
        lo, hi = interval(k, len(ts))
        mech = collections.Counter(t["conditions"].get("mechanism", "") for t in ts
                                   if not t["metrics"]["success"])
        rows.append(dict(suite=suite, task=task, k=k, n=len(ts), lo=lo, hi=hi,
                         grade=grade(lo), mechanisms=dict(mech)))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", nargs="*", default=["libero_object", "libero_spatial", "libero_goal"])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=500)
    ap.add_argument("--evidence", default="runs/evidence/reliability")
    ap.add_argument("--from", dest="reuse", default="", help="read existing trials, do not run")
    ap.add_argument("--ingest", action="store_true")
    args = ap.parse_args()

    trials: list[dict] = []
    for suite in args.suites:
        path = Path(args.reuse or args.evidence) / f"{suite}.trials.json"
        if not args.reuse:
            print(f"[run] {suite}: {args.episodes} episodes x {SUITE_TASKS.get(suite, 10)} tasks",
                  flush=True)
            collect(suite, args.episodes, args.horizon, ROOT / path)
        p = ROOT / path
        if not p.exists():
            print(f"  (no trials at {path})")
            continue
        trials += json.loads(p.read_text())["trials"]

    rows = report(trials)
    print(f"\n{'task':28s} {'rate':>9s} {'94% interval':>15s}  {'grade':9s} mechanisms")
    for r in rows:
        mech = ", ".join(f"{k} x{v}" for k, v in sorted(r["mechanisms"].items(), key=lambda x: -x[1]))
        print(f"{r['suite'][7:]:>8s}[{r['task']:2d}]{'':14s}"[:28]
              + f" {r['k']:3d}/{r['n']:<5d} [{r['lo']:.2f}, {r['hi']:.2f}]  {r['grade']:9s} {mech}")
    by_grade = collections.Counter(r["grade"] for r in rows)
    worst = [r for r in rows if r["grade"] != "reliable"]
    n = sum(r["n"] for r in rows)
    k = sum(r["k"] for r in rows)
    lo, hi = interval(k, n)
    print(f"\noverall {k}/{n} = {k / max(n, 1):.3f}  [{lo:.2f}, {hi:.2f}]   "
          + "  ".join(f"{g} {by_grade[g]}" for g in ("reliable", "marginal", "broken")))
    if worst:
        print("\nnot reliable, worst first:")
        for r in sorted(worst, key=lambda r: r["lo"]):
            print(f"  {r['suite']}[{r['task']}] {r['k']}/{r['n']} lo {r['lo']:.2f}: "
                  + ", ".join(f"{m} x{c}" for m, c in sorted(r["mechanisms"].items(),
                                                             key=lambda x: -x[1])))
    agg = collections.Counter()
    for r in rows:
        agg.update(r["mechanisms"])
    if agg:
        print("\nfailures by mechanism: " + ", ".join(f"{m} {c}" for m, c in agg.most_common()))

    if args.ingest:
        revs = sorted({t["repro"].get("teacher_revision", "?") for t in trials})
        print(f"\ningest ({', '.join(revs)}):")
        print(ingest(trials, str(ROOT / (args.reuse or args.evidence))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
