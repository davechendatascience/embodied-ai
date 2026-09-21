#!/usr/bin/env python
"""Is the teacher reliable enough to collect data with, and where is it not?

  teacher_report.py --suites libero_object libero_spatial libero_goal --episodes 20
  teacher_report.py --from runs/evidence            # re-read trials already collected
  teacher_report.py --episodes 20 --out $OUT        # as component-belief's TST-teacher-reliability

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
TEACHER_CODE = ["screwhead/skills.py", "screwhead/skill_teacher.py", "screwhead/grasp_planner.py",
                "screwhead/reach.py", "screwhead/frames.py", "screwhead/contacts.py", "screwhead/scene.py",
                "screwhead/task_env.py", "screwhead/sim_arm.py", "screwhead/task_spec.py",
                "screwhead/servo.py", "screwhead/gripper_servo.py", "screwhead/kin_np.py",
                "screwhead/episode_log.py"]


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
RELIABLE_LO = 0.90        # the contract's bar (CTR-teacher-reliable), on the lower bound
MARGINAL_LO = 0.50
Z94 = 1.881               # normal quantile for a two-sided 94% interval
TIMELINE_MAX = 700        # characters of a timeline shown before it is elided
TIMELINE_KEEP = 340
EVENTS_SHOWN = 10


def interval(k: int, n: int, z: float = Z94) -> tuple[float, float]:
    """Wilson interval, 94% by default -- the width the ledger's contracts are written to."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - r) / d), min(1.0, (c + r) / d)


def grade(lo: float) -> str:
    return "reliable" if lo >= RELIABLE_LO else ("marginal" if lo >= MARGINAL_LO else "broken")


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


def _kind(ev: str) -> str:
    """An event with its numbers taken out, so the same thing in different episodes counts once."""
    import re
    text = ev.split(" ", 1)[1] if ev.startswith("t") else ev
    text = re.sub(r"[-+]?\d+(\.\d+)?", "#", text)
    for cut in (":", ",", " by "):
        if cut in text and not text.startswith("PUSHED"):
            text = text.split(cut)[0]
    return text.strip()


def _counts(counter: collections.Counter | dict, limit: int | None = None) -> str:
    items = counter.most_common(limit) if isinstance(counter, collections.Counter) else \
        sorted(counter.items(), key=lambda x: -x[1])
    return ", ".join(f"{k} x{c}" for k, c in items) or "-"


def _episode_lines(t: dict, d: dict) -> list[str]:
    """One failed episode in full: timeline, events, grasp, false predicate."""
    tl = d.get("timeline", "")
    lines = [f"  -- ep{d.get('episode')} ({d.get('steps')} steps) [{t['conditions'].get('mechanism')}]",
             "     timeline: " + (tl if len(tl) < TIMELINE_MAX else tl[:TIMELINE_KEEP] + " ... " + tl[-TIMELINE_KEEP:])]
    lines += ["     " + e for e in d.get("events", [])[:EVENTS_SHOWN]]
    lines += [f"     grasp {obj}: " + ", ".join(f"{k}={v}" for k, v in g.items())
              for obj, g in d.get("grasp", {}).items()]
    lines += ["     final: " + f for f in d.get("final", [])]
    return lines


def _task_section(r: dict, fails: list[dict], examples: int) -> list[str]:
    """What a task's failed episodes have in common, and the first few in full."""
    det = [t.get("detail", {}) for t in fails]
    lang = det[0].get("language", "") if det else ""
    kinds = collections.Counter(k for d in det for k in {_kind(e) for e in d.get("events", [])})
    tiers = collections.Counter(g.get("tier") for d in det for g in d.get("grasp", {}).values())
    lines = [f"\n### {r['suite']}[{r['task']}] {r['k']}/{r['n']}  {lang}",
             "  mechanisms: " + _counts(r["mechanisms"]),
             "  events in failed episodes: " + _counts(kinds, 8),
             "  grasp tiers chosen: " + _counts(tiers)]
    for t, d in list(zip(fails, det, strict=True))[:examples]:
        lines += _episode_lines(t, d)
    return lines


def diagnose(rows: list[dict], trials: list[dict], examples: int, out_md: Path | None) -> None:
    """Per task that is not perfect: what the failed episodes have in common, and two of
    them in full; every failed episode goes to `out_md`."""
    by = collections.defaultdict(list)
    for t in trials:
        by[(t["conditions"]["suite"], t["conditions"]["task"])].append(t)
    md = []
    for r in sorted(rows, key=lambda r: r["lo"]):
        if r["k"] == r["n"]:
            continue
        fails = [t for t in by[(r["suite"], r["task"])] if not t["metrics"]["success"]]
        section = _task_section(r, fails, examples)
        print("\n".join(section))
        md += section
        for t in fails[examples:]:                    # the full record, in the file only
            md += _episode_lines(t, t.get("detail", {}))
    if out_md is not None:
        out_md.write_text("\n".join(md) + "\n")
        print(f"\nfull record of every failed episode -> {out_md}")


def _parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", nargs="*", default=["libero_object", "libero_spatial", "libero_goal"])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=500)
    ap.add_argument("--evidence", default="runs/evidence/reliability")
    ap.add_argument("--from", dest="reuse", default="", help="read existing trials, do not run")
    ap.add_argument("--ingest", action="store_true")
    ap.add_argument("--examples", type=int, default=2, help="failed episodes shown in full per task")
    ap.add_argument("--out", default="", help="write the trials for component-belief's run_test ($OUT)")
    return ap.parse_args()


def write_out(trials: list[dict], path: str) -> None:
    """The trials as component-belief's run_test reads them: metrics, conditions, repro.
    The per-episode diagnosis stays in the evidence directory, not in the ledger."""
    Path(path).write_text(json.dumps({"trials": [
        {"metrics": t["metrics"], "conditions": t["conditions"], "repro": t["repro"]} for t in trials]}))


def _gather(args) -> list[dict]:
    """Run each suite (unless --from) and read its trials."""
    trials: list[dict] = []
    for suite in args.suites:
        path = ROOT / (args.reuse or args.evidence) / f"{suite}.trials.json"
        if not args.reuse:
            print(f"[run] {suite}: {args.episodes} episodes x {SUITE_TASKS.get(suite, 10)} tasks", flush=True)
            collect(suite, args.episodes, args.horizon, path)
        if not path.exists():
            print(f"  (no trials at {path})")
            continue
        trials += json.loads(path.read_text())["trials"]
    return trials


def _print_table(rows: list[dict]) -> None:
    print(f"\n{'task':28s} {'rate':>9s} {'94% interval':>15s}  {'grade':9s} mechanisms")
    for r in rows:
        print(f"{r['suite'][7:]:>8s}[{r['task']:2d}]{'':14s}"[:28]
              + f" {r['k']:3d}/{r['n']:<5d} [{r['lo']:.2f}, {r['hi']:.2f}]  {r['grade']:9s} "
              + (_counts(r["mechanisms"]) if r["mechanisms"] else ""))
    by_grade = collections.Counter(r["grade"] for r in rows)
    n, k = sum(r["n"] for r in rows), sum(r["k"] for r in rows)
    lo, hi = interval(k, n)
    print(f"\noverall {k}/{n} = {k / max(n, 1):.3f}  [{lo:.2f}, {hi:.2f}]   "
          + "  ".join(f"{g} {by_grade[g]}" for g in ("reliable", "marginal", "broken")))
    worst = sorted((r for r in rows if r["grade"] != "reliable"), key=lambda r: r["lo"])
    if worst:
        print("\nnot reliable, worst first:")
        for r in worst:
            print(f"  {r['suite']}[{r['task']}] {r['k']}/{r['n']} lo {r['lo']:.2f}: {_counts(r['mechanisms'])}")
    agg = collections.Counter()
    for r in rows:
        agg.update(r["mechanisms"])
    if agg:
        print("\nfailures by mechanism: " + ", ".join(f"{m} {c}" for m, c in agg.most_common()))


def main() -> int:
    args = _parse()
    trials = _gather(args)
    rows = report(trials)
    _print_table(rows)
    print("\n==== diagnosis, worst task first ====")
    diagnose(rows, trials, args.examples, ROOT / (args.reuse or args.evidence) / "diagnosis.md")
    if args.out:
        write_out(trials, args.out)
        print(f"\n{len(trials)} trials -> {args.out}")
    if args.ingest:
        revs = sorted({t["repro"].get("teacher_revision", "?") for t in trials})
        print(f"\ningest ({', '.join(revs)}):")
        print(ingest(trials, str(ROOT / (args.reuse or args.evidence))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
