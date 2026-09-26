#!/usr/bin/env python
"""The skill teacher on LIBERO-Variations: each task generated from the benchmark's metadata and a split seed,
one episode from its kept scene, scored by LIBERO's predicate on the generated task file.

    PYTHONPATH=third_party/LIBERO:. MUJOCO_GL=egl .venv-libero/bin/python tools/lv_eval.py \
        --bench screwhead/variations/benchmarks/v0.yaml --seed 1 --tasks 40 --trials runs/lv/v0_s1.trials.json

Scored rounds use a fresh --seed each (teacher-push protocol); a seed once scored is not developed on.
Tasks run on --cpus in a rolling pool, one process per core (skill_eval._run).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
from skill_eval import Job, _crashed_row, _failed_row, _run, play  # noqa: E402

LV_CPUS = "5,6,7,8,9,15,16,17"


def _worker(remote, job_fields: dict) -> None:
    job = Job(**job_fields)
    os.sched_setaffinity(0, {job.cpu})
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT))
    from screwhead.teacher.skill_teacher import SkillTeacher
    from screwhead.variations import generator
    bench = generator.Benchmark.load(job.bench)
    t0 = time.time()
    try:
        env, kept, _history = generator.generate(bench, job.seed, job.task, tempfile.mkdtemp(), render=False,
                                                 horizon=job.horizon, robot=job.robot)
    except Exception as e:  # noqa: BLE001  a generation that keeps nothing is the generator's failure, not the teacher's
        row = _failed_row(job.task, job.task, 0, "", f"generation: {type(e).__name__}: {e}"[:400], "generation")
        remote.send(dict(row, seed=job.seed, env_episode=0, generated=False))
        remote.send(None)
        return
    generated_s = time.time() - t0
    teacher = SkillTeacher(env)
    teacher.skills.reach.refuse_when_empty = job.refuse
    try:
        row, _frames = play(env, teacher, job, job.task, record=False)
    except Exception as e:  # noqa: BLE001  a worker boundary: a crash is one failed, recorded episode
        row = _crashed_row(job, 0, env, e)
    d = kept.draft
    row.update(generated=True, template=d.template, moved=d.categories[d.a], target=d.categories[d.target],
               objects=[p.category for p in d.placements], task_seed=kept.seed, attempt=kept.attempt,
               task_digest=kept.digest, placed_digest=kept.placed_digest, fingerprint=kept.fingerprint,
               generated_s=round(generated_s, 1))
    remote.send(row)
    remote.send(None)
    env.close()


def _parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=str(ROOT / "screwhead/variations/benchmarks/v0.yaml"))
    ap.add_argument("--seed", type=int, required=True, help="the split seed; fresh for every scored round")
    ap.add_argument("--tasks", type=int, default=40)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--cpus", default=LV_CPUS)
    ap.add_argument("--horizon", type=int, default=0, help="0: the benchmark's")
    ap.add_argument("--trials", default="")
    ap.add_argument("--check-against", default="",
                    help="trials of an earlier generation of this split: each task's digests must reproduce")
    return ap.parse_args()


def _print_row(r: dict) -> None:
    what = "ok  " if r["success"] else "FAIL"
    print(f"  task {r['task']:3d} {what} {r.get('template', '-'):7s} {r['language'][:60]!r:62s} "
          f"steps {r['steps']:3d} {'' if r['success'] else r.get('mechanism', '')}", flush=True)


def _summarise(rows: list[dict], seconds: float) -> None:
    scored = [r for r in rows if r.get("generated") is not False]     # a lost task counts as a failure
    not_generated = len(rows) - len(scored)
    by = collections.defaultdict(list)
    for r in scored:
        by[r.get("template", "lost")].append(r)
    print()
    for t in sorted(by):
        ok = sum(r["success"] for r in by[t])
        fails = collections.Counter(r.get("mechanism") or r["last_phase"] for r in by[t] if not r["success"])
        print(f"  {t}: {ok}/{len(by[t])}" + (f"  failures: {dict(fails)}" if fails else ""))
    ok = sum(r["success"] for r in scored)
    print(f"LIBERO-Variations: {ok}/{len(scored)} = {ok / max(1, len(scored)):.3f}"
          + (f", {not_generated} not generated" if not_generated else "") + f"   ({seconds:.0f}s)")


def _reproduced(rows: list[dict], path: str) -> None:
    """CTR-lv-regenerates: a task reproduces when its task file, model and placed state digest as they did in the
    earlier generation of the same split."""
    earlier = {t["repro"]["task"]: t["repro"] for t in json.loads(Path(path).read_text())["trials"]}
    for r in rows:
        e = earlier.get(r["task"], {})
        r["reproduced"] = bool(r.get("generated")) and e.get("task_digest") == r.get("task_digest") \
            and e.get("placed_digest") == r.get("placed_digest")
    print(f"reproduced {sum(r['reproduced'] for r in rows)}/{len(rows)} against {path}")


def _write_trials(rows: list[dict], args, bench_meta: dict, rev: str, gen_rev: str) -> None:
    suite = f"libero_variations_v{bench_meta['version']}"
    Path(args.trials).parent.mkdir(parents=True, exist_ok=True)
    Path(args.trials).write_text(json.dumps({"trials": [
        {"metrics": dict({"success": r["success"], "generated": r.get("generated")},
                         **({"reproduced": r["reproduced"]} if "reproduced" in r else {})),
         "conditions": {"task": r["task"], "suite": suite, "template": r.get("template", ""),
                        "last_phase": r["last_phase"], "mechanism": r.get("mechanism", "")},
         "detail": dict(r.get("detail", {}), steps=r["steps"], language=r["language"],
                        objects=r.get("objects", []), moved=r.get("moved", ""), target=r.get("target", "")),
         "repro": {"bench": os.path.relpath(args.bench, ROOT), "bench_version": bench_meta["version"],
                   "task_suite": suite, "generator_revision": gen_rev,
                   "split_seed": args.seed, "task": r["task"], "task_seed": r.get("task_seed"),
                   "attempt": r.get("attempt"), "task_digest": r.get("task_digest"),
                   "placed_digest": r.get("placed_digest"), "fingerprint": r.get("fingerprint"),
                   "horizon": args.horizon, "teacher_revision": rev}} for r in rows]}, indent=1))
    print("->", args.trials)


def main() -> int:
    import yaml
    args = _parse()
    from teacher_report import teacher_revision

    from screwhead.variations.generator import generator_revision
    rev, gen_rev = teacher_revision(), generator_revision(args.bench)
    with open(args.bench) as f:
        meta = yaml.safe_load(f)
    args.horizon = args.horizon or int(meta["horizon"])
    cpus = [int(c) for c in args.cpus.split(",")]
    jobs = [Job(suite=f"libero_variations_v{meta['version']}", task=i, episodes=1, seed=args.seed,
                cpu=cpus[k % len(cpus)], ep_offset=0, horizon=args.horizon, refuse=False, start={}, video="",
                max_videos=0, video_px=0, bench=os.path.abspath(args.bench))
            for k, i in enumerate(range(args.first, args.first + args.tasks))]
    t0 = time.time()
    rows = _run(jobs, jobs[0].suite, worker=_worker, printer=_print_row)
    have = {r["task"] for r in rows}
    # a task a dead worker never reported is a failure, not an absence (skill_eval._fill_lost)
    rows += [dict(_failed_row(j.task, j.task, 0, "", "lost: the worker died before sending it", "lost"),
                  seed=j.seed, env_episode=0, generated=None) for j in jobs if j.task not in have]
    _summarise(rows, time.time() - t0)
    if args.check_against:
        _reproduced(rows, args.check_against)
    if args.trials:
        _write_trials(rows, args, meta, rev, gen_rev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
