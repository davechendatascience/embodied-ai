#!/usr/bin/env python
"""Run the geometry-driven teacher (screwhead/teacher/skill_teacher.py) on LIBERO tasks.

  skill_eval.py --suite libero_object --episodes 5
  skill_eval.py --suite libero_spatial --tasks 4 --episodes 10 --split 5 -v
  skill_eval.py --suite libero_object --tasks 0 1 --episodes 3 --video videos/skill

Per task: success, and for each failure the mechanism, a one-line summary and (with -v)
the full account from screwhead/teacher/episode_log.py -- timeline, events, the grasp chosen, and
the false predicate term. Trials go to --trials for tools/teacher_report.py and the ledger.
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PERF_CORES = "5,6,7,8,9,15,16,17,18,19"
VIDEO_FPS = 20


@dataclass(frozen=True)
class Job:
    """One worker: a task, a share of its episodes, a seed and a core."""
    suite: str
    task: int
    episodes: int
    seed: int
    cpu: int
    ep_offset: int
    horizon: int
    start: dict
    video: str
    max_videos: int


def _observe(errors: list, fn, *args):
    """Run a diagnostic. Its failure is reported, never scored: inside the episode's own
    guard, a diagnostic TypeError once recorded episodes LIBERO had not scored as failures."""
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001  the observer boundary
        errors.append(f"{type(e).__name__}: {e}")
        return None


def _run_episode(env, teacher, job: Job, ep: int, record: bool):
    """One episode: the row for the report, and frames if recording."""
    from screwhead.teacher.episode_log import EpisodeLog
    env.reset()
    errors: list[str] = []
    log = _observe(errors, EpisodeLog, env, teacher)
    frames, done, info, missing = [], False, {}, ""
    phases = collections.Counter()
    while not done:
        s = env.snapshot()
        try:
            a = teacher.act(s)
        except NotImplementedError as e:          # a skill this suite needs and we lack:
            missing = str(e).split("(")[0].strip()   # report it, do not kill the worker
            teacher.phase = f"unimplemented:{missing}"
            a = np.zeros(7)
        phases[teacher.phase] += 1
        if log is not None and not errors:
            _observe(errors, log.step, s)
        if record:
            ag, wr = env.images()
            frames.append(np.concatenate([ag[::-1], wr[::-1]], axis=1))
        _, _, done, info = env.step(a)
    ok = bool(info["success"])
    diag = mechanism = ""
    detail = dict(timeline="", events=[], grasp={}, final=[])
    if log is not None and not errors:
        if not ok:
            diag = _observe(errors, log.summary, missing) or ""
            mechanism = _observe(errors, log.mechanism, missing) or ""
        detail = _observe(errors, log.finish, ok) or detail
    detail["observer_errors"] = errors
    track = log.track if log is not None else {}
    row = dict(task=job.task, episode=job.ep_offset + ep, success=ok, steps=env.t,
               last_phase=teacher.phase, step_index=teacher.step_index, phases=dict(phases),
               language=env.language, diag=diag, mechanism=mechanism, detail=detail,
               **{k: round(float(v), 4) if isinstance(v, float) else v
                  for k, v in track.items() if k not in ("skill", "regressed")})
    return row, frames


def _failed_row(task: int, episode: int, steps: int, language: str, what: str, mechanism: str) -> dict:
    return dict(task=task, episode=episode, success=False, steps=steps, last_phase=mechanism,
                step_index=-1, phases={}, language=language, diag=what, mechanism=mechanism,
                detail=dict(timeline="", events=[what], grasp={}, final=[]))


def _crashed_row(job: Job, ep: int, env, e: Exception) -> dict:
    import traceback
    where = traceback.extract_tb(e.__traceback__)[-1]
    what = f"crash: {type(e).__name__}: {e} at {Path(where.filename).name}:{where.lineno}"
    return _failed_row(job.task, job.ep_offset + ep, env.t, env.language, what, "crash")


def _fill_lost(rows: list[dict], jobs: list[Job]) -> list[dict]:
    """Every episode a job owed is a row. One a dead worker never sent is a failure, not an
    absence: dropped, it biased the rate toward the episodes that got to finish."""
    have = {(r["task"], r["episode"]) for r in rows}
    return rows + [_failed_row(j.task, ep, 0, "", "lost: the worker died before sending it", "lost")
                   for j in jobs for ep in range(j.ep_offset, j.ep_offset + j.episodes)
                   if (j.task, ep) not in have]


def _worker(remote, job_fields: dict) -> None:
    job = Job(**job_fields)
    os.sched_setaffinity(0, {job.cpu})
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT))
    from screwhead.teacher.skill_teacher import SkillTeacher
    from screwhead.sim.task_env import StartNoise, TaskEnv
    env = TaskEnv(job.suite, job.task, horizon=job.horizon, seed=job.seed, render=True,
                  start=StartNoise(**job.start))
    teacher = SkillTeacher(env)
    videos = 0
    for ep in range(job.episodes):
        record = bool(job.video) and videos < job.max_videos
        try:
            row, frames = _run_episode(env, teacher, job, ep, record)
        except Exception as e:  # noqa: BLE001  a worker boundary: a crash is one failed, recorded
            #                     episode, not a lost worker -- one diagnostic TypeError once
            #                     silently removed all 20 episodes of a task from an assessment
            row, frames = _crashed_row(job, ep, env, e), []
        if record and frames and not row["success"]:
            import imageio.v2 as imageio
            Path(job.video).mkdir(parents=True, exist_ok=True)
            imageio.mimsave(Path(job.video) / f"{job.suite}_t{job.task}_ep{ep}_fail.mp4", frames,
                            fps=VIDEO_FPS, macro_block_size=1)
            videos += 1
        remote.send(row)
    remote.send(None)
    env.close()
    remote.close()


def _parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--tasks", type=int, nargs="*", default=None)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=600)
    ap.add_argument("--start-xy", type=float, default=0.0)
    ap.add_argument("--start-z", type=float, default=0.0)
    ap.add_argument("--start-yaw", type=float, default=0.0)
    ap.add_argument("--start-tilt", type=float, default=0.0)
    ap.add_argument("--start-null", type=float, default=0.0)
    ap.add_argument("--cpus", default=PERF_CORES)
    ap.add_argument("--seed", type=int, default=555)
    ap.add_argument("--split", type=int, default=1,
                    help="workers per task, each running a share of the episodes on its own "
                         "seed -- for iterating on one task without waiting on one core")
    ap.add_argument("--video", default="")
    ap.add_argument("--max-videos", type=int, default=1)
    ap.add_argument("--trials", default="")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="timeline, events, grasp, and the false predicate term for every failure")
    return ap.parse_args()


def _jobs(args) -> list[Job]:
    tasks = args.tasks if args.tasks is not None else list(range(10))
    cpus = [int(c) for c in args.cpus.split(",")]
    start = dict(xy_m=args.start_xy, z_m=args.start_z, yaw_deg=args.start_yaw,
                 tilt_deg=args.start_tilt, null_rad=args.start_null)
    per = -(-args.episodes // args.split)
    jobs = []
    for t in tasks:
        for j in range(args.split):
            n = min(per, args.episodes - j * per)
            if n > 0:
                seed = args.seed * 100 + t if args.split == 1 else (args.seed * 100 + t) * 1000 + j
                jobs.append(Job(args.suite, t, n, seed, cpus[len(jobs) % len(cpus)], j * per,
                                args.horizon, start, args.video, args.max_videos))
    return jobs


def _run(jobs: list[Job], suite: str) -> list[dict]:
    """Launch one process per job and print rows as they arrive."""
    ctx = mp.get_context("spawn")
    procs, remotes = [], []
    for job in jobs:
        a, b = ctx.Pipe()
        p = ctx.Process(target=_worker, args=(b, asdict(job)), daemon=True)
        p.start()
        b.close()
        procs.append(p)
        remotes.append(a)
    rows, open_pipes = [], set(range(len(remotes)))
    while open_pipes:
        for i in list(open_pipes):
            try:
                r = remotes[i].recv()
            except EOFError:
                r = None
            if r is None:
                open_pipes.discard(i)
                continue
            rows.append(r)
            print(f"  {suite} task {r['task']} ep {r['episode']}: {'ok' if r['success'] else 'fail'} "
                  f"steps {r['steps']} last {r['last_phase']}" + (f" | {r['diag']}" if r["diag"] else ""),
                  flush=True)
    for p in procs:
        p.join(timeout=10)
    return rows


def _print_failure(r: dict, verbose: bool) -> None:
    print(f"      ep{r['episode']} {r['last_phase']:22s} [{r['mechanism']}] {r['diag']}")
    d = r.get("detail", {})
    if not (verbose and d):
        return
    print(f"         timeline: {d['timeline']}")
    for ev in d["events"]:
        print(f"         {ev}")
    for obj, gl in d["grasp"].items():
        print(f"         grasp {obj}: {gl}")
    for line in d["final"]:
        print(f"         final: {line}")


def _summarise(rows: list[dict], args, seconds: float) -> None:
    by = collections.defaultdict(list)
    for r in rows:
        by[r["task"]].append(r)
    print()
    for t in sorted(by):
        ok = sum(r["success"] for r in by[t])
        fails = collections.Counter(r["last_phase"] for r in by[t] if not r["success"])
        print(f"  task {t}: {ok}/{len(by[t])}  {by[t][0]['language'][:54]!r}"
              + (f"  failures: {dict(fails)}" if fails else ""))
        for r in by[t]:
            if not r["success"]:
                _print_failure(r, args.verbose)
    ok = sum(r["success"] for r in rows)
    print(f"{args.suite}: {ok}/{len(rows)} = {ok / max(len(rows), 1):.2f}   ({seconds:.0f}s)")


def _write_trials(rows: list[dict], args) -> None:
    Path(args.trials).parent.mkdir(parents=True, exist_ok=True)
    Path(args.trials).write_text(json.dumps({"trials": [
        {"metrics": {"success": r["success"]},
         "conditions": {"task": r["task"], "suite": args.suite, "last_phase": r["last_phase"],
                        "mechanism": r.get("mechanism", "")},
         "detail": dict(r.get("detail", {}), episode=r["episode"], steps=r["steps"], language=r["language"]),
         "repro": {"seed": args.seed * 100 + r["task"], "task": r["task"], "task_suite": args.suite,
                   "episode": r["episode"], "horizon": args.horizon,
                   "teacher_revision": "skill_teacher"}} for r in rows]}, indent=1))
    print("->", args.trials)


def main() -> int:
    args = _parse()
    t0 = time.time()
    jobs = _jobs(args)
    rows = _fill_lost(_run(jobs, args.suite), jobs)
    _summarise(rows, args, time.time() - t0)
    if args.trials:
        _write_trials(rows, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
