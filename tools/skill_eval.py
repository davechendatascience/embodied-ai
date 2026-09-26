#!/usr/bin/env python
"""Run the geometry-driven teacher (screwhead/teacher/skill_teacher.py) on LIBERO tasks.

  skill_eval.py --suite libero_object --episodes 5
  skill_eval.py --suite libero_spatial --tasks 4 --episodes 10 --split 5 -v
  skill_eval.py --suite libero_object --tasks 0 1 --episodes 3 --video videos/skill
  skill_eval.py --suite libero_goal --episodes 50 --wide 1.5 --seed 2701   # widened randomized starts

Per task: success, and for each failure the mechanism, a one-line summary and (with -v)
the full account from screwhead/teacher/episode_log.py -- timeline, events, the grasp chosen, and
the false predicate term. Trials go to --trials for tools/teacher_report.py and the ledger.

With --wide S every episode starts from a fresh layout LIBERO's own samplers draw inside the task file's regions
(TaskEnv.reset_fresh), not one of its 50 stored initial states, with the tool's start randomized at S times the
randomized test set's start bounds (screwhead/sim/task_env_place.RANDOMIZER); each episode seeded by (seed, task,
episode) alone. A fresh --seed per sweep keeps fixes from being tuned to the episodes of an earlier one.
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
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
    refuse: bool
    start: dict
    video: str
    max_videos: int
    video_px: int
    stride: int = 1      # this job's episodes are ep_offset, ep_offset + stride, ... of one stream
    init_order: bool = False   # LIBERO's protocol: episode e starts from initial state e (mod their number)
    robot: str = "Panda"       # the embodiment (Execution.robot, Execution.gripper)
    gripper: str = "PandaGripper"
    diagnostics: bool = False  # record the servo's tracking per episode (SimArm.diagnostics) in its trial
    wide: float = 0.0          # > 0: fresh LIBERO layouts, the start randomized at this multiple of RANDOMIZER


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
    from screwhead.teacher.grip_watch import GripWatch
    from screwhead.teacher.refusal import Refusal
    # Drawn from the seeded stream (with replacement), 50 episodes covered 26-34 of a task's 50 initial
    # states and repeated the rest exactly; --init-order runs each once, in order, as LIBERO does
    episode = job.ep_offset + ep * job.stride
    if job.wide > 0:
        env.reset_fresh(job.seed, episode)
    elif job.init_order:
        env.reset(init_index=episode % len(env.init_states))
    else:
        env.reset()
    errors: list[str] = []
    log = _observe(errors, EpisodeLog, env, teacher)
    # CTR-teacher-gentle's metrics, kept on their own error list: a failing watch must not
    # silence the log's diagnosis, nor a failing log the watch
    watch_errors: list[str] = []
    watch = _observe(watch_errors, GripWatch, env, teacher)
    frames, done, info, missing = [], False, {}, ""
    phases = collections.Counter()
    refusal = None
    while not done:
        s = env.snapshot()
        try:
            a = teacher.act(s)
        except Refusal as r:
            # DEF-refusal-is-an-outcome: a precondition with no witness ends the episode here,
            # naming the predicate. Running on to the horizon spends it and disturbs a scene a
            # later attempt would otherwise find as it was -- libero_goal 9 tipped the bottle over
            # in the four hundred steps after its grasp screen had already scored feasible 0.
            refusal = r
            teacher.phase = f"refused:{r.predicate}"
            break
        except NotImplementedError as e:          # a skill this suite needs and we lack:
            missing = str(e).split("(")[0].strip()   # report it, do not kill the worker
            teacher.phase = f"unimplemented:{missing}"
            a = np.zeros(7)
        phases[teacher.phase] += 1
        if log is not None and not errors:
            _observe(errors, log.step, s)
        if watch is not None and not watch_errors:
            _observe(watch_errors, watch.step)
        if record:
            frames.append(_frame(env, job, teacher.phase) if job.video_px else
                          np.concatenate([im[::-1] for im in env.images()], axis=1))
        _, _, done, info = env.step(a)
    # A refusal on the first step leaves info empty -- the loop broke before env.step ran -- so
    # the refusal is checked first and the scorer is never asked about an episode nobody attempted.
    ok = refusal is None and bool(info.get("success"))
    diag = mechanism = ""
    detail = dict(timeline="", events=[], grasp={}, final=[])
    if log is not None and not errors:
        if not ok:
            diag = _observe(errors, log.summary, missing) or ""
            mechanism = _observe(errors, log.mechanism, missing) or ""
        detail = _observe(errors, log.finish, ok) or detail
    detail["observer_errors"] = errors + watch_errors
    gentle = watch.metrics() if watch is not None and not watch_errors else {}
    track = log.track if log is not None else {}
    forced = {o: g.get("rejected", {}) for o, g in
              (detail.get("grasp") or {}).items() if g.get("forced")}
    row = dict(task=job.task, episode=job.ep_offset + ep * job.stride, success=ok, steps=env.t,
               refused=refusal is not None, **(refusal.as_row() if refusal else {}),
               forced_grasp=bool(forced), rejected_by=forced, **gentle,
               last_phase=teacher.phase, step_index=teacher.step_index, phases=dict(phases),
               language=env.language, diag=diag, mechanism=mechanism, detail=detail,
               init_index=getattr(env, "init_index", None),
               **{k: round(float(v), 4) if isinstance(v, float) else v
                  for k, v in track.items() if k not in ("skill", "regressed")})
    return row, frames


def _frame(env, job: Job, phase: str) -> np.ndarray:
    """Agentview and wrist at video_px, with the step, time, phase and task written on top."""
    import cv2
    img = np.concatenate([env.render("agentview", job.video_px),
                          env.render("robot0_eye_in_hand", job.video_px)], axis=1)
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    label = f"{job.suite}[{job.task}]   step {env.t:3d}   {env.t / VIDEO_FPS:5.1f} s   {phase}"
    cv2.putText(img, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _failed_row(task: int, episode: int, steps: int, language: str, what: str, mechanism: str) -> dict:
    return dict(task=task, episode=episode, success=False, steps=steps, last_phase=mechanism,
                step_index=-1, phases={}, language=language, diag=what, mechanism=mechanism,
                detail=dict(timeline="", events=[what], grasp={}, final=[]))


def _crashed_row(job: Job, ep: int, env, e: Exception) -> dict:
    import traceback
    where = traceback.extract_tb(e.__traceback__)[-1]
    what = f"crash: {type(e).__name__}: {e} at {Path(where.filename).name}:{where.lineno}"
    row = _failed_row(job.task, job.ep_offset + ep * job.stride, env.t, env.language, what, "crash")
    row["init_index"] = getattr(env, "init_index", None)
    return row


def _fill_lost(rows: list[dict], jobs: list[Job]) -> list[dict]:
    """Every episode a job owed is a row. One a dead worker never sent is a failure, not an
    absence: dropped, it biased the rate toward the episodes that got to finish."""
    have = {(r["task"], r["episode"]) for r in rows}
    return rows + [_failed_row(j.task, ep, 0, "", "lost: the worker died before sending it", "lost")
                   for j in jobs for ep in range(j.ep_offset, j.ep_offset + j.episodes * j.stride, j.stride)
                   if (j.task, ep) not in have]


def _worker(remote, job_fields: dict) -> None:
    job = Job(**job_fields)
    os.sched_setaffinity(0, {job.cpu})
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT))
    from screwhead.teacher.skill_teacher import SkillTeacher
    from screwhead.sim.sim_arm import Execution
    from screwhead.sim.task_env import StartNoise, TaskEnv
    # cameras only for video: the teacher reads no image, trajectories are identical either way, and
    # ten workers rendering every step held the GPU at 33-39% and the box at 93-95 C
    start = StartNoise(**job.start)
    if job.wide > 0:
        from screwhead.sim.task_env_place import RANDOMIZER
        start = StartNoise(**{k.removeprefix("start_"): v * job.wide
                              for k, v in RANDOMIZER.items() if k.startswith("start_")})
    env = TaskEnv(job.suite, job.task, horizon=job.horizon, seed=job.seed, render=bool(job.video),
                  start=start, execution=Execution(robot=job.robot, gripper=job.gripper))
    teacher = SkillTeacher(env)
    teacher.skills.reach.refuse_when_empty = job.refuse
    videos = 0
    for ep in range(job.episodes):
        if job.stride > 1 and not job.wide:   # the stream's episodes before this one belong to other workers
            for _ in range(job.ep_offset if ep == 0 else job.stride - 1):
                env.skip_episode()
        record = bool(job.video) and videos < job.max_videos
        if job.diagnostics:
            env.enable_diagnostics()
        try:
            row, frames = _run_episode(env, teacher, job, ep, record)
            if job.diagnostics:
                row.setdefault("detail", {})["servo"] = env.diagnostics()
        except Exception as e:  # noqa: BLE001  a worker boundary: a crash is one failed, recorded
            #                     episode, not a lost worker -- one diagnostic TypeError once
            #                     silently removed all 20 episodes of a task from an assessment
            row, frames = _crashed_row(job, ep, env, e), []
        if record and frames and not row["success"]:
            import imageio.v2 as imageio
            Path(job.video).mkdir(parents=True, exist_ok=True)
            imageio.mimsave(Path(job.video) / f"{job.suite}_t{job.task}_ep{job.ep_offset + ep * job.stride}_fail.mp4", frames,
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
    ap.add_argument("--refuse", action="store_true",
                    help="refuse when the reach screen finds nothing, instead of forcing its first "
                         "candidate (DEF-witness-or-refusal). Off until the screen is calibrated: "
                         "it is declared conservative, and refusing on it cost 56 successes over "
                         "1500 episodes, 50 of them on a task the rejected grasp solves 50/50")
    ap.add_argument("--at-once", type=int, default=0,
                    help="processes alive at a time (0: one per core in --cpus)")
    ap.add_argument("--seed", type=int, default=555)
    ap.add_argument("--init-order", action="store_true",
                    help="LIBERO's protocol: episode e starts from the task's initial state e, each once "
                         "(default: drawn from the seeded stream, with replacement)")
    ap.add_argument("--split", type=int, default=1,
                    help="workers per task, each running a share of the episodes on its own "
                         "seed -- for iterating on one task without waiting on one core")
    ap.add_argument("--interleave", type=int, default=1,
                    help="workers per task on the task's one seed, worker j running episodes j, "
                         "j+N, ...: the same episodes as one worker, episode for episode, N times "
                         "sooner")
    ap.add_argument("--video", default="")
    ap.add_argument("--max-videos", type=int, default=1)
    ap.add_argument("--video-px", type=int, default=0, help="record at this size, with a caption (0: the 128 px observation cameras)")
    ap.add_argument("--robot", default="Panda", help="the arm, by robosuite's name (Panda, UR5e, IIWA, Jaco, Kinova3)")
    ap.add_argument("--gripper", default="PandaGripper", help="the gripper, by robosuite's name")
    ap.add_argument("--diagnostics", action="store_true", help="record the servo's tracking per episode in its trial")
    ap.add_argument("--wide", type=float, default=0.0,
                    help="fresh LIBERO layouts, the start randomized at this multiple of the randomized test set's bounds")
    ap.add_argument("--trials", default="")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="timeline, events, grasp, and the false predicate term for every failure")
    return ap.parse_args()


def _jobs(args) -> list[Job]:
    tasks = args.tasks if args.tasks is not None else list(range(10))
    cpus = [int(c) for c in args.cpus.split(",")]
    start = dict(xy_m=args.start_xy, z_m=args.start_z, yaw_deg=args.start_yaw,
                 tilt_deg=args.start_tilt, null_rad=args.start_null)
    if args.interleave > 1 and args.split > 1:
        raise SystemExit("--interleave and --split are exclusive")
    per = -(-args.episodes // args.split)
    jobs = []
    for t in tasks:
        for j in range(args.interleave if args.interleave > 1 else 0):
            n = len(range(j, args.episodes, args.interleave))
            if n > 0:
                jobs.append(Job(args.suite, t, n, args.seed * 100 + t, cpus[len(jobs) % len(cpus)], j,
                                args.horizon, args.refuse, start, args.video,
                                args.max_videos, args.video_px, stride=args.interleave,
                                init_order=args.init_order, robot=args.robot, gripper=args.gripper,
                                diagnostics=args.diagnostics, wide=args.wide))
        for j in range(args.split if args.interleave <= 1 else 0):
            n = min(per, args.episodes - j * per)
            if n > 0:
                # --wide draws every episode from (seed, episode) alone, so the task's seed must not change with the split
                seed = args.seed * 100 + t if args.split == 1 or args.wide else (args.seed * 100 + t) * 1000 + j
                jobs.append(Job(args.suite, t, n, seed, cpus[len(jobs) % len(cpus)], j * per,
                                args.horizon, args.refuse, start, args.video,
                                args.max_videos, args.video_px, init_order=args.init_order, robot=args.robot,
                                gripper=args.gripper, diagnostics=args.diagnostics, wide=args.wide))
    return jobs


def _run(jobs: list[Job], suite: str, at_once: int = 0) -> list[dict]:
    """Run the jobs, at most `at_once` processes alive at a time, printing rows as they arrive.

    Every job is a whole LIBERO environment: a process of its own, above a gigabyte of it, and
    several threads whatever OMP_NUM_THREADS says. The job list is tasks x split, so a full suite
    at --split 10 is a hundred of them, which is ten pinned to each core and more memory than the
    machine has. Launching them all at once froze this machine twice. The default cap is one
    process per core named in --cpus.

    A rolling pool: a job starts on a core as soon as the one before it there has finished. Run in
    waves, every wave waited for its slowest task, and a libero_90 sweep kept 3 of 10 cores busy.
    """
    from multiprocessing.connection import wait
    cpus = list(dict.fromkeys(j.cpu for j in jobs))
    at_once = min(at_once or len(cpus), len(cpus)) or 1
    ctx = mp.get_context("spawn")
    pending, free, live, rows = list(jobs), cpus[:at_once], {}, []
    while pending or live:
        while pending and free:
            job = replace(pending.pop(0), cpu=free.pop(0))
            a, b = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(b, asdict(job)), daemon=True)
            proc.start()
            b.close()
            live[a] = (proc, job.cpu)
        for conn in wait(list(live)):
            try:
                r = conn.recv()
            except EOFError:
                r = None
            if r is None:                       # the worker is done (or died): free its core
                proc, cpu = live.pop(conn)
                proc.join(timeout=10)
                free.append(cpu)
                continue
            rows.append(r)
            _print_row(suite, r)
    return rows


def _print_row(suite: str, r: dict) -> None:
    verdict = ("refused " + r["refused_predicate"] if r.get("refused")
               else "ok" if r["success"] else "fail")
    print(f"  {suite} task {r['task']} ep {r['episode']}: {verdict} "
          f"steps {r['steps']} last {r['last_phase']}"
          + (f" drops {r['drop_count']}" if r.get("drop_count") else "")
          + (f" | {r['diag']}" if r["diag"] else ""), flush=True)


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
    """Report attempts and refusals apart.

    DEF-refusal-is-an-outcome: a rate that adds them cannot tell a teacher that does not try from
    one that tries and botches, and improving either moves the same number the same way. So the
    rate is over episodes the teacher attempted, and refusals are counted beside it with the
    predicate that had no witness -- a failure is evidence about execution, a refusal about
    coverage, and they are fixed by different work.
    """
    by = collections.defaultdict(list)
    for r in rows:
        by[r["task"]].append(r)
    print()
    for t in sorted(by):
        tried = [r for r in by[t] if not r.get("refused")]
        gave_up = [r for r in by[t] if r.get("refused")]
        ok = sum(r["success"] for r in tried)
        fails = collections.Counter(r["last_phase"] for r in tried if not r["success"])
        # an episode can succeed with the object dropped on to its target: count it apart
        dropped = sum(bool(r.get("drop_count")) for r in by[t])
        why = collections.Counter(f"{r['refused_predicate']}({r['refused_subject']})" for r in gave_up)
        rate = f"{ok}/{len(tried)}" if tried else "0/0"
        print(f"  task {t}: {rate}  {by[t][0]['language'][:54]!r}"
              + (f"  refused {len(gave_up)}: {dict(why)}" if gave_up else "")
              + (f"  dropped in {dropped}" if dropped else "")
              + (f"  failures: {dict(fails)}" if fails else ""))
        for r in tried:
            if not r["success"]:
                _print_failure(r, args.verbose)
    tried = [r for r in rows if not r.get("refused")]
    gave_up = len(rows) - len(tried)
    ok = sum(r["success"] for r in tried)
    rate = ok / len(tried) if tried else float("nan")
    print(f"{args.suite}: {ok}/{len(tried)} attempted = {rate:.2f}"
          + (f", {gave_up}/{len(rows)} refused" if gave_up else "")
          + f"   ({seconds:.0f}s)")


def _write_trials(rows: list[dict], args, rev: str) -> None:
    """Write the trials, with refusals kept out of the reliability rate.

    A refused episode carries no `success` key at all -- not false, absent. The ledger excludes a
    trial missing a rule's metric rather than failing it, and that exclusion is the whole mechanism
    keeping refusals out of CTR-teacher-reliable; they are counted by CTR-teacher-attempts instead.
    Writing success: false here would look like completing a record, and would quietly fold
    coverage back into reliability. The assertion below is there because that edit is tempting.
    """
    # Stamp the teacher that actually ran: `rev` is hashed when the run starts (main). This used to
    # write the constant "skill_teacher", which pools every revision into one slice -- the
    # compatibility key reads teacher_revision, so a constant makes a change to the teacher
    # invisible and averages it with the teacher it replaced. teacher_code() walks the import
    # closure, so a new file in the teacher enters the hash by itself.
    Path(args.trials).parent.mkdir(parents=True, exist_ok=True)
    # the embodiment, recorded only where it is not the Panda with its gripper, so Panda trials keep their format
    embodiment = ({} if (args.robot, args.gripper) == ("Panda", "PandaGripper")
                  else {"robot": args.robot, "gripper": args.gripper})
    Path(args.trials).write_text(json.dumps({"trials": [
        {"metrics": dict({"refused": True} if r.get("refused") else
                         {"success": r["success"], "refused": False},
                         **{k: r[k] for k in ("release_gap_mm", "drop_count") if k in r}),
         "conditions": {"task": r["task"], "suite": args.suite, "last_phase": r["last_phase"],
                        "mechanism": r.get("mechanism", ""),
                        "refused_predicate": r.get("refused_predicate", "")},
         "detail": dict(r.get("detail", {}), episode=r["episode"], steps=r["steps"], language=r["language"]),
         "repro": {"seed": args.seed * 100 + r["task"], "task": r["task"], "task_suite": args.suite,
                   "episode": r["episode"], "horizon": args.horizon, "init_index": r.get("init_index"),
                   "teacher_revision": rev, **embodiment}} for r in rows]}, indent=1))
    written = json.loads(Path(args.trials).read_text())["trials"]
    leaked = [t for t in written if t["metrics"].get("refused") and "success" in t["metrics"]]
    assert not leaked, (f"{len(leaked)} refused trials carry a success metric; they would be "
                        "scored as failures against CTR-teacher-reliable instead of excluded")
    print("->", args.trials)


def main() -> int:
    args = _parse()
    # Hashed before anything runs: hashed when the trials were written, a file edited during a
    # 20-minute sweep stamped the episodes of the code that ran with the code that had not
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from teacher_report import teacher_revision
    rev = teacher_revision()
    t0 = time.time()
    jobs = _jobs(args)
    rows = _fill_lost(_run(jobs, args.suite, args.at_once), jobs)
    _summarise(rows, args, time.time() - t0)
    if args.trials:
        _write_trials(rows, args, rev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
