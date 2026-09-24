#!/usr/bin/env python
"""Does a success settle? Every placed object released, at rest, and no more tipped than
LIBERO's own humans leave it, after the teacher has finished.

  teacher_settled.py --reference                                     # the humans' final tilts, once
  teacher_settled.py --episodes 20 --seed 557 --out $OUT             # as TST-teacher-settled

LIBERO ends an episode at the first step its goal predicate holds. On episode 0 of all 130
tasks, 93 of the 104 objects the successful episodes placed were at that step still held,
falling or rocking; 12 came to rest tipped (a moka pot at 64 deg on the stove), and 2
successes did not survive two seconds of stillness (a wine bottle rolling off its rack). A
demonstration that stops at the first success teaches the drop and the tip.

So each episode runs as skill_eval runs it up to LIBERO's first success, and then the teacher
carries on with its own plan -- its snapshot's success flag held false, so it lets go of what
it still holds and retreats -- until the plan is complete, the robot touches no placed object
and every free object is at rest, or until the episode's horizon. Then, per episode:

  success          LIBERO's goal held at some step (the benchmark's own score)
  success_final    and still holds at the end
  finished         the plan completed (the teacher's "settle" phase) before the horizon
  released         the robot touches no placed object
  at_rest          no free object's surface moves faster than REST_SPEED, sim_arm's own test
                   for a settled scene: |v| + |w| r, r the object's bounding radius
  tilt_excess_deg  over the placed objects with a reference, the tilt from world up minus the
                   largest final tilt among LIBERO's human demonstrations of the task (<= 0
                   passes; one-sided: the humans leave the basket's bottles on their side)
  settled          all of the above
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

PERF = "5,6,7,8,9,15,16,17,18,19"
TASKS = {"libero_spatial": 10, "libero_object": 10, "libero_goal": 10, "libero_10": 10, "libero_90": 90}
HORIZON = {"libero_10": 800}          # the sweeps' horizons: 800 on libero_10, 600 elsewhere
REFERENCE = ROOT / "runs/demo_survey/final_tilt.json"


def _tilt(q) -> float:
    """Degrees between a body's z axis and world up, from its (w, x, y, z) quaternion."""
    _w, x, y, _z = (float(v) for v in q)
    return float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1.0, 1.0))))


def _placed(spec) -> list[str]:
    """The objects an On or In goal moves, in goal order."""
    out = []
    for g in spec.goals:
        if g[0] in ("on", "in") and g[1] in spec.objects and g[1] not in out:
            out.append(g[1])
    return out


def _free_joint(m, scene, obj: str) -> int | None:
    """The free joint an object's root body hangs on."""
    root = scene._root_id(obj)
    for j in range(m.njnt):
        if int(m.jnt_bodyid[j]) == root and int(m.jnt_type[j]) == 0:
            return j
    return None


# -- the humans' reference -----------------------------------------------------------------------
def reference(suites: list[str], demos: int) -> None:
    """Each placed object's tilt from world up in the last recorded state of every human demo.
    The object's quaternion is read from the state vector, so nothing is stepped or forwarded."""
    import h5py
    from demo_survey import DATASETS
    from screwhead.sim.task_env import StartNoise, TaskEnv
    out = json.loads(REFERENCE.read_text()) if REFERENCE.exists() else {}
    for suite in suites:
        for task in range(TASKS[suite]):
            env = TaskEnv(suite, task, horizon=10, seed=1, render=False, start=StartNoise())
            m, spec = env.scene.m, env.task_spec
            addr = {}
            for o in _placed(spec):
                j = _free_joint(m, env.scene, o)
                if j is not None:
                    addr[o] = int(m.jnt_qposadr[j])
            path = DATASETS / suite / f"{spec.name}_demo.hdf5"
            tilts = {o: [] for o in addr}
            if path.exists():
                with h5py.File(path) as f:
                    keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))[:demos]
                    for k in keys:
                        last = f["data"][k]["states"][-1]
                        for o, a in addr.items():                 # state = [time, qpos, qvel]
                            tilts[o].append(round(_tilt(last[1 + a + 3: 1 + a + 7]), 1))
            env.close()
            out.setdefault(suite, {})[str(task)] = {o: t for o, t in tilts.items() if t}
            print(f"{suite}[{task}] " + "  ".join(f"{o}: n={len(t)} max {max(t):.1f}"
                                                  for o, t in tilts.items() if t), flush=True)
    REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE.write_text(json.dumps(out, indent=1))
    print("->", REFERENCE.relative_to(ROOT))


# -- the episodes -----------------------------------------------------------------------------
def _task(item: tuple) -> list[dict]:
    suite, task, episodes, seed, horizon, ref, cores = item
    cpu = cores.get()
    try:
        os.sched_setaffinity(0, {cpu})
        return _episodes(suite, task, episodes, seed, horizon, ref, cpu)
    finally:
        cores.put(cpu)


def _episodes(suite: str, task: int, episodes: int, seed: int, horizon: int, ref: dict, cpu: int) -> list[dict]:
    from skill_eval import Job, _run_episode
    from screwhead.sim import contacts
    from screwhead.sim.sim_arm import REST_SPEED
    from screwhead.sim.task_env import TaskEnv
    from screwhead.teacher.refusal import Refusal
    from screwhead.teacher.skill_teacher import SkillTeacher
    job = Job(suite, task, episodes, seed * 100 + task, cpu, 0, horizon, False, {}, "", 1, 0)
    env = TaskEnv(suite, task, horizon=horizon, seed=job.seed, render=False)
    teacher = SkillTeacher(env)
    sc, m, d = env.scene, env.scene.m, env.scene.d
    placed = _placed(env.task_spec)
    dof, radius, owner = {}, {}, {}
    for o in env.task_spec.objects:
        j = _free_joint(m, sc, o)
        if j is None:
            continue
        dof[o] = int(m.jnt_dofadr[j])
        radius[o] = float(np.linalg.norm(sc.object_box(o).half))
        owner[int(m.jnt_bodyid[j])] = o
    qadr = {o: int(m.jnt_qposadr[_free_joint(m, sc, o)]) for o in placed if o in dof}

    def of(b: int) -> str | None:
        while b > 0:
            if b in owner:
                return owner[b]
            b = int(m.body_parentid[b])
        return None

    def touched() -> set[str]:
        return {of(b) for b in contacts.robot_contacts(m, d, penetrating=False)} & set(placed)

    def moving() -> list[str]:
        return [o for o, a in dof.items()
                if float(np.linalg.norm(d.qvel[a:a + 3])) + float(np.linalg.norm(d.qvel[a + 3:a + 6])) * radius[o]
                > REST_SPEED]

    rows = []
    for ep in range(episodes):
        row, _ = _run_episode(env, teacher, job, ep, record=False)
        if row.get("refused"):
            rows.append(dict(task=task, episode=ep, refused=True))
            continue
        first, t_first, finished = bool(row["success"]), env.t, False
        if first:
            while env.t < horizon:
                s = env.snapshot()
                try:
                    a = teacher.act(dict(s, success=False))
                except (Refusal, NotImplementedError):
                    break
                env.step(a)
                if teacher.phase == "settle" and not touched() and not moving():
                    finished = True
                    break
        held, still = touched(), moving()
        excess, tipped = [], []
        for o in placed:
            humans = (ref.get(suite, {}).get(str(task), {}) or {}).get(o)
            if o in qadr and humans:
                tilt = _tilt(d.qpos[qadr[o] + 3: qadr[o] + 7])
                excess.append(tilt - max(humans))
                if excess[-1] > 0:
                    tipped.append(f"{o} {tilt:.0f}>{max(humans):.0f}")
        final = env.success()
        tilt_excess = max(excess) if excess else 0.0
        settled = first and final and finished and not held and not still and tilt_excess <= 0
        why = [w for w, bad in (("never succeeded", not first), ("lost", first and not final),
                                ("unfinished", first and not finished), ("held", bool(held)),
                                ("moving", bool(still)), ("tipped", bool(tipped))) if bad]
        rows.append(dict(task=task, episode=ep, refused=False, success=first, success_final=final,
                         finished=finished, released=not held, at_rest=not still,
                         tilt_excess_deg=round(float(tilt_excess), 1), settled=settled,
                         settle_steps=env.t - t_first if first else -1, why=why, tipped=tipped,
                         references=len(excess), placed=len(placed)))
    env.close()
    return rows


def _trial(r: dict, suite: str, seed: int, horizon: int, rev: str) -> dict:
    keys = ("success", "success_final", "finished", "released", "at_rest", "tilt_excess_deg", "settled",
            "settle_steps")
    metrics = {"refused": True} if r["refused"] else dict({k: r[k] for k in keys}, refused=False)
    return {"metrics": metrics,
            "conditions": {"task": r["task"], "suite": suite, "why": ",".join(r.get("why", [])),
                           "tipped": "; ".join(r.get("tipped", []))},
            "repro": {"seed": seed * 100 + r["task"], "task": r["task"], "task_suite": suite,
                      "episode": r["episode"], "horizon": horizon, "teacher_revision": rev}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", nargs="*", default=list(TASKS))
    ap.add_argument("--tasks", type=int, nargs="*", default=None, help="default: every task of each suite")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=557)
    ap.add_argument("--reference", action="store_true", help="build the humans' reference and stop")
    ap.add_argument("--demos", type=int, default=50)
    ap.add_argument("--cpus", default=PERF)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.reference:
        reference(args.suites, args.demos)
        return 0
    from teacher_report import teacher_revision
    rev = teacher_revision()                       # the code that runs, stamped before it runs
    if not REFERENCE.exists():                     # derived from LIBERO's datasets, rebuilt when absent
        reference(list(TASKS), args.demos)
    ref = json.loads(REFERENCE.read_text())
    cpus = [int(c) for c in args.cpus.split(",")]
    cores = mp.Manager().Queue()
    for c in cpus:
        cores.put(c)
    items = [(s, t, args.episodes, args.seed, HORIZON.get(s, 600), ref, cores)
             for s in args.suites for t in (args.tasks if args.tasks is not None else range(TASKS[s]))]
    trials, table = [], []
    with mp.get_context("spawn").Pool(len(cpus), maxtasksperchild=1) as pool:
        for (suite, task, *_), job in [(it, pool.apply_async(_task, (it,))) for it in items]:
            rows = job.get()
            scored = [r for r in rows if not r["refused"]]
            why = collections.Counter(w for r in scored for w in r["why"])
            k, n = sum(r["settled"] for r in scored), len(scored)
            ok = sum(r["success"] for r in scored)
            table.append((suite, task, k, ok, n))
            print(f"{suite:>14s}[{task:2d}] settled {k:2d}/{n:<2d} (success {ok:2d})  "
                  + ", ".join(f"{w} {c}" for w, c in why.most_common()), flush=True)
            trials += [_trial(r, suite, args.seed, HORIZON.get(suite, 600), rev) for r in rows]
    by = collections.defaultdict(lambda: [0, 0, 0])
    for suite, _t, k, ok, n in table:
        by[suite][0] += k
        by[suite][1] += ok
        by[suite][2] += n
    for suite, (k, ok, n) in by.items():
        print(f"{suite}: settled {k}/{n}, success {ok}/{n}")
    if args.out:
        Path(args.out).write_text(json.dumps({"trials": trials}))
        print("->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
