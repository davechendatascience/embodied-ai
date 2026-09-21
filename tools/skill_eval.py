#!/usr/bin/env python
"""Run the geometry-driven teacher (screwhead/skill_teacher.py) on LIBERO tasks.

  skill_eval.py --suite libero_object --episodes 5
  skill_eval.py --suite libero_object --tasks 0 1 --episodes 3 --video videos/skill

Per-task success and the phase each failure ended in, so a missing skill is visible as a
phase rather than a number. Trials go to --trials for the ledger (CTR-skill-teacher-solves-unseen).
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _track(env, teacher, s, track) -> None:
    """The four numbers that say WHY an episode ended where it did.

    A phase name says where the teacher stopped. Whether it ever had the object, whether
    it was ever pressed against something bolted down, how close it came to the grasp,
    and how close the object came to its target say what stopped it.
    """
    step = teacher.step or teacher.plan[min(teacher.step_index, len(teacher.plan) - 1)]
    if 0 <= teacher.step_index < track["step"]:
        # went BACK a step: the object was lost after it had been taken (dropped in the
        # carry, knocked out at release). Remember it -- the reset below would hide it.
        track["regressed"] = track.get("skill", "?")
    if track["step"] != teacher.step_index or track.get("skill") != step.skill:
        # per plan step: holding the drawer handle in step 1 is not holding the bowl in
        # step 2, and carrying these over labelled a failed pick "delivered-but-unscored"
        track.update(held=False, fixture=0, to_grasp=9.9, to_place=9.9, step=teacher.step_index,
                     skill=step.skill)
    try:
        if step.obj:
            _R, p_g, _w = teacher.skills.grasp_for(step.obj)
            track["to_grasp"] = min(track["to_grasp"], float(np.linalg.norm(p_g - s["p_tool"])))
            track["held"] = track["held"] or teacher.skills.held(step.obj)
        if step.skill in ("place_in", "place_on"):
            q, t = teacher.skills.place_target(step.obj, step.region, step.skill == "place_in")
            track["to_place"] = min(track["to_place"], float(np.linalg.norm(t - q)))
        elif step.skill in ("articulate", "turn"):
            art = env.scene.articulation(step.region)
            track["held"] = track["held"] or teacher.skills.holding(art["body"])
    except Exception:
        pass
    m, d = env.scene.m, env.scene.d
    for i in range(d.ncon):
        c = d.contact[i]
        if c.dist >= 0:
            continue
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        n1, n2 = m.body_id2name(b1) or "", m.body_id2name(b2) or ""
        r1, r2 = n1.startswith(("robot", "gripper")), n2.startswith(("robot", "gripper"))
        if r1 == r2:
            continue
        other = b2 if r1 else b1
        if not teacher.skills._movable(other) and (m.body_id2name(other) or "") != "table":
            track["fixture"] += 1
            break


class EpisodeLog:
    """Everything the assessment needs to say what went wrong in one episode.

    A mechanism label says which way an episode failed; fixing it needs the story: how
    long each phase took and whether the arm was tracking, when the object was taken
    and where it was lost, what was pushed before it was grasped, which fixtures the arm
    ran into, which grasp was chosen and from how many feasible ones, and -- at the end
    -- which term of LIBERO's own predicate is false.
    """
    STILL = ("squeeze", "release", "settle", "done", "retreat")

    def __init__(self, env, teacher):
        self.env, self.teacher = env, teacher
        self.timeline: list[list] = []          # [phase, t0, t1, reanchors0, clamps0]
        self.events: list[tuple[int, str]] = []
        self.track = dict(held=False, fixture=0, to_grasp=9.9, to_place=9.9, step=-1)
        self.p_hist = collections.deque(maxlen=30)
        self.stalled: set = set()
        self.touched: set = set()
        self.held_prev: dict = {}
        self.pushed: set = set()
        self.start = {st.obj: env.scene.body_pose(st.obj)[1].copy()
                      for st in teacher.plan if st.obj}

    def event(self, text: str) -> None:
        self.events.append((self.env.t, text))

    def step(self, s: dict) -> None:
        env, t = self.env, self.teacher
        ph = t.phase
        if not self.timeline or self.timeline[-1][0] != ph:
            if self.timeline:
                self.timeline[-1][2] = env.t
            self.timeline.append([ph, env.t, env.t, env.servo.reanchors, env.servo.limit_clamps])
        self.p_hist.append(np.asarray(s["p_tool"]).copy())
        if (len(self.p_hist) == self.p_hist.maxlen and ph not in self.stalled
                and not any(k in ph for k in self.STILL)):
            if float(np.ptp(np.array(self.p_hist), axis=0).max()) < 0.003:
                self.stalled.add(ph)
                self.event(f"STALL in {ph}: tool still for 30 steps, {self._where(s)}")
        if env.t % 2:
            return
        if env.t % 4 == 0:
            _track(env, t, s, self.track)
        step = t.step
        if step is not None and step.obj:
            try:
                h = t.skills.held(step.obj)
            except Exception:
                h = False
            was = self.held_prev.get(step.obj, False)
            if h and not was:
                self.event(f"grasped {step.obj} in {ph}, aperture {1000 * s['aperture']:.0f} mm")
            elif was and not h:
                q = env.scene.body_pose(step.obj)[1]
                at = False
                for st in t.plan:
                    if st.obj == step.obj and st.skill in ("place_in", "place_on"):
                        q_, tq = t.skills.place_target(st.obj, st.region, st.skill == "place_in")
                        at = t.skills.at_place(q_, tq)
                if not at:
                    self.event(f"LOST {step.obj} in {ph}: object z {q[2]:.3f}, tool z "
                               f"{s['p_tool'][2]:.3f}, aperture {1000 * s['aperture']:.0f} mm")
            self.held_prev[step.obj] = h
        if not any(self.held_prev.values()):
            for obj, p0 in self.start.items():
                if obj in self.pushed:
                    continue
                d = float(np.linalg.norm(env.scene.body_pose(obj)[1] - p0))
                if d > 0.02:
                    self.pushed.add(obj)
                    self.event(f"PUSHED {obj} {1000 * d:.0f} mm before any grasp, in {ph}, "
                               f"by {','.join(self._touching(obj)) or '?'}")
        if env.t % 6 == 0:
            for name in self._fixture_contacts():
                if name not in self.touched:
                    self.touched.add(name)
                    self.event(f"touched fixture {name} in {ph}, {self._where(s)}")

    # -- helpers ----------------------------------------------------------------------
    def _where(self, s) -> str:
        t, sk = self.teacher, self.teacher.skills
        st = t.step
        try:
            if st is None:
                return ""
            if st.skill == "pick":
                _R, p_g, _w = sk.grasp_for(st.obj)
                d = np.asarray(s["p_tool"]) - p_g
                return f"{1000 * np.linalg.norm(d):.0f} mm from grasp (dz {1000 * d[2]:+.0f})"
            if st.skill in ("place_in", "place_on"):
                q, tq = sk.place_target(st.obj, st.region, st.skill == "place_in")
                d = q - tq
                return f"object {1000 * np.linalg.norm(d):.0f} mm from target (dz {1000 * d[2]:+.0f})"
            if st.skill in ("articulate", "turn"):
                a = self.env.scene.articulation(st.region)
                hp = self.env.scene.d.geom_xpos[a["handle_geom"]] - self.env.scene.base
                return (f"{1000 * np.linalg.norm(np.asarray(s['p_tool']) - hp):.0f} mm from handle, "
                        f"joint {a['qpos']:+.3f}")
        except Exception as e:
            return f"({type(e).__name__})"
        return ""

    def _touching(self, obj) -> list[str]:
        m, d = self.env.scene.m, self.env.scene.d
        bid = self.env.scene.body_id(obj)
        out = set()
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if bid in (b1, b2):
                out.add((m.body_id2name(b2 if b1 == bid else b1) or "?").replace("_main", ""))
        return sorted(out)

    def _fixture_contacts(self) -> list[str]:
        m, d = self.env.scene.m, self.env.scene.d
        out = set()
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist >= 0:
                continue
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            n1, n2 = m.body_id2name(b1) or "", m.body_id2name(b2) or ""
            r1, r2 = n1.startswith(("robot", "gripper")), n2.startswith(("robot", "gripper"))
            if r1 != r2:
                other = b2 if r1 else b1
                if not self.teacher.skills._movable(other):
                    out.add(m.body_id2name(other) or "?")
        return sorted(out)

    def why_false(self) -> list[str]:
        """Each goal conjunct, and the term of LIBERO's predicate that is false."""
        env, t = self.env, self.teacher
        e = env.scene.env
        out = []
        for g in env.task_spec.goals:
            pred = g[0].lower()
            ok = t.satisfied(g)
            line = f"{g[0]}({', '.join(g[1:])}) = {ok}"
            try:
                if pred in ("in", "on"):
                    obj, tgt = g[1], g[2]
                    op = e.sim.data.body_xpos[e.obj_body_id[obj]]
                    if tgt in e.object_sites_dict:
                        so = e.object_sites_dict[tgt]
                        sp, sm = e.sim.data.get_site_xpos(tgt), e.sim.data.get_site_xmat(tgt)
                        size = np.asarray(so.size, float)
                        if pred == "in":
                            tot = np.abs(sm @ size)
                            rel = op - sp
                            line += (f" | contain: rel {np.round(rel, 3)} within +-{np.round(tot, 3)} "
                                     f"(z floor -{tot[2] + 0.01:.3f})")
                        else:
                            dl = sm @ (op - sp)
                            parent = e.object_states_dict[tgt].parent_name
                            contact = e.check_contact(e.get_object(parent), e.get_object(obj))
                            line += (f" | under: dz {dl[2]:+.3f} needs ({size[2] - 0.005:.3f}, "
                                     f"{size[2] + 0.10:.3f}), |dxy| {np.round(np.abs(dl[:2]), 3)} < "
                                     f"{np.round(size[:2], 3)}; contact with {parent} {contact}")
                    else:
                        tp = e.sim.data.body_xpos[e.obj_body_id[tgt]]
                        contact = e.check_contact(e.get_object(tgt), e.get_object(obj))
                        line += (f" | dxy {1000 * np.linalg.norm(op[:2] - tp[:2]):.0f} mm (< 30), "
                                 f"dz {op[2] - tp[2]:+.3f} (>= 0), contact {contact}")
                    R = e.sim.data.body_xmat[e.obj_body_id[obj]].reshape(3, 3)
                    tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
                    if tilt > 30:
                        line += f" | {obj} TIPPED {tilt:.0f} deg"
                elif pred in ("open", "close", "turnon", "turnoff"):
                    a = env.scene.articulation(g[1])
                    line += f" | joint {a['qpos']:+.3f}, thresholds {a['thresholds']}"
            except Exception as ex:
                line += f" | ({type(ex).__name__}: {ex})"
            out.append(line)
        return out

    def finish(self, success: bool) -> dict:
        if self.timeline:
            self.timeline[-1][2] = self.env.t
        rs, cl = self.env.servo.reanchors, self.env.servo.limit_clamps
        parts = []
        for i, (ph, t0, t1, r0, c0) in enumerate(self.timeline):
            r1 = self.timeline[i + 1][3] if i + 1 < len(self.timeline) else rs
            c1 = self.timeline[i + 1][4] if i + 1 < len(self.timeline) else cl
            extra = []
            if r1 - r0 > 3:
                extra.append(f"reanchor {r1 - r0}")
            if c1 - c0 > 3:
                extra.append(f"clamp {c1 - c0}")
            parts.append(f"{ph} {t1 - t0}" + (f" [{', '.join(extra)}]" if extra else ""))
        return dict(timeline=" > ".join(parts),
                    events=[f"t{t} {x}" for t, x in self.events],
                    grasp={k: v for k, v in self.teacher.skills.grasp_log.items()},
                    final=[] if success else self.why_false())


def _mechanism(track, missing: str) -> str:
    """One of five ways an episode fails, from those numbers, prefixed by the plan step
    it failed in (pick, place_in, articulate, ...)."""
    if missing:
        return "unimplemented"
    skill = track.get("skill", "?")
    if skill in ("articulate", "turn"):
        return f"{skill}/" + ("never-held-handle" if not track["held"] else "held-but-not-moved")
    if not track["held"]:
        if track["to_grasp"] > 0.03:
            m = "blocked-reaching" if track["fixture"] else "never-reached-grasp"
        else:
            m = "reached-but-no-grip"
    elif track["to_place"] > 0.05:
        m = "held-but-not-delivered"
    else:
        m = "delivered-but-unscored"
    lost = track.get("regressed")
    return f"{skill}/{m}" + (f" (after losing it in {lost})" if lost else "")


def _diagnose(env, teacher, missing: str) -> str:
    """Why this episode ended where it did, in one line.

    A phase name says where the teacher stopped, not what stopped it. These four say what:
    whether the servo was tracking at all (re-anchors), what the robot was touching, how
    far the tool was from the pose it was asking for, and whether the jaws were moving.
    """
    if missing:
        return f"unimplemented {missing}"
    m, d = env.scene.m, env.scene.d
    touching = set()
    for i in range(d.ncon):
        c = d.contact[i]
        if c.dist >= 0:
            continue
        n1 = m.body_id2name(m.geom_bodyid[c.geom1]) or ""
        n2 = m.body_id2name(m.geom_bodyid[c.geom2]) or ""
        r1, r2 = n1.startswith(("robot", "gripper")), n2.startswith(("robot", "gripper"))
        if r1 != r2:
            touching.add((n2 if r1 else n1).replace("_main", ""))
    s = env.snapshot()
    bits = [f"reanchor {env.servo.reanchors}", f"ap {s['aperture'] * 1000:.0f}mm"]
    step = teacher.plan[min(teacher.step_index, len(teacher.plan) - 1)]
    try:
        if step.skill == "pick" or step.obj:
            _R, p_g, _w = teacher.skills.grasp_for(step.obj)
            bits.append(f"to_grasp {np.linalg.norm(p_g - s['p_tool']) * 1000:.0f}mm")
        if step.skill in ("place_in", "place_on"):
            q, t = teacher.skills.place_target(step.obj, step.region, step.skill == "place_in")
            bits.append(f"to_place {np.linalg.norm(t - q) * 1000:.0f}mm")
    except Exception as e:                       # geometry gone (object off the table)
        bits.append(f"geom? {type(e).__name__}")
    bits.append("touch " + (",".join(sorted(touching)[:3]) or "-"))
    return " ".join(bits)


def _worker(remote, suite, task, episodes, seed, cpu, kw, video_dir, max_videos, ep_offset=0):
    os.sched_setaffinity(0, {cpu})
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT))
    from screwhead.skill_teacher import SkillTeacher
    from screwhead.task_env import TaskEnv
    env = TaskEnv(suite, task, seed=seed, render=True, **kw)
    teacher = SkillTeacher(env)
    rows, videos = [], 0
    for ep in range(episodes):
        env.reset()
        frames, done, info = [], False, {}
        phases = collections.Counter()
        missing = ""
        log = EpisodeLog(env, teacher)
        track = log.track
        while not done:
            s = env.snapshot()
            try:
                a = teacher.act(s)
            except NotImplementedError as e:      # a skill this suite needs and we lack:
                missing = str(e).split("(")[0].strip()   # report it, do not kill the worker
                teacher.phase = f"unimplemented:{missing}"
                a = np.zeros(7)
            phases[teacher.phase] += 1
            log.step(s)
            if video_dir and videos < max_videos:
                ag, wr = env.images()
                frames.append(np.concatenate([ag[::-1], wr[::-1]], axis=1))
            _, _, done, info = env.step(a)
        rows.append(dict(task=task, episode=ep_offset + ep, success=bool(info["success"]), steps=env.t,
                         last_phase=teacher.phase, step_index=teacher.step_index,
                         phases=dict(phases), language=env.language,
                         diag="" if info["success"] else _diagnose(env, teacher, missing),
                         mechanism="" if info["success"] else _mechanism(track, missing),
                         detail=log.finish(bool(info["success"])),
                         **{k: round(float(v), 4) if isinstance(v, float) else v
                            for k, v in track.items() if k not in ("skill", "regressed")}))
        if video_dir and frames and videos < max_videos and not info["success"]:
            import imageio.v2 as imageio
            Path(video_dir).mkdir(parents=True, exist_ok=True)
            imageio.mimsave(Path(video_dir) / f"{suite}_t{task}_ep{ep}_fail.mp4", frames, fps=20,
                            macro_block_size=1)
            videos += 1
        remote.send(rows[-1])
    remote.send(None)
    env.close()
    remote.close()


def main() -> int:
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
    ap.add_argument("--cpus", default="5,6,7,8,9,15,16,17,18,19")
    ap.add_argument("--seed", type=int, default=555)
    ap.add_argument("--split", type=int, default=1,
                    help="workers per task, each running a share of the episodes on its own "
                         "seed -- for iterating on one task without waiting on one core")
    ap.add_argument("--video", default="")
    ap.add_argument("--max-videos", type=int, default=1)
    ap.add_argument("--trials", default="")
    ap.add_argument("-v", "--verbose", action="store_true", help="timeline, events, grasp, "
                    "and the false predicate term for every failed episode")
    args = ap.parse_args()

    tasks = args.tasks if args.tasks is not None else list(range(10))
    cpus = [int(c) for c in args.cpus.split(",")]
    kw = dict(horizon=args.horizon, gripper_mode="target", start_xy_m=args.start_xy,
              start_z_m=args.start_z, start_yaw_deg=args.start_yaw, start_tilt_deg=args.start_tilt,
              start_null_rad=args.start_null)
    ctx = mp.get_context("spawn")
    procs, remotes = [], []
    jobs = []
    per = -(-args.episodes // args.split)
    for t in tasks:
        for j in range(args.split):
            n = min(per, args.episodes - j * per)
            if n > 0:
                seed = args.seed * 100 + t if args.split == 1 else (args.seed * 100 + t) * 1000 + j
                jobs.append((t, n, seed, j * per))
    for i, (t, n, seed, off) in enumerate(jobs):
        a, b = ctx.Pipe()
        p = ctx.Process(target=_worker, args=(b, args.suite, t, n, seed, cpus[i % len(cpus)], kw,
                                              args.video, args.max_videos, off), daemon=True)
        p.start(); b.close(); procs.append(p); remotes.append(a)
    rows, open_pipes, t0 = [], set(range(len(remotes))), time.time()
    while open_pipes:
        for i in list(open_pipes):
            try:
                r = remotes[i].recv()
            except EOFError:
                open_pipes.discard(i); continue
            if r is None:
                open_pipes.discard(i); continue
            rows.append(r)
            print(f"  {args.suite} task {r['task']} ep {r['episode']}: "
                  f"{'ok' if r['success'] else 'fail'} steps {r['steps']} last {r['last_phase']}"
                  + (f" | {r['diag']}" if r.get("diag") else ""), flush=True)
    for p in procs:
        p.join(timeout=10)
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
                print(f"      ep{r['episode']} {r['last_phase']:22s} [{r['mechanism']}] {r['diag']}")
                d = r.get("detail", {})
                if args.verbose and d:
                    print(f"         timeline: {d['timeline']}")
                    for ev in d["events"]:
                        print(f"         {ev}")
                    for obj, gl in d["grasp"].items():
                        print(f"         grasp {obj}: {gl}")
                    for line in d["final"]:
                        print(f"         final: {line}")
    ok = sum(r["success"] for r in rows)
    print(f"{args.suite}: {ok}/{len(rows)} = {ok / max(len(rows), 1):.2f}   ({time.time() - t0:.0f}s)")
    if args.trials:
        Path(args.trials).parent.mkdir(parents=True, exist_ok=True)
        Path(args.trials).write_text(json.dumps({"trials": [
            {"metrics": {"success": r["success"]},
             "conditions": {"task": r["task"], "suite": args.suite, "last_phase": r["last_phase"],
                            "mechanism": r.get("mechanism", "")},
             "detail": dict(r.get("detail", {}), episode=r["episode"], steps=r["steps"],
                            language=r["language"]),
             "repro": {"seed": args.seed * 100 + r["task"], "task": r["task"], "task_suite": args.suite,
                       "teacher_revision": "skill_teacher"}} for r in rows]}, indent=1))
        print("->", args.trials)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
