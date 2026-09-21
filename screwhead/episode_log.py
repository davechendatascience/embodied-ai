"""What happened in one teacher episode, and why it failed.

A mechanism label says which way an episode failed; fixing it needs the story: how long
each phase took and whether the arm was tracking, when the object was taken and where it
was lost, what was pushed before it was grasped, which fixtures the arm ran into, which
grasp was chosen from how many feasible ones, and -- at the end -- which term of LIBERO's
own predicate is false. tools/skill_eval.py records one of these per episode and
tools/teacher_report.py aggregates them.
"""
from __future__ import annotations

import collections
import contextlib

import numpy as np

from . import contacts

STALL_STEPS = 30          # the tool still for this many steps ...
STALL_SPAN = 0.003        # ... within this many metres is a stall
PUSH_DIST = 0.02          # an object moved this far before any grasp was pushed
REACHED = 0.03            # the tool came within this of the grasp
DELIVERED = 0.05          # the object came within this of its target
UNSEEN = 9.9              # sentinel distance: never measured
TILTED_DEG = 30.0         # an object tilted more than this has tipped over
BIG_DELTA = 3             # re-anchors / clamps per phase worth showing in a timeline
ON_BAND_BELOW = 0.005     # LIBERO's under(): the object origin within this below ...
ON_BAND_ABOVE = 0.10      # ... and this above the region's top
IN_FLOOR = 0.01           # LIBERO's in_box() lowers the region's floor by this
STILL = ("squeeze", "release", "settle", "done", "retreat")
GEOMETRY_ERRORS = (ValueError, KeyError, IndexError)     # an object or region not found


class EpisodeLog:
    def __init__(self, env, teacher):
        self.env, self.teacher = env, teacher
        self.timeline: list[list] = []          # [phase, t0, t1, reanchors0, clamps0]
        self.events: list[tuple[int, str]] = []
        self.track = dict(held=False, fixture=0, to_grasp=UNSEEN, to_place=UNSEEN, step=-1)
        self.p_hist = collections.deque(maxlen=STALL_STEPS)
        self.stalled: set = set()
        self.touched: set = set()
        self.held_prev: dict = {}
        self.pushed: set = set()
        self.ever_held = False
        self.start = {st.obj: env.scene.body_pose(st.obj)[1].copy() for st in teacher.plan if st.obj}

    def event(self, text: str) -> None:
        self.events.append((self.env.t, text))

    # -- per step ------------------------------------------------------------------------
    def step(self, s: dict) -> None:
        env, ph = self.env, self.teacher.phase
        self._timeline(ph)
        self._stall(s, ph)
        if env.t % 2:
            return
        if env.t % 4 == 0:
            self._track(s)
        self._held_events(s, ph)
        if not self.ever_held:                           # after a drop, a moved object was not "pushed"
            self._pushes(ph)
        if env.t % 6 == 0:
            self._fixtures(s, ph)

    def _timeline(self, ph: str) -> None:
        env = self.env
        if not self.timeline or self.timeline[-1][0] != ph:
            if self.timeline:
                self.timeline[-1][2] = env.t
            self.timeline.append([ph, env.t, env.t, env.servo.reanchors, env.servo.limit_clamps])

    def _stall(self, s: dict, ph: str) -> None:
        self.p_hist.append(np.asarray(s["p_tool"]).copy())
        if (len(self.p_hist) == self.p_hist.maxlen and ph not in self.stalled
                and not any(k in ph for k in STILL)
                and float(np.ptp(np.array(self.p_hist), axis=0).max()) < STALL_SPAN):
            self.stalled.add(ph)
            self.event(f"STALL in {ph}: tool still for {STALL_STEPS} steps, {self._where(s)}")

    def _held_events(self, s: dict, ph: str) -> None:
        t, step = self.teacher, self.teacher.step
        if step is None or not step.obj:
            return
        h = t.skills.held(step.obj)
        was = self.held_prev.get(step.obj, False)
        if h and not was:
            self.ever_held = True
            self.event(f"grasped {step.obj} in {ph}, aperture {1000 * s['aperture']:.0f} mm")
        elif was and not h and not self._delivered(step.obj):
            q = self.env.scene.body_pose(step.obj)[1]
            self.event(f"LOST {step.obj} in {ph}: object z {q[2]:.3f}, tool z "
                       f"{s['p_tool'][2]:.3f}, aperture {1000 * s['aperture']:.0f} mm")
        self.held_prev[step.obj] = h

    def _delivered(self, obj: str) -> bool:
        """A release at the target is the plan; anywhere else it is a loss."""
        sk, at = self.teacher.skills, False
        for st in self.teacher.plan:
            if st.obj == obj and st.skill in ("place_in", "place_on"):
                tgt = sk.place_target(st.obj, st.region, st.skill == "place_in", decide=False)
                at = tgt is not None and sk.at_place(*tgt)
        return at

    def _pushes(self, ph: str) -> None:
        m, d = self.env.scene.m, self.env.scene.d
        for obj, p0 in self.start.items():
            if obj in self.pushed:
                continue
            dist = float(np.linalg.norm(self.env.scene.body_pose(obj)[1] - p0))
            if dist > PUSH_DIST:
                self.pushed.add(obj)
                by = sorted({contacts.body_name(m, b).replace("_main", "") or "?"
                             for b in contacts.touching(m, d, self.env.scene.body_id(obj))})
                self.event(f"PUSHED {obj} {1000 * dist:.0f} mm before any grasp, in {ph}, "
                           f"by {','.join(by) or '?'}")

    def _fixtures(self, s: dict, ph: str) -> None:
        m, d = self.env.scene.m, self.env.scene.d
        names = sorted({contacts.body_name(m, b) or "?" for b in contacts.robot_contacts(m, d)
                        if not contacts.is_movable(m, b)})
        for name in names:
            if name not in self.touched:
                self.touched.add(name)
                self.event(f"touched fixture {name} in {ph}, {self._where(s)}")

    def _where(self, s) -> str:
        """How far the tool (or the object) is from what the current step is aiming at."""
        st, sk = self.teacher.step, self.teacher.skills
        try:
            if st is None:
                return ""
            if st.skill == "pick":
                p_g = sk.cached_grasp(st.obj)
                if p_g is None:
                    return "no grasp chosen yet"
                d = np.asarray(s["p_tool"]) - p_g
                return f"{1000 * np.linalg.norm(d):.0f} mm from grasp (dz {1000 * d[2]:+.0f})"
            if st.skill in ("place_in", "place_on"):
                tgt = sk.place_target(st.obj, st.region, st.skill == "place_in", decide=False)
                if tgt is None:
                    return "no drop point chosen yet"
                d = tgt[0] - tgt[1]
                return f"object {1000 * np.linalg.norm(d):.0f} mm from target (dz {1000 * d[2]:+.0f})"
            if st.skill in ("articulate", "turn"):
                a = self.env.scene.articulation(st.region)
                hp = self.env.scene.d.geom_xpos[a["handle_geom"]] - self.env.scene.base
                return (f"{1000 * np.linalg.norm(np.asarray(s['p_tool']) - hp):.0f} mm from handle, "
                        f"joint {a['qpos']:+.3f}")
        except GEOMETRY_ERRORS as e:
            return f"({type(e).__name__})"
        return ""

    # -- the numbers the mechanism is read from ------------------------------------------------
    def _track(self, s: dict) -> None:
        """Per plan step: whether the object was ever held, whether the arm was ever
        pressed against something bolted down, and how close the tool came to the grasp
        and the object to its target."""
        t, track = self.teacher, self.track
        step = t.step or t.plan[min(t.step_index, len(t.plan) - 1)]
        if 0 <= t.step_index < track["step"]:
            # went BACK a step: the object was lost after it had been taken. Remember it --
            # the reset below would hide it.
            track["regressed"] = track.get("skill", "?")
        if track["step"] != t.step_index or track.get("skill") != step.skill:
            # holding the drawer handle in step 1 is not holding the bowl in step 2
            track.update(held=False, fixture=0, to_grasp=UNSEEN, to_place=UNSEEN,
                         step=t.step_index, skill=step.skill)
        with contextlib.suppress(*GEOMETRY_ERRORS):      # the object has left the scene
            self._track_distances(step, s)
        m, d = self.env.scene.m, self.env.scene.d
        if any(not contacts.is_movable(m, b) and contacts.body_name(m, b) != "table"
               for b in contacts.robot_contacts(m, d)):
            track["fixture"] += 1

    def _track_distances(self, step, s: dict) -> None:
        sk, track = self.teacher.skills, self.track
        if step.obj:
            p_g = sk.cached_grasp(step.obj)
            if p_g is not None:
                track["to_grasp"] = min(track["to_grasp"], float(np.linalg.norm(p_g - s["p_tool"])))
            track["held"] = track["held"] or sk.held(step.obj)
        if step.skill in ("place_in", "place_on"):
            tgt = sk.place_target(step.obj, step.region, step.skill == "place_in", decide=False)
            if tgt is not None:
                track["to_place"] = min(track["to_place"], float(np.linalg.norm(tgt[1] - tgt[0])))
        elif step.skill in ("articulate", "turn"):
            art = self.env.scene.articulation(step.region)
            w = sk.handle_width(step.region)
            track["held"] = track["held"] or (w is not None and sk.holding(art["body"], w))

    def mechanism(self, missing: str) -> str:
        """One of five ways an episode fails, prefixed by the plan step it failed in."""
        if missing:
            return "unimplemented"
        track = self.track
        skill = track.get("skill", "?")
        if skill in ("articulate", "turn"):
            return f"{skill}/" + ("never-held-handle" if not track["held"] else "held-but-not-moved")
        if not track["held"]:
            if track["to_grasp"] > REACHED:
                m = "blocked-reaching" if track["fixture"] else "never-reached-grasp"
            else:
                m = "reached-but-no-grip"
        elif track["to_place"] > DELIVERED:
            m = "held-but-not-delivered"
        else:
            m = "delivered-but-unscored"
        lost = track.get("regressed")
        return f"{skill}/{m}" + (f" (after losing it in {lost})" if lost else "")

    def summary(self, missing: str) -> str:
        """Why the episode ended where it did, in one line: servo tracking, jaws, distance
        to the grasp and target, and what the robot is touching."""
        env, t = self.env, self.teacher
        if missing:
            return f"unimplemented {missing}"
        m, d = env.scene.m, env.scene.d
        touching = sorted({contacts.body_name(m, b).replace("_main", "") for b in contacts.robot_contacts(m, d)})
        s = env.snapshot()
        bits = [f"reanchor {env.servo.reanchors}", f"ap {s['aperture'] * 1000:.0f}mm"]
        step = t.plan[min(t.step_index, len(t.plan) - 1)]
        try:
            p_g = t.skills.cached_grasp(step.obj) if step.obj else None
            if p_g is not None:
                bits.append(f"to_grasp {np.linalg.norm(p_g - s['p_tool']) * 1000:.0f}mm")
            if step.skill in ("place_in", "place_on"):
                tgt = t.skills.place_target(step.obj, step.region, step.skill == "place_in", decide=False)
                if tgt is not None:
                    bits.append(f"to_place {np.linalg.norm(tgt[1] - tgt[0]) * 1000:.0f}mm")
        except GEOMETRY_ERRORS as e:                     # the object has left the scene
            bits.append(f"geom? {type(e).__name__}")
        bits.append("touch " + (",".join(touching[:3]) or "-"))
        return " ".join(bits)

    # -- the end -------------------------------------------------------------------------------
    def why_false(self) -> list[str]:
        """Each goal conjunct, and the term of LIBERO's predicate that is false."""
        out = []
        for g in self.env.task_spec.goals:
            pred = g[0].lower()
            line = f"{g[0]}({', '.join(g[1:])}) = {self.teacher.satisfied(g)}"
            try:
                if pred in ("in", "on"):
                    line += self._explain_placement(pred, g[1], g[2])
                elif pred in ("open", "close", "turnon", "turnoff"):
                    a = self.env.scene.articulation(g[1])
                    line += f" | joint {a['qpos']:+.3f}, thresholds {a['thresholds']}"
            except (*GEOMETRY_ERRORS, AttributeError) as ex:
                line += f" | ({type(ex).__name__}: {ex})"
            out.append(line)
        return out

    def _explain_placement(self, pred: str, obj: str, tgt: str) -> str:
        e = self.env.scene.env
        op = e.sim.data.body_xpos[e.obj_body_id[obj]]
        if tgt in e.object_sites_dict:
            sp, sm = e.sim.data.get_site_xpos(tgt), e.sim.data.get_site_xmat(tgt)
            size = np.asarray(e.object_sites_dict[tgt].size, float)
            if pred == "in":
                tot = np.abs(sm @ size)
                line = (f" | contain: rel {np.round(op - sp, 3)} within +-{np.round(tot, 3)} "
                        f"(z floor -{tot[2] + IN_FLOOR:.3f})")
            else:
                dl = sm @ (op - sp)
                parent = e.object_states_dict[tgt].parent_name
                # a region of the table itself has no parent, and LIBERO's On then checks
                # position only -- asking robosuite for contact with None raised TypeError
                contact = (e.check_contact(e.get_object(parent), e.get_object(obj))
                           if parent is not None else "not checked")
                line = (f" | under: dz {dl[2]:+.3f} needs ({size[2] - ON_BAND_BELOW:.3f}, "
                        f"{size[2] + ON_BAND_ABOVE:.3f}), |dxy| {np.round(np.abs(dl[:2]), 3)} < "
                        f"{np.round(size[:2], 3)}; contact with {parent} {contact}")
        else:
            tp = e.sim.data.body_xpos[e.obj_body_id[tgt]]
            contact = e.check_contact(e.get_object(tgt), e.get_object(obj))
            line = (f" | dxy {1000 * np.linalg.norm(op[:2] - tp[:2]):.0f} mm (< 30), "
                    f"dz {op[2] - tp[2]:+.3f} (>= 0), contact {contact}")
        R = e.sim.data.body_xmat[e.obj_body_id[obj]].reshape(3, 3)
        tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
        return line + (f" | {obj} TIPPED {tilt:.0f} deg" if tilt > TILTED_DEG else "")

    def finish(self, success: bool) -> dict:
        if self.timeline:
            self.timeline[-1][2] = self.env.t
        ends = [row[3:5] for row in self.timeline[1:]] + [[self.env.servo.reanchors, self.env.servo.limit_clamps]]
        parts = []
        for (ph, t0, t1, r0, c0), (r1, c1) in zip(self.timeline, ends, strict=True):
            extra = ([f"reanchor {r1 - r0}"] if r1 - r0 > BIG_DELTA else []) + \
                    ([f"clamp {c1 - c0}"] if c1 - c0 > BIG_DELTA else [])
            parts.append(f"{ph} {t1 - t0}" + (f" [{', '.join(extra)}]" if extra else ""))
        return dict(timeline=" > ".join(parts), events=[f"t{t} {x}" for t, x in self.events],
                    grasp=dict(self.teacher.skills.grasp_log), final=[] if success else self.why_false())
