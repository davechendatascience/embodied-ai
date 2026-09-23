"""The optimization teacher's verdicts on a simulated state: settled success and the violations
(BRN-optimization-teacher; the RL environment's checks, corrected after review TRL-0214..0216).

Settled: LIBERO accepts; no robot geom touches a moved object or a goal joint's body; each moved
free object is in static equilibrium inside its contacts' friction cones (settle.py) -- at the
period's end, or, when a Watch is given, at any substep of the period (MuJoCo's contact set can
chatter: a cheese resting in a bowl cycles 2 and 3 contacts every 5 substeps) -- with too little
kinetic energy to tip over an edge of its support or to slide out of LIBERO's acceptance set;
each goal joint's predicted rest satisfies its predicate. Settled means the object cannot leave
acceptance by tipping or sliding; it does not mean motionless (a bowl may still rock flat).

Violations:
  release      -- a moved object loses its last robot contact and does not regain one within the
                  time a free fall takes to cover DEF-gentle-placement's 5 mm gap
                  (sqrt(2 * 0.005 m / g) = 31.9 ms, 16 physics substeps). Such a loss is a release,
                  and its gap beneath and vertical speed, read at the moment of loss, must meet
                  DEF-gentle-placement. The Watch carries the contact state and the open losses
                  across periods; it is part of the state a caller carries and forks for rollouts.
                  A loss still open is undecided (pending) until the watch finishes, which judges
                  it. The gap is a downward ray against collidable geoms only.
  disturbance  -- a movable body the goal does not name has a support -- the body bearing the
                  largest upward share of its weight through contact forces -- and a face down
                  that match none of the given references (e.g. the episode start's and the
                  search's start state's, so undoing a disturbance is not one); judged when asked.
                  The face is whichever body-frame face points most nearly down, so it changes at
                  a 45 degree tilt.
  lost         -- a moved object's centre of mass below the arena's support surface.
"""
from __future__ import annotations

import math
from collections import defaultdict

import mujoco
import numpy as np

_NO_CONTACTS = np.zeros((0, 2), int)

from ..geometry.box_distance import BoxSet
from ..sim import contacts
from . import settle

GENTLE_GAP = 0.005          # m, DEF-gentle-placement
GENTLE_SPEED = 0.05         # m/s, DEF-gentle-placement
DOWN = np.array([0.0, 0.0, -1.0])


class Verdicts:
    def __init__(self, env, loss):
        self.env, self.loss = env, loss
        self.lib = env.env.env
        self.m, self.d = env.scene.raw()
        m = self.m
        self.free = {}                                   # name -> root body of every free object
        for name, body in self.lib.obj_body_id.items():
            j = int(m.body_jntadr[body])
            if j >= 0 and int(m.jnt_type[j]) == mujoco.mjtJoint.mjJNT_FREE:
                self.free[name] = int(body)
        goal_objects = {g[1] for g in loss.goals if g[0] in ("in", "on")}
        named = goal_objects | {g[2] for g in loss.goals if g[0] in ("in", "on")} | {g[1] for g in loss.goals}
        self.moved = sorted(goal_objects & set(self.free))
        self.others = sorted(set(self.free) - named)
        self.goal_joints = {g: [m.joint(p["joint"]).id for p in loss._joint[g]["joints"]]
                            for g in loss.goals if g in loss._joint}
        self.goal_joint_bodies = sorted({int(m.jnt_bodyid[j]) for js in self.goal_joints.values() for j in js})
        self.robot = {b for b in range(m.nbody) if contacts.is_robot(m.body(b).name)}
        self.collidable = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
        self.fall_substeps = math.ceil(math.sqrt(2 * GENTLE_GAP / float(np.linalg.norm(m.opt.gravity)))
                                       / float(m.opt.timestep))
        self.floor = self._arena_surface()
        self._robot_mask = np.zeros(m.nbody, bool)
        self._robot_mask[list(self.robot)] = True
        self._masks: dict[int, np.ndarray] = {}      # body-id masks, one per subtree, built once
        self._subtrees = {root: settle.subtree_bodies(m, root)
                          for root in set(self.free.values()) | set(self.goal_joint_bodies)}
        self._boxes = {n: BoxSet(m, [m.geom(g).id for g in self.lib.get_object(n).contact_geoms]) for n in self.moved}

    # -- settled ------------------------------------------------------------------------------
    def settled(self, watch: Watch | None = None) -> bool:
        if not self.env.success():
            return False
        if watch is not None and watch.open:            # a contact loss not yet judged: not settled
            return False
        if any(self.robot_touches(b) for b in self.goal_joint_bodies):
            return False
        g_mag = float(np.linalg.norm(self.m.opt.gravity))
        for name in self.moved:
            root = self.free[name]
            if self.robot_touches(root):
                return False
            if not (settle.in_equilibrium(self.m, self.d, root) or (watch is not None and name in watch.supported)):
                return False
            energy = settle.kinetic_energy(self.m, self.d, root)
            if energy >= settle.tipping_barrier(self.m, self.d, root):
                return False
            goal = next(c for c in self.loss.goals if c[0] in ("in", "on") and c[1] == name)
            slide = settle.min_friction(self.m, self.d, root) * float(self.m.body_subtreemass[root]) * g_mag \
                * self.loss.slide_margin(goal)
            if energy >= slide:
                return False
        for goal, joints in self.goal_joints.items():
            for j, probe in zip(joints, self.loss._joint[goal]["joints"], strict=True):
                if probe["side"] * (settle.joint_rest(self.m, self.d, j) - probe["theta"]) <= 0:
                    return False
        return True

    def mask(self, root: int) -> np.ndarray:
        """A body-id mask for a subtree, built once. Contact queries run once per physics substep
        of every rollout, so they are array indexing rather than a Python loop over contacts."""
        m = self._masks.get(root)
        if m is None:
            m = np.zeros(self.m.nbody, bool)
            m[list(self._subtrees.get(root) or settle.subtree_bodies(self.m, root))] = True
            self._masks[root] = m
        return m

    def contact_bodies(self) -> np.ndarray:
        """The two body ids of every current contact, as an (ncon, 2) array."""
        n = self.d.ncon
        return self.m.geom_bodyid[self.d.contact.geom[:n]] if n else _NO_CONTACTS

    def robot_touches(self, root: int) -> bool:
        b = self.contact_bodies()
        if not len(b):
            return False
        obj, rob = self.mask(root)[b], self._robot_mask[b]
        return bool(np.any((obj[:, 0] & rob[:, 1]) | (obj[:, 1] & rob[:, 0])))

    # -- violations ---------------------------------------------------------------------------
    def watch(self) -> Watch:
        return Watch(self)

    def reference(self) -> dict:
        """(support, face down) of every unnamed movable body, now."""
        return {n: (self.support(n), self.face_down(n)) for n in self.others}

    def disturbed(self, *references: dict) -> str:
        """A movable body the goal does not name rests somewhere none of the references allows."""
        for n in self.others:
            now = (self.support(n), self.face_down(n))
            if all(now != r[n] for r in references):
                return f"disturbed {n}: support/face {now} matches none of {[r[n] for r in references]}"
        return ""

    def unsettled(self) -> list[str]:
        """Which unnamed movable bodies carry enough kinetic energy to leave the rest they are in
        (the tipping barrier settled() uses). Not a verdict: a nudge that settles back breaks
        nothing, but a search that ranks these lower stops knocking things over before they land."""
        return [n for n in self.others
                if settle.kinetic_energy(self.m, self.d, self.free[n])
                >= settle.tipping_barrier(self.m, self.d, self.free[n])]

    def lost(self) -> str:
        for n in self.moved:
            if float(self.d.subtree_com[self.free[n]][2]) < self.floor:
                return f"lost {n} below the arena surface"
        return ""

    # -- scene queries ------------------------------------------------------------------------
    def support(self, name: str) -> int:
        """The root body bearing the largest upward share of the object's weight through contact
        forces; -1 when nothing pushes it up."""
        root = self.free[name]
        b = self.contact_bodies()
        if not len(b):
            return -1
        inside = self.mask(root)[b]
        touching = np.flatnonzero(inside[:, 0] != inside[:, 1])   # exactly one side is the object
        share: dict[int, float] = defaultdict(float)
        f = np.zeros(6)
        for i in touching:                                        # only the contacts that matter
            i = int(i)
            c = self.d.contact[i]
            mujoco.mj_contactForce(self.m, self.d, i, f)
            on_geom2 = np.asarray(c.frame, float).reshape(3, 3).T @ f[:3]     # the force geom1 puts on geom2
            second_is_object = bool(inside[i, 1])
            up = float(on_geom2[2]) if second_is_object else -float(on_geom2[2])
            share[self._root(int(b[i, 0] if second_is_object else b[i, 1]))] += up
        best = max(share, key=share.get, default=-1)
        return best if best >= 0 and share[best] > 0 else -1

    def face_down(self, name: str) -> int:
        """Which of the object's six body-frame faces points most nearly down."""
        R = np.asarray(self.d.xmat[self.free[name]], float).reshape(3, 3)
        down = R.T @ DOWN
        axis = int(np.argmax(np.abs(down)))
        return axis * 2 + int(down[axis] > 0)

    def gap_below(self, name: str) -> float:
        """Height of the object's lowest contact-geom point above the first collidable surface
        straight beneath it (robot and the object's own geoms excluded)."""
        root = self.free[name]
        boxes = self._boxes[name]
        c, R = boxes.pose(self.d)
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
        verts = (c[:, None, :] + np.einsum("pij,pkj->pki", R, corners[None] * boxes.h[:, None, :])).reshape(-1, 3)
        low = verts[np.argmin(verts[:, 2])]
        # cast from the centre of mass's height at the lowest point's xy: a resting object sits a few
        # microns into its support, and a ray started at its lowest point would begin inside the support
        start = np.array([low[0], low[1], float(self.d.subtree_com[root][2])])
        own = self._subtrees[root]
        hit = np.inf
        for g in self.collidable:
            b = int(self.m.geom_bodyid[g])
            if b in own or b in self.robot:
                continue
            if int(self.m.geom_type[g]) == mujoco.mjtGeom.mjGEOM_MESH:
                dist = mujoco.mj_rayMesh(self.m, self.d, g, start, DOWN)
            else:
                dist = mujoco.mju_rayGeom(self.d.geom_xpos[g], self.d.geom_xmat[g], self.m.geom_size[g], start, DOWN,
                                          int(self.m.geom_type[g]))
            if dist >= 0:
                hit = min(hit, dist)
        return max(float(low[2] - (start[2] - hit)), 0.0) if np.isfinite(hit) else np.inf

    def _root(self, body: int) -> int:
        while body > 0 and int(self.m.body_parentid[body]) != 0:
            body = int(self.m.body_parentid[body])
        return body

    def _arena_surface(self) -> float:
        """The table's collision top when the arena has a table, else the floor."""
        body = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "table")
        if body < 0:
            return 0.0
        tops = [float(self.d.geom_xpos[g][2] + np.abs(self.d.geom_xmat[g].reshape(3, 3) @ self.m.geom_size[g])[2])
                for g in self.collidable
                if int(self.m.geom_bodyid[g]) == body and int(self.m.geom_type[g]) == mujoco.mjtGeom.mjGEOM_BOX]
        return max(tops) if tops else 0.0


class Watch:
    """What the verdicts carry across substeps and periods: the release check's contact state and
    open losses, and which moved objects were in equilibrium at some substep of the last period.
    substep() is SimArm._advance's substep hook (it reads the state at the start of each
    substep); period_end() reads the period's final, forwarded state; fork() copies the watch
    for a rollout; pending() is the verdict so far with open losses undecided; finish() judges
    them."""

    def __init__(self, v: Verdicts):
        self.v = v
        self.touching = {n: v.robot_touches(v.free[n]) for n in v.moved}
        self.open: dict[str, tuple[int, float, float]] = {}       # name -> (substep of loss, gap, vertical speed)
        self.count = 0
        self.violation = ""
        self.armed = False                                          # record equilibrium this period
        self.supported: set[str] = set()                            # in equilibrium at a substep this period
        self._next: bool | None = None                              # arm the next period (set at period_end)
        self._read_at_end = False                                   # period_end already read this state

    def fork(self) -> Watch:
        """A copy for a rollout: it carries the contact record and the open losses, but not a
        violation already decided on the executed trajectory -- a rollout judges what it causes."""
        w = Watch.__new__(Watch)
        w.v, w.count, w.armed, w._next, w._read_at_end = self.v, self.count, self.armed, self._next, self._read_at_end
        w.violation = ""
        w.touching, w.open, w.supported = dict(self.touching), dict(self.open), set(self.supported)
        return w

    def substep(self, _i=None) -> None:
        if self._next is not None:                                  # a new period: start its record
            self.armed, self.supported, self._next = self._next, set(), None
        if self._read_at_end:                                       # period_end already read this state
            self._read_at_end = False
            return
        self.count += 1
        self._read()
        if self.armed:
            for n in self.v.moved:
                if n not in self.supported and not self.touching[n] \
                        and settle.in_equilibrium(self.v.m, self.v.d, self.v.free[n]):
                    self.supported.add(n)

    def period_end(self, success: bool) -> None:
        """After the period's closing forward: read its final state; the period's equilibrium
        record stays readable by settled() until the next period starts, which records only if
        LIBERO accepted at this period's end (settled needs it only then). The state read here is
        the one the next period's first substep would read, so it is counted once, here."""
        self.count += 1
        self._read()
        self._read_at_end = True
        self._next = success

    def pending(self) -> str:
        return self.violation

    def doomed(self) -> str:
        """The violation an open loss would become: it was recorded at the moment of loss, so a
        plan that ends with one has already failed DEF-gentle-placement unless contact returns."""
        if self.violation:
            return self.violation
        for n, (_k, gap, speed) in self.open.items():
            if gap > GENTLE_GAP or speed > GENTLE_SPEED:
                return f"release {n} (open): gap {gap * 1000:.1f} mm, vertical speed {speed:.3f} m/s"
        return ""

    def finish(self) -> str:
        for n, (_k, gap, speed) in list(self.open.items()):
            self._judge(n, gap, speed)
        return self.violation

    def _read(self) -> None:
        for n in self.v.moved:
            now = self.v.robot_touches(self.v.free[n])
            if self.touching[n] and not now:
                mujoco.mj_subtreeVel(self.v.m, self.v.d)
                speed = abs(float(self.v.d.subtree_linvel[self.v.free[n]][2]))     # the centre of mass's
                self.open[n] = (self.count, self.v.gap_below(n), speed)
            elif now:
                self.open.pop(n, None)
            self.touching[n] = now
        for n, (k, gap, speed) in list(self.open.items()):
            if self.count - k >= self.v.fall_substeps:
                self._judge(n, gap, speed)   # the first violation is kept; contacts keep being tracked

    def _judge(self, n: str, gap: float, speed: float) -> None:
        del self.open[n]
        if not self.violation and (gap > GENTLE_GAP or speed > GENTLE_SPEED):
            self.violation = f"release {n}: gap {gap * 1000:.1f} mm, vertical speed {speed:.3f} m/s"
