"""The optimization teacher's verdicts on a simulated state: settled success and the violations
(BRN-optimization-teacher; the RL environment's checks, corrected after review TRL-0214..0216).

Settled: LIBERO accepts; no robot geom touches a moved object or a goal joint's body; each moved
free object is in static equilibrium inside its contacts' friction cones (settle.py), with too
little kinetic energy to tip over an edge of its support or to slide out of LIBERO's acceptance
set; each goal joint's predicted rest satisfies its predicate.

Violations:
  release      -- a moved object loses its last robot contact and does not regain one within the
                  time a free fall takes to cover DEF-gentle-placement's 5 mm gap
                  (sqrt(2 * 0.005 m / g) = 31.9 ms, 16 physics substeps). Such a loss is a release,
                  and its gap beneath and downward speed, read at the moment of loss, must meet
                  DEF-gentle-placement. A loss still open when the watch finishes is judged as a
                  release. The gap is a downward ray against collidable geoms only.
  disturbance  -- a movable body the goal does not name has a different support -- the body
                  bearing the largest upward share of its weight through contact forces -- or a
                  different face down than at the reference; judged when asked (the search asks at
                  the end of each control period).
  lost         -- a moved object's centre of mass below the arena's support surface.
"""
from __future__ import annotations

import math
from collections import defaultdict

import mujoco
import numpy as np

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
        self._subtrees = {root: settle.subtree_bodies(m, root)
                          for root in set(self.free.values()) | set(self.goal_joint_bodies)}
        self._boxes = {n: BoxSet(m, [m.geom(g).id for g in self.lib.get_object(n).contact_geoms]) for n in self.moved}

    # -- settled ------------------------------------------------------------------------------
    def settled(self) -> bool:
        if not self.env.success():
            return False
        if any(self.robot_touches(b) for b in self.goal_joint_bodies):
            return False
        g_mag = float(np.linalg.norm(self.m.opt.gravity))
        for name in self.moved:
            root = self.free[name]
            if self.robot_touches(root) or not settle.in_equilibrium(self.m, self.d, root):
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

    def robot_touches(self, root: int) -> bool:
        bodies = self._subtrees.get(root) or settle.subtree_bodies(self.m, root)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            b1, b2 = int(self.m.geom_bodyid[c.geom1]), int(self.m.geom_bodyid[c.geom2])
            if (b1 in bodies and b2 in self.robot) or (b2 in bodies and b1 in self.robot):
                return True
        return False

    # -- violations ---------------------------------------------------------------------------
    def release_watch(self) -> ReleaseWatch:
        return ReleaseWatch(self)

    def reference(self, start: dict | None = None) -> dict:
        """(support, face down) of every unnamed movable body: now, or -- given the episode start's
        reference -- the start's for each body whose support and face are still the start's and
        the present one for a body already changed (so a search from a disturbed state judges only
        the disturbances its own plan causes)."""
        now = {n: (self.support(n), self.face_down(n)) for n in self.others}
        if start is None:
            return now
        return {n: start[n] if now[n] == start[n] else now[n] for n in self.others}

    def disturbed(self, reference: dict) -> str:
        for n in self.others:
            now = (self.support(n), self.face_down(n))
            if now != reference[n]:
                return f"disturbed {n}: support {reference[n][0]} -> {now[0]}, face {reference[n][1]} -> {now[1]}"
        return ""

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
        bodies = self._subtrees[root]
        share: dict[int, float] = defaultdict(float)
        f = np.zeros(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            b1, b2 = int(self.m.geom_bodyid[c.geom1]), int(self.m.geom_bodyid[c.geom2])
            if (b1 in bodies) == (b2 in bodies):
                continue
            mujoco.mj_contactForce(self.m, self.d, i, f)
            on_geom2 = np.asarray(c.frame, float).reshape(3, 3).T @ f[:3]     # the force geom1 puts on geom2
            up = float(on_geom2[2]) if b2 in bodies else -float(on_geom2[2])
            share[self._root(b1 if b2 in bodies else b2)] += up
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


class ReleaseWatch:
    """Per-substep release check (the substep hook of SimArm._advance). A contact loss is judged
    once it has stayed lost for the free-fall time of the gentle gap; finish() judges the rest."""

    def __init__(self, v: Verdicts):
        self.v = v
        self.touching = {n: v.robot_touches(v.free[n]) for n in v.moved}
        self.open: dict[str, tuple[int, float, float]] = {}       # name -> (substep of loss, gap, falling speed)
        self.count = 0
        self.violation = ""

    def substep(self, _i=None) -> None:
        self.count += 1
        if self.violation:
            return
        for n in self.v.moved:
            now = self.v.robot_touches(self.v.free[n])
            if self.touching[n] and not now:
                mujoco.mj_subtreeVel(self.v.m, self.v.d)
                falling = -float(self.v.d.subtree_linvel[self.v.free[n]][2])     # the centre of mass's
                self.open[n] = (self.count, self.v.gap_below(n), falling)
            elif now:
                self.open.pop(n, None)
            self.touching[n] = now
        for n, (k, gap, down) in list(self.open.items()):
            if self.count - k >= self.v.fall_substeps:
                self._judge(n, gap, down)

    def finish(self) -> str:
        for n, (_k, gap, down) in list(self.open.items()):
            self._judge(n, gap, down)
        return self.violation

    def _judge(self, n: str, gap: float, down: float) -> None:
        del self.open[n]
        if not self.violation and (gap > GENTLE_GAP or down > GENTLE_SPEED):
            self.violation = f"release {n}: gap {gap * 1000:.1f} mm, falling {down:.3f} m/s"
