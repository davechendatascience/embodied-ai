"""Object-layout randomisation that keeps each LIBERO-spatial instruction true.

Every task has the same five free objects -- the target bowl (t), the other bowl
(o), the cookie box, the ramekin and the plate -- plus two fixtures, the stove and
the wooden cabinet. What an instruction names is a RELATION ("between the plate
and the ramekin", "next to the cookie box", "on the ramekin"), and in this suite
that relation is what tells the two black bowls apart. So objects move, and a
layout is kept only if the relation still singles out the target.

Groups move rigidly: a bowl resting on the cookie box or the ramekin moves with
it. Bowls on fixtures (drawer, stove, cabinet top) shift only a little within
their support. Everything on the table shifts within a disc and turns.

A proposal is accepted only if it survives, in order:
  before settling  no two objects' footprints overlap; all stay on the table
                   and off the fixtures' footprints
  after settling   nothing slid or was ejected (10 mm), the relation holds on the
                   poses that physics produced, and the other bowl is not on or
                   crowding the plate
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NAMES = dict(t="akita_black_bowl_1", o="akita_black_bowl_2", cookies="cookies_1",
             ramekin="glazed_rim_porcelain_ramekin_1", plate="plate_1")


@dataclass(frozen=True)
class Group:
    members: tuple          # keys of NAMES; first member is the reference
    mode: str               # "table" | "local" | "fixed"
    radius: float = 0.0     # local shift radius (m); table groups use the sampler radius
    yaw: bool = True


def _xy(P, k):
    return P[k][:2]


def _between(P, a, b, k, u_rng=(0.2, 0.8), d_max=0.05):
    A, B, X = _xy(P, a), _xy(P, b), _xy(P, k)
    ab = B - A
    u = float((X - A) @ ab / (ab @ ab))
    d = float(np.linalg.norm(A + u * ab - X))
    return u_rng[0] <= u <= u_rng[1] and d <= d_max, (u, d)


def relation_ok(task: int, P: dict) -> bool:
    """P: key -> base-frame position after settling."""
    dist = lambda a, b: float(np.linalg.norm(_xy(P, a) - _xy(P, b)))
    if dist("o", "plate") < 0.12:                      # the other bowl must not be on/at the plate
        return False
    if task == 0:
        ok_t, _ = _between(P, "plate", "ramekin", "t")
        ok_o, (u_o, d_o) = _between(P, "plate", "ramekin", "o", u_rng=(0.05, 0.95), d_max=0.09)
        return ok_t and not ok_o
    if task == 1:
        return dist("t", "ramekin") < 0.16 and dist("o", "ramekin") > dist("t", "ramekin") + 0.06
    if task == 2:
        c = np.array([0.585, 0.015])                   # LIBERO's own table-centre placement
        return float(np.linalg.norm(_xy(P, "t") - c)) < 0.05 and float(np.linalg.norm(_xy(P, "o") - c)) > 0.15
    if task == 6:
        return dist("t", "cookies") < 0.14 and dist("o", "cookies") > dist("t", "cookies") + 0.06
    if task == 8:
        return dist("t", "plate") < 0.16 and dist("o", "plate") > dist("t", "plate") + 0.06
    return True     # 3, 4, 5, 7, 9: the relation is a support, kept by grouping and checked below


SUPPORT = {3: "cookies", 5: "ramekin"}                 # target must still rest on this after settling

GROUPS = {
    0: [Group(("t",), "table"), Group(("o",), "table"), Group(("cookies",), "table"),
        Group(("ramekin",), "table"), Group(("plate",), "table")],
    1: [Group(("t",), "table"), Group(("o",), "table"), Group(("cookies",), "table"),
        Group(("ramekin",), "table"), Group(("plate",), "table")],
    2: [Group(("t",), "local", 0.04), Group(("o",), "table"), Group(("cookies",), "table"),
        Group(("ramekin",), "table"), Group(("plate",), "table")],
    3: [Group(("cookies", "t"), "table", yaw=False), Group(("o",), "fixed"),
        Group(("ramekin",), "table"), Group(("plate",), "table")],
    4: [Group(("t",), "fixed"), Group(("o",), "fixed"), Group(("cookies",), "table"),
        Group(("ramekin",), "table"), Group(("plate",), "table")],
    5: [Group(("ramekin", "t"), "table", yaw=False), Group(("cookies", "o"), "table", yaw=False),
        Group(("plate",), "table")],
    6: [Group(("t",), "table"), Group(("o",), "fixed"), Group(("cookies",), "table"),
        Group(("ramekin",), "table"), Group(("plate",), "table")],
    7: [Group(("t",), "local", 0.01), Group(("o",), "fixed"), Group(("cookies",), "table"),
        Group(("ramekin",), "table"), Group(("plate",), "table")],
    8: [Group(("t",), "table"), Group(("o",), "table"), Group(("cookies",), "table"),
        Group(("ramekin",), "table"), Group(("plate",), "table")],
    9: [Group(("t",), "local", 0.02), Group(("o",), "fixed"), Group(("cookies",), "table"),
        Group(("ramekin",), "table"), Group(("plate",), "table")],
}

TABLE_X, TABLE_Y = (0.33, 0.88), (-0.36, 0.42)        # base frame, reachable table region


def _yaw_quat(a):
    return np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])


def _qmul(q, r):
    w0, x0, y0, z0 = q; w1, x1, y1, z1 = r
    return np.array([w0*w1 - x0*x1 - y0*y1 - z0*z1, w0*x1 + x0*w1 + y0*z1 - z0*y1,
                     w0*y1 - x0*z1 + y0*w1 + z0*x1, w0*z1 + x0*y1 - y0*x1 + z0*w1])


class LayoutSampler:
    def __init__(self, env, radius: float = 0.08, max_tries: int = 60):
        self.env, self.radius, self.max_tries = env, radius, max_tries
        self.task = env.ti
        self.groups = GROUPS[self.task]
        self.last_tries = 0
        import collections
        self.reasons = collections.Counter()

    # -- geometry read from the model ------------------------------------------------
    def _handles(self):
        sim = self.env.env.sim; m, d = sim.model, sim.data
        h = {}
        for k, n in NAMES.items():
            j = m.joint_name2id(f"{n}_joint0")
            h[k] = (int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j]), int(m.jnt_bodyid[j]))
        return h

    def _footprint(self, body):
        sim = self.env.env.sim; m, d = sim.model, sim.data
        c = d.body_xpos[body][:2]; r = 0.0
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) != body or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            r = max(r, float(np.linalg.norm(d.geom_xpos[g][:2] - c)) + float(np.max(m.geom_size[g][:2])))
        return r

    def _fixture_boxes(self, tb):
        sim = self.env.env.sim; m, d = sim.model, sim.data
        boxes = []
        for prefix in ("flat_stove_1", "wooden_cabinet_1"):
            pts = [d.geom_xpos[g][:2] - tb[:2] for g in range(m.ngeom)
                   if (m.body_id2name(int(m.geom_bodyid[g])) or "").startswith(prefix)]
            if pts:
                P = np.array(pts); boxes.append((P.min(0) - 0.03, P.max(0) + 0.03))
        return boxes

    def compatible(self) -> bool:
        """The settled init state touches only where this task's grouping expects.

        6 of LIBERO's 500 init states lean one object against another (tasks 0, 2, 8:
        a bowl against the plate or the ramekin). Moving such a pair apart drops the
        leaner -- measured, 1200 of 1200 proposals rejected -- so those init states
        are redrawn in layout mode rather than retried forever.
        """
        sim = self.env.env.sim; m, d = sim.model, sim.data
        h = self._handles()
        for k, (_, v, b) in h.items():
            tilt = np.degrees(np.arccos(np.clip(d.body_xmat[b].reshape(3, 3)[2, 2], -1.0, 1.0)))
            if tilt > 10.0 or np.linalg.norm(d.qvel[v:v + 3]) > 0.01:
                return False                  # knocked over, or still moving
        body_key = {b: k for k, (_, _, b) in h.items()}
        same_group = {frozenset((a, b)) for g in self.groups for a in g.members for b in g.members if a != b}
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if b1 in body_key and b2 in body_key and body_key[b1] != body_key[b2]:
                if frozenset((body_key[b1], body_key[b2])) not in same_group:
                    return False
        return True

    # -- sampling ------------------------------------------------------------------
    def sample(self, rng, settle) -> bool:
        """Apply a valid layout to the environment's current (settled) state. Returns
        False if none was found; the caller keeps LIBERO's layout in that case."""
        env = self.env; sim = env.env.sim; m, d = sim.model, sim.data
        h = self._handles()
        tb = d.body_xpos[m.body_name2id("robot0_base")].copy()
        saved = np.asarray(sim.get_state().flatten()).copy()
        q0 = {k: d.qpos[a:a + 7].copy() for k, (a, _, _) in h.items()}
        foot = {k: self._footprint(b) for k, (_, _, b) in h.items()}
        fixtures = self._fixture_boxes(tb)
        for attempt in range(self.max_tries):
            sim.set_state_from_flattened(saved); sim.forward()
            intended = {}
            table_items = []
            for g in self.groups:
                if g.mode == "fixed":
                    for k in g.members:
                        intended[k] = q0[k][:3].copy()
                    continue
                R = self.radius if g.mode == "table" else g.radius
                r = R * np.sqrt(rng.random()); th = 2 * np.pi * rng.random()
                shift = np.array([r * np.cos(th), r * np.sin(th), 0.0])
                yaw = rng.uniform(-np.pi, np.pi) if (g.yaw and g.mode == "table") else 0.0
                ref = q0[g.members[0]][:3]
                for k in g.members:
                    a, v, _ = h[k]
                    rel = q0[k][:3] - ref
                    c, s = np.cos(yaw), np.sin(yaw)
                    rel = np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1], rel[2]])
                    d.qpos[a:a + 3] = ref + shift + rel
                    d.qpos[a + 3:a + 7] = _qmul(_yaw_quat(yaw), q0[k][3:7])
                    d.qvel[v:v + 6] = 0.0
                    intended[k] = d.qpos[a:a + 3].copy()
                if g.mode == "table":
                    table_items.append((g.members, (ref + shift)[:2] - tb[:2], max(foot[k] for k in g.members)))
            # before settling: footprints, table bounds, fixtures
            ok = True
            for i, (_, c1, r1) in enumerate(table_items):
                if not (TABLE_X[0] <= c1[0] <= TABLE_X[1] and TABLE_Y[0] <= c1[1] <= TABLE_Y[1]):
                    ok = False; self.reasons["off table"] += 1; break
                if any((lo - r1 <= c1).all() and (c1 <= hi + r1).all() for lo, hi in fixtures):
                    ok = False; self.reasons["on a fixture"] += 1; break
                for _, c2, r2 in table_items[i + 1:]:
                    if np.linalg.norm(c1 - c2) < r1 + r2 + 0.01:
                        ok = False; self.reasons["overlap"] += 1; break
                if not ok:
                    break
            if not ok:
                continue
            sim.forward()
            settle(10)
            # after settling: nothing slid or was ejected, and the relation holds
            now = {k: d.qpos[h[k][0]:h[k][0] + 3].copy() for k in h}
            if any(np.linalg.norm(now[k][:2] - intended[k][:2]) > 0.010 or abs(now[k][2] - intended[k][2]) > 0.010
                   for k in intended):
                self.reasons["slid or ejected"] += 1
                worst = max(intended, key=lambda k: max(np.linalg.norm(now[k][:2] - intended[k][:2]), abs(now[k][2] - intended[k][2])))
                self.reasons[f"  worst mover: {worst}"] += 1
                self.last_slide = {k: (round(float(np.linalg.norm(now[k][:2] - intended[k][:2])) * 1000, 1),
                                       round(float(now[k][2] - intended[k][2]) * 1000, 1)) for k in intended}
                continue
            P = {k: now[k] - tb for k in now}
            if not relation_ok(self.task, P):
                self.reasons["relation false"] += 1
                continue
            if self.task in SUPPORT and not self._touching("t", SUPPORT[self.task], h):
                self.reasons["support lost"] += 1
                continue
            self.last_tries = attempt + 1
            return True
        sim.set_state_from_flattened(saved); sim.forward()
        self.last_tries = self.max_tries
        return False

    def _touching(self, a, b, h):
        sim = self.env.env.sim; m, d = sim.model, sim.data
        ba, bb = h[a][2], h[b][2]
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if {g1, g2} == {ba, bb}:
                return True
        return False
