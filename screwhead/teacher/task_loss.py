"""A task's loss, generic over bddl goals: how far the simulator state is from what LIBERO's
scorer accepts (BRN-task-loss-matches-scorer).

For each goal conjunct the term is zero when LIBERO's own predicate accepts it
(`_eval_predicate`, the function `_check_success` conjoins) and otherwise the distance of
the state to that predicate's acceptance set, measured from the quantities the predicate
reads:

  in(o, site)        o's origin to the site's containment box (|R size| about the site,
                     1 cm lower at the bottom); a site needs no contact (it returns True)
  on(o, site)        o's origin to the site's `under` band (R (o - site) within the
                     footprint, between size_z - 5 mm and size_z + 10 cm above), plus
                     CONTACT_GAP while o does not touch the site's parent object
  on(o, object x)    x's origin must not be above o's, their origins within 3 cm in xy,
                     plus CONTACT_GAP while they do not touch
  open/close(site)   the site's joint to LIBERO's threshold for its category
  turnon/off(fix.)   the fixture's joint to its threshold

Gating by the scorer makes the zero set LIBERO's by construction; the distance is what a
learner descends, so its agreement with the scorer is measured separately (a satisfied
conjunct with a large distance, or an unsatisfied one with none, means the geometry here
has drifted from LIBERO's). Nothing is keyed by task.

`reach` is shaping, not loss: the tool point's distance to what must move -- the object
of an in/on conjunct, the handle of an articulated one -- zero once the conjunct holds or
the object is held in both fingers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sim import contacts

CONTACT_GAP = 0.01       # m: what a missing required contact adds to a distance
ONTOP_XY = 0.03          # m: LIBERO's ObjectState.check_ontop centre tolerance
IN_FLOOR_SLACK = 0.01    # m: LIBERO's in_box lowers the box bottom by this
UNDER_BELOW = 0.005      # m: SiteObject.under accepts from size_z - this ...
UNDER_ABOVE = 0.10       # m: ... to size_z + this
MIN_TERM = 1e-4          # an unsatisfied conjunct never contributes less than this


@dataclass(frozen=True)
class Term:
    satisfied: bool      # LIBERO's own predicate
    distance: float      # to the predicate's acceptance set (m; joint units for joints)
    reach: float         # m, tool point to what must move; 0 once satisfied or held
    held: bool           # both finger groups touch what must move

    @property
    def loss(self) -> float:
        return 0.0 if self.satisfied else max(self.distance, MIN_TERM)


class TaskLoss:
    """The loss of one LIBERO task, read from a live TaskEnv."""

    def __init__(self, env):
        self.env = env
        self.lib = env.env.env               # the LIBERO domain: predicates, sites, objects
        self.scene = env.scene
        self.goals = [tuple(g) for g in env.task_spec.goals]
        for g in self.goals:
            if g[0] not in ("in", "on", "open", "close", "turnon", "turnoff"):
                raise NotImplementedError(f"no distance for predicate {g[0]!r}")

    # -- the loss -------------------------------------------------------------------
    def terms(self) -> list[Term]:
        p_tool = self.env.tool_state()["p_tool"] + self.scene.base
        return [self._term(g, p_tool) for g in self.goals]

    def __call__(self) -> float:
        return float(sum(t.loss for t in self.terms()))

    def _term(self, g: tuple, p_tool: np.ndarray) -> Term:
        satisfied = bool(self.lib._eval_predicate(list(g)))
        if g[0] in ("in", "on"):
            distance = self._placement(g)
            body = self.lib.obj_body_id[g[1]]
            held = contacts.finger_sides(self.scene.m, self.scene.d, body) == {0, 1}
            target = self.scene.object_box(g[1]).world_centre + self.scene.base
        else:
            a = self.scene.articulation(g[1])
            distance = self._joint(g[0], a)
            held = contacts.finger_sides_on_geom(self.scene.m, self.scene.d, a["handle_geom"]) == {0, 1}
            target = a["handle_world"] + self.scene.base
        reach = 0.0 if satisfied or held else float(np.linalg.norm(p_tool - target))
        return Term(satisfied=satisfied, distance=float(distance), reach=reach, held=held)

    # -- distances to each predicate's acceptance set ---------------------------------
    def _pos(self, obj: str) -> np.ndarray:
        return np.asarray(self.lib.sim.data.body_xpos[self.lib.obj_body_id[obj]], float)

    def _touch(self, a: str, b: str) -> bool:
        return bool(self.lib.check_contact(self.lib.get_object(a), self.lib.get_object(b)))

    def _placement(self, g: tuple) -> float:
        pred, obj, where = g
        p = self._pos(obj)
        sites = self.lib.object_sites_dict
        if where in sites:
            site = sites[where]
            c = np.asarray(self.lib.sim.data.get_site_xpos(where), float)
            R = np.asarray(self.lib.sim.data.get_site_xmat(where), float).reshape(3, 3)
            size = np.asarray(site.size, float)
            if pred == "in":
                return _outside(p, *_containment(c, R, size))
            if not hasattr(site, "under"):
                return 0.0                            # LIBERO accepts such an On always
            dp = R @ (p - c)                          # as SiteObject.under computes it
            gap = np.array([max(abs(dp[0]) - size[0], 0.0), max(abs(dp[1]) - size[1], 0.0),
                            max(size[2] - UNDER_BELOW - dp[2], 0.0, dp[2] - size[2] - UNDER_ABOVE)])
            parent = getattr(site, "parent_name", None)
            touch = parent is None or self.lib.get_object(parent) is None or self._touch(parent, obj)
            return float(np.linalg.norm(gap)) + (0.0 if touch else CONTACT_GAP)
        if pred == "in":
            raise NotImplementedError(f"in({obj}, {where}): containment in an object, not a site")
        q = self._pos(where)                          # ObjectState.check_ontop, `where` the support
        gap = np.array([max(np.linalg.norm(q[:2] - p[:2]) - ONTOP_XY, 0.0), max(q[2] - p[2], 0.0)])
        return float(np.linalg.norm(gap)) + (0.0 if self._touch(where, obj) else CONTACT_GAP)

    @staticmethod
    def _joint(pred: str, a: dict) -> float:
        th, s, q = a["thresholds"], a["sign"], a["qpos"]
        goal = {"open": th.get("open"), "close": th.get("close"),
                "turnon": th.get("on"), "turnoff": th.get("off")}[pred]
        if goal is None:
            raise ValueError(f"{a['category']}: no {pred} threshold")
        toward = s if pred in ("open", "turnon") else -s   # the direction that satisfies pred
        return max(toward * (goal - q), 0.0)


def _containment(c: np.ndarray, R: np.ndarray, size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SiteObject.in_box's bounds: |R size| about the centre, the bottom 1 cm lower."""
    half = np.abs(R @ size)
    lb, ub = c - half, c + half
    lb[2] -= IN_FLOOR_SLACK
    return lb, ub


def _outside(p: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> float:
    return float(np.linalg.norm(np.maximum(lb - p, 0.0) + np.maximum(p - ub, 0.0)))
