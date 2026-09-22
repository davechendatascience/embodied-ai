"""A task's loss, generic over bddl goals: zero exactly where LIBERO's scorer accepts the
state, and otherwise how far the state is from what it accepts (BRN-task-loss-core).

Per goal conjunct the term is 0 when LIBERO's own predicate accepts it (the _eval_predicate
that _check_success conjoins) and otherwise d + M:

  in(o, site)        d = a, o's origin to the site's containment box
  on(o, site)        d = a + b: a to the site's `under` band (footprint and height band);
                     b the exact separation between o and the site's parent while LIBERO
                     sees no contact (a site without a parent object needs none)
  on(o, object x)    d = a + b: a from LIBERO's on-top radius and height offset; b likewise
  open/close/        d = the joint's distance to its threshold, in the joint coordinate
  turnon/turnoff

Every tolerance is LIBERO's own and none is copied from its source: the on-top radius, the
band heights and the containment slack are found by probing its predicate functions with
synthetic poses, and each joint's threshold and satisfied side by bisecting the predicate
over the joint's range. M is any positive constant: it keeps a rejected state above zero and,
for a single-conjunct goal (all of libero_object, spatial and goal), changes neither the zero
set nor how rejected states compare. For a translated object d / 2 <= max(a, b) <= D*, the
translation it needs to reach a pose LIBERO accepts. Not covered: an in-region carried by a
slide joint (libero_goal 3's drawer), obstacles, and goals with several conjuncts.

`reach` is shaping for a learner, not loss: the tool point's distance to what must move, zero
once the conjunct holds or the thing is held in both fingers.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from ..geometry.box_distance import BoxSet, separation
from ..sim import contacts

M_DEFAULT = 1e-4
_JOINT_PREDICATES = ("open", "close", "turnon", "turnoff")


@dataclass(frozen=True)
class Term:
    satisfied: bool      # LIBERO's own predicate
    distance: float      # d: 0 on acceptance (and on LIBERO's strict boundaries)
    reach: float         # m, tool point to what must move; 0 once satisfied or held
    held: bool           # both finger groups touch what must move
    m: float = M_DEFAULT

    @property
    def loss(self) -> float:
        return 0.0 if self.satisfied else self.distance + self.m


class TaskLoss:
    """The loss of one LIBERO task, read from a live TaskEnv."""

    def __init__(self, env, m: float = M_DEFAULT):
        if m <= 0:
            raise ValueError("M must be positive")
        self.env, self.m = env, m
        self.lib = env.env.env                                   # the LIBERO domain
        self.scene = env.scene
        self.goals = [tuple(g) for g in env.task_spec.goals]
        self._boxes: dict[str, BoxSet] = {}
        self._ontop = probe_ontop()
        self._site: dict[str, dict] = {}
        self._joint: dict[tuple, dict] = {}
        for g in self.goals:
            if g[0] in _JOINT_PREDICATES:
                self._joint[g] = probe_joint(self.lib, g[0], g[1])
            elif g[0] not in ("in", "on"):
                raise NotImplementedError(f"no distance for predicate {g[0]!r}")
            elif g[2] in self.lib.object_sites_dict:
                self._site[g[2]] = probe_site(self.lib.object_sites_dict[g[2]])
            elif g[0] == "in":
                raise NotImplementedError(f"{g}: containment in an object, not a site")

    # -- the loss ---------------------------------------------------------------------
    def terms(self) -> list[Term]:
        p_tool = self.env.tool_state()["p_tool"] + self.scene.base
        return [self._term(g, p_tool) for g in self.goals]

    def __call__(self) -> float:
        return float(sum(t.loss for t in self.terms()))

    def _term(self, g: tuple, p_tool: np.ndarray) -> Term:
        satisfied = bool(self.lib._eval_predicate(list(g)))
        m, d = self.scene.m, self.scene.d                       # robosuite's wrappers, as contacts reads them
        if g[0] in _JOINT_PREDICATES:
            distance = self._joint_distance(g)
            a = self.scene.articulation(g[1])
            held = contacts.finger_sides_on_geom(m, d, a["handle_geom"]) == {0, 1}
            target = a["handle_world"] + self.scene.base
        else:
            distance = self._placement_distance(g)
            held = contacts.finger_sides(m, d, self.lib.obj_body_id[g[1]]) == {0, 1}
            target = self.scene.object_box(g[1]).world_centre + self.scene.base
        reach = 0.0 if satisfied or held else float(np.linalg.norm(p_tool - target))
        return Term(satisfied=satisfied, distance=float(distance), reach=reach, held=held, m=self.m)

    # -- d ------------------------------------------------------------------------------
    def _pos(self, obj: str) -> np.ndarray:
        return np.asarray(self.lib.sim.data.body_xpos[self.lib.obj_body_id[obj]], float)

    def _touch(self, a: str, b: str) -> bool:
        return bool(self.lib.check_contact(self.lib.get_object(a), self.lib.get_object(b)))

    def _separation(self, a: str, b: str) -> float:
        m, d = self.scene.raw()
        for name in (a, b):
            if name not in self._boxes:
                ids = [m.geom(n).id for n in self.lib.get_object(name).contact_geoms]
                self._boxes[name] = BoxSet(m, ids)
        return separation(d, self._boxes[a], self._boxes[b])

    def _placement_distance(self, g: tuple) -> float:
        pred, obj, where = g
        p = self._pos(obj)
        if where not in self.lib.object_sites_dict:          # on(obj, object): ObjectState.check_ontop
            q = self._pos(where)
            gap = np.array([max(np.linalg.norm(q[:2] - p[:2]) - self._ontop["r"], 0.0),
                            max(q[2] - p[2] - self._ontop["dz"], 0.0)])
            b = 0.0 if self._touch(where, obj) else self._separation(obj, where)
            return float(np.linalg.norm(gap)) + b
        site, probe = self.lib.object_sites_dict[where], self._site[where]
        c = np.asarray(self.lib.sim.data.get_site_xpos(where), float)
        R = np.asarray(self.lib.sim.data.get_site_xmat(where), float).reshape(3, 3)
        if pred == "in":                                       # a site needs no contact
            half = np.abs(R @ np.asarray(site.size, float))
            lb, ub = c - half - probe["in_lo_slack"], c + half + probe["in_hi_slack"]
            return float(np.linalg.norm(np.maximum(lb - p, 0.0) + np.maximum(p - ub, 0.0)))
        if probe.get("on_always"):
            return 0.0
        s = probe["size"]
        dp = R @ (p - c)                                       # as SiteObject.under computes it
        foot = s[:2] + probe["foot_slack"]
        gap = np.array([max(abs(dp[0]) - foot[0], 0.0), max(abs(dp[1]) - foot[1], 0.0),
                        max(s[2] - probe["under_below"] - dp[2], 0.0, dp[2] - s[2] - probe["under_above"])])
        parent = getattr(site, "parent_name", None)
        needs_contact = parent is not None and self.lib.get_object(parent) is not None
        b = self._separation(obj, parent) if needs_contact and not self._touch(parent, obj) else 0.0
        return float(np.linalg.norm(gap)) + b

    def slide_margin(self, g: tuple) -> float:
        """How far the moved object's origin can slide horizontally before it leaves the set
        LIBERO accepts for this conjunct (0 when it is not accepted, or for a joint)."""
        if g[0] in _JOINT_PREDICATES or not self.lib._eval_predicate(list(g)):
            return 0.0
        pred, obj, where = g
        p = self._pos(obj)
        if where not in self.lib.object_sites_dict:
            q = self._pos(where)
            return max(self._ontop["r"] - float(np.linalg.norm(q[:2] - p[:2])), 0.0)
        site, probe = self.lib.object_sites_dict[where], self._site[where]
        c = np.asarray(self.lib.sim.data.get_site_xpos(where), float)
        R = np.asarray(self.lib.sim.data.get_site_xmat(where), float).reshape(3, 3)
        if pred == "in":
            half = np.abs(R @ np.asarray(site.size, float))
            lb, ub = c - half - probe["in_lo_slack"], c + half + probe["in_hi_slack"]
            return float(min(p[0] - lb[0], ub[0] - p[0], p[1] - lb[1], ub[1] - p[1]))
        if probe.get("on_always"):
            return np.inf
        dp = R @ (p - c)
        foot = probe["size"][:2] + probe["foot_slack"]
        return float(min(foot[0] - abs(dp[0]), foot[1] - abs(dp[1])))

    def _joint_distance(self, g: tuple) -> float:
        probe = self._joint[g]
        qpos = self.lib.sim.data.qpos
        ds = [max(0.0, j["side"] * (j["theta"] - float(qpos[self.lib.sim.model.get_joint_qpos_addr(j["joint"])])))
              for j in probe["joints"]]
        return min(ds) if probe["any"] else max(ds)


# -- probing LIBERO's predicates -----------------------------------------------------------
def _bisect(accepts, t_in: float, t_out: float) -> tuple[float, float]:
    """Narrow [t_in, t_out] to adjacent floats, accepts(t_in) true and accepts(t_out) false."""
    if not accepts(t_in) or accepts(t_out):
        raise ValueError("the bracket does not straddle the predicate's boundary")
    for _ in range(200):
        mid = 0.5 * (t_in + t_out)
        if mid in (t_in, t_out):
            break
        if accepts(mid):
            t_in = mid
        else:
            t_out = mid
    return t_in, t_out


class _Stub:
    """Just enough of a LIBERO domain for its ObjectState and SiteObjectState: synthetic
    positions and joint values, contact always true, so a probe never touches the live sim."""

    def __init__(self, sites=None, objects=None):
        self.objects_dict = {"_o": None, "_s": None}
        self.fixtures_dict = {}
        self.object_sites_dict = dict(sites or {})
        self._objects = dict(objects or {})
        self.obj_body_id = {"_o": 0, "_s": 1}
        data = SimpleNamespace(body_xpos=np.zeros((2, 3)), qpos=np.zeros(1),
                               get_site_xpos=lambda _name: np.zeros(3), get_site_xmat=lambda _name: np.eye(3))
        self.sim = SimpleNamespace(data=data, model=SimpleNamespace(get_joint_qpos_addr=lambda _j: 0))

    def get_object(self, name):
        if name in self._objects:
            return self._objects[name]
        if name in ("_o", "_s", "_parent"):
            return SimpleNamespace()
        return self.object_sites_dict.get(name)

    def check_contact(self, *_):
        return True


def probe_ontop() -> dict:
    """ObjectState.check_ontop's tolerances: the xy radius and the height offset of the object
    above its support."""
    from libero.libero.envs.object_states.base_object_states import ObjectState
    from libero.libero.envs.predicates import eval_predicate_fn
    st = _Stub()
    obj, support = ObjectState(st, "_o"), ObjectState(st, "_s")

    def accepts(p):
        st.sim.data.body_xpos[0] = p
        return bool(eval_predicate_fn("on", obj, support))
    radii = [_bisect(lambda r, u=u: accepts(r * u), 0.0, 1.0)[0]
             for u in (np.array([np.cos(a), np.sin(a), 0.0]) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False))]
    dz = -_bisect(lambda z: accepts(np.array([0.0, 0.0, z])), 0.0, -1.0)[0]
    return dict(r=float(np.mean(radii)), dz=float(dz))


def probe_site(site) -> dict:
    """A site's containment slack per face and its `under` band (footprint slack, heights
    above the site's centre), with the site at the origin."""
    from libero.libero.envs.object_states.base_object_states import ObjectState, SiteObjectState
    from libero.libero.envs.predicates import eval_predicate_fn
    st = _Stub(sites={"_site": site})
    obj, region = ObjectState(st, "_o"), SiteObjectState(st, "_site", "_parent")
    size = np.abs(np.asarray(site.size, float))
    out = {"size": size}

    def accepts(pred, p):
        st.sim.data.body_xpos[0] = p
        return bool(eval_predicate_fn(pred, obj, region))
    if hasattr(site, "in_box"):
        lo, hi = np.zeros(3), np.zeros(3)
        for k in range(3):
            e = np.eye(3)[k]
            hi[k] = _bisect(lambda t, e=e: accepts("in", t * e), 0.0, 10.0)[0] - size[k]
            lo[k] = _bisect(lambda t, e=e: accepts("in", -t * e), 0.0, 10.0)[0] - size[k]
        out["in_lo_slack"], out["in_hi_slack"] = lo, hi
    if not hasattr(site, "under"):
        out["on_always"] = accepts("on", np.array([5.0, 5.0, -5.0]))
        return out
    zs = np.linspace(-1.0, 1.0, 20001)
    inside = [z for z in zs if accepts("on", np.array([0.0, 0.0, z]))]
    z0 = float(np.median(inside))
    z_lo = _bisect(lambda z: accepts("on", np.array([0.0, 0.0, z])), z0, -1.0)[0]
    z_hi = _bisect(lambda z: accepts("on", np.array([0.0, 0.0, z])), z0, 1.0)[0]
    foot = np.zeros(2)
    for k in range(2):
        e = np.eye(3)[k]
        plus = _bisect(lambda t, e=e: accepts("on", np.array([0.0, 0.0, z0]) + t * e), 0.0, 10.0)[0]
        minus = _bisect(lambda t, e=e: accepts("on", np.array([0.0, 0.0, z0]) - t * e), 0.0, 10.0)[0]
        foot[k] = 0.5 * (plus + minus)
    out.update(under_below=float(size[2] - z_lo), under_above=float(z_hi - size[2]), foot_slack=foot - size[:2])
    return out


def probe_joint(lib, pred: str, name: str, scan: int = 4001) -> dict:
    """Threshold and satisfied side of pred(name) for each joint the predicate iterates over,
    by bisecting the predicate over the joint's MuJoCo range. The distance to satisfaction is
    then max(0, side * (theta - q)), taken as the least over joints for the any-joint
    predicates (open, turnon) and the most for the all-joint ones. A turn_on probe rewrites the
    fixture's vis_site_names, which is restored."""
    from libero.libero.envs.object_states.base_object_states import ObjectState, SiteObjectState
    from libero.libero.envs.predicates import eval_predicate_fn
    sites, model = lib.object_sites_dict, lib.sim.model
    if name in sites:
        owner = lib.get_object(sites[name].parent_name)
        joints = list(sites[name].joints)
        st = _Stub(sites={"_site": sites[name]}, objects={"_parent": owner})
        state = SiteObjectState(st, "_site", "_parent")
    else:
        owner = lib.get_object(name)
        joints = list(owner.joints)
        st = _Stub(objects={"_x": owner})
        state = ObjectState(st, "_x", is_fixture=True)
    vis = copy.deepcopy(owner.object_properties.get("vis_site_names", {}))

    def accepts(q):
        st.sim.data.qpos[0] = q
        return bool(eval_predicate_fn(pred, state))
    out = []
    for joint in joints:
        jid = model.joint_name2id(joint)
        lo, hi = (float(x) for x in model.jnt_range[jid])
        qs = np.linspace(lo, hi, scan)
        v = np.array([accepts(q) for q in qs])
        switch = np.nonzero(np.diff(v.astype(int)))[0]
        if len(switch) != 1:
            raise ValueError(f"{pred}({name}) joint {joint}: {len(switch)} switches over [{lo}, {hi}]")
        i = switch[0]
        side = 1.0 if v[i + 1] else -1.0                         # accepted above or below
        t_in, t_out = (_bisect(accepts, float(qs[i + 1]), float(qs[i])) if side > 0
                       else _bisect(accepts, float(qs[i]), float(qs[i + 1])))
        out.append(dict(joint=joint, theta=0.5 * (t_in + t_out), side=side))
    if "vis_site_names" in owner.object_properties:
        owner.object_properties["vis_site_names"] = vis
    return dict(joints=out, any=pred in ("open", "turnon"))
