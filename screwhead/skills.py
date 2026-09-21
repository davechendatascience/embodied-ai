"""Skills, parameterised by measured geometry, sequenced from a task's goal.

The ten libero_spatial programs are hand-written and their constants tuned per task. This
is the replacement the held-out splits argued for: pick and place read the object's own
bounding box and the target region's own site, so a task nobody wrote a program for still
gets a demonstration.

Each skill is a Markov feedback law -- its action is a function of the current state, with
no stored progress -- so the teacher can still label any state a student reaches, which is
what DAgger needs (belief.yaml, IFC-teacher__expert).

Kept from the spatial programs, because they were measured there:
  funnel approach   the target height above the grasp shrinks as the tool aligns, so
                    nearby states get nearby labels (a switched approach stalled students)
  squeeze           close until the jaws stop closing, rather than to a fixed aperture
  speed floor       5 cm/s minimum while approaching; proportional homing ends at
                    1-3 cm/s where a student's direction agreed only to cosine 0.68
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gripper_servo import A_OPEN, target_to_channel

Z = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class SkillConfig:
    k_lin: float = 5.0
    k_rot: float = 3.0
    v_max: float = 0.25
    w_max: float = 1.2
    v_min: float = 0.05            # approach speed floor
    approach: float = 0.10         # pre-grasp height above the grasp
    funnel_xy: float = 0.02
    funnel_rot: float = 0.15
    at_pos: float = 0.006
    at_rot: float = 0.10
    grip_margin: float = 0.012     # jaws this much wider than the object before descending
    squeeze_rate: float = 0.005    # jaw speed below which a squeeze counts as settled
    lift: float = 0.12             # carry height above the support
    place_clearance: float = 0.015 # object bottom above the target surface before release
    release_steps: int = 6
    max_grip: float = 0.075        # widest jaw opening used (A_OPEN is the hard limit)


def rotvec(R: np.ndarray) -> np.ndarray:
    c = (np.trace(R) - 1) / 2
    th = float(np.arccos(np.clip(c, -1.0, 1.0)))
    if th < 1e-8:
        return np.zeros(3)
    return th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def rot_angle(R: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))


def _axis_rot(axis: np.ndarray, angle: float) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / (np.linalg.norm(a) + 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def tool_frame(jaw_dir: np.ndarray, approach: np.ndarray) -> np.ndarray:
    """Tool frame that closes along `jaw_dir` and advances along `approach`.

    The Panda gripper closes along its own y axis and looks along +z, so z_tool is the
    approach direction and y_tool the jaw axis, squared against it.
    """
    z = np.asarray(approach, float)
    z = z / (np.linalg.norm(z) + 1e-12)
    y = np.asarray(jaw_dir, float)
    y = y - z * (y @ z)
    n = np.linalg.norm(y)
    if n < 1e-6:                                  # jaw parallel to the approach: any square axis
        y = np.cross(z, Z if abs(z @ Z) < 0.9 else np.array([1.0, 0.0, 0.0]))
        n = np.linalg.norm(y)
    y = y / n
    return np.column_stack([np.cross(y, z), y, z])


def top_down(yaw_dir: np.ndarray) -> np.ndarray:
    """The common case: straight down on to an object lying on a support."""
    y = np.asarray(yaw_dir, float)
    y = y - Z * (y @ Z)
    return tool_frame(y if np.linalg.norm(y) > 1e-6 else np.array([1.0, 0.0, 0.0]), -Z)


class Skills:
    """Feedback laws over a TaskEnv + Scene. Stateless apart from the cached grasp choice,
    which is re-derived whenever the object has moved more than a centimetre."""

    def __init__(self, env, config: SkillConfig | None = None):
        self.env = env
        self.scene = env.scene
        self.k = config or SkillConfig()
        self.phase = ""
        self._grasp_cache: dict[str, tuple] = {}
        self._palm: float | None = None
        self._episode = None
        self.grasp_log: dict = {}
        self._drop_cache: dict = {}
        self._carry_cache: dict = {}
        self._handle_cache: dict[str, tuple[np.ndarray, float, np.ndarray]] = {}
        self.last_choice: dict = {}
        self._geom_mask = None
        self._geom_half = None

    # -- twist helper ---------------------------------------------------------------
    def twist_to(self, R, p, R_goal, p_goal, v_max=None, v_min=None):
        k, spec = self.k, self.env.spec
        v_max = k.v_max if v_max is None else v_max
        v_min = k.v_min if v_min is None else v_min
        w = rotvec(R.T @ R_goal) * k.k_rot
        v = R.T @ (p_goal - p) * k.k_lin
        if np.linalg.norm(w) > k.w_max:
            w *= k.w_max / np.linalg.norm(w)
        if np.linalg.norm(v) > v_max:
            v *= v_max / np.linalg.norm(v)
        dist = float(np.linalg.norm(p_goal - p))
        if v_min > 0 and dist > 0.001 and 0 < np.linalg.norm(v) < v_min:
            v *= v_min / np.linalg.norm(v)
        return np.concatenate([w / (spec.rot_scale * spec.control_hz),
                               v / (spec.pos_scale * spec.control_hz)])

    def action(self, twist, target_aperture: float) -> np.ndarray:
        return np.concatenate([twist, [target_to_channel(np.clip(target_aperture, 0.0, A_OPEN))]])

    # -- geometry -------------------------------------------------------------------
    def new_episode_check(self) -> None:
        """Forget every decision made in a previous episode.

        The grasp cache is invalidated when the object moves more than a centimetre, and
        LIBERO often starts the next episode with the object within a centimetre of where
        the last one started -- so a new episode silently inherited the previous one's
        grasp, crossing height and handle frame, chosen from a different arm pose. On the
        cream cheese one bad choice in episode 0 then failed 17 more episodes in a row.
        A teacher whose labels depend on an earlier episode is not the Markov teacher
        DAgger needs.
        """
        ep = getattr(self.env, "episode", None)
        if ep != self._episode:
            self._episode = ep
            self._grasp_cache.clear()
            self._handle_cache.clear()
            self.grasp_log.clear()
            self._drop_cache.clear()
            self._carry_cache.clear()

    def grasp_for(self, obj: str, via: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float]:
        """Top-down grasp on an object, from its measured box.

        The height and the jaw width come from the box; the yaw is CHOSEN BY REACHABILITY,
        not by a rule. Measured on the floor scene: a jaw axis held tangential to the base
        put the arm on a joint limit at every height over the basket (margin 0.000, IK not
        converging), while other yaws had 1.1-1.4 rad of margin -- so the teacher scores
        candidate yaws by IK at the grasp, above it, and at the place pose.
        """
        self.new_episode_check()
        box = self.scene.object_box(obj)
        centre = box.world_centre
        cached = self._grasp_cache.get(obj)
        if cached is not None and np.linalg.norm(cached[3] - centre) < 0.01:
            return cached[0], cached[1], cached[2]
        bid = self.scene.body_id(obj)
        chosen, used = None, "forced"
        tiers = self._tiers(obj, box)
        for name, tier in tiers:
            chosen = self._choose(tier, via, allow=bid, strict=True)
            if chosen is not None:
                used = name
                break
        if chosen is None:                      # nothing feasible anywhere: take the best try
            chosen = self._choose(sum((t for _, t in tiers), []), via, allow=bid)
        R_grasp, p_grasp, width, app = chosen
        h = self._reachable_transit(R_grasp, p_grasp - app * self.k.approach, bid)
        self._grasp_cache[obj] = (R_grasp, p_grasp, width, centre, h, app)
        # what was decided and why, for the assessment (tools/skill_eval.py) to report
        prev = self.grasp_log.get(obj, {})
        self.grasp_log[obj] = dict(
            tier=used, offered={n: len(t) for n, t in tiers}, via=via is not None,
            feasible=self.last_choice.get("feasible"), score=self.last_choice.get("score"),
            approach=[round(float(v), 2) for v in app], jaw=[round(float(v), 2) for v in R_grasp[:, 1]],
            width_mm=round(1000 * float(width), 1), open_mm=round(1000 * min(self.k.max_grip, width + self.k.grip_margin), 1),
            grasp_above_centre_mm=round(1000 * float(p_grasp[2] - centre[2]), 1),
            transit=round(float(h), 3), choices=prev.get("choices", 0) + 1)
        return R_grasp, p_grasp, width

    def transit_for(self, obj: str) -> float:
        """The cached crossing height for this object's grasp."""
        c = self._grasp_cache.get(obj)
        return float(c[4]) if c is not None else 0.0

    def approach_for(self, obj: str) -> np.ndarray:
        """The direction the tool advances on this object: down, or in from the side."""
        c = self._grasp_cache.get(obj)
        return c[5] if c is not None else -Z

    def _reachable_transit(self, R: np.ndarray, p_g: np.ndarray, body: int) -> float:
        """The highest crossing height the ARM can actually hold, at or below the one the
        obstacles ask for.

        Clearing the wooden cabinet means a pose 30 cm over its top, and the Panda cannot
        hold it: the servo re-anchored 565 times in one episode with the tool touching
        nothing at all and the jaws never closing. Converging is not enough either --
        the highest converging crossing put the arm at sigma 0.021, nearly straight,
        where it cannot track a twist at all. So walk the height down until the arm has
        room to move in every direction.
        """
        from .kin_np import NpChain, sigma_min, solve_ik
        s = self.env.snapshot()
        top = self.transit_height(s["p_tool"], p_g, body)
        floor = float(p_g[2] + self.k.approach)
        hs = list(np.arange(top, floor - 1e-9, -0.05)) or [floor]
        if hs[-1] > floor:
            hs.append(floor)
        Ts = []
        for h in hs:
            T = np.eye(4); T[:3, :3] = R; T[:3, 3] = np.array([p_g[0], p_g[1], h])
            Ts.append(T)
        c = NpChain.of(self.env.chain)
        q0 = np.asarray(self.env.raw["robot0_joint_pos"], float)
        res = solve_ik(c, np.stack(Ts), q0[None], lam=0.02, max_iters=200, trust=0.2)
        conv, th = res["converged"], res["theta"]
        sig = sigma_min(c, th)
        lim = c.limits
        margin = np.minimum(th - lim[:, 0], lim[:, 1] - th).min(1)
        for i, h in enumerate(hs):
            if (conv[i] and sig[i] > 0.05 and margin[i] > 0.15
                    and self._collides(th[i], body) < 2):
                return float(h)
        return floor

    def _tiers(self, obj: str, box) -> list[list[tuple[np.ndarray, np.ndarray, float, np.ndarray]]]:
        """Candidate grasps in order of preference, each tier tried on its own.

        Whole-object faces when one fits. When none does -- a 107 mm bowl against 75 mm of
        jaw -- grasp a PART: LIBERO decomposes a bowl into 2.6 mm wall segments 38 mm tall,
        and a segment near the rim fits easily. That is the rim grasp the hand-written
        spatial programs did by hand, derived here from the collision geometry.

        The last tier comes in from the side, because a bowl inside a drawer has the next
        shelf over it exactly as the pull bar did, and a tier that offers nothing FEASIBLE
        must hand over rather than force: forcing left the teacher reaching for a 107 mm
        grasp it could never close, 0/3 with the arm re-anchoring and touching nothing.
        """
        bid = self.scene.body_id(obj)
        half_h = float(np.abs(box.R @ np.diag(box.half)).sum(1)[2])
        faces, parts, sides = [], [], []
        for w, d in box.width_axes():                     # narrow face first
            if w + self.k.grip_margin <= self.k.max_grip and self._clear_body(box.world_centre, -Z, bid):
                faces += self._at(box.world_centre, half_h, d, w, body=bid)
        for c, hw, d, w in self._parts(obj, box):
            if self._clear_body(c, -Z, bid):
                parts += self._at(c, hw, d, w, body=bid)
        for g in self._part_geoms(obj, box):
            sides += self._handle_grasps(g)
        if not (faces or parts or sides):
            d0 = box.width_axes()[0][1] if box.width_axes() else np.array([1.0, 0.0, 0.0])
            faces = self._at(box.world_centre, half_h, d0, self.k.max_grip - self.k.grip_margin)
        return [(n, t) for n, t in (("faces", faces), ("parts", parts), ("sides", sides)) if t]

    def _at(self, centre: np.ndarray, half_h: float, d: np.ndarray, w: float,
            body: int = -1) -> list[tuple[np.ndarray, np.ndarray, float, np.ndarray]]:
        """Top-down grasps on a body of this height, jaws both ways round."""
        depth = min(max(0.02, 0.66 * half_h), self.palm_clearance() - 0.010)
        p = centre + Z * max(half_h - depth, -half_h + 0.005)
        h = np.array([d[0], d[1], 0.0])
        n = np.linalg.norm(h)
        if n < 1e-6:
            return []
        h = h / n
        if body >= 0:
            w = self._width_at(body, p, h) or w
        return [(top_down(h), p, w, -Z), (top_down(-h), p, w, -Z)]

    def _width_at(self, body: int, p: np.ndarray, jaw: np.ndarray) -> float:
        """How wide the object is WHERE THE JAWS WILL BE, not overall.

        A wine bottle's bounding box is 43 mm across and its neck, which is what a grasp
        30 mm below the top actually holds, is far narrower. Opening to the box width
        means the jaws travel the difference before they touch anything, and swing wide
        enough to catch the shoulder on the way down.

        What the jaws span is the material CONNECTED to the grasp point, not everything
        the ray crosses: taking the outer extent returned a bowl's full 107 mm diameter
        for a grasp on its 2.6 mm wall -- the far wall is a separate piece, 100 mm along
        the same axis -- and no candidate was feasible at all.
        """
        from .scene import _geom_half
        m, d = self.scene.m, self.scene.d
        spans = []
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) != body or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            Rg = d.geom_xmat[g].reshape(3, 3)
            hl = _geom_half(m, g)
            hw = np.abs(Rg @ np.diag(hl)).sum(1)
            c = d.geom_xpos[g] - self.scene.base
            if not (c[2] - hw[2] - 0.005 <= p[2] <= c[2] + hw[2] + 0.005):
                continue                       # this geom is not at the jaws' height
            across = np.cross(Z, jaw)          # along the jaws' line of sight
            # the pads are ~20 mm wide: material more than 10 mm off the jaw line is never
            # between them. A 30 mm allowance let a rim grasp count neighbouring wall
            # segments and measured a 2.6 mm wall as 29 mm, opening the jaws to 41.
            if abs(float((c - p) @ across)) > 0.01 + float(np.abs(across @ (Rg @ np.diag(hl))).sum()):
                continue
            extent = float(np.abs(jaw @ (Rg @ np.diag(hl))).sum())
            m_c = float(c @ jaw)
            spans.append((m_c - extent, m_c + extent))
        if not spans:
            return 0.0
        at = float(p @ jaw)
        lo = hi = None
        for a, b in sorted(spans):                     # merge, keep the piece under the jaws
            if lo is None or a > hi + 0.002:
                if lo is not None and lo - 0.002 <= at <= hi + 0.002:
                    return float(hi - lo)
                lo, hi = a, b
            else:
                hi = max(hi, b)
        return float(hi - lo) if lo is not None and lo - 0.002 <= at <= hi + 0.002 else 0.0

    def _handle_grasps(self, g: int) -> list[tuple[np.ndarray, np.ndarray, float, np.ndarray]]:
        """Grasps on a handle geom, from every pairing of its own axes.

        A handle is not approached from above: the cabinet's pull bar has the next drawer
        directly over it, and a top-down funnel never converged (the teacher sat in
        `reach` for 500 steps). The bar is 15 x 16 mm across and 89 mm long, so the jaws
        take it from the front and close vertically -- one pairing of the geom's axes,
        found the same way for any handle.
        """
        from .scene import _geom_half
        m, d = self.scene.m, self.scene.d
        Rg = d.geom_xmat[g].reshape(3, 3)
        hl = _geom_half(m, g)
        c = d.geom_xpos[g] - self.scene.base
        out, blocked = [], []
        for j in range(3):
            w = 2 * float(hl[j])
            if w + self.k.grip_margin > self.k.max_grip:
                continue
            for i in range(3):
                if i == j:
                    continue
                for sgn in (1.0, -1.0):
                    app = sgn * Rg[:, i]
                    if app[2] > 0.3:               # nothing is approached from underneath
                        continue
                    cands = [(tool_frame(jaw, app), c, w, app) for jaw in (Rg[:, j], -Rg[:, j])]
                    (out if self._clear_geom(c, app, g) else blocked).extend(cands)
        return out or blocked

    def _collides(self, theta: np.ndarray, allow: int = -1, aperture: float | None = None) -> int:
        """Would the ARM be in something at this joint configuration?

        IK converging says nothing about the rest of the arm: reaching a bowl beside the
        wooden cabinet, every candidate converged with 0.75 rad of joint margin and the
        forearm sat in `wooden_cabinet_1_base` -- the servo re-anchored 89 times in 200
        steps and the tool never closed the last 90 mm. The simulator answers exactly, so
        put the configuration in, look, and put the state back.
        """
        sim = self.env.env.sim
        robot = self.env.env.env.robots[0]
        idx = robot._ref_joint_pos_indexes
        qpos, qvel = sim.data.qpos.copy(), sim.data.qvel.copy()
        sim.data.qpos[idx] = np.asarray(theta, float)
        if aperture is not None:
            # the fingers where the grasp will have them, not where they happen to be:
            # screened fully open (77 mm), every rim grasp on the bowl in the drawer hit
            # the drawer walls, none was feasible, and a forced grasp stalled 43 mm short
            sim.data.qpos[robot._ref_gripper_joint_pos_indexes] = [aperture / 2, -aperture / 2]
        sim.forward()
        hit = self._touching_other(allow)
        sim.data.qpos[:] = qpos
        sim.data.qvel[:] = qvel
        sim.forward()
        return hit

    def _touching_other(self, allow: int) -> int:
        """0 clear, 1 touching something loose, 2 touching something bolted down.

        Touching the target is the point of a grasp, so the target does not count. Nor
        should a plate: vetoing any contact left the drawer handle with no candidate at
        all and the teacher fell back to a blocked top-down approach (0/5, having opened
        the drawer in 78 steps before the screen). A fixture is different in kind -- the
        forearm in `wooden_cabinet_1_base` stopped the arm dead, 89 re-anchors in 200
        steps -- so a fixture vetoes and a loose object only costs.
        """
        m, d = self.scene.m, self.scene.d
        worst = 0
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist >= 0:
                continue
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            n1 = m.body_id2name(b1) or ""
            n2 = m.body_id2name(b2) or ""
            r1 = n1.startswith(("robot", "gripper"))
            r2 = n2.startswith(("robot", "gripper"))
            if r1 == r2:
                continue
            other = b2 if r1 else b1
            if other == allow or int(m.body_parentid[other]) == allow:
                continue
            worst = max(worst, 1 if self._movable(other) else 2)
        return worst

    def _movable(self, body: int) -> bool:
        """A body the arm can push out of the way: one that hangs on a free joint."""
        m = self.scene.m
        n = int(m.body_jntnum[body])
        adr = int(m.body_jntadr[body])
        return any(int(m.jnt_type[adr + i]) == 0 for i in range(n))

    def _clear_geom(self, p: np.ndarray, app: np.ndarray, geom: int) -> bool:
        """Can the tool come in along `app` and reach this geom, or is something in the way?

        IK does not know about the cabinet. Straight down on to the drawer's pull bar is a
        perfectly reachable pose, and the shelf 38 mm above the bar makes it impossible --
        the teacher hovered there for 300 steps. A ray down the approach answers it.
        """
        import mujoco
        m, d = self.scene.m, self.scene.d
        mm = m._model if hasattr(m, "_model") else m
        dd = d._data if hasattr(d, "_data") else d
        v = np.asarray(app, float)
        v = v / (np.linalg.norm(v) + 1e-12)
        start = p + self.scene.base - v * self.k.approach
        gid = np.zeros(1, np.int32)
        dist = float(mujoco.mj_ray(mm, dd, start, v, None, 1, -1, gid))
        hit = int(gid[0])
        if hit < 0 or dist < 0:
            return False                       # not even the handle: the ray missed it
        if hit == geom:
            return True
        # another geom of the same moving part is fine if it is where the handle is
        return (int(m.geom_bodyid[hit]) == int(m.geom_bodyid[geom])
                and dist > self.k.approach - 0.03)

    def _clear_body(self, p: np.ndarray, app: np.ndarray, body: int) -> bool:
        """The tool can come down this line and meet the object, not something else.

        The same blindness as the drawer shelf, on the table: a bowl beside the cookie box
        has rim segments the arm cannot come down on to, and choosing one of those left
        the teacher in `approach` for the whole episode.
        """
        import mujoco
        m, d = self.scene.m, self.scene.d
        mm = m._model if hasattr(m, "_model") else m
        dd = d._data if hasattr(d, "_data") else d
        v = np.asarray(app, float)
        v = v / (np.linalg.norm(v) + 1e-12)
        gid = np.zeros(1, np.int32)
        dist = float(mujoco.mj_ray(mm, dd, p + self.scene.base - v * self.k.approach, v,
                                   None, 1, -1, gid))
        hit = int(gid[0])
        return hit >= 0 and dist >= 0 and int(m.geom_bodyid[hit]) == body

    def _geom_part(self, g: int) -> tuple[np.ndarray, float, np.ndarray, float] | None:
        """(centre, half height, thin horizontal direction, width) of one collision geom."""
        from .scene import _geom_half
        m, d = self.scene.m, self.scene.d
        Rg = d.geom_xmat[g].reshape(3, 3)
        hl = _geom_half(m, g)
        hw = np.abs(Rg @ np.diag(hl)).sum(1)
        axes = [(2 * float(hl[i]), Rg[:, i]) for i in range(3) if abs(Rg[2, i]) < 0.7]
        axes.sort(key=lambda a: a[0])
        if not axes:
            return None
        return (d.geom_xpos[g] - self.scene.base, float(hw[2]), axes[0][1], axes[0][0])

    def _part_geoms(self, obj: str, box) -> list[int]:
        """Collision geoms near the object's top that the jaws could fit around."""
        from .scene import _geom_half
        m, d = self.scene.m, self.scene.d
        bid = self.scene.body_id(obj)
        top = box.p[2] + (box.R @ box.centre)[2] + float(np.abs(box.R @ np.diag(box.half)).sum(1)[2])
        out = []
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) != bid or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            Rg = d.geom_xmat[g].reshape(3, 3)
            hl = _geom_half(m, g)
            hw = np.abs(Rg @ np.diag(hl)).sum(1)
            if float(d.geom_xpos[g][2] - self.scene.base[2] + hw[2]) < top - 0.02:
                continue                               # the foot ring, not the rim
            if min(2 * float(hl[i]) for i in range(3)) + self.k.grip_margin > self.k.max_grip:
                continue
            out.append(g)
        return out[:16]

    def _parts(self, obj: str, box) -> list[tuple[np.ndarray, float, np.ndarray, float]]:
        """Top-down grasp sites on those geoms, the tallest wall first."""
        out = []
        for g in self._part_geoms(obj, box):
            part = self._geom_part(g)
            if part is not None and part[3] + self.k.grip_margin <= self.k.max_grip:
                out.append(part)
        out.sort(key=lambda o: -o[1])                  # a taller wall gives the pads more to hold
        return out[:16]

    def _choose(self, cands: list[tuple[np.ndarray, np.ndarray, float, np.ndarray]],
                via: np.ndarray | None, allow: int = -1, strict: bool = False):
        """The first candidate grasp the arm can actually reach -- at the grasp, at the
        pre-grasp above it, and (when known) at the place pose the same grasp must reach.

        Shape proposes, reachability disposes. Measured: a jaw axis chosen on reachability
        alone closed across the cream cheese's diagonal, and one chosen on shape alone sat
        on a joint limit over the basket (margin 0.000 at every height).
        """
        from .kin_np import NpChain, sigma_min, solve_ik
        c = NpChain.of(self.env.chain)
        q0 = np.asarray(self.env.raw["robot0_joint_pos"], float)
        lim = c.limits
        Ts, meta = [], []
        for i, (R, p_g, _w, app) in enumerate(cands):
            pre = p_g - app * self.k.approach
            # the column the tool comes down, not only its ends: an endpoint-only check
            # passed grasps whose descent clamped on a joint limit halfway down
            column = [pre + Z * 0.05, pre + Z * 0.10]
            for pt in [p_g, pre] + column + ([via] if via is not None else []):
                T = np.eye(4); T[:3, :3] = R; T[:3, 3] = pt
                Ts.append(T); meta.append(i)
        res = solve_ik(c, np.stack(Ts), q0[None], lam=0.02, max_iters=200, trust=0.2)
        th, conv = res["theta"], res["converged"]
        sig = sigma_min(c, th)
        margin = np.minimum(th - lim[:, 0], lim[:, 1] - th).min(1)
        scores = {}
        for i in range(len(cands)):
            sel = [j for j, mi in enumerate(meta) if mi == i]
            if not all(conv[j] for j in sel):
                continue
            ap = min(self.k.max_grip, cands[i][2] + self.k.grip_margin)
            touch = max(self._collides(th[j], allow, ap) for j in sel)
            if touch == 2:                   # reachable on paper, the arm in the cabinet
                continue
            scores[i] = min(min(margin[j] for j in sel), 8 * min(sig[j] for j in sel)) - 0.3 * touch
        if not scores:
            if strict:
                return None
            self.last_choice = dict(n=len(cands), feasible=0, score=None, forced=True)
            return cands[0]
        # Shape proposes in list order (narrow faces first), conditioning decides how far
        # down that list to look. Taking the first merely-feasible candidate put the arm
        # where the damped solve clamps on a limit and the tool settled 20 mm off; taking
        # the best-conditioned one ignored the shape and cost four object episodes. So:
        # the first candidate that is comfortably conditioned, then progressively less.
        pick = None
        for bar in (0.35, 0.20, 0.05):
            ok = [i for i in sorted(scores) if scores[i] >= bar]
            if ok:
                pick = ok[0]
                break
        if pick is None:
            pick = max(scores, key=scores.get)
        self.last_choice = dict(n=len(cands), feasible=len(scores), score=float(scores[pick]),
                                forced=False)
        return cands[pick]

    def palm_clearance(self) -> float:
        """How far below the palm the tool point sits, measured from the model.

        The Panda's palm shell bottoms out 40 mm above the tool point and the fingertips
        reach 5 mm below it. A grasp deeper than that under an object's top drives the palm
        into the object: a 146 mm bottle grasped 48 mm down stalled with the palm in contact
        and the arm 28 mm short of its target, with no finger contact at all.
        """
        if self._palm is not None:
            return self._palm
        m, d = self.scene.m, self.scene.d
        from .scene import _geom_half
        p_tool = self.env.snapshot()["p_tool"] + self.scene.base
        lo = np.inf
        for g in range(m.ngeom):
            b = m.body_id2name(int(m.geom_bodyid[g])) or ""
            if not b.startswith("gripper0") or "finger" in b:
                continue
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            h = np.abs(d.geom_xmat[g].reshape(3, 3) @ np.diag(_geom_half(m, g))).sum(1)
            lo = min(lo, float(d.geom_xpos[g][2] - h[2] - p_tool[2]))
        self._palm = 0.04 if not np.isfinite(lo) else lo
        return self._palm

    def at_grasp(self, p: np.ndarray, p_g: np.ndarray, app: np.ndarray | None = None) -> bool:
        """The object is inside the jaw envelope, so squeeze.

        A tight ball around the grasp point is not the right test: the grasp point is
        re-derived from the object, the object shifts when the pads touch it, and the
        tool then falls out of the ball and REOPENS. Measured on the cream cheese, that
        alternated squeeze and descend for 200 steps with the jaws cycling 47-55 mm.
        The envelope is the one the fingers actually sweep: 15 mm laterally, and from
        35 mm above the grasp point (pads) to 15 mm below it (tips).
        """
        app = -Z if app is None else app
        d = p - p_g
        along = float(d @ app)                         # negative once the tool is short of it
        return bool(np.linalg.norm(d - app * along) < 0.015 and -0.035 < along < 0.015)

    def _grasp_width(self, box) -> float:
        widths = box.width_axes()
        return float(widths[0][0]) if widths else 0.04

    def held(self, obj: str) -> bool:
        """The object is between the pads and moving with them: both finger groups touch it
        and the jaws have stopped closing."""
        m, d = self.scene.m, self.scene.d
        try:
            bid = self.scene.body_id(obj)
        except ValueError:
            return False
        sides = set()
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            other = b2 if b1 == bid else (b1 if b2 == bid else None)
            if other is None:
                continue
            name = m.body_id2name(other) or ""
            # the Panda's pads are gripper0_{left,right}finger and their tips are
            # gripper0_finger_joint{1,2}_tip: contact usually lands on the tips
            if "leftfinger" in name or "finger_joint1" in name:
                sides.add(0)
            elif "rightfinger" in name or "finger_joint2" in name:
                sides.add(1)
        if len(sides) < 2:
            return False
        # jaws that have stopped closing mean the squeeze has settled -- the test that
        # decides when to LIFT. Once the object is off its support that test flickers
        # with every acceleration (held True/False on alternate steps through a carry,
        # and the teacher switching between squeeze and lift), so an object touching
        # nothing but the fingers counts as held, whatever the jaws are doing.
        return (abs(self.env.snapshot()["aperture_rate"]) < self.k.squeeze_rate
                or self._airborne(bid))

    def _airborne(self, body: int) -> bool:
        """Touching the gripper and nothing else."""
        m, d = self.scene.m, self.scene.d
        touching = False
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if body not in (b1, b2):
                continue
            other = m.body_id2name(b2 if b1 == body else b1) or ""
            if not other.startswith("gripper"):
                return False
            touching = True
        return touching

    # -- skills ---------------------------------------------------------------------
    def region_pose(self, region: str):
        """(R, p, half) of a region site, or of an object used as one (On(x, plate))."""
        try:
            return self.scene.region(region)
        except Exception:
            box = self.scene.object_box(region)
            return box.R, box.world_centre, box.half

    def via_for(self, region: str) -> np.ndarray:
        """Where the tool must be to deliver into this region -- the point the grasp yaw
        has to stay reachable at, not just the grasp itself."""
        _, p_reg, half = self.region_pose(region)
        return np.array([p_reg[0], p_reg[1], p_reg[2] + float(half[2]) + self.k.lift])

    def pick(self, obj: str, s: dict, via: np.ndarray | None = None) -> np.ndarray:
        """Approach a synthesised grasp through a funnel, then squeeze."""
        k = self.k
        R, p = s["R_tool"], s["p_tool"]
        R_g, p_g, width = self.grasp_for(obj, via)
        app = self.approach_for(obj)
        open_to = min(k.max_grip, width + k.grip_margin)
        if self.held(obj):
            self.phase = "lift"
            # the first 2 cm slowly, so friction takes the load before the jaws accelerate
            # it -- a rim pinch lost the bowl mid-carry at 29 cm and it left the table
            slow = float(self.scene.object_box(obj).world_centre[2] - self._grasp_cache[obj][3][2]) < 0.02
            return self.action(self.twist_to(R, p, R_g, p + Z * 0.05, v_max=0.06 if slow else 0.12), 0.0)
        e_rot = rot_angle(R.T @ R_g)
        if self.at_grasp(p, p_g, app) and e_rot < k.at_rot:
            self.phase = "squeeze"
            return self.action(self.twist_to(R, p, R_g, p_g, v_min=0.0), 0.0)
        off = p - p_g
        lateral = float(np.linalg.norm(off - app * (off @ app)))
        along = float(-(off @ app))                    # still to advance toward the grasp
        misalign = max(lateral / k.funnel_xy, e_rot / k.funnel_rot)
        pre = p_g - app * k.approach                   # where the advance starts
        if lateral <= k.funnel_xy and along <= k.approach + 0.02:
            self.phase = "descend"                     # on the approach axis: advance
            target = p_g - app * (k.approach * float(np.clip(misalign, 0.0, 1.0)))
            return self.action(self.twist_to(R, p, R_g, target), open_to)
        leg = self.path_to(R, p, R_g, pre, max(self.transit_for(obj), pre[2]))
        if leg is not None:
            self.phase, R_t, target = leg
            return self.action(self.twist_to(R, p, R_t, target), open_to)
        self.phase = "down"                            # above the start: straight down to it
        return self.action(self.twist_to(R, p, R_g, pre), open_to)

    def place(self, obj: str, region: str, s: dict, inside: bool) -> np.ndarray:
        """Carry a held object over the region and release it there.

        Controlled on the OBJECT, not the tool: the tool's offset from the object is
        whatever the grasp happened to be, and LIBERO tests the object's own origin
        against the region box (In) or its surface (On).
        """
        k = self.k
        R, p = s["R_tool"], s["p_tool"]
        R_reg, p_reg, half = self.region_pose(region)
        q, target_q = self.place_target(obj, region, inside)
        if not self.held(obj):
            if self.at_place(q, target_q):     # let go of it, do not take it back
                self.phase = "settle"
                return self.retreat(s)
            self.phase = "regrasp"
            return self.pick(obj, s, via=self.via_for(region))
        delta = target_q - q
        over = float(np.linalg.norm(delta[:2]))
        carry_z = self._carry_height(obj, region, q, target_q, R, p)
        R_carry = R                                 # the grasp yaw was chosen to reach here too
        if over > 0.02 and q[2] < carry_z - 0.02:
            self.phase = "lift"
            c = self._grasp_cache.get(obj)
            slow = c is not None and float(self.scene.object_box(obj).world_centre[2] - c[3][2]) < 0.02
            return self.action(self.twist_to(R, p, R_carry, p + Z * (carry_z - q[2]),
                                             v_max=0.06 if slow else 0.15), 0.0)
        if over > 0.02:
            self.phase = "carry"
            goal = p + np.array([delta[0], delta[1], max(0.0, carry_z - q[2])])
            # 0.15 m/s, not the 0.25 of an empty hand: a bowl held by a 2.6 mm rim pinch
            # left the jaws at 30 cm during a full-speed carry
            return self.action(self.twist_to(R, p, R_carry, goal, v_max=0.15), 0.0)
        if delta[2] < -0.004:
            self.phase = "lower"
            return self.action(self.twist_to(R, p, R_carry, p + Z * delta[2], v_max=0.10, v_min=0.03), 0.0)
        self.phase = "release"
        return self.action(np.zeros(6), A_OPEN)

    # -- articulated fixtures --------------------------------------------------------
    def holding(self, body: int) -> bool:
        """Both finger groups touch this body and the jaws have stopped closing."""
        m, d = self.scene.m, self.scene.d
        sides = set()
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            other = b2 if b1 == body else (b1 if b2 == body else None)
            if other is None:
                continue
            name = m.body_id2name(other) or ""
            if "leftfinger" in name or "finger_joint1" in name:
                sides.add(0)
            elif "rightfinger" in name or "finger_joint2" in name:
                sides.add(1)
        return len(sides) == 2 and abs(self.env.snapshot()["aperture_rate"]) < self.k.squeeze_rate

    def articulate(self, region: str, mode: str, s: dict) -> np.ndarray:
        """Drive a hinge or a slide to its goal by taking the handle with it.

        The drawer and the stove knob are the same skill: LIBERO's fixtures move one joint,
        the handle is the collision geom furthest along the opening direction, and the tool
        pose that opens the fixture is the current tool pose carried by that joint's own
        motion -- a translation along the slide axis, or a rotation about the hinge anchor.
        Nothing here is per-fixture except the thresholds LIBERO's own predicates use.
        """
        k = self.k
        R, p = s["R_tool"], s["p_tool"]
        a = self.scene.articulation(region)
        th, sign = a["thresholds"], a["sign"]
        q_goal = {"open": th.get("open"), "close": th.get("close"),
                  "on": th.get("on"), "off": th.get("off")}[mode]
        q_goal = float(q_goal) + sign * (0.03 if mode in ("open", "on") else -0.01)
        dq = q_goal - a["qpos"]
        # choose the frame ONCE: re-solving every step let the winner flip between
        # candidates, the wrist target jumped, and the jaws cycled 15-35 mm without ever
        # closing on the bar. The handle itself moves as the fixture opens, so the POINT
        # is re-read each step and only the frame is held.
        self.new_episode_check()
        held_frame = self._handle_cache.get(region)
        if held_frame is None:
            cands = self._handle_grasps(a["handle_geom"])
            if not cands:
                raise NotImplementedError(f"{region}: no graspable handle geom")
            R_h, _p, w, app = self._choose(cands, None, allow=a["body"])
            self._handle_cache[region] = (R_h, w, app)
            self.grasp_log[region] = dict(
                tier="handle", offered={"handle": len(cands)}, feasible=self.last_choice.get("feasible"),
                score=self.last_choice.get("score"), forced=self.last_choice.get("forced"),
                approach=[round(float(v), 2) for v in app], jaw=[round(float(v), 2) for v in R_h[:, 1]],
                width_mm=round(1000 * float(w), 1), handle_geom=self.scene.m.geom_id2name(a["handle_geom"]))
        else:
            R_h, w, app = held_frame
        p_h = self.scene.d.geom_xpos[a["handle_geom"]] - self.scene.base
        if not self.holding(a["body"]):
            open_to = min(k.max_grip, w + k.grip_margin)
            off = p - p_h
            along = float(off @ (-app))                    # how far short of the handle
            lateral = float(np.linalg.norm(off + app * along))
            e_rot = rot_angle(R.T @ R_h)
            if lateral < 0.012 and -0.015 < along < 0.035 and e_rot < k.at_rot:
                self.phase = "squeeze"
                return self.action(self.twist_to(R, p, R_h, p_h, v_min=0.0), 0.0)
            misalign = max(lateral / k.funnel_xy, e_rot / k.funnel_rot)
            target = p_h - app * (k.approach * float(np.clip(misalign, 0.0, 1.0)))
            self.phase = "reach" if misalign >= 0.5 else "descend"
            return self.action(self.twist_to(R, p, R_h, target), open_to)
        self.phase = "drive"
        if a["jnt_type"] == 2:                                     # slide: translate
            R_goal, p_goal = R, p + a["axis"] * dq
        else:                                                      # hinge: turn about the anchor
            Rr = _axis_rot(a["axis"], dq)
            R_goal, p_goal = Rr @ R, a["anchor"] + Rr @ (p - a["anchor"])
        return self.action(self.twist_to(R, p, R_goal, p_goal, v_max=0.12, v_min=0.02), 0.0)

    def transit_height(self, p_from: np.ndarray, p_to: np.ndarray, exclude: int) -> float:
        """A height that clears everything standing between here and there.

        Measured from the scene: every geom whose footprint is within 8 cm of the line,
        minus the robot and the target itself, taken at its top. The bounding radius is
        enough -- being generous costs a few centimetres of travel and avoids a stall.
        """
        from .scene import _geom_half
        m, d = self.scene.m, self.scene.d
        if self._geom_mask is None:
            names = [m.body_id2name(int(m.geom_bodyid[g])) or "" for g in range(m.ngeom)]
            self._geom_mask = np.array([not n.startswith(("robot", "gripper")) for n in names])
            self._geom_half = np.stack([_geom_half(m, g) for g in range(m.ngeom)])
        pos = d.geom_xpos - self.scene.base
        xmat = d.geom_xmat.reshape(-1, 3, 3)
        # the exact top of each geom, not its bounding radius: the table's radius is a
        # metre, and asking to clear that sent the tool to the ceiling (0/50 everywhere)
        tops = pos[:, 2] + (np.abs(xmat[:, 2, :]) * self._geom_half).sum(1)
        rad = np.linalg.norm(self._geom_half[:, :2], axis=1)
        a, b = np.asarray(p_from[:2], float), np.asarray(p_to[:2], float)
        ab = b - a
        L2 = float(ab @ ab)
        t = np.clip(((pos[:, :2] - a) @ ab) / L2, 0.0, 1.0) if L2 > 1e-9 else np.zeros(len(pos))
        near = np.linalg.norm(pos[:, :2] - (a + t[:, None] * ab), axis=1) - rad < 0.08
        keep = near & self._geom_mask & (np.asarray(m.geom_bodyid) != exclude)
        floor = float(p_to[2] + self.k.approach)
        if not keep.any():
            return floor
        return float(min(max(np.max(tops[keep]) + 0.05, floor), p_to[2] + 0.30))

    def path_to(self, R: np.ndarray, p: np.ndarray, R_goal: np.ndarray, start: np.ndarray,
                safe_h: float) -> tuple[str, np.ndarray, np.ndarray] | None:
        """Up, over, down on to `start`; None once the tool is in the column above it.

        Aiming straight at a target cuts a diagonal through whatever stands between, and
        rotates the wrist while the fingers are next to things. The path a person would
        draw instead: rise vertically to a plane measured to clear the scene, cross it,
        turn the wrist up there where nothing is near, and come straight down. Each leg
        is chosen from the current state, so any state still gets a label.
        """
        k = self.k
        lat = float(np.linalg.norm((p - start)[:2]))
        # entering the column needs 2 cm; once below the crossing plane, keep going down
        # and correct sideways unless 5 cm off. One threshold for both sent the tool back
        # up whenever it drifted 2 cm on the way down -- up/over/down/up dozens of times
        # in one episode on the bowl on the cabinet.
        if lat <= k.funnel_xy or (p[2] < safe_h - 0.02 and lat <= 0.05):
            return None
        if p[2] < safe_h - 0.02:
            return "up", R, np.array([p[0], p[1], safe_h])
        return "over", R_goal, np.array([start[0], start[1], safe_h])

    def place_target(self, obj: str, region: str, inside: bool) -> tuple[np.ndarray, np.ndarray]:
        """(object origin now, where that origin has to end up)."""
        self.new_episode_check()
        R_reg, p_reg, half = self.region_pose(region)
        box = self.scene.object_box(obj)
        q = self.scene.body_pose(obj)[1]                      # the origin LIBERO tests
        half_h = float(np.abs(box.R @ np.diag(box.half)).sum(1)[2])
        centre_off = float((box.world_centre - q)[2])         # origin to box centre, world z
        xy = self._drop_xy(obj, region, R_reg, p_reg, half, box)
        if inside:
            target_q = p_reg.copy()
            target_q[:2] = xy
            target_q[2] = p_reg[2] + max(0.0, float(half[2]) - half_h) + self.k.place_clearance
        else:
            target_q = p_reg + R_reg @ np.array([0.0, 0.0, float(half[2])])
            target_q[:2] = xy
            target_q[2] += half_h - centre_off + self.k.place_clearance
        return q, target_q

    def _drop_xy(self, obj: str, region: str, R_reg, p_reg, half, box) -> np.ndarray:
        """The point of the region nearest its centre that is open from above.

        A drawer opened 14 cm keeps the back of its interior under the cabinet top, and
        its region's centre can be under there too: the teacher held the bowl over a spot
        it could not come down on and stalled, 0/20 on "open the top drawer and put the
        bowl inside". A vertical ray says which parts of the region are reachable.
        """
        key = (obj, region)
        if key in self._drop_cache:
            return self._drop_cache[key]
        import mujoco
        m, d = self.scene.m, self.scene.d
        mm = m._model if hasattr(m, "_model") else m
        dd = d._data if hasattr(d, "_data") else d
        top = float(p_reg[2] + abs(float(half[2])))
        ohw = np.abs(box.R @ np.diag(box.half)).sum(1)[:2]      # object half-footprint
        span = np.maximum(np.abs(np.asarray(half[:2], float)) - ohw, 0.0)
        best, best_d = np.asarray(p_reg[:2], float), np.inf
        exclude = self.scene.body_id(obj)
        gid = np.zeros(1, np.int32)
        for fx in np.linspace(-1, 1, 5):
            for fy in np.linspace(-1, 1, 5):
                local = np.array([fx * span[0], fy * span[1], 0.0])
                pt = p_reg + R_reg @ local
                start = np.array([pt[0], pt[1], top + 0.30]) + self.scene.base
                dist = float(mujoco.mj_ray(mm, dd, start, np.array([0.0, 0.0, -1.0]), None, 1,
                                           exclude, gid))
                hit_z = start[2] - dist - self.scene.base[2] if dist >= 0 else -np.inf
                if hit_z > top + 0.01:
                    continue                    # something over this spot
                dc = float(np.linalg.norm(local[:2]))
                if dc < best_d:
                    best, best_d = pt[:2].copy(), dc
        self._drop_cache[key] = best
        return best

    def _carry_height(self, obj: str, region: str, q, target_q, R, p) -> float:
        """Object-origin height for the carry: its bottom clears what is between here and
        the drop, and the arm can hold the tool there with room to move.

        A fixed "region top + 12 cm" put the carry for a top drawer where the arm is
        nearly straight -- stuck in `lift` 200-400 mm short on most of 20 episodes.
        """
        key = (obj, region)
        if key in self._carry_cache:
            return self._carry_cache[key]
        from .kin_np import NpChain, sigma_min, solve_ik
        box = self.scene.object_box(obj)
        hang = float(q[2] - (box.world_centre[2] - np.abs(box.R @ np.diag(box.half)).sum(1)[2]))
        tool_off = p - q
        clear = self.transit_height(q, target_q, self.scene.body_id(obj)) - 0.05 + hang + 0.03
        lo = float(target_q[2] + 0.05)
        hi = float(min(max(clear, lo), target_q[2] + 0.30))
        hs = list(np.arange(hi, lo - 1e-9, -0.03)) or [lo]
        Ts = []
        for h in hs:
            T = np.eye(4); T[:3, :3] = R
            T[:3, 3] = np.array([target_q[0], target_q[1], h]) + tool_off
            Ts.append(T)
        c = NpChain.of(self.env.chain)
        res = solve_ik(c, np.stack(Ts), np.asarray(self.env.raw["robot0_joint_pos"], float)[None],
                       lam=0.02, max_iters=200, trust=0.2)
        sig = sigma_min(c, res["theta"])
        margin = np.minimum(res["theta"] - c.limits[:, 0], c.limits[:, 1] - res["theta"]).min(1)
        out = lo
        for i, h in enumerate(hs):
            if res["converged"][i] and sig[i] > 0.05 and margin[i] > 0.15:
                out = float(h)
                break
        self._carry_cache[key] = out
        return out

    def at_place(self, q: np.ndarray, target_q: np.ndarray) -> bool:
        """The object is where it was carried to, whether or not it is still in the jaws.

        Without this the release chatters: the jaws open, `held` goes false, the goal is
        not scored yet, and the teacher takes the object back -- measured on the wine
        bottle, release and squeeze alternating every two steps until the horizon.
        """
        d = target_q - q
        return bool(np.linalg.norm(d[:2]) < 0.04 and d[2] < 0.03)

    def retreat(self, s: dict) -> np.ndarray:
        self.phase = "retreat"
        R, p = s["R_tool"], s["p_tool"]
        return self.action(self.twist_to(R, p, R, p + Z * 0.10, v_max=0.15), A_OPEN)
