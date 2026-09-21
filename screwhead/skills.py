"""Skills, parameterised by measured geometry, sequenced from a task's goal.

The ten libero_spatial programs are hand-written and their constants tuned per task. This
is the replacement: pick, place and articulate read the object's own collision geometry
and the target region's own site, so a task nobody wrote a program for still gets a
demonstration. Grasp candidates come from grasp_planner.py, whether the arm can use them
from reach.py; this module holds the feedback laws and the per-episode decisions.

Each skill is a Markov feedback law -- its action is a function of the current state --
so the teacher can label any state a student reaches, which is what DAgger needs
(belief.yaml, IFC-teacher__expert). The only memory is per-episode caches of decisions
that are re-derivable from the state (the grasp, the handle frame, the drop point, the
carry height), cleared at every episode boundary.

Kept from the spatial programs, because they were measured there:
  funnel approach   the target height above the grasp shrinks as the tool aligns
  squeeze           close until the jaws stop closing, rather than to a fixed aperture
  speed floor       5 cm/s minimum while approaching (proportional homing ends at
                    1-3 cm/s, where a student's direction agreed only to cosine 0.68)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import contacts
from .frames import Z, axis_rot, pose, rot_angle, rotvec
from .grasp_planner import GraspPlanner
from .gripper_servo import A_OPEN, target_to_channel
from .reach import Reach

MIN_TWIST_DIST = 0.001    # m: nearer than this, the speed floor does not apply


@dataclass(frozen=True)
class SkillConfig:
    # proportional control and speed limits
    k_lin: float = 5.0
    k_rot: float = 3.0
    v_max: float = 0.25
    w_max: float = 1.2
    v_min: float = 0.05            # approach speed floor
    # approach
    approach: float = 0.10         # pre-grasp distance before the grasp, along the approach
    funnel_xy: float = 0.02        # lateral error at which the approach is back at full height
    funnel_rot: float = 0.15       # rad, same for rotation
    at_rot: float = 0.10           # rad: squeeze only this well aligned
    descend_margin: float = 0.02   # still "on the axis" this far beyond the pre-grasp
    column_leave: float = 0.05     # below the crossing plane, go back up only this far off
    plane_band: float = 0.02       # "below the crossing plane" means this far below it
    # the jaw envelope: lateral error, and how far short of / past the grasp point
    grasp_lateral: float = 0.015   # objects: the fingers sweep 15 mm either side
    handle_lateral: float = 0.012  # handles are thinner
    short_max: float = 0.035       # the pads reach 35 mm back from the tool point
    past_max: float = 0.015        # the tips 15 mm beyond it
    reach_misalign: float = 0.5    # handle approach reads as "reach" above this misalignment
    handle_open_slack: float = 0.012   # jaws have closed ON a handle when the aperture lies in
    handle_min_frac: float = 0.5       # [frac * width, width + slack]; near zero they missed it
    handle_leave_lateral: float = 0.03  # the jaws are still around a handle within this of its axis
    handle_short_max: float = 0.008    # squeeze a handle only with the tool point this close to it:
                                       # the object envelope (35 mm short) let the jaws finish
                                       # closing 17 mm in front of the drawer's bar
    # grip
    grip_margin: float = 0.012     # jaws this much wider than the object before descending
    max_grip: float = 0.075        # widest jaw opening used (A_OPEN is the hard limit)
    squeeze_rate: float = 0.005    # jaw speed below which a squeeze counts as settled
    regrasp_move: float = 0.01     # an object moved this far gets a fresh grasp
    # lift, carry, place
    lift: float = 0.12             # carry height above the support, for the grasp's via point
    lift_dz: float = 0.05          # a pick lifts this far before the place takes over
    lift_speed: float = 0.12
    slow_lift_dz: float = 0.02     # the first 2 cm of any lift slowly, so friction takes the
    slow_lift_speed: float = 0.06  # load before the jaws accelerate it (a rim pinch dropped
                                   # the bowl at 29 cm when lifted at full speed)
    carry_speed: float = 0.15      # a bowl held by a 2.6 mm rim pinch left the jaws at 0.25
    over_xy: float = 0.02          # the object is "over" the target within this
    lower_speed: float = 0.10
    lower_speed_min: float = 0.03
    lower_done: float = 0.004      # lowered to within this of the target height
    place_clearance: float = 0.015 # object bottom above the target surface before release
    at_place_xy: float = 0.04      # a released object this close to its target is left there
    at_place_dz: float = 0.03
    drop_grid: int = 5             # drop points tried across a region, per axis
    drop_ray_height: float = 0.30  # rays cast down from this far above the region top
    drop_clear: float = 0.01       # a hit above the region top by more than this blocks the spot
    retreat_dz: float = 0.10
    retreat_speed: float = 0.15
    # articulation
    drive_past_open: float = 0.03  # drive a joint this far past its "open"/"on" threshold
    drive_past_close: float = 0.01 # and this far past "close"/"off"
    drive_speed: float = 0.12
    drive_lead_rot: float = 0.06   # rad a hinge drive may lead the handle (< at_rot): leading by the
    #                                whole remaining turn spun the tool at w_max while the knob, damped,
    #                                followed at a fifth of that -- the jaws were pried open and let go
    drive_lead_slide: float = 0.03 # m a slide drive may lead it (k_lin x this still saturates drive_speed)
    drive_speed_min: float = 0.02


class Skills:
    """Feedback laws over a TaskEnv + Scene."""

    def __init__(self, env, config: SkillConfig | None = None):
        self.env = env
        self.scene = env.scene
        self.k = config or SkillConfig()
        self.planner = GraspPlanner(env, self.k)
        self.reach = Reach(env, self.k)
        self.phase = ""
        self.grasp_log: dict = {}      # what was decided and why, for tools/skill_eval.py
        self._episode = None
        self._grasp_cache: dict[str, tuple] = {}
        self._handle_cache: dict[str, tuple[np.ndarray, float, np.ndarray, float]] = {}
        self._drop_cache: dict = {}
        self._carry_cache: dict = {}

    # -- commands -------------------------------------------------------------------------
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
        if v_min > 0 and dist > MIN_TWIST_DIST and 0 < np.linalg.norm(v) < v_min:
            v *= v_min / np.linalg.norm(v)
        return np.concatenate([w / spec.max_angular_speed, v / spec.max_linear_speed])

    def action(self, twist, target_aperture: float) -> np.ndarray:
        return np.concatenate([twist, [target_to_channel(np.clip(target_aperture, 0.0, A_OPEN))]])

    # -- per-episode decisions --------------------------------------------------------------
    def new_episode_check(self) -> None:
        """Forget every decision made in a previous episode.

        LIBERO often starts the next episode with the object within a centimetre of where
        the last one started, so a cache keyed on the object's position inherited the
        previous episode's grasp: one bad choice in episode 0 failed 17 more on the cream
        cheese. A teacher whose labels depend on an earlier episode is not Markov.
        """
        ep = getattr(self.env, "episode", None)
        if ep != self._episode:
            self._episode = ep
            for cache in (self._grasp_cache, self._handle_cache, self.grasp_log,
                          self._drop_cache, self._carry_cache):
                cache.clear()

    def grasp_for(self, obj: str, via: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float]:
        """The grasp for an object: shape proposes (grasp_planner), reachability disposes
        (reach), re-chosen when the object has moved."""
        self.new_episode_check()
        box = self.scene.object_box(obj)
        centre = box.world_centre
        cached = self._grasp_cache.get(obj)
        if cached is not None and np.linalg.norm(cached[3] - centre) < self.k.regrasp_move:
            return cached[0], cached[1], cached[2]
        bid = self.scene.body_id(obj)
        tiers = self.planner.tiers(obj, box)
        chosen, used = None, "forced"
        for name, tier in tiers:
            chosen = self.reach.choose(tier, via, allow=bid, strict=True)
            if chosen is not None:
                used = name
                break
        if chosen is None:                      # nothing feasible anywhere: take the best try
            chosen = self.reach.choose(sum((t for _, t in tiers), []), via, allow=bid)
        R_grasp, p_grasp, width, app = chosen
        h = self.reach.reachable_transit(R_grasp, p_grasp - app * self.k.approach, bid)
        self._grasp_cache[obj] = (R_grasp, p_grasp, width, centre, h, app)
        self._log_grasp(obj, used, tiers, via, chosen, centre, h)
        return R_grasp, p_grasp, width

    def _log_grasp(self, obj, tier, tiers, via, chosen, centre, h) -> None:
        R_grasp, p_grasp, width, app = chosen
        choice = self.reach.last_choice
        self.grasp_log[obj] = dict(
            tier=tier, offered={n: len(t) for n, t in tiers}, via=via is not None,
            feasible=choice.get("feasible"), score=choice.get("score"),
            approach=[round(float(v), 2) for v in app], jaw=[round(float(v), 2) for v in R_grasp[:, 1]],
            width_mm=round(1000 * float(width), 1),
            open_mm=round(1000 * min(self.k.max_grip, width + self.k.grip_margin), 1),
            grasp_above_centre_mm=round(1000 * float(p_grasp[2] - centre[2]), 1),
            transit=round(float(h), 3), choices=self.grasp_log.get(obj, {}).get("choices", 0) + 1)

    def cached_grasp(self, obj: str) -> np.ndarray | None:
        """The grasp point already chosen for `obj`, without choosing one.

        For observers: tools/skill_eval.py's diagnostics used to call grasp_for(), which
        re-chose and CACHED a grasp -- without the place point -- whenever the object had
        moved, and the teacher then reused that choice if it had to regrasp. Reading must
        not change what is read.
        """
        c = self._grasp_cache.get(obj)
        return None if c is None else c[1]

    def transit_for(self, obj: str) -> float:
        """The cached crossing height for this object's grasp."""
        c = self._grasp_cache.get(obj)
        return float(c[4]) if c is not None else 0.0

    def approach_for(self, obj: str) -> np.ndarray:
        """The direction the tool advances on this object: down, or in from the side."""
        c = self._grasp_cache.get(obj)
        return c[5] if c is not None else -Z

    # -- state tests -------------------------------------------------------------------------
    def _in_envelope(self, p, p_target, app, lateral_max: float, short_max: float | None = None) -> bool:
        """The target is inside the jaw envelope: `lateral_max` either side, and from
        `short_max` short of the target to `past_max` beyond it."""
        d = p - p_target
        along = float(d @ app)                         # negative while short of the target
        short = self.k.short_max if short_max is None else short_max
        return bool(np.linalg.norm(d - app * along) < lateral_max and -short < along < self.k.past_max)

    def at_grasp(self, p: np.ndarray, p_g: np.ndarray, app: np.ndarray | None = None) -> bool:
        """The object is inside the jaw envelope, so squeeze.

        Not a tight ball around the grasp point: the point is re-derived from the object,
        the object shifts when the pads touch it, and the tool then falls out of the ball
        and REOPENS -- squeeze and descend alternated for 200 steps on the cream cheese.
        """
        return self._in_envelope(p, p_g, -Z if app is None else app, self.k.grasp_lateral)

    def held(self, obj: str) -> bool:
        """Both finger groups touch the object, and either the squeeze has settled or the
        object touches nothing but the fingers.

        Jaws that have stopped closing is the test for when to LIFT; mid-carry it flickered
        with every acceleration and the teacher alternated squeeze and lift.
        """
        m, d = self.scene.m, self.scene.d
        try:
            bid = self.scene.body_id(obj)
        except ValueError:
            return False
        if len(contacts.finger_sides(m, d, bid)) < 2:
            return False
        return (abs(self.env.snapshot()["aperture_rate"]) < self.k.squeeze_rate
                or contacts.only_gripper(m, d, bid))

    def holding(self, body: int, width: float) -> bool:
        """Both finger groups touch this body and the jaws have closed down to its width.

        Not "the jaws have stopped moving": pulling a handle moves the jaws, so that test
        went false on the first step of every pull, the teacher re-centred on the handle,
        and squeeze and drive alternated step by step. And not "both fingers touch the
        body": replayed, the jaws had closed to 1 mm in front of the drawer's bar, fingertips
        pressed on its face, and that passed as holding. The aperture has to be stopped by
        the handle itself.
        """
        ap = self.env.snapshot()["aperture"]
        return (len(contacts.finger_sides(self.scene.m, self.scene.d, body)) == 2
                and self.k.handle_min_frac * width <= ap <= width + self.k.handle_open_slack)

    # -- regions -------------------------------------------------------------------------------
    def region_pose(self, region: str):
        """(R, p, half) of a region site, or of an object used as one (On(x, plate))."""
        try:
            return self.scene.region(region)
        except ValueError:                             # not a site: an object
            box = self.scene.object_box(region)
            return box.R, box.world_centre, box.half

    def via_for(self, region: str) -> np.ndarray:
        """Where the tool must be to deliver into this region -- the point the grasp has to
        stay reachable at, not just the grasp itself."""
        _, p_reg, half = self.region_pose(region)
        return np.array([p_reg[0], p_reg[1], p_reg[2] + float(half[2]) + self.k.lift])

    # -- pick ------------------------------------------------------------------------------------
    def pick(self, obj: str, s: dict, via: np.ndarray | None = None) -> np.ndarray:
        """Up, over and down to the pre-grasp, advance along the approach, squeeze, lift."""
        k = self.k
        R, p = s["R_tool"], s["p_tool"]
        R_g, p_g, width = self.grasp_for(obj, via)
        app = self.approach_for(obj)
        open_to = min(k.max_grip, width + k.grip_margin)
        if self.held(obj):
            self.phase = "lift"
            slow = self._risen(obj) < k.slow_lift_dz
            return self.action(self.twist_to(R, p, R_g, p + Z * k.lift_dz,
                                             v_max=k.slow_lift_speed if slow else k.lift_speed), 0.0)
        e_rot = rot_angle(R.T @ R_g)
        if self.at_grasp(p, p_g, app) and e_rot < k.at_rot:
            self.phase = "squeeze"
            return self.action(self.twist_to(R, p, R_g, p_g, v_min=0.0), 0.0)
        off = p - p_g
        lateral = float(np.linalg.norm(off - app * (off @ app)))
        along = float(-(off @ app))                    # still to advance toward the grasp
        misalign = max(lateral / k.funnel_xy, e_rot / k.funnel_rot)
        pre = p_g - app * k.approach                   # where the advance starts
        if lateral <= k.funnel_xy and along <= k.approach + k.descend_margin:
            self.phase = "descend"                     # on the approach axis: advance
            target = p_g - app * (k.approach * float(np.clip(misalign, 0.0, 1.0)))
            return self.action(self.twist_to(R, p, R_g, target), open_to)
        leg = self.path_to(R, p, R_g, pre, max(self.transit_for(obj), pre[2]))
        if leg is not None:
            self.phase, R_t, target = leg
            return self.action(self.twist_to(R, p, R_t, target), open_to)
        self.phase = "down"                            # above the start: straight down to it
        return self.action(self.twist_to(R, p, R_g, pre), open_to)

    def _risen(self, obj: str) -> float:
        """How far the object has risen since its grasp was chosen."""
        c = self._grasp_cache.get(obj)
        return np.inf if c is None else float(self.scene.object_box(obj).world_centre[2] - c[3][2])

    def path_to(self, R: np.ndarray, p: np.ndarray, R_goal: np.ndarray, start: np.ndarray,
                safe_h: float) -> tuple[str, np.ndarray, np.ndarray] | None:
        """Up, over, down on to `start`; None once the tool is in the column above it.

        Aiming straight at a target cuts a diagonal through whatever stands between and
        turns the wrist next to things. Instead: rise to a plane measured to clear the
        scene, cross it, turn the wrist up there, come straight down. Entering the column
        needs `funnel_xy`; once below the plane the tool keeps descending unless
        `column_leave` off -- one threshold for both went up/over/down/up dozens of times.
        """
        k = self.k
        lat = float(np.linalg.norm((p - start)[:2]))
        below = p[2] < safe_h - k.plane_band
        if lat <= k.funnel_xy or (below and lat <= k.column_leave):
            return None
        if below:
            return "up", R, np.array([p[0], p[1], safe_h])
        return "over", R_goal, np.array([start[0], start[1], safe_h])

    # -- place -----------------------------------------------------------------------------------
    def place(self, obj: str, region: str, s: dict, inside: bool) -> np.ndarray:
        """Carry a held object over the region and release it there.

        Controlled on the OBJECT, not the tool: the tool's offset from the object is
        whatever the grasp happened to be, and LIBERO tests the object's own origin.
        """
        k = self.k
        R, p = s["R_tool"], s["p_tool"]
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
        if over > k.over_xy and q[2] < carry_z - k.plane_band:
            self.phase = "lift"
            slow = self._risen(obj) < k.slow_lift_dz
            return self.action(self.twist_to(R, p, R, p + Z * (carry_z - q[2]),
                                             v_max=k.slow_lift_speed if slow else k.carry_speed), 0.0)
        if over > k.over_xy:
            self.phase = "carry"
            goal = p + np.array([delta[0], delta[1], max(0.0, carry_z - q[2])])
            return self.action(self.twist_to(R, p, R, goal, v_max=k.carry_speed), 0.0)
        if delta[2] < -k.lower_done:
            self.phase = "lower"
            return self.action(self.twist_to(R, p, R, p + Z * delta[2], v_max=k.lower_speed,
                                             v_min=k.lower_speed_min), 0.0)
        self.phase = "release"
        return self.action(np.zeros(6), A_OPEN)

    def place_target(self, obj: str, region: str, inside: bool,
                     decide: bool = True) -> tuple[np.ndarray, np.ndarray] | None:
        """(object origin now, where that origin has to end up). With decide=False, None if
        the drop point has not been chosen this episode: observers read it, never choose it."""
        if not decide and (getattr(self.env, "episode", None) != self._episode
                           or (obj, region) not in self._drop_cache):
            return None
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

        A drawer opened 14 cm keeps the back of its interior under the cabinet top, and its
        region's centre can be under there too. A vertical ray says which parts are open.
        """
        key = (obj, region)
        if key in self._drop_cache:
            return self._drop_cache[key]
        k = self.k
        top = float(p_reg[2] + abs(float(half[2])))
        ohw = np.abs(box.R @ np.diag(box.half)).sum(1)[:2]      # object half-footprint
        span = np.maximum(np.abs(np.asarray(half[:2], float)) - ohw, 0.0)
        best, best_d = np.asarray(p_reg[:2], float), np.inf
        exclude = self.scene.body_id(obj)
        down = np.array([0.0, 0.0, -1.0])
        for fx in np.linspace(-1, 1, k.drop_grid):
            for fy in np.linspace(-1, 1, k.drop_grid):
                local = np.array([fx * span[0], fy * span[1], 0.0])
                pt = p_reg + R_reg @ local
                start = np.array([pt[0], pt[1], top + k.drop_ray_height])
                _hit, dist = self.planner._ray(start, down, exclude)
                hit_z = start[2] - dist if dist >= 0 else -np.inf
                dc = float(np.linalg.norm(local[:2]))
                if hit_z <= top + k.drop_clear and dc < best_d:
                    best, best_d = pt[:2].copy(), dc
        self._drop_cache[key] = best
        return best

    def _carry_height(self, obj: str, region: str, q, target_q, R, p) -> float:
        key = (obj, region)
        if key not in self._carry_cache:
            self._carry_cache[key] = self.reach.carry_height(
                self.scene.object_box(obj), q, target_q, R, p, self.scene.body_id(obj))
        return self._carry_cache[key]

    def at_place(self, q: np.ndarray, target_q: np.ndarray) -> bool:
        """The object is where it was carried to, whether or not it is still in the jaws.

        Without this the release chatters: the jaws open, `held` goes false, the goal is
        not scored yet, and the teacher takes the object back.
        """
        d = target_q - q
        return bool(np.linalg.norm(d[:2]) < self.k.at_place_xy and d[2] < self.k.at_place_dz)

    def retreat(self, s: dict) -> np.ndarray:
        self.phase = "retreat"
        R, p = s["R_tool"], s["p_tool"]
        return self.action(self.twist_to(R, p, R, p + Z * self.k.retreat_dz, v_max=self.k.retreat_speed),
                           A_OPEN)

    # -- articulated fixtures ------------------------------------------------------------------
    def articulate(self, region: str, mode: str, s: dict) -> np.ndarray:
        """Drive a hinge or a slide to its goal by taking the handle with it.

        The drawer and the stove knob are the same skill: the tool pose that opens the
        fixture is the current one carried by that joint's own motion -- a translation
        along the slide axis, or a rotation about the hinge anchor. Nothing here is
        per-fixture except the thresholds LIBERO's own predicates use.
        """
        k = self.k
        R, p = s["R_tool"], s["p_tool"]
        a = self.scene.articulation(region)
        R_h, w, app = self._handle_frame(region, a, mode)
        p_h = self.scene.d.geom_xpos[a["handle_geom"]] - self.scene.base
        if self.holding(a["body"], w):
            return self._drive(a, mode, R, p, R_h, p_h)
        e_rot = rot_angle(R.T @ R_h)
        if self._in_envelope(p, p_h, app, k.handle_lateral, k.handle_short_max) and e_rot < k.at_rot:
            self.phase = "squeeze"
            return self.action(self.twist_to(R, p, R_h, p_h, v_min=0.0), 0.0)
        off = p - p_h
        short = float(off @ (-app))
        misalign = max(float(np.linalg.norm(off + app * short)) / k.funnel_xy, e_rot / k.funnel_rot)
        target = p_h - app * (k.approach * float(np.clip(misalign, 0.0, 1.0)))
        self.phase = "reach" if misalign >= k.reach_misalign else "descend"
        return self.action(self.twist_to(R, p, R_h, target), min(k.max_grip, w + k.grip_margin))

    def leave_handle(self, s: dict) -> np.ndarray | None:
        """Back out of a driven handle the way the jaws went in, before anything else moves
        the arm; None once clear of every handle.

        The top drawer's bar is pinched with the jaw axis vertical, so rising first -- the
        next pick's opening move -- hooked it with the lower finger and jolted the drawer
        back past its open margin: the precondition flickered open/shut for 25 steps.
        """
        k, p = self.k, s["p_tool"]
        for region in list(self._handle_cache):
            a = self.scene.articulation(region)
            R_h, w, app = self._handle_frame(region, a, "open")
            p_h = self.scene.d.geom_xpos[a["handle_geom"]] - self.scene.base
            off = p - p_h
            ahead = float(off @ app)                       # 0 at the handle, -approach at the pre-grasp
            if np.linalg.norm(off - app * ahead) < k.handle_leave_lateral and abs(ahead) < k.approach:
                self.phase = "leave"
                return self.action(self.twist_to(s["R_tool"], p, s["R_tool"], p_h - app * (k.approach + k.lift_dz)),
                                   min(k.max_grip, w + k.grip_margin))
        return None

    def handle_width(self, region: str) -> float | None:
        """The width of the handle grasp chosen this episode, if one has been."""
        c = self._handle_cache.get(region)
        return None if c is None else c[1]

    def _handle_frame(self, region: str, a: dict, mode: str) -> tuple[np.ndarray, float, np.ndarray]:
        """The handle grasp's frame, chosen ONCE per episode: re-solving every step let the
        winner flip between candidates and the jaws cycled 15-35 mm without closing.

        What is held is the frame relative to the joint, carried by the joint's motion since
        it was chosen: a slide does not turn its handle, a hinge does. Held fixed in the world,
        the frame went stale as the stove knob turned, and a tool turning with the knob read
        as misaligned (0.15 rad at a knob angle of 0.02) and let go to re-reach."""
        self.new_episode_check()
        if region not in self._handle_cache:
            self._choose_handle(region, a, mode)
        R_h0, w, app0, q0 = self._handle_cache[region]
        R_h, _ = self._joint_motion(a, a["qpos"] - q0, R_h0, np.zeros(3))
        app, _ = self._joint_motion(a, a["qpos"] - q0, app0[:, None], np.zeros(3))
        return R_h, w, app[:, 0]

    def _choose_handle(self, region: str, a: dict, mode: str) -> None:
        cands = self.planner.handle_grasps(a["handle_geom"])
        if not cands:
            raise NotImplementedError(f"{region}: no graspable handle geom")
        dq = self._drive_dq(a, mode)

        def along_the_motion(cand):
            # the grasp has to stay usable while the joint moves, not only where it starts:
            # a stove-knob grasp with 1 feasible candidate in 10 turned the knob to 0.35 of
            # its 0.5 and ran the arm into its limits, re-anchoring, every time
            R, p = cand[0], cand[1]
            return [pose(*self._joint_motion(a, f * dq, R, p)) for f in (0.5, 1.0)]

        R_h, _p, w, app = self.reach.choose(cands, None, allow=a["body"], extra=along_the_motion)
        self._handle_cache[region] = (R_h, w, app, float(a["qpos"]))
        choice = self.reach.last_choice
        self.grasp_log[region] = dict(
            tier="handle", offered={"handle": len(cands)}, feasible=choice.get("feasible"),
            score=choice.get("score"), forced=choice.get("forced"),
            approach=[round(float(v), 2) for v in app], jaw=[round(float(v), 2) for v in R_h[:, 1]],
            width_mm=round(1000 * float(w), 1), handle_geom=self.scene.m.geom_id2name(a["handle_geom"]))

    def _drive(self, a: dict, mode: str, R, p, R_h, p_h) -> np.ndarray:
        """Carry the held handle along the joint's own motion toward just past its goal, a
        bounded step ahead of where the HANDLE is now: aimed from the tool's own pose, the
        target ran on however far the handle lagged."""
        k = self.k
        self.phase = "drive"
        lead = k.drive_lead_slide if a["jnt_type"] == 2 else k.drive_lead_rot
        dq = float(np.clip(self._drive_dq(a, mode), -lead, lead))
        R_goal, p_goal = self._joint_motion(a, dq, R_h, p_h)
        return self.action(self.twist_to(R, p, R_goal, p_goal, v_max=k.drive_speed,
                                         v_min=k.drive_speed_min), 0.0)

    def _drive_dq(self, a: dict, mode: str) -> float:
        """How far the joint still has to move: to just past its goal threshold."""
        th, sign = a["thresholds"], a["sign"]
        past = self.k.drive_past_open if mode in ("open", "on") else -self.k.drive_past_close
        return float(th[mode]) + sign * past - a["qpos"]

    @staticmethod
    def _joint_motion(a: dict, dq: float, R, p):
        """A pose carried by the joint's own motion: along the slide, or about the hinge."""
        if a["jnt_type"] == 2:                                     # slide: translate
            return R, p + a["axis"] * dq
        Rr = axis_rot(a["axis"], dq)                               # hinge: turn about the anchor
        return Rr @ R, a["anchor"] + Rr @ (p - a["anchor"])
