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

from ..sim import contacts
from .clearing import Clearing
from ..geometry.frames import Z, axis_rot, pose, rot_angle, rotvec, top_down
from .grasp_planner import GraspPlanner
from ..sim.gripper_servo import A_OPEN, target_to_channel
from ..sim.scene import VERTICAL_COS, geom_world_box
from .reach import Reach
from .refusal import Refusal

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
    handle_open_slack: float = 0.004   # jaws have closed ON a handle when the aperture lies in
    handle_min_frac: float = 0.5       # [frac * width, width + slack]; near zero they missed it.
    #                                    Held, it reads 0.9-1.4 mm over the width (drawer and knob,
    #                                    278 drive steps); the approach opening is width + 12 mm
    handle_leave_lateral: float = 0.03  # the jaws are still around a handle within this of its axis
    handle_leave_past: float = 0.01    # the retreat aims this far beyond where it stops, so the
    #                                    proportional approach does not stall short of it
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
    lift_speed: float = 0.15       # (0.12 before the carry heights were fixed; LIBERO's humans carry at 0.17-0.23)
    slow_lift_dz: float = 0.02     # the first 2 cm of any lift slowly, so friction takes the
    slow_lift_speed: float = 0.06  # load before the jaws accelerate it (a rim pinch dropped
                                   # the bowl at 29 cm when lifted at full speed)
    carry_speed: float = 0.18      # a bowl held by a 2.6 mm rim pinch left the jaws at 0.25; at 0.22,
    #                                libero_spatial 2 dropped it twice as often and fell 49 -> 36 of 50
    over_xy: float = 0.02          # the object is "over" the target within this
    lower_speed: float = 0.15
    lower_speed_min: float = 0.03
    lower_done: float = 0.004      # lowered to within this of the target height
    place_clearance: float = 0.015 # object bottom above the target surface before release
    at_place_xy: float = 0.04      # a released object this close to its target is left there
    at_place_dz: float = 0.03
    at_place_above: float = 0.03   # ... and no higher than this above it: with no bound, a bowl 8 cm
    #                                over libero_10 3's drawer counted as delivered the moment a
    #                                contact made `held` flicker, and was let go there
    drop_grid: int = 5             # drop points tried across a region, per axis
    drop_ray_height: float = 0.30  # rays cast down from this far above the region top
    drop_clear: float = 0.01       # a hit above the region top by more than this blocks the spot
    fit_tol: float = 0.005         # m an object's footprint may exceed a region's and still fit it
    drop_margin: float = 0.02      # m around the object's footprint that must be open from above too: at
    #                                0.01 a bowl carried 10 deg tilted and drifting 5-11 mm reached the face
    #                                of the drawer above libero_10 3's
    retreat_dz: float = 0.10
    # push (BRN-push-cages-a-dish), each measured on LIBERO's 50 human demos of libero_goal 5
    push_open: float = 0.062        # m the jaws stay open: 62 mm median, never commanded closed
    push_lead: float = 0.44         # tool point ahead of the dish's centre, x its plan radius
    push_floor: float = 0.0         # m the fingertips ride above the dish's floor -- on it, as the human
    #                                 hand does (fingertips 1.7 mm below its top): 3 mm above, they touched
    #                                 the plate on 1-4 steps of 400 and never moved it; 6 mm below the
    #                                 floor tilted the plate 8-12 degrees and pinned it to the table
    push_speed: float = 0.10        # m/s: at the humans' 0.06 the plate stick-slipped and averaged 0.019;
    #                                 at 0.10, 8 of 8 within LIBERO's 300 steps, median 182 against 265
    push_ahead: float = 0.08        # m the aim runs ahead of the held point: aimed 10-30 mm ahead,
    #                                 the reference stopped where the stuck plate held it, and the
    #                                 push had no force left; the human hand moves at a steady rate
    push_corridor: float = 0.015    # m off the held point, across the push, before re-entering
    push_slip: float = 0.04         # m ahead of the held point before the fingers count as out of it
    retreat_speed: float = 0.15
    # articulation
    drive_past_open: float = 0.03  # drive a joint this far past its "open"/"on" threshold
    open_margin: float = 0.012     # a container counts as open for filling this far past LIBERO's threshold
    drive_past_close: float = 0.01 # and this far past "close"/"off"
    drive_speed: float = 0.12
    # hook (a sliding drawer opened with open jaws from above), measured on LIBERO's 50 human demos
    # of libero_goal 3: the tool 42 mm on the opening side of the bar and 17 mm above its centre,
    # pointing down, the jaws along the opening direction and wide open; never closed
    hook_lead: float = 0.042
    hook_above: float = 0.017      # (lower or closer, 9-12 mm and 27-35 mm, never engaged: 3 of 30)
    hook_ahead: float = 0.05        # m the drag's aim runs ahead of the hook point (speed-limited)
    hook_corridor: float = 0.015    # m off the hook point, across the pull, before re-entering
    hook_band: float = 0.004        # m of the hook height the tool must be at before it drags: at
    #                                 17 mm off it the finger swept over the bar
    # closing a drawer by pushing its bar (libero_10 3: LIBERO's humans close the bottom drawer with
    # no grasp in 50 of 50 demos; the side pinch the teacher reached for stalled against the cabinet)
    push_close_lead: float = 0.03   # m the tool stands in front of the bar's centre, on the opening side
    push_close_ahead: float = 0.05  # m the press's aim runs ahead of the push point (speed-limited)
    push_close_corridor: float = 0.02   # m off the press line before re-approaching
    # the front hook, for a drawer whose bar is neither open from above nor graspable (libero_90 6:
    # the bottom drawer, the middle one's bar over it). LIBERO's humans open it with no grasp in 50 of 50:
    # jaws open 80 mm along the bar, fingers 31 deg below level into the cabinet, the tool at the bar
    # 26 mm above its centre (medians of 20 demos), dragged along the opening
    front_hook_pitch: float = 0.54      # rad below level
    front_hook_above: float = 0.018     # m above the bar's centre: 18 opened libero_90 6 in 4 of 4; 0, 6, 12 and the humans' 26 in 0 of 4
    front_hook_stage: float = 0.10      # m in front of the hook point the tool comes down to first
    front_hook_engage: float = 0.003    # m in front of the hook point at most before the drag starts
    push_close_pitch: float = 0.785     # rad the fingers point from straight down toward the push: down
    #                                     (0), the forearm met the cabinet top 4 cm short of closed; level
    #                                     (pi/2), the push pose was out of reach in 44 of 50
    drive_lead_rot: float = 0.06   # rad a hinge drive may lead the handle (< at_rot): leading by the
    #                                whole remaining turn spun the tool at w_max while the knob, damped,
    #                                followed at a fifth of that -- the jaws were pried open and let go
    drive_lead_slide: float = 0.03 # m a slide drive may lead it (k_lin x this still saturates drive_speed)
    drive_speed_min: float = 0.02
    # leaving an overhang (BRN-lift-leaves-overhang)
    exit_margin: float = 0.01       # m added to the object's plan half-extent for its column
    exit_step: float = 0.02         # m between the distances an exit is looked for at
    exit_max: float = 0.14          # m: the farthest exit examined
    exit_directions: int = 16


class Skills:
    """Feedback laws over a TaskEnv + Scene."""

    def __init__(self, env, config: SkillConfig | None = None):
        self.env = env
        self.scene = env.scene
        self.k = config or SkillConfig()
        self.planner = GraspPlanner(env, self.k)
        self.reach = Reach(env, self.k)
        self.clearing = Clearing(self.scene, self.planner, self.reach)
        self.phase = ""
        self.grasp_log: dict = {}      # what was decided and why, for tools/skill_eval.py
        self._episode = None
        self._grasp_cache: dict[str, tuple] = {}
        self._handle_cache: dict[str, tuple[np.ndarray, float, np.ndarray, float]] = {}
        self._drop_cache: dict = {}
        self._carry_cache: dict = {}
        self._spots: dict[str, tuple] = {}         # synthetic regions: where a crowded object goes
        self._clearing: dict[tuple, str | None] = {}

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
                          self._drop_cache, self._carry_cache, self._spots, self._clearing):
                cache.clear()

    def grasp_for(self, obj: str, via: np.ndarray | None = None,
                  place: tuple[str, bool] | None = None) -> tuple[np.ndarray, np.ndarray, float]:
        """The grasp for an object: shape proposes (grasp_planner), reachability disposes
        (reach), re-chosen when the object has moved. `place` (region, inside): the arm is also
        screened where it will let go -- the drop point plus the grasp's offset in the hand."""
        self.new_episode_check()
        box = self.scene.object_box(obj)
        centre = box.world_centre
        cached = self._grasp_cache.get(obj)
        if cached is not None and np.linalg.norm(cached[3] - centre) < self.k.regrasp_move:
            return cached[0], cached[1], cached[2]
        bid = self.scene.body_id(obj)
        tiers = self.planner.tiers(obj, box, self._held_by(obj))
        extra = self._release_probe(obj, *place) if place is not None else None
        chosen, used = None, ""
        for name, tier in tiers:
            chosen = self.reach.choose(tier, via, allow=bid, extra=extra)
            if chosen is not None:
                used = name
                break
        if chosen is None:
            # Every tier came up empty. Refusing is what DEF-witness-or-refusal asks, and it is off
            # until the screen is calibrated, because the screen is conservative and a forced grasp
            # it rejected solves libero_goal 6 fifty times out of fifty. Off, this is the behaviour
            # the teacher had before refusal existed: the best try over every tier at once.
            if self.reach.refuse_when_empty:
                raise Refusal("graspable", obj,
                              f"0 of {sum(len(t) for _, t in tiers)} candidates over "
                              f"{len(tiers)} tiers passed the reach screen")
            chosen = self.reach.choose(sum((t for _, t in tiers), []), via, allow=bid, extra=extra, force=True)
            used = "forced"
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
            forced=choice.get("forced", False), rejected=choice.get("rejected", {}),
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

    def holding(self, geom: int, width: float) -> bool:
        """Both finger groups touch the handle geom and the jaws have closed down to its width.

        Not "the jaws have stopped moving": pulling a handle moves the jaws, so that test
        went false on the first step of every pull, the teacher re-centred on the handle,
        and squeeze and drive alternated step by step. And not "both fingers touch the
        body": replayed, the jaws had closed to 1 mm in front of the drawer's bar, fingertips
        pressed on its face, and that passed as holding. The aperture has to be stopped by
        the handle itself -- and with an upper bound at the approach opening (width + 12 mm)
        and contact anywhere on the drawer's body, open jaws brushing its face also passed.
        """
        ap = self.env.snapshot()["aperture"]
        return (len(contacts.finger_sides_on_geom(self.scene.m, self.scene.d, geom)) == 2
                and self.k.handle_min_frac * width <= ap <= width + self.k.handle_open_slack)

    # -- regions -------------------------------------------------------------------------------
    def region_pose(self, region: str):
        """(R, p, half) of a region site, of an object used as one (On(x, plate)), or of a
        spot chosen to move a crowded object to."""
        if region in self._spots:
            return self._spots[region]
        try:
            return self.scene.region(region)
        except ValueError:                             # not a site: an object
            box = self.scene.object_box(region)
            return box.R, box.world_centre, box.half

    def clearing_spot(self, obj: str, container: str, R_tool: np.ndarray) -> str | None:
        """The region to move `obj` to before `container` is opened, if opening it would
        crowd the object (clearing.py); decided once per episode, from where it rests."""
        self.new_episode_check()
        key = (obj, container)
        if key not in self._clearing:
            a = self.scene.articulation(container)
            dq = self._drive_dq(a, "open")
            # clear_of is judged where the plan leaves the container, open by open_margin: at the drive
            # target, 18 mm further, the rim pinch was lost on all 8 of 8 layouts probed, against 4 of 8
            dq_plan = float(a["thresholds"]["open"]) + a["sign"] * self.k.open_margin - float(a["qpos"])
            # clear_of(o, j) (DEF-skill-contract): o stays graspable with j at its open target. The
            # footprint test alone moved the bowl in every libero_goal 3 episode (~120 steps); LIBERO's
            # humans open the drawer and pinch the bowl beside it, 20 of 20
            crowded = self.clearing.crowds(obj, a, dq) and not self._graspable_open(obj, a, dq_plan, container)
            xy = self.clearing.spot(obj, a, dq, R_tool) if crowded else None
            name = None
            if xy is not None:
                box = self.scene.object_box(obj)
                ext = np.abs(box.R) @ box.half
                surface = float(box.world_centre[2] - ext[2])
                name = f"clear:{obj}"
                self._spots[name] = (np.eye(3), np.array([xy[0], xy[1], surface]),
                                     np.array([ext[0], ext[1], 0.0]))
            self._clearing[key] = name
            self.grasp_log[f"clearing {obj}"] = dict(      # grasp_log entries are dicts: the report reads them
                decision="clear of it" if xy is None and not crowded
                else "crowded, no spot" if xy is None else f"move to {np.round(xy, 3).tolist()}")
        return self._clearing[key]

    def _graspable_open(self, obj: str, a: dict, dq: float, container: str) -> bool:
        """obj keeps the grasp tier it has now with the container's joint written to its open target;
        the simulator state is restored whatever the answer. Any tier at all was not enough: with the
        top drawer open, libero_goal 3's bowl fell from rim pinches to side grasps in 25 of 50
        episodes, and none of those succeeded (23 of 25 did on a rim pinch)."""
        via = self.via_for(container)
        bid = self.scene.body_id(obj)

        def first_tier():
            for name, t in self.planner.tiers(obj, self.scene.object_box(obj), self._held_by(obj)):
                if self.reach.choose(t, via, allow=bid) is not None:
                    return name
            return None

        now = first_tier()
        sim = self.env.env.sim
        qpos, qvel = sim.data.qpos.copy(), sim.data.qvel.copy()
        try:
            sim.data.qpos[a["qposadr"]] = float(qpos[a["qposadr"]]) + dq
            sim.forward()
            opened = first_tier()
        finally:
            sim.data.qpos[:] = qpos
            sim.data.qvel[:] = qvel
            sim.forward()
        return opened is not None and opened == now

    def at_spot(self, obj: str, spot: str) -> bool:
        return self.at_place(*self.place_target(obj, spot, inside=False))

    def via_for(self, region: str) -> np.ndarray:
        """Where the tool must be to deliver into this region -- the point the grasp has to
        stay reachable at, not just the grasp itself."""
        _, p_reg, half = self.region_pose(region)
        return np.array([p_reg[0], p_reg[1], p_reg[2] + float(half[2]) + self.k.lift])

    # -- pick ------------------------------------------------------------------------------------
    def pick(self, obj: str, s: dict, via: np.ndarray | None = None,
             place: tuple[str, bool] | None = None) -> np.ndarray:
        """Up, over and down to the pre-grasp, advance along the approach, squeeze, lift."""
        k = self.k
        R, p = s["R_tool"], s["p_tool"]
        R_g, p_g, width = self.grasp_for(obj, via, place)
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

    # -- push ------------------------------------------------------------------------------------
    def push(self, obj: str, region: str, s: dict) -> np.ndarray:
        """Move a dish without grasping it (BRN-push-cages-a-dish): jaws open and aligned with the
        motion, fingertips inside its rim ahead of its centre, dragged toward the region.

        As LIBERO's human demos of libero_goal 5 do it, all 50: the leading finger bears on the
        inside of the leading rim, so the rim cages the dish and re-aiming from its pose every
        step is enough. Picked instead, the 137 mm plate scored 3 of 50 on forced grasps.
        """
        k = self.k
        R, p = s["R_tool"], s["p_tool"]
        box = self.scene.object_box(obj)
        ext = np.abs(box.R) @ box.half
        c = box.world_centre
        bottom, top = float(c[2] - ext[2]), float(c[2] + ext[2])
        _R_reg, p_reg, _half = self.region_pose(region)
        d = (p_reg - c)[:2]
        if np.linalg.norm(d) < 1e-6:
            self.phase = "hold"
            return self.action(np.zeros(6), k.push_open)
        u = np.r_[d / np.linalg.norm(d), 0.0]
        reach = self.planner.gripper().reach
        radius = float(max(ext[0], ext[1]))
        z_push = self._dish_floor(obj, c, radius) + k.push_floor + reach
        held = np.r_[c[:2] + u[:2] * k.push_lead * radius, z_push]
        # the jaws along the motion, one finger leading: of the two frames that do it, the one the
        # wrist is nearer to now -- a function of the state, and no half turn of the wrist
        frames = sorted((top_down(u), top_down(-u)), key=lambda Rc: rot_angle(R.T @ Rc))
        R_push = frames[0]
        off = (p - held)[:2]
        along = float(off @ u[:2])
        lateral = float(np.linalg.norm(off - u[:2] * along))
        inside = p[2] - reach < top and lateral < k.push_corridor and along < k.push_slip
        if inside and rot_angle(R.T @ R_push) < k.at_rot:
            self.phase = "push"
            aim = held + u * k.push_ahead
            return self.action(self.twist_to(R, p, R_push, aim, v_max=k.push_speed), k.push_open)
        # enter -- only through a push the arm can hold: the entry, halfway and the end are
        # reachable poses (DEF-reachable-pose), the three stops a carry is screened at
        end = np.r_[p_reg[:2] + u[:2] * k.push_lead * radius, z_push]
        stops = [held, (held + end) / 2, end]
        passing = [Rc for Rc in frames if self.reach.all_reachable([pose(Rc, x) for x in stops])]
        if passing:
            R_push = passing[0]
        elif self.reach.refuse_when_empty:
            raise Refusal("reachable", obj, "no push entry whose entry, halfway and end poses "
                          "are reachable, for either jaw direction")
        leg = self.path_to(R, p, R_push, np.r_[held[:2], z_push], top + k.lift_dz)
        if leg is not None:
            self.phase, R_t, target = leg
            return self.action(self.twist_to(R, p, R_t, target), k.push_open)
        self.phase = "enter"
        return self.action(self.twist_to(R, p, R_push, held), k.push_open)

    def _dish_floor(self, obj: str, c: np.ndarray, radius: float) -> float:
        """Height of a dish's floor: the top of its collision geoms near its axis (the plate's
        floor disc tops out 8.6 mm above its bottom, its rim rises to 19)."""
        m, d = self.scene.m, self.scene.d
        bid = self.scene.body_id(obj)
        tops = []
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) != bid or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            cc, Rg, hl = geom_world_box(m, d, g, self.scene.base)
            if np.linalg.norm((cc - c)[:2]) < 0.2 * radius:
                tops.append(float(cc[2] + (np.abs(Rg) @ hl)[2]))
        box = self.scene.object_box(obj)
        return max(tops) if tops else float(c[2] - (np.abs(box.R) @ box.half)[2])

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
            return self.pick(obj, s, via=self.via_for(region), place=(region, inside))
        delta = target_q - q
        over = float(np.linalg.norm(delta[:2]))
        carry_z = self._carry_height(obj, region, q, target_q, R, p)
        if over > k.over_xy and q[2] < carry_z - k.plane_band:
            exit_q = self._overhang_exit(obj, q)
            if exit_q is not None:                     # BRN-lift-leaves-overhang
                self.phase = "slide"
                goal = p + np.array([exit_q[0] - q[0], exit_q[1] - q[1], 0.0])
                return self.action(self.twist_to(R, p, R, goal, v_max=k.carry_speed), 0.0)
            self.phase = "lift"
            slow = self._risen(obj) < k.slow_lift_dz
            return self.action(self.twist_to(R, p, R, p + Z * (carry_z - q[2]),
                                             v_max=k.slow_lift_speed if slow else k.carry_speed), 0.0)
        R_fit = self._fit_yaw(obj, region, R)
        if over > k.over_xy:
            self.phase = "carry"
            goal = p + np.array([delta[0], delta[1], max(0.0, carry_z - q[2])])
            return self.action(self.twist_to(R, p, R_fit, goal, v_max=k.carry_speed), 0.0)
        R_reg, p_reg, half = self.region_pose(region)
        stop_z = self._on_support(obj, self.scene.object_box(obj), q, target_q[:2], R_reg, p_reg, half,
                                  float(target_q[2]))
        # an object held by its handle hangs tilted and is seated by being pressed on to its support, as
        # before: stopped at the support, the moka pot went 14 -> 8 of 20 (libero_10 2)
        resting = (self._held_by(obj) != "handle"
                   and not contacts.only_gripper(self.scene.m, self.scene.d, self.scene.body_id(obj)))
        if delta[2] < -k.lower_done and not (resting and q[2] - stop_z <= k.lower_done):
            self.phase = "lower"
            # on to the drop point, not only down: lowered straight, the bowl drifted 13 mm toward
            # the cabinet on its way into libero_10 3's drawer
            return self.action(self.twist_to(R, p, R_fit, p + delta, v_max=k.lower_speed,
                                             v_min=k.lower_speed_min), 0.0)
        self.phase = "release"
        return self.action(np.zeros(6), A_OPEN)

    def _fit_yaw(self, obj: str, region: str, R: np.ndarray) -> np.ndarray:
        """The tool turned about the vertical so the held object's footprint fits the region's: unturned
        if it fits as it is or cannot fit either way, else by the smaller turn that lays the object's
        longest horizontal axis along the region's. Re-derived from the object's pose every step.
        Carried as grasped, libero_90 73's book (110 x 29 mm) came down crosswise into a 56 x 124 mm
        caddy compartment, caught its wall and was lost (0 of 20)."""
        R_reg, _p_reg, half = self.region_pose(region)
        th = self._fit_turn(self.scene.object_box(obj), R_reg, half)
        return axis_rot(Z, th) @ R if th else R

    def _fit_turn(self, box, R_reg, half) -> float:
        """The turn about the vertical _fit_yaw gives the object for a region: 0 when it fits as it is
        or cannot fit either way."""
        axes = [(2.0 * float(box.half[j]), box.R[:, j]) for j in range(3) if abs(float(box.R[2, j])) < VERTICAL_COS]
        if not axes:
            return 0.0
        # the region's horizontal axes, whichever of its box axes they are: the caddy's compartment
        # regions have their local y vertical
        flat = [i for i in range(3) if abs(float(R_reg[2, i])) < VERTICAL_COS]
        if len(flat) != 2:
            return 0.0
        rx, ry = R_reg[:, flat[0]], R_reg[:, flat[1]]
        room = (2.0 * abs(float(half[flat[0]])), 2.0 * abs(float(half[flat[1]])))

        def fits(yaw, tol):
            Rz = axis_rot(Z, yaw)
            ex = sum(length * abs(float((Rz @ v) @ rx)) for length, v in axes)
            ey = sum(length * abs(float((Rz @ v) @ ry)) for length, v in axes)
            return ex <= room[0] + tol and ey <= room[1] + tol
        # left as it is only if it fits with no tolerance: libero_90 77's book came down turned 15 deg,
        # 56.5 mm across a 55.5 mm compartment -- inside fit_tol -- and its corner landed on the wall
        if fits(0.0, 0.0):
            return 0.0
        _l, a = max(axes, key=lambda e: e[0])
        r = rx if room[0] >= room[1] else ry
        th = float(np.arctan2(a[0] * r[1] - a[1] * r[0], a[0] * r[0] + a[1] * r[1]))
        th = min((th, th - np.pi, th + np.pi), key=lambda x: (abs(x), -x))   # a tie goes to the positive turn
        return th if fits(th, self.k.fit_tol) else 0.0

    def _overhang_exit(self, obj: str, q: np.ndarray) -> np.ndarray | None:
        """Where the held object's origin must first move, level, before it rises: None when the
        column over its footprint is clear (BRN-lift-leaves-overhang), or when no examined offset
        is both clear overhead and reachable by a level move that meets nothing at the object's own
        height (inside a drawer every such move meets a wall, and the object rises in place)."""
        k = self.k
        box = self.scene.object_box(obj)
        ext = np.abs(box.R) @ box.half
        c = box.world_centre
        radius = float(ext[:2].max()) + k.exit_margin
        top, bottom = float(c[2] + ext[2]), float(c[2] - ext[2])
        mine = {self.scene.body_id(obj)}
        if self.reach.footprint_clear(c[:2], radius, top, mine):
            return None
        angles = np.linspace(0.0, 2 * np.pi, k.exit_directions, endpoint=False)
        for dist in np.arange(k.exit_step, k.exit_max + 1e-9, k.exit_step):
            for th in angles:
                xy = c[:2] + dist * np.array([np.cos(th), np.sin(th)])
                if (self.reach.footprint_clear(xy, radius, top, mine)
                        and self.reach.band_clear(c[:2], xy, radius, bottom, top, mine)):
                    return q[:2] + (xy - c[:2])
        return None

    def _held_by(self, obj: str) -> str | None:
        from .task_spec import held_by
        spec = getattr(self.env, "task_spec", None)
        return held_by((getattr(spec, "objects", {}) or {}).get(obj))

    def _release_probe(self, obj: str, region: str, inside: bool):
        """For a grasp candidate, the tool pose where it will let go: the drop point, computed now
        and not cached (the container may still be shut), plus the candidate's offset in the hand.
        Screened like the via: libero_10 3's bowl went into the bottom drawer with the hand's long
        axis toward the cabinet, the hand met the middle drawer's front, the lower stopped short,
        and the bowl dropped in tilted and jammed the drawer's close (0 of 50)."""
        R_reg, p_reg, half = self.region_pose(region)
        box = self.scene.object_box(obj)
        q = self.scene.body_pose(obj)[1]
        half_h = float(np.abs(box.R @ np.diag(box.half)).sum(1)[2])
        centre_off = float((box.world_centre - q)[2])
        xy, _d = self._open_point(obj, R_reg, p_reg, half, box)
        if inside:
            target = np.array([xy[0], xy[1], p_reg[2] + max(0.0, float(half[2]) - half_h) + self.k.place_clearance])
        else:
            target = p_reg + R_reg @ np.array([0.0, 0.0, float(half[2])])
            target[:2] = xy
            target[2] += half_h - centre_off + self.k.place_clearance
        return lambda cand: [pose(cand[0], target + (cand[1] - q))]

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

    def _on_support(self, obj: str, box, q: np.ndarray, xy: np.ndarray, R_reg, p_reg, half,
                    region_z: float) -> float:
        """The origin height at which the lower phase ends: the region-derived one, raised to where the
        object's bottom is place_clearance above the surface under its footprint's centre at the drop
        point (BRN-place-onto-physical-support). Read only there, and only once the object touches something
        besides the fingers -- not for the carry height, the grasp screen or the delivered test, which keep
        the region-derived height. A moka pot swings level for a step mid-air; let go there, it fell 15 mm
        and tipped (libero_10 2, 14 -> 6 of 20). LIBERO's regions are sites, not surfaces: the basket's
        contain region reaches 3 mm below the table while its floor stands 16 mm above it, so the
        target lay 7 mm below the floor, the lower phase never ended, and the jaws pressed the bottle
        into the floor to the horizon (libero_object 2, 0 of 20 settled with 20 of 20 success). The
        centre, not the highest point under the footprint: a corner over a caddy wall raised a book's
        release 7 cm and it fell out; what lies under the corners is the drop point's test (_open_point)."""
        R = box.R
        up = int(np.argmax(np.abs(R[2])))
        # only for an object carried level: its lowest corner within the lower phase's tolerance of its
        # bottom face's centre. A moka pot hangs 21 deg from its handle, its lowest corner 27 mm under
        # that centre; lowered to the region's height it is pressed on to the burner, which rights it
        # (libero_10 2, 8 of 10), and raised to the burner's it is let go tilted and tips (1 of 10)
        if sum(abs(float(R[2, i])) * float(box.half[i]) for i in range(3) if i != up) > self.k.lower_done:
            return region_z
        bottom = box.world_centre - R[:, up] * float(box.half[up]) * float(np.sign(R[2, up]) or 1.0)
        centre = np.asarray(xy, float)[:2] + (bottom - q)[:2]
        top = float(p_reg[2] + (np.abs(R_reg) @ np.abs(np.asarray(half, float)))[2])
        z_from = top + 2.0 * float((np.abs(R) @ box.half)[2])
        _g, dist = self.planner.ray_scene(np.array([centre[0], centre[1], z_from]), -Z, self.scene.body_id(obj))
        if dist < 0:
            return region_z
        return max(region_z, z_from - dist + float((q - bottom)[2]) + self.k.place_clearance)

    def _drop_xy(self, obj: str, region: str, R_reg, p_reg, half, box) -> np.ndarray:
        """The point of the region nearest its centre that is open from above.

        A drawer opened 14 cm keeps the back of its interior under the cabinet top, and its
        region's centre can be under there too. A vertical ray says which parts are open.
        """
        key = (obj, region)
        if key in self._drop_cache:
            return self._drop_cache[key]
        best, _best_d = self._open_point(obj, R_reg, p_reg, half, box)
        self._drop_cache[key] = best
        return best

    def admits(self, obj: str, region: str) -> bool:
        """admits(C, o) (DEF-skill-contract): some point of the region, the object's footprint
        inside it, is open from above -- read from the state now, never cached."""
        R_reg, p_reg, half = self.region_pose(region)
        return bool(np.isfinite(self._open_point(obj, R_reg, p_reg, half, self.scene.object_box(obj))[1]))

    def _open_point(self, obj: str, R_reg, p_reg, half, box) -> tuple[np.ndarray, float]:
        """(the point of the region nearest its centre open from above, its distance from the
        centre; inf when none is).

        The earlier search stands wherever its answer holds: its point, if the object's own
        footprint there -- as it will be carried, centred on its box -- overhangs nothing. Only
        where it does, or the earlier search found nothing, and only over a level region, is the
        footprint searched as carried: over libero_10 3's drawer the earlier point was 22 mm clear of
        the cabinet's face and the carried search's centre was not (10 -> 4 of 10), and over the
        tilted wine rack the earlier search found nothing and used the centre, where the bottle sits
        in its cradle, while the carried search found a spot 33 mm off it (50 -> 0 of 50)."""
        best, best_d = self._open_point_grid(obj, R_reg, p_reg, half, box)
        if float(np.max(np.abs(R_reg[2]))) < 1.0 - 1e-6:      # a tilted region: no level footprint to test
            return best, best_d
        # the earlier search's footprint is the object's, unturned and centred on its origin; where the
        # carry will not turn it and its box is centred on its origin (a bowl), that is the true one
        off = box.world_centre - self.scene.body_pose(obj)[1]
        if self._fit_turn(box, R_reg, np.abs(np.asarray(half, float))) == 0.0 and float(np.linalg.norm(off[:2])) <= self.k.fit_tol:
            return best, best_d
        if np.isfinite(best_d) and self._footprint_clear(obj, R_reg, p_reg, half, box, best):
            return best, best_d
        carried, carried_d = self._open_point_carried(obj, R_reg, p_reg, half, box)
        return (carried, carried_d) if np.isfinite(carried_d) else (best, best_d)

    def _footprint_clear(self, obj: str, R_reg, p_reg, half, box, xy: np.ndarray) -> bool:
        """The object's footprint with its origin at xy -- turned as the carry will turn it, centred
        on its box, its centre and boundary sampled -- is open from above (the earlier search's top)."""
        half = np.abs(np.asarray(half, float))
        flat = [i for i in range(3) if abs(float(R_reg[2, i])) < VERTICAL_COS]
        flat = flat if len(flat) == 2 else [0, 1]
        ax = [R_reg[:, i] for i in flat]
        Rz = axis_rot(Z, self._fit_turn(box, R_reg, half))
        B = Rz @ box.R @ np.diag(box.half)
        ohw = [float(np.abs(a @ B).sum()) for a in ax]
        off = Rz @ (box.world_centre - self.scene.body_pose(obj)[1])
        centre = np.array([xy[0], xy[1], p_reg[2]]) + ax[0] * float(ax[0] @ off) + ax[1] * float(ax[1] @ off)
        top = float(p_reg[2] + abs(float(half[2])))
        exclude = self.scene.body_id(obj)
        return all(self._open_above(centre + ax[0] * fx + ax[1] * fy, top, exclude)
                   for fx in (-ohw[0], 0.0, ohw[0]) for fy in (-ohw[1], 0.0, ohw[1]))

    def _open_point_grid(self, obj: str, R_reg, p_reg, half, box) -> tuple[np.ndarray, float]:
        """The drop-point search as it stood before BRN-drop-point-footprint-as-carried: origins about
        the region's centre, the footprint centred on the origin, as grasped, along the world's axes."""
        k = self.k
        top = float(p_reg[2] + abs(float(half[2])))
        ohw = np.abs(box.R @ np.diag(box.half)).sum(1)[:2]      # object half-footprint
        span = np.maximum(np.abs(np.asarray(half[:2], float)) - ohw, 0.0)
        best, best_d = np.asarray(p_reg[:2], float), np.inf
        exclude = self.scene.body_id(obj)
        # the whole footprint, with a margin, not only its centre: centred where the bottom drawer's
        # region runs under the cabinet, the bowl's rim came down 4 mm from the cabinet's face, caught
        # its top, and was let go 9 cm up (libero_10 3)
        foot = [np.array([sx * (ohw[0] + k.drop_margin), sy * (ohw[1] + k.drop_margin), 0.0])
                for sx in (-1.0, 0.0, 1.0) for sy in (-1.0, 0.0, 1.0)]
        for fx in np.linspace(-1, 1, k.drop_grid):
            for fy in np.linspace(-1, 1, k.drop_grid):
                local = np.array([fx * span[0], fy * span[1], 0.0])
                dc = float(np.linalg.norm(local[:2]))
                if dc >= best_d:
                    continue
                pt = p_reg + R_reg @ local
                if all(self._open_above(pt + off, top, exclude) for off in foot):
                    best, best_d = pt[:2].copy(), dc
        return best, best_d

    def _open_point_carried(self, obj: str, R_reg, p_reg, half, box) -> tuple[np.ndarray, float]:
        """The search over the footprint as it will be carried (BRN-drop-point-footprint-as-carried)."""
        k = self.k
        half = np.abs(np.asarray(half, float))
        # the region's horizontal axes, whichever of its box axes they are (the caddy's compartments
        # have their local y vertical), and the footprint the object will have over it: turned as the
        # carry will turn it (_fit_turn), and centred where its box is, not at its origin -- a book's
        # box centre is 12 mm from its origin, and centred on the origin in libero_90 77's compartment
        # its end hung 5-7 mm over the wall
        flat = [i for i in range(3) if abs(float(R_reg[2, i])) < VERTICAL_COS]
        flat = flat if len(flat) == 2 else [0, 1]
        ax = [R_reg[:, i] for i in flat]
        top = float(p_reg[2] + (np.abs(R_reg) @ half)[2])
        Rz = axis_rot(Z, self._fit_turn(box, R_reg, half))
        B = Rz @ box.R @ np.diag(box.half)
        ohw = np.array([float(np.abs(a @ B).sum()) for a in ax])       # half-footprint along each axis
        off = Rz @ (box.world_centre - self.scene.body_pose(obj)[1])
        c_loc = np.array([float(a @ off) for a in ax])                  # box centre from origin
        span = np.maximum(half[flat] - ohw, 0.0)
        best, best_d = np.asarray(p_reg[:2], float), np.inf
        exclude = self.scene.body_id(obj)
        # the whole footprint, with a margin, not only its centre: centred where the bottom drawer's
        # region runs under the cabinet, the bowl's rim came down 4 mm from the cabinet's face, caught
        # its top, and was let go 9 cm up (libero_10 3). The footprint's own edge is sampled too: a
        # 7 mm caddy wall under a book's end lay between the centre and the margin and was not seen
        # origins about the region's centre first -- LIBERO scores the origin -- then origins that put
        # the footprint about the centre; with the margin, then, where no spot leaves room for it (a
        # book in a compartment 7 mm longer than itself on each side), without it
        for margin in (k.drop_margin, 0.0):
            foot = [ax[0] * fx + ax[1] * fy
                    for fx in sorted({-(ohw[0] + margin), -ohw[0], 0.0, ohw[0], ohw[0] + margin})
                    for fy in sorted({-(ohw[1] + margin), -ohw[1], 0.0, ohw[1], ohw[1] + margin})]
            for shift in (np.zeros(2), -c_loc):
                for fx in np.linspace(-1, 1, k.drop_grid):
                    for fy in np.linspace(-1, 1, k.drop_grid):
                        local = np.array([fx * span[0], fy * span[1]]) + shift
                        dc = float(np.linalg.norm(local))
                        if dc >= best_d:
                            continue
                        pt = p_reg + ax[0] * local[0] + ax[1] * local[1]
                        centre = pt + ax[0] * c_loc[0] + ax[1] * c_loc[1]
                        if all(self._open_above(centre + o, top, exclude) for o in foot):
                            best, best_d = pt[:2].copy(), dc
                if np.isfinite(best_d):
                    return best, best_d
        return best, best_d

    def _open_above(self, pt: np.ndarray, top: float, exclude: int) -> bool:
        """A ray down from above the region's top meets nothing higher than it (the robot aside)."""
        start = np.array([pt[0], pt[1], top + self.k.drop_ray_height])
        _hit, dist = self.planner.ray_scene(start, np.array([0.0, 0.0, -1.0]), exclude)
        return (start[2] - dist if dist >= 0 else -np.inf) <= top + self.k.drop_clear

    def _carry_height(self, obj: str, region: str, q, target_q, R, p) -> float:
        """Per grasp, not per object: a regrasp holds the object differently, and a height
        checked for the old grasp's rotation and offset says nothing about the new one."""
        g = self._grasp_cache.get(obj)
        key = (obj, region, None if g is None else tuple(np.round(g[1], 4)))
        if key not in self._carry_cache:
            self._carry_cache[key] = self.reach.carry_height(
                self.scene.object_box(obj), q, target_q, R, p, self.scene.body_id(obj))
            self.grasp_log.setdefault(obj, {})["carry"] = self.reach.last_carry
        return self._carry_cache[key]

    def at_place(self, q: np.ndarray, target_q: np.ndarray) -> bool:
        """The object is where it was carried to, whether or not it is still in the jaws.

        Without this the release chatters: the jaws open, `held` goes false, the goal is
        not scored yet, and the teacher takes the object back.
        """
        d = target_q - q
        return bool(np.linalg.norm(d[:2]) < self.k.at_place_xy and -self.k.at_place_above < d[2] < self.k.at_place_dz)

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
        if mode == "open" and a["jnt_type"] == 2 and self._hookable(a):
            return self._hook(a, s)
        if mode == "close" and a["jnt_type"] == 2:
            return self._push_close(a, s)
        R_h, w, app = self._handle_frame(region, a, mode)
        if mode == "open" and a["jnt_type"] == 2 and self.grasp_log.get(region, {}).get("forced"):
            return self._front_hook(a, s)          # no feasible grasp of the bar: hook it from the front
        p_h = self.scene.d.geom_xpos[a["handle_geom"]] - self.scene.base
        if self.holding(a["handle_geom"], w):
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

        And no further than that (BRN-leave-handle-along-approach): backing out to 100-150 mm
        stretched the arm until sigma_min fell 0.108 -> 0.063 and the next motion crossed
        0.003; the servo re-anchored and the arm swept the bowl set down beside it.
        """
        k, p = self.k, s["p_tool"]
        for region in list(self._handle_cache):
            a = self.scene.articulation(region)
            R_h, w, app = self._handle_frame(region, a, "open")
            p_h = self.scene.d.geom_xpos[a["handle_geom"]] - self.scene.base
            off = p - p_h
            ahead = float(off @ app)                       # 0 at the handle, -approach at the pre-grasp
            clear = self.handle_clearance(a["handle_geom"], app)
            if np.linalg.norm(off - app * ahead) < k.handle_leave_lateral and abs(ahead) < clear:
                self.phase = "leave"
                return self.action(self.twist_to(s["R_tool"], p, s["R_tool"],
                                                 p_h - app * (clear + k.handle_leave_past)),
                                   min(k.max_grip, w + k.grip_margin))
        return None

    def handle_clearance(self, geom: int, app: np.ndarray) -> float:
        """How far the tool point must be back from the handle's centre, along the approach,
        before no part of the handle lies between the jaws: the handle's own half-depth along
        that axis plus the finger meshes' reach past the tool point (the gripper as scanned).
        A predicate on where the tool is, not a distance travelled from wherever the grip was."""
        m, d = self.scene.m, self.scene.d
        _c, Rg, hl = geom_world_box(m, d, geom, self.scene.base)
        return float(np.abs(app @ Rg) @ hl) + self.planner.gripper().reach

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
            R, p, app = cand[0], cand[1], cand[3]
            stations = [pose(*self._joint_motion(a, f * dq, R, p)) for f in (0.5, 1.0)]
            # and so does the retreat that ends the drive (BRN-leave-handle-along-approach): the
            # tool backs out along the approach as the motion carried it
            R1, p1 = self._joint_motion(a, dq, R, p)
            app1, _ = self._joint_motion(a, dq, app[:, None], np.zeros(3))
            return stations + [pose(R1, p1 - app1[:, 0] * self.handle_clearance(a["handle_geom"], app1[:, 0]))]

        # Reachable along the drive is a filter, not a ranking, and there is no weaker filter to
        # fall back to: a grasp checked only at the start of the motion is a claim about a
        # different configuration of the world, and a logged caveat is not a witness.
        chosen = self.reach.choose(cands, None, allow=a["body"], extra=along_the_motion)
        drive_checked = chosen is not None
        if chosen is None:
            if self.reach.refuse_when_empty:
                raise Refusal("accessible", region,
                              f"0 of {len(cands)} handle grasps reachable along the joint's motion")
            # the pre-refusal fallback: ranked among grasps reachable at the start of the drive,
            # forcing only if none is reachable even there
            chosen = self.reach.choose(cands, None, allow=a["body"], force=True)
        R_h, _p, w, app = chosen
        self._handle_cache[region] = (R_h, w, app, float(a["qpos"]))
        choice = dict(self.reach.last_choice, drive_checked=drive_checked)
        self.grasp_log[region] = dict(
            tier="handle", offered={"handle": len(cands)}, feasible=choice.get("feasible"),
            drive_checked=choice["drive_checked"], forced=choice.get("forced", False),
            rejected=choice.get("rejected", {}), score=choice.get("score"),
            approach=[round(float(v), 2) for v in app], jaw=[round(float(v), 2) for v in R_h[:, 1]],
            width_mm=round(1000 * float(w), 1), handle_geom=self.scene.m.geom_id2name(a["handle_geom"]))

    def _hook_frame(self, a: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """(bar centre, opening direction in plan, the bar's long axis, the bar's top) now."""
        c, Rg, hl = geom_world_box(self.scene.m, self.scene.d, a["handle_geom"], self.scene.base)
        bar = Rg[:, int(np.argmax(hl))]
        u = a["axis"] * float(np.sign(self._drive_dq(a, "open")) or 1.0)
        u = u - bar * float(u @ bar)
        u = u - Z * float(u @ Z)
        return c, u / np.linalg.norm(u), bar, float(c[2] + (np.abs(Rg) @ hl)[2])

    def _hookable(self, a: dict) -> bool:
        """Nothing of the scene over the bar where it sits closed, the robot aside: the top drawer's
        bar is open from above; the middle drawer's has the top drawer's bar over it. Read where the
        bar is now, the middle drawer's came out from under the top one partway through the pinch
        drive, the mode flipped to the hook mid-drive, and libero_goal 0 took 99 steps for 74."""
        c, _u, _bar, top = self._hook_frame(a)
        closed = c - a["axis"] * float(a["qpos"])                 # a slide's bar at q = 0
        return self.reach.column_clear(closed, self.k.hook_corridor, top, {a["body"]})

    def _hook(self, a: dict, s: dict) -> np.ndarray:
        """Open a sliding drawer as LIBERO's human demos of libero_goal 3 all do: jaws wide open,
        pointing down, the tool on the opening side of the bar so the trailing finger stands over
        its rear edge, dragged along the opening direction at the drive speed, re-aimed from the
        bar's pose every step. Nothing is grasped, so there is nothing to back out of: the next
        skill's first move is up."""
        k = self.k
        R, p = s["R_tool"], s["p_tool"]
        c, u, _bar, top = self._hook_frame(a)
        R_hook = min((top_down(u), top_down(-u)), key=lambda Rc: rot_angle(R.T @ Rc))
        hook = c + u * k.hook_lead + Z * k.hook_above
        off = p - hook
        along = float(off @ u)
        lateral = float(np.linalg.norm((off - u * along)[:2]))
        down = abs(float(off[2])) < k.hook_band
        if down and lateral < k.hook_corridor and -k.hook_corridor < along < k.hook_ahead \
                and rot_angle(R.T @ R_hook) < k.at_rot:
            self.phase = "drag"
            return self.action(self.twist_to(R, p, R_hook, hook + u * k.hook_ahead, v_max=k.drive_speed,
                                             v_min=k.drive_speed_min), A_OPEN)
        leg = self.path_to(R, p, R_hook, hook, top + k.lift_dz)
        if leg is not None:
            self.phase, R_t, target = leg
            return self.action(self.twist_to(R, p, R_t, target), A_OPEN)
        self.phase = "hook"
        return self.action(self.twist_to(R, p, R_hook, hook), A_OPEN)

    def _front_hook(self, a: dict, s: dict) -> np.ndarray:
        """Open a sliding drawer as LIBERO's humans open libero_90 6's bottom drawer: jaws open along
        the bar, the fingers pitched front_hook_pitch below level into the cabinet so their tips go
        over and behind the bar, the tool front_hook_above over the bar's centre; come down in front
        of the drawer, move in level to the bar, then drag along the opening direction, re-aimed
        from the bar's pose every step. Nothing is grasped."""
        k = self.k
        R, p = s["R_tool"], s["p_tool"]
        c, u, bar, top = self._hook_frame(a)                  # u: the opening direction in plan

        def frame(jaw):
            z = -u * np.cos(k.front_hook_pitch) - Z * np.sin(k.front_hook_pitch)
            y = jaw - z * float(jaw @ z)
            y = y / np.linalg.norm(y)
            return np.column_stack([np.cross(y, z), y, z])
        R_f = min((frame(bar), frame(-bar)), key=lambda Rc: rot_angle(R.T @ Rc))
        hook = c + Z * k.front_hook_above
        off = p - hook
        along = float(off @ u)                               # > 0: in front of the hook point
        lateral = float(np.linalg.norm((off - u * along)[:2]))
        level = abs(float(off[2])) < k.hook_band
        aligned = rot_angle(R.T @ R_f) < k.at_rot
        if level and aligned and lateral < k.hook_corridor and -k.hook_corridor < along < k.front_hook_engage:
            self.phase = "drag"
            return self.action(self.twist_to(R, p, R_f, hook + u * k.hook_ahead, v_max=k.drive_speed,
                                             v_min=k.drive_speed_min), A_OPEN)
        stage = hook + u * k.front_hook_stage
        if abs(float(off[2])) < 2 * k.hook_band and lateral < k.hook_corridor and 0.0 < along < k.front_hook_stage + k.hook_corridor:
            self.phase = "in"                                # level, in front of the bar: move in
            return self.action(self.twist_to(R, p, R_f, hook), A_OPEN)
        leg = self.path_to(R, p, R_f, stage, top + k.lift_dz)
        if leg is not None:
            self.phase, R_t, target = leg
            return self.action(self.twist_to(R, p, R_t, target), A_OPEN)
        self.phase = "front"
        return self.action(self.twist_to(R, p, R_f, stage), A_OPEN)

    def _push_close(self, a: dict, s: dict) -> np.ndarray:
        """Close a sliding drawer by pressing its bar along the closing direction: jaws shut, the
        fingers pitched from straight down toward the push (push_close_pitch), the jaw axis along the bar so
        both fingers bear on it, the tool point at the bar's height and push_close_lead in front of
        it; re-aimed from the bar's pose every step. Pointing down instead, the forearm came down on
        the cabinet's top edge 4 cm short of closed (libero_10 3). Nothing is grasped; the next
        skill's first move is up."""
        k = self.k
        R, p = s["R_tool"], s["p_tool"]
        c, u, bar, top = self._hook_frame(a)                  # u: the opening direction in plan

        def fist(jaw):
            z = -u * np.sin(k.push_close_pitch) - Z * np.cos(k.push_close_pitch)
            y = jaw - z * float(jaw @ z)
            y = y / np.linalg.norm(y)
            return np.column_stack([np.cross(y, z), y, z])
        R_push = min((fist(bar), fist(-bar)), key=lambda Rc: rot_angle(R.T @ Rc))
        push = c + u * k.push_close_lead
        off = p - push
        along = float(off @ u)                               # > 0: still in front of the push point
        lateral = float(np.linalg.norm((off - u * along)[:2]))
        level = abs(float(off[2])) < k.hook_band
        if level and lateral < k.push_close_corridor and -k.push_close_ahead < along < k.push_close_corridor \
                and rot_angle(R.T @ R_push) < k.at_rot:
            self.phase = "press"
            return self.action(self.twist_to(R, p, R_push, push - u * k.push_close_ahead, v_max=k.drive_speed,
                                             v_min=k.drive_speed_min), 0.0)
        leg = self.path_to(R, p, R_push, push, top + k.lift_dz)
        if leg is not None:
            self.phase, R_t, target = leg
            return self.action(self.twist_to(R, p, R_t, target), 0.0)
        self.phase = "front"
        return self.action(self.twist_to(R, p, R_push, push), 0.0)

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
