"""Demonstration programs: one per task, built from shared, verified skills.

Each program is a FEEDBACK LAW -- its action is a function of the current
simulator state alone, so it can label any state a student reaches (DAgger).
Randomisation flows through the actual poses: every target is computed from
where the bowl, plate and arm are this episode.

Why per task. A single generic program trades tasks off against each other:
checking the whole descent path fixed the drawer (11 -> 14/20) while dropping the
ramekin (20 -> 16) and the stove (8 -> 3). Scenes differ in geometry that a
generic rule cannot see -- a cabinet top over the drawer, a stove at the edge of
the workspace. So the SKILLS are shared and the STRATEGY is per task, and a
change to one task's program cannot move another task's result.

Skills (shared, each one verified on demonstrations or in simulation):
  twist_to     proportional control on SE(3), speed-limited, through TwistServo
  pinch grasp  across the 3 mm bowl wall, tool z down, closing axis radial
  squeeze      until both jaws touch and the fingers stop closing (the settle
               aperture varies: 3.8-5.3 mm on demo carries, 8.7 mm on the ramekin)
  lift         first 20 mm slowly, so friction takes the load
  carry        keeping the grip transform measured at the current step
  lower, release

Strategy (per task, in ProgramConfig): grasp selection (endpoint or whole-path
checks), approach lengths, tolerances, carry clearance, and an optional exit
direction for bowls that must leave a confined space before rising.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np
import torch

from . import contacts
from .progress import is_held, rot_angle, rotvec
from .teacher_env import TARGET

HOME_Q = np.array([0.0, -0.161, 0.0, -2.4446, 0.0, 2.2268, np.pi / 4])   # LIBERO Panda home posture
Z_UP = np.array([0, 0, 1.0])
IK_KW = dict(lam=0.02, max_iters=150, trust=0.2)

# grasp selection
REUSE_VERIFIED = 0.015     # m: a verified grasp is carried along with a bowl that moved less than this
OUTER_FINGER = 0.047       # m beyond the rim radius: where the outer finger lands, for the neighbour check
CLOSED_APERTURE = 0.008    # m: the jaws closed on the wall, for check_closed
PLACE_MIN_SIGMA = 0.03     # the carry and place poses must be at least this far from singular
# approach
SPEED_FLOOR_MIN_DIST = 0.001   # m: no speed floor inside the last millimetre
ABOVE_MIN = -0.002         # m: the tool counts as above the grasp down to this
PRESHAPE_DONE = 0.004      # m of aperture error at which pre-shaping is done
PRESHAPE_STILL = 0.01      # m/s: ... and the fingers have stopped (switched approach)
DESCEND_MISALIGN = 0.5     # funnel misalignment below which the phase is "descend"
PRE_SLACK = 0.01           # m above the pre-grasp still counted as at it (switched approach)
FAR_XY = 0.04              # m: farther than this from the pre-grasp column, rise before crossing
RISE_ABOVE_PRE = 0.03      # m: rise to this above the pre-grasp before crossing
RISE_BAND = 0.03           # m below the pre-grasp over which the funnel blends rising into heading over
NEAR_XY = 0.02             # m: the funnel heads straight for a target this close ...
FAR_BAND = 0.04            # m: ... and rises in place for one this much farther,
RISE_STEP = 0.05           # m: rising at most this far above the tool
# transport
RISE_GRIP_DZ = 0.003       # m: with rise_counts_as_grip, a bowl raised this far counts as gripped
CARRY_BAND = 0.01          # m: a bowl this far below the carry height is still rising
EXIT_REACHED = 0.01        # m from the exit point
EXIT_SLACK = 0.005         # m short of exit_dist at which the bowl has left its pocket
LIFT_MIN_XY = 0.03         # m: this close to the plate, no separate lift


@dataclass(frozen=True)
class ProgramConfig:
    # motion
    k_lin: float = 5.0
    k_rot: float = 3.0
    v_max: float = 0.25
    w_max: float = 1.2
    # Minimum approach speed toward the grasp (0 = proportional all the way in). Proportional
    # homing ends at 1-3 cm/s, where a student's predicted direction agreed with the teacher
    # only to cosine 0.68 and closed-loop rollouts crept or froze a few mm short.
    v_min_approach: float = 0.0
    # grasp selection: "endpoints" checks pre-grasp and grasp only; "path" also the descent
    selection: str = "endpoints"
    min_sigma: float = 0.02
    approaches: tuple = (0.08,)
    face_weight: float = 1.0          # prefer the rim side facing the robot
    travel_weight: float = 0.2        # prefer small joint travel from the current arm
    sector: tuple | None = None       # (centre_deg, half_width_deg) of allowed rim angles, base frame
    # tolerances
    at_grasp_pos: float = 0.004
    at_grasp_rot: float = 0.08
    column_xy: float = 0.006
    column_rot: float = 0.10
    # grip, lift, carry, place
    squeeze_aperture: float = 0.006
    settled_rate: float = 0.005
    rise_counts_as_grip: bool = False
    lift_slow_dz: float = 0.02
    lift_slow_v: float = 0.06
    carry_clearance: float = 0.12
    over_plate_xy: float = 0.008
    release_dz: float = 0.006
    preshape_aperture: float | None = None   # close the jaws to this before descending (confined bowls)
    preshape_tau: float = 0.12        # gripper lag: act on aperture + rate * tau (measured: settles in 8 steps)
    preshape_band: float = 0.003
    sector_rel_robot: tuple | None = None    # (offset_deg, half_width_deg) around the robot-facing rim angle
    posture_gain: float = 0.0         # null-space pull toward LIBERO's home posture
    limit_margin: float = 0.0         # rad: reject grasp/path solutions this close to a joint limit
    neighbor_clearance: float = 0.0   # m: outer finger must clear other objects' footprints by this
    check_closed: bool = False        # also collision-check the grasp pose with the jaws closed
    clear_first: bool = True          # below the pre-grasp and far off: rise before traversing
    funnel: bool = True               # continuous approach/descend (False: the original switched version)
    funnel_xy: float = 0.02           # lateral error at which the target is back at full approach height
    funnel_rot: float = 0.15          # rad, same for rotation
    funnel_aperture: float = 0.012    # m, same for pre-shaping error
    exit_dir_deg: float | None = None  # leave along this base-frame direction before rising
    exit_dist: float = 0.0
    exit_height: float = 0.015


BASE = ProgramConfig()          # v2 behaviour: 20/20 on tasks 1, 2, 3, 5, 6 with random starts
PROGRAMS: dict[int, ProgramConfig] = {t: BASE for t in range(10)}

# Task 4 -- bowl in the open top drawer. The bowl fits the drawer with little
# room and the drawer walls stand 5-7 mm above its rim. Measured with the arm
# placed at the grasp: open jaws (78 mm) collide at every reachable rim angle;
# jaws pre-shaped to 26 mm are clear across 135-202 deg (the robot-facing side),
# sigma_min 0.135-0.143 there.
# Tasks 0, 1, 6, 8 name a neighbour ("between", "next to"), so randomised layouts
# put an object within finger range. Measured on task 0: the outer finger and the
# hand closed onto the ramekin (117-124 mm from the bowl), one jaw never reached
# the bowl, and the squeeze stalled at 13-14 mm.
for _t in (0, 1, 6, 8):
    PROGRAMS[_t] = replace(BASE, neighbor_clearance=0.03, check_closed=True)

PROGRAMS[4] = replace(BASE, preshape_aperture=0.026, sector_rel_robot=(0.0, 40.0),
                      selection="path", min_sigma=0.05, approaches=(0.08, 0.05),
                      # switched approach, not the funnel: in a bowl-sized pocket the funnel's
                      # blended sideways-and-down motion meets the drawer walls. Same seeds and
                      # layouts, 6 episodes: funnel 0/6, switched 6/6.
                      funnel=False)

# Task 7 -- bowl on the stove, at the edge of the workspace. Endpoint grasps are
# well conditioned from the nominal posture (sigma_min 0.10-0.18 on the robot-facing
# side) but from randomised starts the elbow drifts in the null space and descends
# into near-singularity (0.22 -> 0.015). A null-space pull toward the home posture
# keeps it out.
# ... and that pull alone is not enough: the stove bowl is only 0.42 m from the
# base, so a grasp on the robot-facing rim folds the elbow to within ~0.2 rad of
# its limit (q4 -2.82..-2.88 vs -3.07) and the servo stalls short of the grasp.
# Same seeds, same random starts, 20 episodes each:
#   robot-facing 12/20, far side 5/20, sideways -90 4/20, sideways +90 17/20
# and on fresh seeds, sideways +90 +-45 with a 0.25 rad limit margin: 28/30.
PROGRAMS[7] = replace(BASE, posture_gain=0.5, limit_margin=0.25, selection="path", min_sigma=0.05,
                      sector_rel_robot=(90.0, 45.0))


def _pose(R, p):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p
    return T


def _joint_margin(th, lim):
    """Distance of each solution to its nearest joint limit (rad)."""
    return np.minimum(th - lim[:, 0], lim[:, 1] - th).min(axis=1)


class _Approach(NamedTuple):
    """The tool against the chosen grasp, measured once per step. Base frame."""
    R: np.ndarray
    p: np.ndarray
    R_g: np.ndarray
    p_g: np.ndarray
    p_pre: np.ndarray
    e_pos: float
    e_rot: float
    off: np.ndarray            # tool minus grasp
    lateral: float             # off, less its vertical part
    above: bool
    approach_len: float        # pre-grasp height above the grasp

    @classmethod
    def measure(cls, R, p, R_g, p_g, p_pre):
        off = p - p_g
        return cls(R, p, R_g, p_g, p_pre, e_pos=float(np.linalg.norm(p_g - p)), e_rot=rot_angle(R.T @ R_g),
                   off=off, lateral=float(np.linalg.norm(off - Z_UP * (off @ Z_UP))),
                   above=float(off @ Z_UP) > ABOVE_MIN, approach_len=float((p_pre - p_g) @ Z_UP))


class ScriptedTeacher:
    def __init__(self, env, config: ProgramConfig | None = None, n_angles: int = 16):
        self.env, self.g = env, env.geom
        self.k = config or PROGRAMS[env.ti]
        self.angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        self._grasp_cache: dict = {}
        self._verified: list = []
        self._episode = None
        self.phase = ""
        if self.k.posture_gain > 0:
            env.servo.posture = HOME_Q.copy()
            env.servo.posture_gain = self.k.posture_gain

    # ------------------------------------------------------------ grasp selection
    def _grasp_candidates(self, R_bowl, p_bowl):
        g, k = self.g, self.k
        z_b = R_bowl[:, 2]
        facing = np.degrees(np.arctan2(-p_bowl[1], -p_bowl[0]))
        out = []
        for phi in self.angles:
            if k.sector is not None:
                d = (np.degrees(phi) - k.sector[0] + 180) % 360 - 180
                if abs(d) > k.sector[1]:
                    continue
            if k.sector_rel_robot is not None:
                d = (np.degrees(phi) - (facing + k.sector_rel_robot[0]) + 180) % 360 - 180
                if abs(d) > k.sector_rel_robot[1]:
                    continue
            radial = np.array([np.cos(phi), np.sin(phi), 0.0])
            radial = radial - z_b * (radial @ z_b); radial /= np.linalg.norm(radial)
            p = p_bowl + radial * g.rim_radius + z_b * (g.rim_top - g.grasp_depth)
            for sgn in (1.0, -1.0):
                y_t, z_t = sgn * radial, -z_b
                out.append((phi, np.stack([np.cross(y_t, z_t), y_t, z_t], axis=1), p))
        return out

    def choose_grasp(self, s: dict):
        """Memoised on the bowl pose. A verified choice is reused for a bowl that moved
        less than 15 mm, carried along with it: re-selecting after a nudge let a
        verified grasp be replaced by an unverified one mid-approach."""
        # Caches are valid within ONE episode only. A grasp verified against last
        # episode's neighbours says nothing about this layout's; keyed on the bowl pose
        # alone it would be reused, collision checks and all skipped.
        if getattr(self.env, "episode", None) != self._episode:
            self._grasp_cache.clear(); self._verified.clear()
            self._episode = getattr(self.env, "episode", None)
        key = (tuple(np.round(s["p_bowl"] / 0.003).astype(int)),
               int(round(np.degrees(np.arctan2(s["R_bowl"][1, 0], s["R_bowl"][0, 0])) / 5)))
        if key in self._grasp_cache:
            return self._grasp_cache[key]
        for (p_b, choice) in self._verified:
            if choice[3] and float(np.linalg.norm(s["p_bowl"] - p_b)) < REUSE_VERIFIED:
                shift = s["p_bowl"] - p_b
                moved = (choice[0], choice[1] + shift, choice[2] + shift, True)
                self._grasp_cache[key] = moved
                return moved
        choice = self._select(s)
        self._grasp_cache[key] = choice
        if choice[3]:
            self._verified.append((s["p_bowl"].copy(), choice))
        return choice

    def _select(self, s):
        """The best verified grasp (R, p, p_pre, True), else the rim point nearest the
        tool (R, p, p_pre, False)."""
        env, k = self.env, self.k
        sim = env.env.sim
        q_now = sim.data.qpos[env.joint_indexes].copy()
        cands = self._grasp_candidates(s["R_bowl"], s["p_bowl"])
        if k.neighbor_clearance > 0:
            # geometry first: on "between"/"next to" layouts most rim angles put the
            # outer finger on a neighbour, and solving IK for them first was most of
            # the cost of a reset
            others = self._neighbours()
            cands = [c for c in cands if self._neighbor_gap(s, others, c[0]) >= k.neighbor_clearance]
        saved = np.asarray(sim.get_state().flatten()).copy()
        try:
            # none left: every rim angle put the outer finger on a neighbour
            choice = None if k.neighbor_clearance > 0 and not cands else self._verified_grasp(s, cands, q_now)
        finally:
            sim.set_state_from_flattened(saved); sim.forward()
        if choice is None:
            from .progress import grasp_frames
            R, p, pre = grasp_frames(s["p_tool"], s["p_bowl"], s["R_bowl"], self.g)
            choice = (R, p, pre, False)
        return choice

    def _verified_grasp(self, s, cands, q_now):
        """The best-scoring candidate whose grasp and approach poses all solve IK and are
        collision-free, as (R, p, p_pre, True), trying each approach length in turn; None
        if there is none. Moves the simulator; the caller restores it."""
        k = self.k
        z_b = s["R_bowl"][:, 2]
        towards = -s["p_bowl"][:2] / (np.linalg.norm(s["p_bowl"][:2]) + 1e-9)
        fr = (1.0,) if k.selection == "endpoints" else (1.0, 2 / 3, 1 / 3)
        th_g, ok_g, sig_g = self._ik([_pose(R, p) for _, R, p in cands], q_now)
        alive = [c for c in range(len(cands)) if ok_g[c]]
        for approach in k.approaches:
            if not alive:
                break
            poses = [_pose(cands[c][1], cands[c][2] + z_b * approach * f) for c in alive for f in fr]
            th_p, ok_p, sig_p = self._ik(poses, q_now)
            scored = []
            for j, c in enumerate(alive):
                sl = slice(j * len(fr), (j + 1) * len(fr))
                if (not ok_p[sl].all() or not self._clear(list(th_p[sl]))
                        or not self._clear([th_g[c]], closed_too=k.check_closed)):
                    continue
                scored.append((self._score(cands[c][0], towards, th_p[sl], sig_p[sl], sig_g[c], q_now), c))
            if scored:
                _, c = max(scored)
                _, R, p = cands[c]
                return (R, p, p + z_b * approach, True)
        return None

    def _score(self, phi, towards, th_p, sig_p, sig_g, q_now) -> float:
        k = self.k
        face = float(np.array([np.cos(phi), np.sin(phi)]) @ towards)
        travel = float(np.linalg.norm(th_p[0] - q_now))
        if k.selection == "endpoints":
            return k.face_weight * face - k.travel_weight * travel
        return min(float(sig_p.min()), float(sig_g)) + 0.02 * k.face_weight * face

    def _ik(self, poses, q_now):
        """IK per pose from two seeds, the current arm and LIBERO's home posture: from one
        seed alone the same scene could verify at one step and fail at the next as the arm
        moved. Returns (theta, ok, sigma_min); ok = converged, conditioned, off the limits."""
        from .ik import sigma_min, solve_ik
        chain, k = self.env.chain, self.k
        lim = chain.limits.numpy()
        best = None
        for seed in (q_now, HOME_Q):
            res = solve_ik(chain, torch.tensor(np.stack(poses)), torch.tensor(np.tile(seed, (len(poses), 1))),
                           **IK_KW)
            th = res["theta"].numpy()
            sig = sigma_min(chain, res["theta"]).numpy()
            ok = res["converged"].numpy() & (sig >= k.min_sigma) & (_joint_margin(th, lim) >= k.limit_margin)
            if best is None:
                best = [th, ok, sig]
            else:
                take = ok & ~best[1]
                best[0][take] = th[take]; best[1] = best[1] | ok; best[2][take] = sig[take]
        return best[0], best[1], best[2]

    def _neighbours(self):
        """(centre xy, footprint radius) of every free object but the robot and the target,
        base frame."""
        mdl, dat = self.env.env.sim.model, self.env.env.sim.data
        tb = dat.body_xpos[mdl.body_name2id("robot0_base")]
        others = []
        for jn in range(mdl.njnt):
            if int(mdl.jnt_type[jn]) != 0:
                continue
            nm = mdl.joint_id2name(jn) or ""
            if contacts.is_robot(nm) or nm.startswith(TARGET):
                continue
            b = int(mdl.jnt_bodyid[jn]); c = dat.body_xpos[b][:2] - tb[:2]; r = 0.0
            for gg in range(mdl.ngeom):
                if int(mdl.geom_bodyid[gg]) == b and (mdl.geom_contype[gg] or mdl.geom_conaffinity[gg]):
                    r = max(r, float(np.linalg.norm(dat.geom_xpos[gg][:2] - tb[:2] - c))
                            + float(np.max(mdl.geom_size[gg][:2])))
            others.append((c, r))
        return others

    def _neighbor_gap(self, s, others, phi) -> float:
        """Clearance from the outer finger at rim angle phi to the nearest neighbour's footprint."""
        finger = s["p_bowl"][:2] + np.array([np.cos(phi), np.sin(phi)]) * (self.g.rim_radius + OUTER_FINGER)
        return min((float(np.linalg.norm(finger - c)) - r for c, r in others), default=1.0)

    def _clear(self, qs, closed_too=False) -> bool:
        """The arm at each of `qs`, jaws at the pre-shape (and closed too, if asked), penetrates
        nothing. Leaves the simulator in the last pose tried; the caller restores it."""
        env, k = self.env, self.k
        sim = env.env.sim
        idx, gidx = env.joint_indexes, env.gripper_indexes
        for q in qs:
            apertures = [k.preshape_aperture] if k.preshape_aperture is not None else [None]
            if closed_too:
                apertures.append(CLOSED_APERTURE)
            for ap in apertures:
                sim.data.qpos[idx] = q
                if ap is not None:
                    sim.data.qpos[gidx] = [ap / 2, -ap / 2]
                sim.forward()
                if env._robot_contact():
                    return False
        return True

    def layout_feasible(self, env) -> bool:
        """Accept a layout only if this program has a verified grasp AND can reach the
        place pose over the plate holding the bowl that way. Called by the environment
        after it samples a layout, so impossible scenes are redrawn, not kept."""
        from .ik import sigma_min, solve_ik
        s = dict(env.snapshot(), **env.ref)
        R, p, _, verified = self.choose_grasp(s)
        if not verified:
            return False
        g, k = self.g, self.k
        z_goal = float(s["p_plate"][2] + g.plate_top - g.bowl_bottom)
        z_carry = max(s["rest_z"], z_goal) + k.carry_clearance
        offset = p - s["p_bowl"]
        poses = [_pose(R, np.array([s["p_plate"][0], s["p_plate"][1], h]) + offset) for h in (z_carry, z_goal)]
        q_now = env.env.sim.data.qpos[env.joint_indexes].copy()
        res = solve_ik(env.chain, torch.tensor(np.stack(poses)), torch.tensor(np.tile(q_now, (2, 1))), **IK_KW)
        margin = _joint_margin(res["theta"].numpy(), env.chain.limits.numpy())
        return bool(res["converged"].all() and (sigma_min(env.chain, res["theta"]) >= PLACE_MIN_SIGMA).all()
                    and (margin >= k.limit_margin).all())

    # -------------------------------------------------------------------- skills
    def twist_to(self, R, p, R_goal, p_goal, v_max=None, v_min=0.0):
        k, spec = self.k, self.env.spec
        v_max = k.v_max if v_max is None else v_max
        w = rotvec(R.T @ R_goal) * k.k_rot
        v = R.T @ (p_goal - p) * k.k_lin
        if np.linalg.norm(w) > k.w_max: w *= k.w_max / np.linalg.norm(w)
        if np.linalg.norm(v) > v_max: v *= v_max / np.linalg.norm(v)
        dist = float(np.linalg.norm(p_goal - p))
        # floor the speed, but not inside the last millimetre, where a floor would chatter
        # (one control step at 5 cm/s is 2.5 mm, under the 4 mm grasp tolerance)
        if v_min > 0 and dist > SPEED_FLOOR_MIN_DIST and np.linalg.norm(v) < v_min:
            v *= v_min / max(np.linalg.norm(v), 1e-9)
        return np.concatenate([w / spec.max_angular_speed, v / spec.max_linear_speed])

    def act(self, s: dict | None = None) -> np.ndarray:
        env, g, k = self.env, self.g, self.k
        s = dict(s or env.snapshot(), **env.ref)
        R, p = s["R_tool"], s["p_tool"]
        if s["success"]:
            self.phase = "done"
            return np.concatenate([np.zeros(6), [-1.0]])
        if is_held(s, g):
            return self._transport(s, R, p)
        R_g, p_g, p_pre, _ = self.choose_grasp(s)
        a = _Approach.measure(R, p, R_g, p_g, p_pre)
        if s["aperture"] < g.hold_min and k.preshape_aperture is None:
            self.phase = "reopen"
            return np.concatenate([self.twist_to(R, p, R_g, p_pre, v_min=k.v_min_approach), [-1.0]])
        if a.e_pos < k.at_grasp_pos and a.e_rot < k.at_grasp_rot:
            self.phase = "close"
            return np.concatenate([self.twist_to(R, p, R_g, p_g), [1.0]])     # closing: no floor
        grip_open, ap_err = self._preshape(s)
        if not k.funnel:
            return self._switched_approach(s, a, grip_open, ap_err)
        return self._funnel(a, grip_open, ap_err)

    def _preshape(self, s):
        """(gripper command, aperture error) while pre-shaping the jaws; (-1 open, 0) without."""
        k = self.k
        if k.preshape_aperture is None:
            return -1.0, 0.0
        pred = s["aperture"] + s.get("aperture_rate", 0.0) * k.preshape_tau
        grip_open = 1.0 if pred > k.preshape_aperture + k.preshape_band else \
            (-1.0 if pred < k.preshape_aperture - k.preshape_band else 0.0)
        # position only: the finger-velocity term spikes whenever the arm moves (the
        # fingers lag), which kept raising the funnel target and bobbed the tool
        return grip_open, abs(s["aperture"] - k.preshape_aperture)

    def _funnel(self, a: _Approach, grip_open, ap_err):
        """FUNNEL. The target height above the grasp shrinks continuously as the tool
        aligns -- laterally, in rotation, and (when pre-shaping) in aperture -- so nearby
        states get nearby labels. The switched version descended only inside 12 mm /
        0.12 rad and otherwise returned to the pre-grasp: measured in closed loop, a
        student arriving just outside that region got labels it had never seen, averaged
        "descend" with "correct sideways", and stalled at 18% of the teacher's speed
        (direction agreement 0.98 for 25 steps, then 0.08)."""
        k = self.k
        misalign = max(a.lateral / k.funnel_xy, a.e_rot / k.funnel_rot,
                       ap_err / k.funnel_aperture if k.preshape_aperture is not None else 0.0)
        height = a.approach_len * float(np.clip(misalign, 0.0, 1.0))
        target = a.p_g + Z_UP * height
        if k.clear_first:
            target = self._rise_before_crossing(a.p, a.p_pre, target)
        self.phase = ("descend" if misalign < DESCEND_MISALIGN else "approach") \
            if k.preshape_aperture is None or ap_err < PRESHAPE_DONE else "preshape"
        return np.concatenate([self.twist_to(a.R, a.p, a.R_g, target, v_min=k.v_min_approach), [grip_open]])

    @staticmethod
    def _rise_before_crossing(p, p_pre, target):
        """Rising before crossing, also continuous: well below the pre-grasp the target
        stays over the tool (rise in place); at pre-grasp height it is over the grasp."""
        lift_w = float(np.clip((p[2] - (p_pre[2] - RISE_BAND)) / RISE_BAND, 0.0, 1.0))
        far_w = float(np.clip((float(np.linalg.norm((p - target)[:2])) - NEAR_XY) / FAR_BAND, 0.0, 1.0))
        w = 1.0 - far_w * (1.0 - lift_w)           # 1: head for the target; 0: rise in place
        return np.array([p[0] + w * (target[0] - p[0]), p[1] + w * (target[1] - p[1]),
                         max(target[2], min(p_pre[2] + RISE_ABOVE_PRE, p[2] + RISE_STEP)) if w < 1.0
                         else target[2]])

    def _switched_approach(self, s, a: _Approach, grip_open, ap_err):
        """The original switched approach, kept for comparison (funnel=False)."""
        k = self.k
        settled = k.preshape_aperture is None or (ap_err < PRESHAPE_DONE
                                                  and abs(s.get("aperture_rate", 1.0)) < PRESHAPE_STILL)
        if (a.lateral < k.column_xy and a.e_rot < k.column_rot and a.above
                and float(a.off @ Z_UP) <= a.approach_len + PRE_SLACK and settled):
            self.phase = "descend"
            return np.concatenate([self.twist_to(a.R, a.p, a.R_g, a.p_g, v_min=k.v_min_approach), [grip_open]])
        self.phase = "approach" if settled else "preshape"
        target = a.p_pre
        far = float(np.linalg.norm((a.p - a.p_pre)[:2])) > FAR_XY
        if k.clear_first and far and a.p[2] < a.p_pre[2] + PRE_SLACK:
            target = np.array([a.p[0], a.p[1], a.p_pre[2] + RISE_ABOVE_PRE])
            self.phase = "rise"
        return np.concatenate([self.twist_to(a.R, a.p, a.R_g, target, v_min=k.v_min_approach), [grip_open]])

    def _transport(self, s, R, p):
        g, k = self.g, self.k
        T_rel = np.linalg.inv(_pose(s["R_bowl"], s["p_bowl"])) @ _pose(R, p)
        z_goal = float(s["p_plate"][2] + g.plate_top - g.bowl_bottom)
        z_carry = max(s["rest_z"], z_goal) + k.carry_clearance
        yaw = np.arctan2(s["R_bowl"][1, 0], s["R_bowl"][0, 0])
        R_up = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.0]])
        h = float(np.linalg.norm((s["p_bowl"] - s["p_plate"])[:2]))
        dz = s["p_bowl"][2] - s["rest_z"]
        if dz < g.lifted_dz and not self._gripped(s, dz) and h > k.over_plate_xy:
            self.phase = "squeeze"
            return np.concatenate([np.zeros(6), [1.0]])
        target, grip, v_cap = self._carry_target(s, h, dz, z_goal, z_carry)
        T_tool = _pose(R_up, target) @ T_rel
        return np.concatenate([self.twist_to(R, p, T_tool[:3, :3], T_tool[:3, 3], v_cap), [grip]])

    def _gripped(self, s, dz) -> bool:
        k = self.k
        return bool((s["side1"] and s["side2"] and abs(s.get("aperture_rate", 1.0)) < k.settled_rate)
                    or s["aperture"] <= k.squeeze_aperture or (k.rise_counts_as_grip and dz > RISE_GRIP_DZ))

    def _carry_target(self, s, h, dz, z_goal, z_carry):
        """Where the held bowl goes next -- lower and release over the plate, leave its
        pocket, lift, or carry -- as (target, gripper command, speed cap); sets the phase."""
        k = self.k
        grip, v_cap = 1.0, None
        exit_xy = None
        if k.exit_dir_deg is not None and k.exit_dist > 0:
            d = np.radians(k.exit_dir_deg)
            exit_xy = s["rest_xy"] + k.exit_dist * np.array([np.cos(d), np.sin(d)]) if "rest_xy" in s else None
        if h <= k.over_plate_xy:
            target = np.array([s["p_plate"][0], s["p_plate"][1], z_goal])
            if s["p_bowl"][2] - z_goal < k.release_dz:
                self.phase, grip = "release", -1.0
            else:
                self.phase = "lower"
        elif exit_xy is not None and float(np.linalg.norm(s["p_bowl"][:2] - exit_xy)) > EXIT_REACHED \
                and s["p_bowl"][2] < z_carry - CARRY_BAND \
                and float(np.linalg.norm(s["p_bowl"][:2] - s["rest_xy"])) < k.exit_dist - EXIT_SLACK:
            target = np.array([exit_xy[0], exit_xy[1], s["rest_z"] + k.exit_height]); self.phase = "exit"
            v_cap = k.lift_slow_v * 2
        elif s["p_bowl"][2] < z_carry - CARRY_BAND and h > LIFT_MIN_XY:
            target = np.array([s["p_bowl"][0], s["p_bowl"][1], z_carry]); self.phase = "lift"
            if dz < k.lift_slow_dz:
                v_cap = k.lift_slow_v
        else:
            target = np.array([s["p_plate"][0], s["p_plate"][1], max(z_carry, s["p_bowl"][2])]); self.phase = "carry"
        return target, grip, v_cap
