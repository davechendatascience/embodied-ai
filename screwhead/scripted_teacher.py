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

from dataclasses import dataclass, field, replace

import numpy as np
import torch

from .progress import is_held, rot_angle, rotvec


@dataclass(frozen=True)
class ProgramConfig:
    # motion
    k_lin: float = 5.0
    k_rot: float = 3.0
    v_max: float = 0.25
    w_max: float = 1.2
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
PROGRAMS[4] = replace(BASE, preshape_aperture=0.026, sector_rel_robot=(0.0, 40.0),
                      selection="path", min_sigma=0.05, approaches=(0.08, 0.05))

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


class ScriptedTeacher:
    def __init__(self, env, config: ProgramConfig | None = None, n_angles: int = 16):
        self.env, self.g = env, env.geom
        self.k = config or PROGRAMS[env.ti]
        self.angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        self._grasp_cache: dict = {}
        self.phase = ""
        if self.k.posture_gain > 0:
            env.servo.posture = np.array([0.0, -0.161, 0.0, -2.4446, 0.0, 2.2268, np.pi / 4])   # LIBERO Panda home
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
        key = (tuple(np.round(s["p_bowl"] / 0.003).astype(int)),
               int(round(np.degrees(np.arctan2(s["R_bowl"][1, 0], s["R_bowl"][0, 0])) / 5)))
        if key not in self._grasp_cache:
            self._grasp_cache[key] = self._select(s)
        return self._grasp_cache[key]

    def _select(self, s):
        from .ik import sigma_min, solve_ik
        env, g, k = self.env, self.g, self.k
        sim = env.env.sim
        idx = env.env.env.robots[0]._ref_joint_pos_indexes
        q_now = sim.data.qpos[idx].copy()
        z_b = s["R_bowl"][:, 2]
        cands = self._grasp_candidates(s["R_bowl"], s["p_bowl"])
        towards = -s["p_bowl"][:2] / (np.linalg.norm(s["p_bowl"][:2]) + 1e-9)

        lim = env.chain.limits.numpy()

        def ik(poses):
            res = solve_ik(env.chain, torch.tensor(np.stack(poses)), torch.tensor(np.tile(q_now, (len(poses), 1))),
                           lam=0.02, max_iters=150, trust=0.2)
            th = res["theta"].numpy()
            sig = sigma_min(env.chain, res["theta"]).numpy()
            margin = np.minimum(th - lim[:, 0], lim[:, 1] - th).min(axis=1)
            return th, res["converged"].numpy() & (sig >= k.min_sigma) & (margin >= k.limit_margin), sig

        gidx = env.env.env.robots[0]._ref_gripper_joint_pos_indexes

        def clear(qs):
            for q in qs:
                sim.data.qpos[idx] = q
                if k.preshape_aperture is not None:
                    sim.data.qpos[gidx] = [k.preshape_aperture / 2, -k.preshape_aperture / 2]
                sim.forward()
                if env._robot_contact():
                    return False
            return True

        saved = np.asarray(sim.get_state().flatten()).copy()
        choice = None
        try:
            fr = (1.0,) if k.selection == "endpoints" else (1.0, 2 / 3, 1 / 3)
            th_g, ok_g, sig_g = ik([_pose(R, p) for _, R, p in cands])
            alive = [c for c in range(len(cands)) if ok_g[c]]
            for approach in k.approaches:
                if not alive:
                    break
                poses = [_pose(cands[c][1], cands[c][2] + z_b * approach * f) for c in alive for f in fr]
                th_p, ok_p, sig_p = ik(poses)
                scored = []
                for j, c in enumerate(alive):
                    sl = slice(j * len(fr), (j + 1) * len(fr))
                    if not ok_p[sl].all() or not clear(list(th_p[sl]) + [th_g[c]]):
                        continue
                    phi = cands[c][0]
                    face = float(np.array([np.cos(phi), np.sin(phi)]) @ towards)
                    travel = float(np.linalg.norm(th_p[sl][0] - q_now))
                    if k.selection == "endpoints":
                        score = k.face_weight * face - k.travel_weight * travel
                    else:
                        score = min(float(sig_p[sl].min()), float(sig_g[c])) + 0.02 * k.face_weight * face
                    scored.append((score, c))
                if scored:
                    _, c = max(scored)
                    _, R, p = cands[c]
                    choice = (R, p, p + z_b * approach, True)
                    break
        finally:
            sim.set_state_from_flattened(saved); sim.forward()
        if choice is None:
            from .progress import grasp_frames
            R, p, pre = grasp_frames(s["p_tool"], s["p_bowl"], s["R_bowl"], g)
            choice = (R, p, pre, False)
        return choice

    # -------------------------------------------------------------------- skills
    def twist_to(self, R, p, R_goal, p_goal, v_max=None):
        k, spec = self.k, self.env.spec
        v_max = k.v_max if v_max is None else v_max
        w = rotvec(R.T @ R_goal) * k.k_rot
        v = R.T @ (p_goal - p) * k.k_lin
        if np.linalg.norm(w) > k.w_max: w *= k.w_max / np.linalg.norm(w)
        if np.linalg.norm(v) > v_max: v *= v_max / np.linalg.norm(v)
        return np.concatenate([w / (spec.rot_scale * spec.control_hz), v / (spec.pos_scale * spec.control_hz)])

    def act(self, s: dict | None = None) -> np.ndarray:
        env, g, k = self.env, self.g, self.k
        s = dict(s or env.snapshot(), **env.ref)
        R, p = s["R_tool"], s["p_tool"]
        z_up = np.array([0, 0, 1.0])
        if s["success"]:
            self.phase = "done"
            return np.concatenate([np.zeros(6), [-1.0]])
        if is_held(s, g):
            return self._transport(s, R, p)
        R_g, p_g, p_pre, _ = self.choose_grasp(s)
        e_pos = float(np.linalg.norm(p_g - p)); e_rot = rot_angle(R.T @ R_g)
        off = p - p_g
        lateral = float(np.linalg.norm(off - z_up * (off @ z_up)))
        above = float(off @ z_up) > -0.002
        approach_len = float((p_pre - p_g) @ z_up)
        if s["aperture"] < g.hold_min and k.preshape_aperture is None:
            self.phase = "reopen"
            return np.concatenate([self.twist_to(R, p, R_g, p_pre), [-1.0]])
        if e_pos < k.at_grasp_pos and e_rot < k.at_grasp_rot:
            self.phase = "close"
            return np.concatenate([self.twist_to(R, p, R_g, p_g), [1.0]])
        grip_open, settled = -1.0, True
        if k.preshape_aperture is not None:
            pred = s["aperture"] + s.get("aperture_rate", 0.0) * k.preshape_tau
            grip_open = 1.0 if pred > k.preshape_aperture + k.preshape_band else \
                (-1.0 if pred < k.preshape_aperture - k.preshape_band else 0.0)
            settled = abs(s["aperture"] - k.preshape_aperture) < 0.004 and abs(s.get("aperture_rate", 1.0)) < 0.01
        if lateral < k.column_xy and e_rot < k.column_rot and above and float(off @ z_up) <= approach_len + 0.01 and settled:
            self.phase = "descend"
            return np.concatenate([self.twist_to(R, p, R_g, p_g), [grip_open]])
        self.phase = "approach" if settled else "preshape"
        return np.concatenate([self.twist_to(R, p, R_g, p_pre), [grip_open]])

    def _transport(self, s, R, p):
        g, k = self.g, self.k
        T_rel = np.linalg.inv(_pose(s["R_bowl"], s["p_bowl"])) @ _pose(R, p)
        z_goal = float(s["p_plate"][2] + g.plate_top - g.bowl_bottom)
        z_carry = max(s["rest_z"], z_goal) + k.carry_clearance
        yaw = np.arctan2(s["R_bowl"][1, 0], s["R_bowl"][0, 0])
        R_up = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.0]])
        h = float(np.linalg.norm((s["p_bowl"] - s["p_plate"])[:2]))
        dz = s["p_bowl"][2] - s["rest_z"]
        grip, v_cap = 1.0, None
        gripped = ((s["side1"] and s["side2"] and abs(s.get("aperture_rate", 1.0)) < k.settled_rate)
                   or s["aperture"] <= k.squeeze_aperture or (k.rise_counts_as_grip and dz > 0.003))
        if dz < g.lifted_dz and not gripped and h > k.over_plate_xy:
            self.phase = "squeeze"
            return np.concatenate([np.zeros(6), [1.0]])
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
        elif exit_xy is not None and float(np.linalg.norm(s["p_bowl"][:2] - exit_xy)) > 0.01 \
                and s["p_bowl"][2] < z_carry - 0.01 \
                and float(np.linalg.norm(s["p_bowl"][:2] - s["rest_xy"])) < k.exit_dist - 0.005:
            target = np.array([exit_xy[0], exit_xy[1], s["rest_z"] + k.exit_height]); self.phase = "exit"
            v_cap = k.lift_slow_v * 2
        elif s["p_bowl"][2] < z_carry - 0.01 and h > 0.03:
            target = np.array([s["p_bowl"][0], s["p_bowl"][1], z_carry]); self.phase = "lift"
            if dz < k.lift_slow_dz:
                v_cap = k.lift_slow_v
        else:
            target = np.array([s["p_plate"][0], s["p_plate"][1], max(z_carry, s["p_bowl"][2])]); self.phase = "carry"
        T_tool = _pose(R_up, target) @ T_rel
        return np.concatenate([self.twist_to(R, p, T_tool[:3, :3], T_tool[:3, 3], v_cap), [grip]])
