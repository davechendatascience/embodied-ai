"""A LIBERO task the teacher can see into.

The teacher is allowed what the deployed policy is not: the simulator's own
object poses. Everything privileged is confined to `obs()`, so the boundary
between teacher and student is one function, and nothing downstream can mistake
one for the other.

Two things make this environment a test of perception rather than of memory:

  PLACEMENT. The goal object is displaced from LIBERO's recorded start by a
  uniform draw over a disc. LIBERO-spatial alone varies it by 12 mm (sd), inside
  the grasp basin of a trajectory aimed at the mean, which is why a blind policy
  solves it. A draw that is ejected on settling -- placed intersecting a
  neighbour -- is rejected and redrawn, since it tests a collision.

  ACTION SPACE. The teacher emits exactly what the student will: a body twist
  normalised by the action-space scale, and a gripper command, decoded to joints
  by damped least squares on the arm's own chain. A teacher that acted in joint
  space could not be distilled into a twist head without a second, unverified
  conversion.

Frames: positions are in the robot base frame. FK on the chain gives the tool
there directly; objects arrive in world and are shifted by the base pose, which
is measured from the simulator rather than assumed (residual against the grip
site: 0.00 mm).
"""
from __future__ import annotations

import os

import numpy as np
import torch

from .interface import ActionSpec
from .ik import decode_twist
from .kinematics import fk

TARGET = "akita_black_bowl_1"
RECEPTACLE = "plate_1"
N_TASKS = 10
OBS_DIM = 7 + 10 + 9 + 3 + 3 + 3 + N_TASKS      # see obs()
ACT_DIM = 7
EJECT_MM = 10.0


def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation about a unit axis."""
    a = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


class PrivilegedEnv:
    def __init__(self, task_index: int, suite: str = "libero_spatial", radius_m: float = 0.0,
                 horizon: int = 300, seed: int = 0, kp: float = 4000.0, render: bool = False,
                 hard_reset: bool = False, servo_iters: int = 1, settle_steps: int = 5,
                 start_xy_m: float = 0.0, start_z_m: float = 0.0, start_yaw_deg: float = 0.0,
                 start_tilt_deg: float = 0.0, start_null_rad: float = 0.0,
                 shaping: bool = False, gamma: float = 0.99, success_bonus: float = 10.0,
                 rich_obs: bool = False, shaping_gamma: float = 1.0):
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from .libero_env import build_chain, gripper_geom, register_ur5e
        register_ur5e()

        self.ti, self.radius, self.horizon, self.kp = task_index, radius_m, horizon, kp
        self.rng = np.random.default_rng(seed)
        self.spec = ActionSpec()
        self.scale = np.array([self.spec.rot_scale * self.spec.control_hz] * 3 +
                              [self.spec.pos_scale * self.spec.control_hz] * 3)
        bm = benchmark.get_benchmark_dict()[suite]()
        task = bm.get_task(task_index)
        self.language = task.language
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        kw = (dict(camera_heights=128, camera_widths=128) if render else
              dict(use_camera_obs=False, has_offscreen_renderer=False))
        # A hard reset reloads the MuJoCo model: 911 ms of a 1058 ms reset,
        # measured. set_init_state overwrites the full state afterwards, so the
        # reload buys nothing and stalls every synchronous worker.
        kw["hard_reset"] = hard_reset
        self.env = OffScreenRenderEnv(bddl_file_name=bddl, robots=["Panda"],
                                      gripper_types="PandaGripper",
                                      controller="JOINT_POSITION", **kw)
        _load = torch.load
        torch.load = lambda *a, **k: _load(*a, **{**k, "weights_only": False})
        try:
            self.init_states = bm.get_task_init_states(task_index)
        finally:
            torch.load = _load
        self.env.reset()
        flange, _ = gripper_geom(self.env)
        self.chain = build_chain("panda", flange)
        from .libero_env import JOINT_ACTION_SCALE
        from .servo import TwistServo
        self.servo = TwistServo(self.chain, self.spec, JOINT_ACTION_SCALE, iters=servo_iters)
        self.onehot = np.eye(N_TASKS, dtype=np.float32)[task_index]
        self.t = 0
        self.last_obs = None
        self._settled: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.settle_steps = settle_steps
        self.start = dict(xy=start_xy_m, z=start_z_m, yaw=np.deg2rad(start_yaw_deg),
                          tilt=np.deg2rad(start_tilt_deg), null=start_null_rad)
        self.start_offset = None
        from .progress import PickPlaceGeometry
        self.geom = PickPlaceGeometry()
        self.shaping, self.gamma, self.success_bonus = shaping, gamma, success_bonus
        self.rich_obs = rich_obs
        self.shaping_gamma = shaping_gamma
        self.ref = None            # rest_z, d0_reach, d0_carry: fixed at the start of an episode
        self.ever_lifted = False
        self.phi = 0.0

    # -- simulator handles (re-fetched: a reset can rebuild the sim) -----------------
    def _ids(self):
        m = self.env.sim.model
        j = m.joint_name2id(f"{TARGET}_joint0")
        return (int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j]), int(m.jnt_bodyid[j]),
                int(m.jnt_bodyid[m.joint_name2id(f"{RECEPTACLE}_joint0")]),
                m.body_name2id("robot0_base"))

    def _gains(self):
        from .libero_env import set_joint_gains
        set_joint_gains(self.env, self.kp)

    def _settle(self, n=3, gripper: float = -1.0):
        """Hold the arm where it is, as an ABSOLUTE joint target.

        A zero action on robosuite's joint controller is a zero DELTA, and right
        after set_init_state it does not hold: measured, the tool moves 16 mm on
        the first step and 22 mm by the third (only 6 mm of it vertical, so not
        sag), while an absolute hold keeps it within 0.0 mm of where it was put.
        A servo armed after a zero-delta settle integrates every later twist from
        a start 22 mm off the demonstrations'.
        """
        from .libero_env import JOINT_ACTION_SCALE
        hold = np.asarray(self.env.env._get_observations(force_update=True)["robot0_joint_pos"])
        for _ in range(n):
            cur = np.asarray(self.env.env._get_observations(force_update=True)["robot0_joint_pos"])
            cmd = np.zeros(self.env.env.action_dim)
            cmd[:7] = np.clip((hold - cur) / JOINT_ACTION_SCALE, -1, 1)
            cmd[-1] = gripper
            self._gains()
            self.env.step(cmd)


    # -- start pose -----------------------------------------------------------------
    def _robot_contact(self) -> bool:
        """True if any robot or gripper geom penetrates something that is not the robot."""
        sim = self.env.sim; m, d = sim.model, sim.data
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist >= 0:
                continue
            n1 = m.body_id2name(m.geom_bodyid[c.geom1]) or ""
            n2 = m.body_id2name(m.geom_bodyid[c.geom2]) or ""
            r1 = n1.startswith(("robot0", "gripper0")); r2 = n2.startswith(("robot0", "gripper0"))
            if r1 != r2:
                return True
        return False

    def _randomize_start(self, batch: int = 16, max_rounds: int = 10) -> None:
        """Move the arm to a random TOOL pose near LIBERO's start, via IK.

        Sampled in task space -- position box, yaw about vertical, tilt about a
        random horizontal axis -- because that is what the approach depends on;
        jittering joints directly gives a tool distribution nobody chose. The IK
        seed is perturbed so the elbow (the null space of a 7-DoF arm) varies too.
        A candidate is kept only if Newton converged, it is away from
        singularity, and the arm penetrates nothing when placed there.
        """
        from .ik import sigma_min, solve_ik
        sim = self.env.sim
        idx = self.env.env.robots[0]._ref_joint_pos_indexes
        q0 = sim.data.qpos[idx].copy()
        T0 = fk(self.chain, torch.tensor(q0)[None])[0].numpy()
        st = self.start
        for _ in range(max_rounds):
            targets, seeds, meta = [], [], []
            for _ in range(batch):
                dp = np.array([self.rng.uniform(-st["xy"], st["xy"]), self.rng.uniform(-st["xy"], st["xy"]),
                               self.rng.uniform(-st["z"], st["z"])])
                yaw = self.rng.uniform(-st["yaw"], st["yaw"])
                ax = self.rng.normal(size=2); ax = np.array([*ax / (np.linalg.norm(ax) + 1e-12), 0.0])
                tilt = self.rng.uniform(-st["tilt"], st["tilt"])
                T = T0.copy()
                T[:3, :3] = _rot(np.array([0, 0, 1.0]), yaw) @ _rot(ax, tilt) @ T0[:3, :3]
                T[:3, 3] = T0[:3, 3] + dp
                T[2, 3] = max(T[2, 3], 0.08)                      # keep the tool clear of the table
                targets.append(T)
                seeds.append(q0 + self.rng.normal(0, st["null"], size=len(q0)))
                meta.append((dp * 1000, np.rad2deg(yaw), np.rad2deg(tilt)))
            res = solve_ik(self.chain, torch.tensor(np.stack(targets)), torch.tensor(np.stack(seeds)),
                           lam=0.02, max_iters=200, trust=0.2)
            ok = res["converged"] & (sigma_min(self.chain, res["theta"]) > 0.02)
            for k in torch.nonzero(ok).flatten().tolist():
                sim.data.qpos[idx] = res["theta"][k].numpy()
                sim.data.qvel[:] = 0.0
                sim.forward()
                if not self._robot_contact():
                    self._settle(self.settle_steps)
                    self.start_offset = meta[k]
                    return
            sim.data.qpos[idx] = q0; sim.forward()
        raise RuntimeError(f"task {self.ti}: no collision-free reachable start pose found")

    # -- episode --------------------------------------------------------------------
    def reset(self, init_index: int | None = None, max_tries: int = 20) -> np.ndarray:
        """Start a LIBERO init state, then displace the goal object from where it RESTS.

        LIBERO's bundled init states spawn the bowl 72 mm above the table; it
        falls to rest (898.4 mm) in the first few steps. Displacing it before
        that and comparing against the spawn height rejects every valid draw as
        an ejection. So: settle, record the resting pose, displace from it,
        settle again, and accept only if the bowl stayed on the table surface
        and landed within EJECT_MM of where it was put.
        """
        from .libero_env import remap_init_state
        k = int(self.rng.integers(len(self.init_states))) if init_index is None else init_index
        if k not in self._settled:
            # The drop from LIBERO's spawn height is the same every time for a
            # given init state: pay for it once. Resets were 145 ms, 124 of them
            # settling, and a synchronous vector env waits on its slowest reset.
            self.env.reset()
            self.env.set_init_state(remap_init_state(self.init_states[k], self.env.sim))
            self._settle(10)
            qadr, *_ = self._ids()
            self._settled[k] = (np.asarray(self.env.sim.get_state().flatten()).copy(),
                                self.env.sim.data.qpos[qadr:qadr + 3].copy())
        settled, rest = self._settled[k]
        self.rejected = 0
        for _ in range(max_tries):
            self.env.reset()
            self.env.set_init_state(settled)
            sim = self.env.sim
            qadr, vadr, *_ = self._ids()
            want = np.zeros(2)
            if self.radius > 0:
                r = self.radius * np.sqrt(self.rng.random())
                th = 2 * np.pi * self.rng.random()
                want = r * np.array([np.cos(th), np.sin(th)])
                sim.data.qpos[qadr:qadr + 2] = rest[:2] + want
                sim.data.qvel[vadr:vadr + 6] = 0.0
                sim.forward()
            self._settle(self.settle_steps)
            now = self.env.sim.data.qpos[qadr:qadr + 3]
            dz = abs(now[2] - rest[2]) * 1000
            dxy = np.linalg.norm(now[:2] - rest[:2] - want) * 1000
            if dz <= EJECT_MM and dxy <= EJECT_MM:
                break
            self.rejected += 1
        else:
            raise RuntimeError(f"task {self.ti}: no valid placement in {max_tries} draws "
                               f"at radius {self.radius} m")
        if any(v > 0 for v in self.start.values()):
            self._randomize_start()
        self.begin()
        self.placement = now - rest
        self.ever_lifted = False
        self.last_obs = self.obs()            # refreshes self.raw, which snapshot() reads
        self.set_reference()
        if self.rich_obs:
            self.last_obs = np.concatenate([self.last_obs, self._rich(self.snapshot())])
        return self.last_obs

    def obs(self, raw: dict | None = None) -> np.ndarray:
        """PRIVILEGED. The only place simulator object state enters the teacher.

        `raw` is the observation dict the environment just returned. Without it
        observables are refreshed with force_update -- they are cached on a
        sampling interval, and a state set without stepping (BC extraction, a
        reset) would otherwise read the previous frame -- which on a rendering
        environment renders both cameras a second time.
        """
        sim = self.env.sim
        o = raw if raw is not None else self.env.env._get_observations(force_update=True)
        self.raw = o
        q = np.asarray(o["robot0_joint_pos"], np.float64)
        gq = o["robot0_gripper_qpos"]
        T = fk(self.chain, torch.tensor(q)[None])[0].numpy()
        _, _, bowl, plate, base = self._ids()
        tb = sim.data.body_xpos[base]
        p_tool = T[:3, 3]
        p_bowl = sim.data.body_xpos[bowl] - tb
        p_plate = sim.data.body_xpos[plate] - tb
        R_bowl = sim.data.body_xmat[bowl].reshape(3, 3)
        vec = np.concatenate([
            q,                                              # 7
            p_tool, T[:3, :2].T.reshape(-1), [gq[0] - gq[1]],  # 10
            p_bowl, R_bowl[:3, :2].T.reshape(-1),           # 9
            p_plate,                                        # 3
            p_bowl - p_tool,                                # 3
            p_plate - p_bowl,                               # 3
            self.onehot,                                    # 10
        ])
        return vec.astype(np.float32)

    def step(self, action: np.ndarray):
        """action: 6 normalised body-twist + 1 gripper, each in [-1, 1].

        Executed through TwistServo: integrated onto an absolute joint reference,
        because per-step deltas lose 18% of every step to controller lag and the
        loss compounds (see servo.py)."""
        a = np.clip(np.asarray(action, np.float64), -1, 1)
        o = self.env.env._get_observations(force_update=True)
        cmd = np.zeros(self.env.env.action_dim)
        cmd[:7] = self.servo.command(np.asarray(o["robot0_joint_pos"]), a[:6] * self.scale)
        cmd[-1] = a[6]
        self._gains()
        raw, _, done, _ = self.env.step(cmd)
        self.t += 1
        success = bool(done)
        self.last_obs = self.obs(raw)
        truncated = self.t >= self.horizon
        info = {"success": success, "truncated": truncated}
        reward = float(success)
        if self.shaping:
            snap = self.snapshot()
            phi, stage, _ = self.progress(snap)
            if stage >= 3 and snap["p_bowl"][2] - self.ref["rest_z"] > self.geom.lifted_dz:
                self.ever_lifted = True
            # task reward only for a bowl that was actually picked up: pushing or
            # flipping it onto the plate satisfies On(bowl, plate) without a grasp
            task = self.success_bonus * float(success and self.ever_lifted)
            # Undiscounted difference. gamma*phi' - phi = (phi' - phi) - (1-gamma)*phi':
            # the second term is a per-step drag proportional to progress already
            # made. Measured, it collapsed a from-scratch policy's peak progress
            # from 0.82 (random) to 0.04 in 50 iterations -- random motion's
            # progress changes average to zero, the drag never does, so the cheapest
            # state was phi = 0. For an episodic task the undiscounted difference
            # telescopes to phi(s_T) - phi(s_0) and carries no drag.
            reward = task + self.shaping_gamma * phi - self.phi
            info.update(phi=phi, stage=stage, ever_lifted=self.ever_lifted,
                        success_lifted=bool(success and self.ever_lifted))
            self.phi = phi
            if self.rich_obs:
                self.last_obs = np.concatenate([self.last_obs, self._rich(snap, phi, stage)])
        return self.last_obs, reward, success or truncated, info

    def begin(self) -> None:
        """Arm the servo at the current joints. Called by reset(); call it yourself
        after placing the sim in a state by any other route."""
        o = self.env.env._get_observations(force_update=True)
        self.servo.reset(np.asarray(o["robot0_joint_pos"]))
        self.t = 0

    def obs_at(self, flat_state: np.ndarray) -> np.ndarray:
        """PRIVILEGED observation at a recorded simulator state (for BC)."""
        from .libero_env import remap_init_state
        sim = self.env.sim
        sim.set_state_from_flattened(remap_init_state(flat_state, sim))
        sim.forward()
        return self.obs()


    # -- task progress (privileged) ---------------------------------------------------
    def snapshot(self) -> dict:
        """Everything progress() needs, read from the simulator, base frame."""
        from .progress import grasp_frames
        sim = self.env.sim; m, d = sim.model, sim.data
        raw = getattr(self, "raw", None) or self.env.env._get_observations(force_update=True)
        q = torch.tensor(np.asarray(raw["robot0_joint_pos"]), dtype=torch.float64)[None]
        T = fk(self.chain, q)[0].numpy()
        tb = d.body_xpos[m.body_name2id("robot0_base")]
        _, _, bowl, plate, _ = self._ids()
        bowl_geoms = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == bowl}
        side = {m.geom_name2id("gripper0_finger1_pad_collision"): 1, m.geom_name2id("gripper0_finger1_collision"): 1,
                m.geom_name2id("gripper0_finger2_pad_collision"): 2, m.geom_name2id("gripper0_finger2_collision"): 2}
        sides, any_grip, supported = set(), False, False
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist > 0.0005:
                continue
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if b not in bowl_geoms or a in bowl_geoms:
                    continue
                name = m.geom_id2name(a) or ""
                if a in side:
                    sides.add(side[a])
                if name.startswith("gripper0") or name.startswith("robot0"):
                    any_grip = True
                else:
                    supported = True
        gq = raw["robot0_gripper_qpos"]
        gv = raw.get("robot0_gripper_qvel", np.zeros(2))
        return dict(R_tool=T[:3, :3], p_tool=T[:3, 3], aperture=float(gq[0] - gq[1]),
                    aperture_rate=float(gv[0] - gv[1]),
                    R_bowl=d.body_xmat[bowl].reshape(3, 3).copy(), p_bowl=(d.body_xpos[bowl] - tb).copy(),
                    p_plate=(d.body_xpos[plate] - tb).copy(), side1=1 in sides, side2=2 in sides,
                    any_grip=any_grip, supported=supported, success=bool(self.env.check_success()))

    RICH_DIM = 1 + 7 + 6

    def _rich(self, snap: dict, phi: float | None = None, stage: int | None = None) -> np.ndarray:
        """PRIVILEGED progress features: phi/6, stage one-hot, tool->grasp waypoint error."""
        from .progress import grasp_error
        if phi is None:
            phi, stage, _ = self.progress(snap)
        oh = np.zeros(7); oh[int(stage)] = 1.0
        return np.concatenate([[phi / 6.0], oh, grasp_error(snap, self.geom)]).astype(np.float32)

    def set_reference(self, snap: dict | None = None) -> None:
        """Episode constants: bowl rest height and the two stage normalisers that depend on layout."""
        from .progress import bowl_distance, reach_distance
        s = snap or self.snapshot(); g = self.geom
        d0_reach, _, _ = reach_distance(dict(s, rest_z=float(s["p_bowl"][2])), g)
        from .progress import transport_remaining
        ref0 = dict(s, rest_z=float(s["p_bowl"][2]), d0_reach=float(d0_reach), d0_carry=1.0)
        d0_carry, _ = transport_remaining(ref0, g)       # the whole transport, from grasped at rest
        self.ref = dict(rest_z=float(s["p_bowl"][2]), d0_reach=float(d0_reach), d0_carry=float(d0_carry),
                        rest_xy=s["p_bowl"][:2].copy())
        self.phi = self.progress(s)[0]

    def progress(self, snap: dict | None = None):
        from .progress import progress
        s = dict(snap or self.snapshot(), **self.ref)
        return progress(s, self.geom)

    # -- what the STUDENT is allowed to see ------------------------------------------
    def images(self) -> tuple[np.ndarray, np.ndarray]:
        """Agentview and wrist frames of the current state (render=True only)."""
        return self.raw["agentview_image"], self.raw["robot0_eye_in_hand_image"]

    def student_state(self) -> np.ndarray:
        """Tool pose + gripper aperture via tool_state, the student's proprioception.
        No object state, and not the teacher's layout of the same quantities."""
        from .state import tool_state
        q = torch.tensor(np.asarray(self.raw["robot0_joint_pos"]), dtype=torch.float64)[None]
        g = self.raw["robot0_gripper_qpos"]
        return tool_state(self.chain, q, torch.tensor([float(g[0] - g[1])], dtype=torch.float64)
                          )[0].float().numpy()

    def close(self):
        self.env.env.close()
