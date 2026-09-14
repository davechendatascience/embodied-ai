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


class PrivilegedEnv:
    def __init__(self, task_index: int, suite: str = "libero_spatial", radius_m: float = 0.0,
                 horizon: int = 300, seed: int = 0, kp: float = 4000.0, render: bool = False,
                 hard_reset: bool = False, servo_iters: int = 1, settle_steps: int = 5):
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
        self.begin()
        self.placement = now - rest
        self.last_obs = self.obs()
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
        return self.last_obs, float(success), success or truncated, {"success": success,
                                                                     "truncated": truncated}

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
