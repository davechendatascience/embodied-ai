"""Any LIBERO task, executed through the same servos as libero_spatial.

PrivilegedEnv (screwhead/teacher_env.py) is tied to one scene: the akita bowl and the
plate are module constants, the layout sampler switches on task index, and progress is
measured against bowl geometry. This environment takes a task specification instead
(screwhead/task_spec.py) and reads what it needs from the scene (screwhead/scene.py),
so the other 120 LIBERO tasks are reachable without new per-task code.

What is kept identical to the spatial pipeline, because gate 2 was measured with it:
  - JOINT_POSITION control at kp=4000 with absolute holds during settling,
  - the arm driven by TwistServo (body twist integrated on SE(3) to a joint reference),
  - the gripper driven by GripperServo from a target aperture,
  - start-pose randomisation in task space via IK, rejecting poses in collision.

Success is LIBERO's own predicate check, so no reward or progress function is reimplemented.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from .interface import ActionSpec
from .kinematics import fk


def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


class TaskEnv:
    def __init__(self, suite: str, task_index: int, horizon: int = 600, seed: int = 0,
                 render: bool = True, kp: float = 4000.0, servo_iters: int = 1,
                 settle_steps: int = 5, hard_reset: bool = False, gripper_mode: str = "target",
                 start_xy_m: float = 0.0, start_z_m: float = 0.0, start_yaw_deg: float = 0.0,
                 start_tilt_deg: float = 0.0, start_null_rad: float = 0.0):
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        from .gripper_servo import GripperServo
        from .libero_env import JOINT_ACTION_SCALE, build_chain, gripper_geom, register_ur5e
        from .scene import Scene
        from .servo import TwistServo
        from .task_spec import parse
        register_ur5e()

        self.suite, self.ti, self.horizon, self.kp = suite, task_index, horizon, kp
        self.gripper_mode = gripper_mode
        self.gripper_servo = GripperServo()
        self.rng = np.random.default_rng(seed)
        self.spec = ActionSpec()
        self.scale = np.array([self.spec.rot_scale * self.spec.control_hz] * 3 +
                              [self.spec.pos_scale * self.spec.control_hz] * 3)
        bm = benchmark.get_benchmark_dict()[suite]()
        task = bm.get_task(task_index)
        self.language = task.language
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        self.task_spec = parse(bddl, suite)
        kw = (dict(camera_heights=128, camera_widths=128) if render else
              dict(use_camera_obs=False, has_offscreen_renderer=False))
        kw["hard_reset"] = hard_reset
        self.env = OffScreenRenderEnv(bddl_file_name=bddl, robots=["Panda"],
                                      gripper_types="PandaGripper", controller="JOINT_POSITION", **kw)
        _load = torch.load
        torch.load = lambda *a, **k: _load(*a, **{**k, "weights_only": False})
        try:
            self.init_states = bm.get_task_init_states(task_index)
        finally:
            torch.load = _load
        self.env.reset()
        flange, _ = gripper_geom(self.env)
        self.chain = build_chain("panda", flange)
        self.servo = TwistServo(self.chain, self.spec, JOINT_ACTION_SCALE, iters=servo_iters,
                                limit_gain=float(os.environ.get("SERVO_LIMIT_GAIN", "0.5")))
        self.scene = Scene(self.env.env)
        self.settle_steps = settle_steps
        self.start = dict(xy=start_xy_m, z=start_z_m, yaw=np.deg2rad(start_yaw_deg),
                          tilt=np.deg2rad(start_tilt_deg), null=start_null_rad)
        self.start_offset = None
        self.episode = 0
        self.t = 0
        self.raw = None
        self._settled: dict[int, np.ndarray] = {}

    # -- plumbing -------------------------------------------------------------------
    def _gains(self):
        from .libero_env import set_joint_gains
        set_joint_gains(self.env, self.kp)

    def _settle(self, n: int = 3, gripper: float = -1.0):
        """Absolute joint hold: a zero delta on this controller drifts 22 mm."""
        from .libero_env import JOINT_ACTION_SCALE
        hold = np.asarray(self.env.env._get_observations(force_update=True)["robot0_joint_pos"])
        for _ in range(n):
            meas = np.asarray(self.env.env._get_observations(force_update=True)["robot0_joint_pos"])
            cmd = np.zeros(self.env.env.action_dim)
            cmd[:7] = np.clip((hold - meas) / JOINT_ACTION_SCALE, -1, 1)
            cmd[-1] = gripper
            self._gains()
            self.env.step(cmd)

    def _max_object_speed(self) -> float:
        m, d = self.env.sim.model, self.env.sim.data
        v = 0.0
        for j in range(m.njnt):
            if int(m.jnt_type[j]) == 0:
                name = m.joint_id2name(j) or ""
                if not name.startswith(("robot", "gripper")):
                    a = int(m.jnt_dofadr[j]); v = max(v, float(np.linalg.norm(d.qvel[a:a + 3])))
        return v

    def _robot_contact(self) -> bool:
        m, d = self.env.sim.model, self.env.sim.data
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist >= 0:
                continue
            n1 = m.body_id2name(m.geom_bodyid[c.geom1]) or ""
            n2 = m.body_id2name(m.geom_bodyid[c.geom2]) or ""
            r1, r2 = n1.startswith(("robot", "gripper")), n2.startswith(("robot", "gripper"))
            if r1 != r2:
                return True
        return False

    def _randomize_start(self, batch: int = 16, max_rounds: int = 10) -> None:
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
                T[2, 3] = max(T[2, 3], 0.08)
                targets.append(T); seeds.append(q0 + self.rng.normal(0, st["null"], size=len(q0)))
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
        raise RuntimeError(f"{self.suite}[{self.ti}]: no collision-free start pose")

    # -- episode --------------------------------------------------------------------
    def reset(self, init_index: int | None = None):
        from .libero_env import remap_init_state
        self.episode += 1
        k = int(self.rng.integers(len(self.init_states))) if init_index is None else init_index
        if k not in self._settled:
            self.env.reset()
            self.env.set_init_state(remap_init_state(self.init_states[k], self.env.sim))
            self._settle(10)
            for _ in range(10):                      # settle until at rest, not a fixed count
                if self._max_object_speed() < 0.005:
                    break
                self._settle(5)
            self._settled[k] = np.asarray(self.env.sim.get_state().flatten()).copy()
        self.env.reset()
        self.env.set_init_state(self._settled[k])
        if any(v > 0 for v in self.start.values()):
            self._randomize_start()
        self.servo.reset(np.asarray(self.env.env._get_observations(force_update=True)["robot0_joint_pos"]))
        self.t = 0
        self.raw = self.env.env._get_observations(force_update=True)
        return self.raw

    def step(self, action: np.ndarray):
        """action: 6 normalised body-twist + 1 gripper (command or target aperture)."""
        a = np.clip(np.asarray(action, np.float64), -1, 1)
        o = self.env.env._get_observations(force_update=True)
        cmd = np.zeros(self.env.env.action_dim)
        cmd[:7] = self.servo.command(np.asarray(o["robot0_joint_pos"]), a[:6] * self.scale)
        if self.gripper_mode == "target":
            from .gripper_servo import channel_to_target
            gq, gv = o["robot0_gripper_qpos"], o.get("robot0_gripper_qvel", np.zeros(2))
            cmd[-1] = self.gripper_servo.command(float(channel_to_target(a[6])),
                                                 float(gq[0] - gq[1]), float(gv[0] - gv[1]))
        else:
            cmd[-1] = a[6]
        self._gains()
        raw, _, done, _ = self.env.step(cmd)
        self.t += 1
        self.raw = raw
        info = {"success": bool(done), "truncated": self.t >= self.horizon}
        return raw, float(done), bool(done) or info["truncated"], info

    # -- what the teacher and the student read --------------------------------------
    def snapshot(self) -> dict:
        """Tool pose and gripper state. Object poses come from Scene, by name."""
        raw = self.raw or self.env.env._get_observations(force_update=True)
        q = torch.tensor(np.asarray(raw["robot0_joint_pos"]), dtype=torch.float64)[None]
        T = fk(self.chain, q)[0].numpy()
        gq = raw["robot0_gripper_qpos"]; gv = raw.get("robot0_gripper_qvel", np.zeros(2))
        return dict(R_tool=T[:3, :3], p_tool=T[:3, 3], aperture=float(gq[0] - gq[1]),
                    aperture_rate=float(gv[0] - gv[1]), success=bool(self.env.env._check_success()))

    def student_state(self) -> np.ndarray:
        from .state import tool_state
        raw = self.raw or self.env.env._get_observations(force_update=True)
        q = torch.tensor(np.asarray(raw["robot0_joint_pos"]), dtype=torch.float64)[None]
        g = raw["robot0_gripper_qpos"]
        return tool_state(self.chain, q, torch.tensor([float(g[0] - g[1])], dtype=torch.float64))[0].float().numpy()

    def images(self):
        raw = self.raw or self.env.env._get_observations(force_update=True)
        return raw["agentview_image"], raw["robot0_eye_in_hand_image"]

    def close(self):
        self.env.env.close()
