"""Any LIBERO task, executed through the same servos as libero_spatial.

PrivilegedEnv (screwhead/teacher_env.py) is tied to one scene: the akita bowl and the
plate are module constants, the layout sampler switches on task index, and progress is
measured against bowl geometry. This environment takes a task specification instead
(screwhead/task_spec.py) and reads what it needs from the scene (screwhead/scene.py),
so the other 120 LIBERO tasks are reachable without new per-task code.

What is kept identical to the spatial pipeline, because gate 2 was measured with it (the
shared parts live in screwhead/sim_arm.py):
  - JOINT_POSITION control at kp=4000 with absolute holds during settling,
  - the arm driven by TwistServo (body twist integrated on SE(3) to a joint reference),
  - the gripper driven by GripperServo from a target aperture,
  - start-pose randomisation in task space via IK, rejecting poses in collision.

Success is LIBERO's own predicate check, so no reward or progress function is reimplemented.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

from .interface import ActionSpec
from .sim_arm import SimArm

REST_SPEED = 0.005        # m/s: the scene is at rest when no free object moves faster
INITIAL_SETTLE = 10       # control steps held after loading an initial state
SETTLE_CHUNK = 5          # then in chunks of this many, until at rest
SETTLE_ROUNDS = 10        # at most this many chunks
CAMERA_PX = 128
SEED_MOD = 2**32 - 1      # numpy's legacy seed range


@dataclass(frozen=True)
class Execution:
    """How actions are executed -- the part that must be identical for teacher and student."""
    kp: float = 4000.0
    servo_iters: int = 1
    settle_steps: int = 5
    gripper_mode: str = "target"      # "target": a target aperture; otherwise a raw command
    hard_reset: bool = False


@dataclass(frozen=True)
class StartNoise:
    """Start-pose randomisation in task space (zero: LIBERO's own start)."""
    xy_m: float = 0.0
    z_m: float = 0.0
    yaw_deg: float = 0.0
    tilt_deg: float = 0.0
    null_rad: float = 0.0

    def as_dict(self) -> dict:
        return dict(xy=self.xy_m, z=self.z_m, yaw=np.deg2rad(self.yaw_deg),
                    tilt=np.deg2rad(self.tilt_deg), null=self.null_rad)


class TaskEnv(SimArm):
    def __init__(self, suite: str, task_index: int, horizon: int = 600, seed: int = 0,
                 render: bool = True, execution: Execution | None = None, start: StartNoise | None = None):
        from .gripper_servo import GripperServo
        from .libero_env import JOINT_ACTION_SCALE, build_chain, gripper_geom, register_ur5e
        from .scene import Scene
        from .servo import TwistServo
        register_ur5e()
        ex = execution or Execution()
        self.suite, self.ti, self.horizon, self.kp = suite, task_index, horizon, ex.kp
        self.gripper_mode, self.settle_steps = ex.gripper_mode, ex.settle_steps
        self.gripper_servo = GripperServo()
        self.rng = np.random.default_rng(seed)
        self._scene_seed = seed
        self.spec = ActionSpec()
        self.scale = np.array([self.spec.max_angular_speed] * 3 + [self.spec.max_linear_speed] * 3)
        self._load_task(suite, task_index, render, ex.hard_reset)
        self.env.reset()
        flange, _ = gripper_geom(self.env)
        self.chain = build_chain("panda", flange)
        self.servo = TwistServo(self.chain, self.spec, JOINT_ACTION_SCALE, iters=ex.servo_iters)
        self.scene = Scene(self.env.env)
        self.start = (start or StartNoise()).as_dict()
        self.episode = 0
        self.t = 0
        self.raw = None
        self._snap = None
        self._settled: dict[int, np.ndarray] = {}

    @property
    def label(self) -> str:
        return f"{self.suite}[{self.ti}]"

    def _load_task(self, suite: str, task_index: int, render: bool, hard_reset: bool) -> None:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        from .task_spec import parse
        bm = benchmark.get_benchmark_dict()[suite]()
        task = bm.get_task(task_index)
        self.language = task.language
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        self.task_spec = parse(bddl, suite)
        kw = (dict(camera_heights=CAMERA_PX, camera_widths=CAMERA_PX) if render else
              dict(use_camera_obs=False, has_offscreen_renderer=False))
        self.env = OffScreenRenderEnv(bddl_file_name=bddl, robots=["Panda"], gripper_types="PandaGripper",
                                      controller="JOINT_POSITION", hard_reset=hard_reset, **kw)
        # LIBERO pickles its init states; torch>=2.6 refuses that unless told otherwise
        _load = torch.load
        torch.load = lambda *a, **k: _load(*a, **{**k, "weights_only": False})
        try:
            self.init_states = bm.get_task_init_states(task_index)
        finally:
            torch.load = _load

    # -- episode --------------------------------------------------------------------
    def _reset_scene(self, k: int) -> None:
        """env.reset(), with LIBERO's fixture placement for init state k made repeatable.

        Every reset re-samples fixture poses (a cabinet within +-1 cm of its region) from
        numpy's global RNG, and writes them to model.body_pos -- which set_init_state does
        not restore. Unseeded, two resets moved the cabinet 12 mm apart, so identical runs
        diverged, and a state settled with the fixture in one place was replayed with it in
        another: an object resting ON the fixture started embedded in it or floating.
        """
        np.random.seed((self._scene_seed * 1009 + k) % SEED_MOD)
        self.env.reset()

    def reset(self, init_index: int | None = None):
        self.episode += 1
        k = int(self.rng.integers(len(self.init_states))) if init_index is None else init_index
        if k not in self._settled:
            self._settled[k] = self._settled_state(k)
        self._reset_scene(k)
        self.env.set_init_state(self._settled[k])
        if any(v > 0 for v in self.start.values()):
            self._randomize_start()
        self.servo.reset(np.asarray(self.observe()["robot0_joint_pos"]))
        self.t = 0
        self.raw = self.observe()
        return self.raw

    def _settled_state(self, k: int) -> np.ndarray:
        """LIBERO's initial state k, held until the objects in it are at rest."""
        from .libero_env import remap_init_state
        self._reset_scene(k)
        self.env.set_init_state(remap_init_state(self.init_states[k], self.env.sim))
        self._settle(INITIAL_SETTLE)
        for _ in range(SETTLE_ROUNDS):                  # settle until at rest, not a fixed count
            if self._max_object_speed() < REST_SPEED:
                break
            self._settle(SETTLE_CHUNK)
        return np.asarray(self.env.sim.get_state().flatten()).copy()

    def step(self, action: np.ndarray):
        """action: 6 normalised body-twist + 1 gripper (command or target aperture)."""
        a = np.clip(np.asarray(action, np.float64), -1, 1)
        o = self.observe()
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
        """Tool pose and gripper state, once per control step. Object poses come from
        Scene, by name."""
        key = (self.episode, self.t)
        if self._snap is not None and self._snap[0] == key:
            return self._snap[1]
        from .kin_np import NpChain
        from .kin_np import fk as fk_np
        raw = self.raw or self.observe()
        T = fk_np(NpChain.of(self.chain), np.asarray(raw["robot0_joint_pos"], float))[0]
        gq, gv = raw["robot0_gripper_qpos"], raw.get("robot0_gripper_qvel", np.zeros(2))
        snap = dict(R_tool=T[:3, :3], p_tool=T[:3, 3], aperture=float(gq[0] - gq[1]),
                    aperture_rate=float(gv[0] - gv[1]), success=bool(self.env.env._check_success()))
        self._snap = (key, snap)
        return snap

    def student_state(self) -> np.ndarray:
        from .state import tool_state
        raw = self.raw or self.observe()
        q = torch.tensor(np.asarray(raw["robot0_joint_pos"]), dtype=torch.float64)[None]
        g = raw["robot0_gripper_qpos"]
        return tool_state(self.chain, q, torch.tensor([float(g[0] - g[1])], dtype=torch.float64))[0].float().numpy()

    def images(self):
        raw = self.raw or self.observe()
        return raw["agentview_image"], raw["robot0_eye_in_hand_image"]

    def close(self):
        self.env.env.close()
