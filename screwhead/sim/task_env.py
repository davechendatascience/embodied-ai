"""Any LIBERO task, executed through the same servos as libero_spatial.

PrivilegedEnv (screwhead/sim/teacher_env.py) is tied to one scene: the akita bowl and the
plate are module constants, the layout sampler switches on task index, and progress is
measured against bowl geometry. This environment takes a task specification instead
(screwhead/teacher/task_spec.py) and reads what it needs from the scene (screwhead/sim/scene.py),
so the other 120 LIBERO tasks are reachable without new per-task code.

What is kept identical to the spatial pipeline, because gate 2 was measured with it (the
shared parts live in screwhead/sim/sim_arm.py):
  - JOINT_POSITION control at kp=4000 with absolute holds during settling,
  - the arm driven by TwistServo (body twist integrated on SE(3) to a joint reference),
  - the gripper driven by GripperServo from a target aperture,
  - start-pose randomisation in task space via IK, rejecting poses in collision.

Success is LIBERO's own predicate check, so no reward or progress function is reimplemented.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .sim_arm import Execution, SimArm

__all__ = ["Execution", "StartNoise", "TaskEnv"]


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
        from .scene import Scene
        from ..teacher.task_spec import parse
        self.suite, self.ti, self.horizon = suite, task_index, horizon
        self.rng = np.random.default_rng(seed)
        self._scene_seed = seed
        self.task_spec = parse(self._open(suite, task_index, render, execution or Execution()), suite)
        self.scene = Scene(self.env.env)
        self.start = (start or StartNoise()).as_dict()
        self.episode = 0
        self.raw = None
        self._snap = None
        self._settled: dict[int, np.ndarray] = {}

    @property
    def label(self) -> str:
        return f"{self.suite}[{self.ti}]"

    # -- episode --------------------------------------------------------------------
    def reset(self, init_index: int | None = None):
        self.episode += 1
        k = int(self.rng.integers(len(self.init_states))) if init_index is None else init_index
        if k not in self._settled:
            self._settled[k] = self._settled_init_state(k)
        self._reset_scene(k)
        self.env.set_init_state(self._settled[k])
        if any(v > 0 for v in self.start.values()):
            self._randomize_start()
        self.servo.reset(np.asarray(self.observe()["robot0_joint_pos"]))
        self.t = 0
        self.raw = self.observe()
        return self.raw

    def step(self, action: np.ndarray):
        """action: 6 normalised body-twist + 1 gripper channel (SimArm.execute)."""
        raw, done = self.execute(action)
        info = {"success": done, "truncated": self.t >= self.horizon}
        return raw, float(done), done or info["truncated"], info

    # -- what the teacher and the student read --------------------------------------
    def snapshot(self) -> dict:
        """Tool pose and gripper state, once per control step. Object poses come from
        Scene, by name."""
        key = (self.episode, self.t)
        if self._snap is not None and self._snap[0] == key:
            return self._snap[1]
        snap = dict(self.tool_state(), success=self.success())
        self._snap = (key, snap)
        return snap

    def student_state(self) -> np.ndarray:
        from ..geometry.state import tool_state
        raw = self.raw or self.observe()
        q = torch.tensor(np.asarray(raw["robot0_joint_pos"]), dtype=torch.float64)[None]
        g = raw["robot0_gripper_qpos"]
        return tool_state(self.chain, q, torch.tensor([float(g[0] - g[1])], dtype=torch.float64))[0].float().numpy()

    def images(self):
        raw = self.raw or self.observe()
        return raw["agentview_image"], raw["robot0_eye_in_hand_image"]

    def close(self):
        self.env.env.close()
