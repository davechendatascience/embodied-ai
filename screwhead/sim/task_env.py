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

POSTURE_GAIN = 0.1     # the start-posture pull (Execution.posture_start): replayed demonstrations kept their joints within
#                        0.18-0.29 rad of the recorded ones with it, against 0.28-0.44 rad with the pull to mid-range


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
                 render: bool | int = True, execution: Execution | None = None, start: StartNoise | None = None):
        from .scene import Scene
        from ..teacher.task_spec import parse
        self.suite, self.ti, self.horizon = suite, task_index, horizon
        self.seed = int(seed)          # an episode's draws are seeded from it and the
        self.rng = np.random.default_rng(seed)   # episode index, never from the stream so far
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
        self.init_index = k                               # which init state this episode drew, for provenance
        if k not in self._settled:
            self._settled[k] = self._settled_init_state(k)
        self._reset_scene(k)
        self.env.set_init_state(self._settled[k])
        self._anchor()
        if any(v > 0 for v in self.start.values()):
            self._randomize_start()
        self.servo.reset(np.asarray(self.observe()["robot0_joint_pos"]))
        if self.execution.posture_start:
            self.servo.posture, self.servo.posture_gain = self.servo.ref.copy(), POSTURE_GAIN
        self.t = 0
        self.raw = self.observe()
        return self.raw

    def place_stored(self, state: np.ndarray, fixtures: dict, fingerprint: str | None = None) -> bool:
        """Start an episode from a stored start (BRN-random-starts-test-set): the stored fixture poses written, then
        the stored state, then the forward pass, execution memory anchored there. False if the model then differs
        from the stored fingerprint or fixture poses -- the episode is not to be scored."""
        from .task_env_place import write_fixtures
        self.episode += 1
        self.init_index = -1
        self._reset_scene(0)
        m = self.env.sim.model._model
        write_fixtures(m, fixtures)
        self.env.set_init_state(np.asarray(state, float))
        self.env.sim.forward()
        self._anchor()
        self.servo.reset(np.asarray(self.observe()["robot0_joint_pos"]))
        if self.execution.posture_start:
            self.servo.posture, self.servo.posture_gain = self.servo.ref.copy(), POSTURE_GAIN
        self.t = 0
        self.raw = self.observe()
        same_fixtures = all(np.array_equal(m.body_pos[m.body(n).id], p) and np.array_equal(m.body_quat[m.body(n).id], q)
                            for n, (p, q) in fixtures.items())
        return same_fixtures and (fingerprint is None or self.model_fingerprint() == fingerprint)

    def skip_episode(self) -> None:
        """Advance the episode stream past one episode without simulating it: an episode's draws
        depend on its index and its init state alone, so the episodes after it are unchanged."""
        self.episode += 1
        self.rng.integers(len(self.init_states))

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
