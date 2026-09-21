"""What both LIBERO environments do the same way: hold the arm, find a start, reach in.

PrivilegedEnv (teacher_env.py, the libero_spatial pipeline) and TaskEnv (task_env.py,
any task) grew the same settling hold, the same object-speed check, the same contact
test and the same start-pose sampler by copying; they live here once. A subclass
provides `env` (the OffScreenRenderEnv), `chain`, `kp`, `rng`, `start`, `settle_steps`
and a `label` for error messages.

It is also the one place that reaches into robosuite's private attributes
(`_ref_joint_pos_indexes`, `_get_observations`), so that when robosuite changes, one
file does.
"""
from __future__ import annotations

import numpy as np
import torch

from . import contacts
from .kinematics import fk

START_MIN_TOOL_Z = 0.08       # m above the base: keep a randomised start clear of the table
START_MIN_SIGMA = 0.02        # a start this close to singular is rejected


def rot(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation about `axis` by `angle`."""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


class SimArm:
    label = "env"

    # -- robosuite internals, touched only here ---------------------------------------
    @property
    def robot(self):
        return self.env.env.robots[0]

    @property
    def joint_indexes(self):
        return self.robot._ref_joint_pos_indexes

    @property
    def gripper_indexes(self):
        return self.robot._ref_gripper_joint_pos_indexes

    def observe(self) -> dict:
        return self.env.env._get_observations(force_update=True)

    # -- holding still ----------------------------------------------------------------
    def _gains(self):
        from .libero_env import set_joint_gains
        set_joint_gains(self.env, self.kp)

    def _settle(self, n: int = 3, gripper: float = -1.0):
        """Hold the arm where it is, as an ABSOLUTE joint target.

        A zero action on robosuite's joint controller is a zero DELTA, and right after
        set_init_state it does not hold: measured, the tool moves 16 mm on the first step
        and 22 mm by the third, while an absolute hold keeps it within 0.0 mm. A servo
        armed after a zero-delta settle integrates every later twist from a start 22 mm
        off the demonstrations'.
        """
        from .libero_env import JOINT_ACTION_SCALE
        hold = np.asarray(self.observe()["robot0_joint_pos"])
        for _ in range(n):
            meas = np.asarray(self.observe()["robot0_joint_pos"])
            cmd = np.zeros(self.env.env.action_dim)
            cmd[:7] = np.clip((hold - meas) / JOINT_ACTION_SCALE, -1, 1)
            cmd[-1] = gripper
            self._gains()
            self.env.step(cmd)

    def _max_object_speed(self) -> float:
        """Fastest linear speed of any free object -- the scene is at rest when it is small."""
        m, d = self.env.sim.model, self.env.sim.data
        v = 0.0
        for j in range(m.njnt):
            if int(m.jnt_type[j]) == 0 and not contacts.is_robot(m.joint_id2name(j) or ""):
                a = int(m.jnt_dofadr[j])
                v = max(v, float(np.linalg.norm(d.qvel[a:a + 3])))
        return v

    def _robot_contact(self) -> bool:
        return contacts.robot_in_contact(self.env.sim.model, self.env.sim.data)

    # -- a random start -----------------------------------------------------------------
    def _randomize_start(self, batch: int = 16, max_rounds: int = 10) -> None:
        """Move the arm to a random TOOL pose near LIBERO's start, via IK.

        Sampled in task space -- position box, yaw about vertical, tilt about a random
        horizontal axis -- because that is what the approach depends on; jittering joints
        gives a tool distribution nobody chose. The IK seed is perturbed so the elbow (the
        null space of a 7-DoF arm) varies too. A candidate is kept only if Newton
        converged, it is away from singularity, and the arm penetrates nothing.
        """
        from .ik import sigma_min, solve_ik
        sim = self.env.sim
        idx = self.joint_indexes
        q0 = sim.data.qpos[idx].copy()
        T0 = fk(self.chain, torch.tensor(q0)[None])[0].numpy()
        for _ in range(max_rounds):
            targets, seeds = self._start_candidates(T0, q0, batch)
            res = solve_ik(self.chain, torch.tensor(np.stack(targets)), torch.tensor(np.stack(seeds)),
                           lam=0.02, max_iters=200, trust=0.2)
            ok = res["converged"] & (sigma_min(self.chain, res["theta"]) > START_MIN_SIGMA)
            for k in torch.nonzero(ok).flatten().tolist():
                sim.data.qpos[idx] = res["theta"][k].numpy()
                sim.data.qvel[:] = 0.0
                sim.forward()
                if not self._robot_contact():
                    self._settle(self.settle_steps)
                    return
            sim.data.qpos[idx] = q0
            sim.forward()
        raise RuntimeError(f"{self.label}: no collision-free reachable start pose found")

    def _start_candidates(self, T0: np.ndarray, q0: np.ndarray, batch: int):
        st = self.start
        targets, seeds = [], []
        for _ in range(batch):
            dp = np.array([self.rng.uniform(-st["xy"], st["xy"]), self.rng.uniform(-st["xy"], st["xy"]),
                           self.rng.uniform(-st["z"], st["z"])])
            yaw = self.rng.uniform(-st["yaw"], st["yaw"])
            ax = self.rng.normal(size=2)
            ax = np.array([*ax / (np.linalg.norm(ax) + 1e-12), 0.0])
            tilt = self.rng.uniform(-st["tilt"], st["tilt"])
            T = T0.copy()
            T[:3, :3] = rot(np.array([0, 0, 1.0]), yaw) @ rot(ax, tilt) @ T0[:3, :3]
            T[:3, 3] = T0[:3, 3] + dp
            T[2, 3] = max(T[2, 3], START_MIN_TOOL_Z)
            targets.append(T)
            seeds.append(q0 + self.rng.normal(0, st["null"], size=len(q0)))
        return targets, seeds
