"""Saving and restoring a TaskEnv's whole execution state, so that a rollout from a restored copy
predicts the executed steps exactly (BRN-execution-state-restore).

The state is what the lean control period (SimArm._advance) reads between periods: MuJoCo's
integration state; the joint controller's ramp (start, goal, k) and the cache
robosuite refreshes only on its new_update flag (joint_pos, joint_vel, mass matrix -- copied
with its memory layout, since robosuite's is Fortran-ordered and np.dot rounds differently on a
C-ordered copy -- and the flag); the twist servo's joint reference, pose reference and
rate-limiter memory; the gripper's finger target; and the episode's step counter. The
controller's goal and robosuite's own clock are not saved: set_goal overwrites the goal at a
period's first substep, and the clock is read only by env.step, which the lean period never calls.
Every restore also copies the model's body poses, which LIBERO re-samples for the fixtures at
every reset (AXM-libero-resamples-fixtures): a state is restorable into any episode of any
instance of the task. State a caller keeps between periods -- the verdicts' release watch
(verdicts.Watch.fork), a teacher's episode-start reference -- is the caller's to carry.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

_INTEGRATION = mujoco.mjtState.mjSTATE_INTEGRATION


@dataclass(frozen=True)
class ExecState:
    mj: np.ndarray
    ramp: tuple[np.ndarray, np.ndarray, int]
    cache: tuple[np.ndarray, np.ndarray, np.ndarray, bool]
    servo: tuple[np.ndarray | None, np.ndarray | None, np.ndarray]
    finger_target: np.ndarray
    t: int
    body_pos: np.ndarray
    body_quat: np.ndarray


def save(te) -> ExecState:
    m, d = te.scene.raw()
    mj = np.empty(mujoco.mj_stateSize(m, _INTEGRATION))
    mujoco.mj_getState(m, d, mj, _INTEGRATION)
    c = te.robot.controller
    ramp = c.interpolator
    s = te.servo
    return ExecState(
        mj=mj,
        ramp=(ramp.start.copy(), ramp.goal.copy(), int(ramp.k)),
        cache=(np.array(c.joint_pos, copy=True), np.array(c.joint_vel, copy=True),
               np.array(c.mass_matrix, order="K", copy=True), bool(c.new_update)),
        servo=(None if s.ref is None else s.ref.copy(), None if s.T_ref is None else s.T_ref.copy(), s.V_prev.copy()),
        finger_target=np.array(te.robot.gripper.current_action, float, copy=True),
        t=int(te.t),
        body_pos=m.body_pos.copy(),
        body_quat=m.body_quat.copy(),
    )


def restore(te, st: ExecState) -> None:
    """Put `st` back into `te` (any episode of any instance of the task it was saved from),
    then forward the model and re-read the observation."""
    m, d = te.scene.raw()
    m.body_pos[:] = st.body_pos
    m.body_quat[:] = st.body_quat
    mujoco.mj_setState(m, d, st.mj, _INTEGRATION)
    c = te.robot.controller
    ramp = c.interpolator
    ramp.start, ramp.goal, ramp.k = st.ramp[0].copy(), st.ramp[1].copy(), st.ramp[2]
    c.joint_pos, c.joint_vel = st.cache[0].copy(), st.cache[1].copy()
    c.mass_matrix = np.array(st.cache[2], order="K", copy=True)
    c.new_update = st.cache[3]
    s = te.servo
    s.ref = None if st.servo[0] is None else st.servo[0].copy()
    s.T_ref = None if st.servo[1] is None else st.servo[1].copy()
    s.V_prev = st.servo[2].copy()
    te.robot.gripper.current_action = st.finger_target.copy()
    te.t = st.t
    mujoco.mj_forward(m, d)
    te._snap = None      # keyed by (episode, t): a rollout may have cached the very key the episode reaches next
    te.raw = te.observe()
