"""Damped-least-squares IK against a raw MuJoCo sim, with no env wrapper.

`MujocoIKSolver` in `environment_libero.py` does the same maths but reads
`env.eef_site_id`, `env.arm_qpos_idx`, `env.arm_joint_limits` and
`env.eef_rot_offset` off a `ReKepLiberoEnv`. The VLA rollout harness drives a
bare `OffScreenRenderEnv`, so it needs the same solve without that wrapper.
The iteration is deliberately identical -- if one is ever fixed, fix both.

Why this exists at all: a UR5e cannot be put where a Panda starts by copying
joint angles, because there is no meaningful map from seven joints to six. It
has to be solved for. Without it the UR5e begins every episode **181 mm** from
the Panda's start end-effector pose, and a closed-loop policy trained from the
Panda's start is being asked a different question from step one.
"""

import numpy as np


def solve_ik(sim, site_id, arm_qpos_idx, target_pos, target_mat=None,
             q_init=None, joint_limits=None, damping=0.05,
             position_tolerance=1e-3, orientation_tolerance=0.02,
             max_iterations=200):
    """Joint angles putting `site_id` at `target_pos` (and optionally orientation).

    Leaves the simulator exactly as it found it -- the search runs on a scratch
    copy of `qpos`, which matters because callers are mid-episode.
    """
    import mujoco

    model, data = sim.model._model, sim.data._data
    arm_qpos_idx = np.asarray(arm_qpos_idx, dtype=int)
    q = (np.array(data.qpos[arm_qpos_idx], dtype=float)
         if q_init is None else np.array(q_init, dtype=float))

    backup = data.qpos.copy()
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    pos_err, rot_err = np.inf, np.inf
    used = 0
    try:
        for used in range(1, max_iterations + 1):
            data.qpos[arm_qpos_idx] = q
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)

            cur_pos = data.site_xpos[site_id].copy()
            cur_mat = data.site_xmat[site_id].reshape(3, 3).copy()
            pos_err_vec = np.asarray(target_pos) - cur_pos
            if target_mat is None:
                rot_err_vec = np.zeros(3)
            else:
                rot_err_vec = 0.5 * (
                    np.cross(cur_mat[:, 0], target_mat[:, 0])
                    + np.cross(cur_mat[:, 1], target_mat[:, 1])
                    + np.cross(cur_mat[:, 2], target_mat[:, 2])
                )
            pos_err = float(np.linalg.norm(pos_err_vec))
            rot_err = float(np.linalg.norm(rot_err_vec))
            if pos_err < position_tolerance and rot_err < orientation_tolerance:
                break

            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
            J = np.vstack([jacp[:, arm_qpos_idx], jacr[:, arm_qpos_idx]])
            err = np.concatenate([pos_err_vec, rot_err_vec])
            JJt = J @ J.T + (damping ** 2) * np.eye(6)
            q = q + J.T @ np.linalg.solve(JJt, err)
            if joint_limits is not None:
                q = np.clip(q, joint_limits[:, 0], joint_limits[:, 1])
    finally:
        data.qpos[:] = backup
        mujoco.mj_forward(model, data)

    return q, pos_err, rot_err, used


def place_arm_at(env, target_pos, target_mat=None, settle=0):
    """Move this env's arm so its grip site sits at `target_pos`.

    Writes the solved joints into the LIVE sim. Returns the achieved position
    error in metres so the caller can refuse to proceed on a bad solve rather
    than run an episode from the wrong place.
    """
    import mujoco

    robot = env.env.robots[0] if hasattr(env, "env") else env.robots[0]
    sim = env.sim
    site_id = sim.model.site_name2id(robot.controller.eef_name)
    arm_qpos_idx = np.asarray(robot._ref_joint_pos_indexes, dtype=int)
    limits = sim.model.jnt_range[robot._ref_joint_indexes]

    q, pos_err, rot_err, iters = solve_ik(
        sim, site_id, arm_qpos_idx, target_pos, target_mat,
        joint_limits=limits)

    sim.data.qpos[arm_qpos_idx] = q
    sim.data.qvel[np.asarray(robot._ref_joint_vel_indexes, dtype=int)] = 0.0
    mujoco.mj_forward(sim.model._model, sim.data._data)

    # The controller caches a goal from the pose it last saw; without this it
    # drives straight back to where the arm used to be on the first step.
    robot.controller.update(force=True)
    robot.controller.reset_goal()

    for _ in range(settle):
        env.step(np.concatenate([np.zeros(6), [-1.0]]))
    return pos_err, rot_err, iters
