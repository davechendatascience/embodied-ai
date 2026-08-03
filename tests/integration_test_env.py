"""Exercise ReKepLiberoEnv against a live LIBERO sim, one method at a time.

Staged deliberately: get_sdf_voxels, _move_to_waypoint and MujocoIKSolver have
never touched a running env, so each stage prints and asserts on its own so a
failure says which method broke rather than "the adapter doesn't work".

    MUJOCO_GL=egl .venv/bin/python tests/integration_test_env.py
"""

import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from rekep_libero import add_rekep_to_path  # noqa: E402
from rekep_libero.config import load_config  # noqa: E402

add_rekep_to_path()

import transform_utils as T  # noqa: E402
from rekep_libero.environment_libero import ReKepLiberoEnv, AGENTVIEW  # noqa: E402

PASS, FAIL = [], []


def stage(name):
    def deco(fn):
        print(f"\n=== {name} ===")
        try:
            fn()
            PASS.append(name)
            print(f"  PASS")
        except Exception as exc:  # noqa: BLE001 - we want every stage to report
            FAIL.append((name, repr(exc)))
            print(f"  FAIL {type(exc).__name__}: {exc}")
        return fn
    return deco


def main():
    config = load_config()
    lib = config["libero"]
    ws = config["workspace"]
    bounds_min, bounds_max = np.array(ws["bounds_min"]), np.array(ws["bounds_max"])

    env_config = dict(config["env"])
    env_config["bounds_min"], env_config["bounds_max"] = ws["bounds_min"], ws["bounds_max"]
    env_config["interpolate_pos_step_size"] = config["main"]["interpolate_pos_step_size"]
    env_config["interpolate_rot_step_size"] = config["main"]["interpolate_rot_step_size"]

    print("=== construct ===")
    t = time.time()
    env = ReKepLiberoEnv(env_config, task_suite=lib["task_suite"], task_id=lib["task_id"],
                         robot=lib["robot"], resolution=lib["resolution"])
    print(f"  built in {time.time()-t:.1f}s | task: {env.instruction}")
    print(f"  reset_joint_pos {np.round(env.reset_joint_pos, 3)}")

    @stage("get_cam_obs")
    def _():
        obs = env.get_cam_obs()
        cam = obs[AGENTVIEW]
        print(f"  rgb {cam['rgb'].shape} seg {cam['seg'].shape} points {cam['points'].shape}")
        print(f"  seg instance ids: {np.unique(cam['seg'])}")
        pts = cam["points"].reshape(-1, 3)
        inside = np.all((pts >= bounds_min) & (pts <= bounds_max), axis=1)
        print(f"  points inside workspace: {inside.sum()}/{len(pts)} ({100*inside.mean():.1f}%)")
        print(f"  point z range {pts[:,2].min():.3f}..{pts[:,2].max():.3f} (table should be ~0.97)")
        assert inside.sum() > 0, "no backprojected points land inside the workspace"

    @stage("get_sdf_voxels")
    def _():
        t0 = time.time()
        sdf = env.get_sdf_voxels(config["main"]["sdf_voxel_size"])
        print(f"  shape {sdf.shape} in {time.time()-t0:.2f}s")
        print(f"  range {sdf.min():.3f}..{sdf.max():.3f} | occupied voxels {(sdf < 0).sum()}")
        assert np.all(np.isfinite(sdf)), "SDF has non-finite values"
        assert (sdf < 0).any(), "SDF has no occupied region — depth backprojection likely wrong"

    @stage("robot state")
    def _():
        pose, q = env.get_ee_pose(), env.get_arm_joint_postions()
        print(f"  ee pose {np.round(pose, 3)}")
        print(f"  joints  {np.round(q, 3)}")
        assert pose.shape == (7,) and q.shape == (7,)

    @stage("IK identity round-trip")
    def _():
        """Solving for the pose the arm is already at must return its own joints."""
        pose = env.get_ee_pose()
        target = T.pose2mat([pose[:3], pose[3:]])
        res = env.ik_solver.solve(target, initial_joint_pos=env.get_arm_joint_postions())
        drift = np.linalg.norm(res.cspace_position - env.get_arm_joint_postions())
        print(f"  success={res.success} pos_err={res.position_error:.5f} "
              f"descents={res.num_descents} joint drift={drift:.5f}")
        assert res.success, "IK failed on the arm's own current pose"
        assert res.position_error < 0.01, f"identity IK position error too large: {res.position_error}"
        # Guards the site-vs-body frame bug: robosuite reports eef position from
        # the grip site but eef orientation from the hand body (90 deg apart on
        # the Panda). Without eef_rot_offset the solver "corrects" a phantom
        # rotation and drifts ~1.5 rad here while still reporting success.
        assert drift < 0.05, f"identity IK moved the joints by {drift:.4f} rad — frame mismatch?"

    @stage("IK does not disturb the sim")
    def _():
        before = env.sim.data.qpos.copy()
        pose = env.get_ee_pose()
        target = T.pose2mat([pose[:3] + np.array([0.05, 0.05, -0.05]), pose[3:]])
        env.ik_solver.solve(target, initial_joint_pos=env.get_arm_joint_postions())
        drift = np.abs(env.sim.data.qpos - before).max()
        print(f"  max qpos drift after IK probing: {drift:.3e}")
        assert drift < 1e-9, "IK solver mutated the live sim state"

    @stage("register/track keypoints")
    def _():
        cam = env.get_cam_obs()[AGENTVIEW]
        poses = env._object_poses()
        print(f"  objects found: {list(poses)}")
        assert poses, "no object poses discovered"
        assert not any("_to_robot0" in n for n in poses), "relative-offset keys leaked into object poses"
        # place a keypoint right on each object
        kps = np.array([p for p, _ in poses.values()])
        env.register_keypoints(kps)
        tracked = env.get_keypoint_positions()
        err = np.abs(tracked - kps).max()
        print(f"  registered {len(kps)} keypoints, recovery error {err:.6f}")
        assert err < 1e-6, "keypoint registration is not identity at registration time"
        # step the sim, keypoints must follow their objects
        for _ in range(10):
            env._step(np.concatenate([np.zeros(6), [env.last_gripper_action]]))
        moved = env.get_keypoint_positions()
        print(f"  after 10 steps, max keypoint shift {np.abs(moved - kps).max():.6f}")
        assert np.all(np.isfinite(moved))

    @stage("execute_action move")
    def _():
        start = env.get_ee_pose()
        target = start.copy()
        target[:3] += np.array([0.03, 0.03, -0.05])
        t0 = time.time()
        pos_err, rot_err = env.execute_action(np.concatenate([target, [env.get_gripper_null_action()]]))
        print(f"  moved in {time.time()-t0:.1f}s ({env.step_counter} sim steps)")
        print(f"  requested {np.round(target[:3],3)} reached {np.round(env.get_ee_pos(),3)}")
        print(f"  pos_err={pos_err:.4f}m rot_err={np.rad2deg(rot_err):.2f}deg")
        assert pos_err < 0.05, f"did not converge to target: {pos_err:.4f}m"

    @stage("gripper")
    def _():
        env.close_gripper()
        closed = np.asarray(env._last_obs["robot0_gripper_qpos"])
        env.open_gripper()
        opened = np.asarray(env._last_obs["robot0_gripper_qpos"])
        print(f"  qpos closed {np.round(closed,4)} -> open {np.round(opened,4)}")
        assert not np.allclose(closed, opened, atol=1e-3), "gripper did not actuate"

    print("\n" + "=" * 60)
    print(f"PASS {len(PASS)}/{len(PASS)+len(FAIL)}: {', '.join(PASS)}")
    for name, err in FAIL:
        print(f"FAIL {name}: {err}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
