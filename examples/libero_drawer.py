"""Run xembody on LIBERO's drawer task with NO privileged information.

Everything the pipeline consumes is a sensor reading or the robot's knowledge
of itself:

  intent    the VLA's achieved end-effector poses and gripper command, from a
            recorded rollout -- the policy saw only RGB and language
  world     depth from the two cameras, backprojected with the camera's own
            intrinsics and extrinsics
  self      the robot's link positions, to remove its own body from the cloud

No object poses, no contact list, no MuJoCo collision geometry, no BDDL goal.

    MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python examples/libero_drawer.py \
        --trace /tmp/vla_jepa_traces/libero_goal_task0_ep0.jsonl
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "third_party", "LIBERO"))

from xembody import frames, keypose, world  # noqa: E402

CAMS = ("agentview", "robot0_eye_in_hand")


def load_trace(path):
    poses, grip = [], []
    for line in open(path):
        r = json.loads(line)
        if r.get("record_type") != "env_step":
            continue
        o = r["obs_after_step"]
        poses.append([*o["robot0_eef_pos"], *o["robot0_eef_quat"]])
        grip.append(r["env_gripper"])
    return np.asarray(poses), np.asarray(grip)


def build_env(suite_name, task_id, res=256):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"),
                        task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl, robots=["Panda"], camera_heights=res,
        camera_widths=res, controller="OSC_POSE", camera_depths=True,
        camera_names=list(CAMS), horizon=10000)
    np.random.seed(0)
    env.reset()
    return env, task


def camera_arrays(env, name, res):
    """Depth in metres, intrinsics, camera-to-world. All public robosuite."""
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
        get_real_depth_map,
    )

    obs = env.env._get_observations()
    raw = obs[f"{name}_depth"]
    d = get_real_depth_map(env.sim, raw)
    if d.ndim == 3:
        d = d[:, :, 0]
    d = d[::-1]                      # MuJoCo renders bottom-left origin
    K = get_camera_intrinsic_matrix(env.sim, name, res, res)
    T = get_camera_extrinsic_matrix(env.sim, name)
    return d, K, T


def robot_self_spheres(env):
    """The robot's own body: self-knowledge, not scene knowledge."""
    m, d = env.sim.model, env.sim.data
    ids = [g for g in range(m.ngeom)
           if (m.body_id2name(m.geom_bodyid[g]) or "").startswith(
               ("robot", "gripper", "mount"))]
    return d.geom_xpos[ids], m.geom_rbound[ids]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--res", type=int, default=256)
    args = ap.parse_args()

    poses, grip = load_trace(args.trace)
    print(f"trace: {len(poses)} steps from a VLA rollout (RGB + language only)")

    # ---- gate: is this task even in scope for a keypose pipeline? --------
    preh = keypose.prehensile(grip)
    print(f"\nprehensile check: gripper ever commanded closed = {preh}")
    marks = keypose.extract(poses, grip)
    print(f"keyposes: {len(marks)} ({len(poses) / len(marks):.1f}x compression)"
          f"  geometric fidelity {keypose.fidelity(poses, marks):.2f}")
    for i, why in marks:
        print(f"    step {i:>4}  {why:<8} {np.round(poses[i, :3], 4)}")

    # ---- world, from depth alone ----------------------------------------
    env, task = build_env(args.suite, args.task_id, args.res)
    print(f"\ntask: \"{task.language}\"")
    depths, Ks, Ts = [], [], []
    for name in CAMS:
        d, K, T = camera_arrays(env, name, args.res)
        depths.append(d)
        Ks.append(K)
        Ts.append(T)
    # bounds derived from the cloud, not hardcoded: the goal suite sits near
    # z=1.0 and the object suite near z=0.15, and constants for one empty the
    # world for the other
    boxes, info = world.from_depth(
        depths, Ks, Ts, robot_spheres=robot_self_spheres(env))
    print(f"world from depth: {info['points']} points -> {info['voxels']} "
          f"voxels -> {len(boxes)} obstacle boxes")

    # ---- goal, in the robot base frame ----------------------------------
    m, d = env.sim.model, env.sim.data
    b = m.body_name2id("robot0_base")
    T_wb = frames.homo(d.body_xpos[b], d.body_xmat[b].reshape(3, 3))
    i_last = marks[-1][0]
    goal = frames.goal_in_base(poses[i_last, :3], poses[i_last, 3:], T_wb)
    print(f"final keypose -> base frame: {np.round(goal[:3, 3], 4)}")
    env.close()

    print()
    if not preh:
        print("VERDICT: OUT OF SCOPE. The policy never closes the gripper on\n"
              "this task -- it opens the drawer by HOOKING it. Keyposes carry\n"
              "an object only when a grasp attaches it to the wrist, so this\n"
              "pipeline cannot execute this solution and says so BEFORE\n"
              "planning rather than after a failed rollout.")
        return 2
    print("VERDICT: in scope -- prehensile solution, keyposes are meaningful.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
