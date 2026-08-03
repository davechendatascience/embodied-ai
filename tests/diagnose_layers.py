"""Isolate which layer is failing: vision, VLM grounding, or grasp mechanics.

Re-running the whole pipeline after each fix confounds these. LIBERO gives us
ground-truth object poses, so each layer can be scored on its own:

  L1 vision    — does the proposer put a keypoint on the target object at all?
  L2 grounding — does the VLM pick THAT keypoint when asked for the object?
  L3 mechanics — driven to the ground-truth keypoint, does the grasp close on
                 the object? (VLM bypassed entirely)

L3 is the one that matters most here: if mechanics fail on a known-correct
keypoint, no amount of VLM quality helps.

    MUJOCO_GL=egl .venv/bin/python tests/diagnose_layers.py
"""

import contextlib
import io
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from rekep_libero import add_rekep_to_path  # noqa: E402
from rekep_libero.config import load_config  # noqa: E402

add_rekep_to_path()

from rekep_libero.environment_libero import ReKepLiberoEnv, AGENTVIEW  # noqa: E402
from keypoint_proposal import KeypointProposer  # noqa: E402

TARGET = sys.argv[1] if len(sys.argv) > 1 else "akita_black_bowl_1"


def build_env(config):
    ws = config["workspace"]
    ec = dict(config["env"])
    ec["bounds_min"], ec["bounds_max"] = ws["bounds_min"], ws["bounds_max"]
    ec["interpolate_pos_step_size"] = config["main"]["interpolate_pos_step_size"]
    ec["interpolate_rot_step_size"] = config["main"]["interpolate_rot_step_size"]
    return ReKepLiberoEnv(ec, task_id=config["libero"]["task_id"])


def main():
    config = load_config()
    env = build_env(config)
    objects = {k: v[0] for k, v in env._object_poses().items()}

    # ---------------- L1: vision ----------------
    print("=== L1  vision: does a keypoint land on the target object? ===")
    cam = env.get_cam_obs()[AGENTVIEW]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        proposer = KeypointProposer(config["keypoint_proposer"])
        keypoints, projected = proposer.get_keypoints(cam["rgb"], cam["points"], cam["seg"])

    dists = np.linalg.norm(keypoints - objects[TARGET], axis=1)
    gt_idx = int(np.argmin(dists))
    print(f"  {len(keypoints)} keypoints proposed")
    print(f"  target {TARGET} at {np.round(objects[TARGET], 3)}")
    print(f"  nearest keypoint is #{gt_idx} at {np.round(keypoints[gt_idx], 3)}, {dists[gt_idx]:.3f} m away")
    l1 = dists[gt_idx] < 0.08
    print(f"  L1 {'PASS' if l1 else 'FAIL'} — {'a usable keypoint exists' if l1 else 'no keypoint near the object'}")

    # ---------------- L3: grasp mechanics (VLM bypassed) ----------------
    # Run before L2 because it is the decisive one and needs no VLM call.
    print("\n=== L3  mechanics: grasp the ground-truth keypoint (no VLM) ===")
    grasp_depth = config["main"]["grasp_depth"]
    target_kp = keypoints[gt_idx]
    pose = env.get_ee_pose()
    approach = env.GRASP_APPROACH_AXIS

    import transform_utils as T
    R = T.quat2mat(pose[3:])
    hover = target_kp + R @ (approach * -grasp_depth / 2.0)
    env.execute_action(np.concatenate([hover, pose[3:], [env.get_gripper_null_action()]]), precise=True)
    print(f"  hover  -> requested {np.round(hover,3)}  reached {np.round(env.get_ee_pos(),3)}")

    grasp = env.get_ee_pose().copy()
    grasp[:3] += T.quat2mat(grasp[3:]) @ (approach * grasp_depth)
    env.execute_action(np.concatenate([grasp, [env.get_gripper_close_action()]]), precise=True)
    print(f"  grasp  -> requested {np.round(grasp[:3],3)}  reached {np.round(env.get_ee_pos(),3)}")

    contacts = env._contacting_objects()
    qpos = np.asarray(env._last_obs["robot0_gripper_qpos"])
    print(f"  gripper qpos {np.round(qpos,4)}  (≈0.0012 both = closed on nothing)")
    print(f"  contacts: {contacts or '{}'}")
    l3 = TARGET in contacts
    print(f"  L3 {'PASS' if l3 else 'FAIL'} — {'object grasped' if l3 else 'nothing in the gripper'}")

    # lift test: does it stay held?
    if l3:
        lift = env.get_ee_pose().copy()
        lift[2] += 0.10
        env.execute_action(np.concatenate([lift, [env.get_gripper_null_action()]]), precise=False)
        held = TARGET in env._contacting_objects()
        dz = env._object_poses()[TARGET][0][2] - objects[TARGET][2]
        print(f"  lift  -> object rose {dz:+.3f} m, still held: {held}")

    print("\n" + "=" * 62)
    print(f"L1 vision    : {'PASS' if l1 else 'FAIL'}")
    print(f"L3 mechanics : {'PASS' if l3 else 'FAIL'}   <- VLM plays no part in this")
    print()
    if l1 and not l3:
        print("=> Mechanics are the blocker. VLM quality is irrelevant until this passes.")
    elif l1 and l3:
        print("=> Vision and mechanics both work. Any remaining failure is VLM grounding")
        print(f"   (does the VLM pick keypoint #{gt_idx} when asked for the bowl?).")
    else:
        print("=> Vision is the blocker: no keypoint lands on the target.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
