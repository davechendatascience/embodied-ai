"""Contact-GraspNet vs the analytic proposer, against ground truth.

The analytic proposer already grasps the robosuite cube correctly, so this is
not asking "which wins" on a box -- PCA is near-optimal on a box. It asks a
narrower question first: does the learned proposer, wired through this env's
frames, land on the object at all? A wrong camera convention shows up here as a
position tens of millimetres off the object, which is exactly how the same bug
presented in the robot project (contacts 61-119 mm away, blamed on a filter
threshold for a day).

Ground truth is MuJoCo's body pose, so "off by X mm" is measured, not eyeballed.

    MUJOCO_GL=egl .venv/bin/python tests/test_grasp_cgn.py
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

from rekep_libero.environment_robosuite import ReKepRobosuiteEnv  # noqa: E402
from rekep_libero.grasp import AnalyticGraspProposer  # noqa: E402
from rekep_libero import grasp_cgn  # noqa: E402
import transform_utils as T  # noqa: E402


def build_env():
    config = load_config()
    ws, ec = config["robosuite_workspace"], dict(config["env"])
    ec["bounds_min"], ec["bounds_max"] = ws["bounds_min"], ws["bounds_max"]
    ec["interpolate_pos_step_size"] = config["main"]["interpolate_pos_step_size"]
    ec["interpolate_rot_step_size"] = config["main"]["interpolate_rot_step_size"]
    return ReKepRobosuiteEnv(ec)


def report(label, result, truth, half_extent, finger_offset):
    if result is None:
        print(f"{label:22s} no grasp proposed")
        return None
    pos, quat, width = result
    # the proposer commands the ee SITE; the fingertips land finger_offset
    # beyond it along the approach, so compare the TIPS against the object
    R = T.quat2mat(quat)
    approach = R[:, 2]
    tips = pos + finger_offset * approach
    lateral = float(np.linalg.norm((tips - truth)[:2]))
    vertical = float(tips[2] - truth[2])
    down = float(np.dot(approach, [0, 0, -1]))
    print(f"{label:22s} lateral {lateral * 1000:6.1f} mm | vertical "
          f"{vertical * 1000:+6.1f} mm | width {width * 1000:5.1f} mm | "
          f"approach.down {down:+.2f}")
    return {"lateral": lateral, "vertical": vertical, "width": width, "down": down}


def main():
    if not grasp_cgn.available():
        print(f"Contact-GraspNet weights not found at {grasp_cgn.CHECKPOINT_DIR}")
        return 1

    env = build_env()
    name = next(iter(env._object_geom_ids))
    truth = np.asarray(env._object_poses()[name][0])
    points = env.object_points(name)
    half_extent = float(np.abs(points - truth)[:, :2].max())
    finger_offset = env.finger_offset()
    print(f"object     : {name}  at {np.round(truth, 4)}")
    print(f"cloud      : {len(points)} pts, half-extent {half_extent * 1000:.1f} mm")
    print(f"mask       : {int(env.object_pixel_mask(name).sum())} px")
    print(f"finger_off : {finger_offset * 1000:.1f} mm\n")

    analytic = AnalyticGraspProposer(finger_offset=finger_offset)
    a = report("analytic (PCA)",
               analytic.propose(points, env.GRASP_APPROACH_AXIS,
                                env.gripper_closing_axis_idx()),
               truth, half_extent, finger_offset)

    t0 = time.time()
    cgn = grasp_cgn.ContactGraspNetProposer(finger_offset=finger_offset)
    c = report("contact-graspnet", cgn.propose_for_object(env, name),
               truth, half_extent, finger_offset)
    elapsed = time.time() - t0
    print(f"\ncandidates : {len(cgn.last_candidates)} above score "
          f"{cgn.score_threshold} (first call {elapsed:.1f}s incl. model load)")
    if cgn.last_candidates:
        top = cgn.last_candidates[0]
        print(f"top score  : {top['score']:.3f}")

    # A grasp is on the object when the fingertips are inside its footprint.
    # Not a quality bar -- a frame bug misses by far more than this.
    tol = half_extent + 0.010
    ok = c is not None and c["lateral"] < tol
    print(f"\nverdict    : CGN {'ON' if ok else 'OFF'} target "
          f"(lateral < {tol * 1000:.0f} mm)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
