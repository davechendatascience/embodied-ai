"""Is the proposed grasp bad, or is the arm failing to reach it?

The physical test (`test_grasp_proposer.py --cgn`) reports FAIL for every LIBERO
object, but two very different things are hiding under that one word:

    akita_black_bowl_1  requested y  0.260  reached y -0.068     <- never got there
    cookies_1           requested   [0.054 0.021 0.969]
                        reached     [0.052 0.016 0.971]          <- got there, held nothing

Only the second is a grasp-quality failure. This measures the grasp itself, with
no controller in the loop: put the jaws where the proposer asked, and count how
much of the object ends up BETWEEN them. A grasp that contains object points is
a good grasp that the arm failed to execute; a grasp that contains none is a bad
grasp regardless of how well the arm tracks it.

    MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/diagnose_grasp_geometry.py
"""

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from rekep_libero import add_rekep_to_path  # noqa: E402
from rekep_libero.config import load_config  # noqa: E402

add_rekep_to_path()

from rekep_libero.environment_libero import ReKepLiberoEnv  # noqa: E402
from rekep_libero.grasp import AnalyticGraspProposer  # noqa: E402
from rekep_libero import grasp_cgn  # noqa: E402
import transform_utils as T  # noqa: E402

FINGER_LENGTH = 0.054      # Panda fingertip pad length along the approach axis


def jaw_occupancy(points, pos, quat, width, approach_axis, closing_idx, finger_offset):
    """How many object points sit inside the volume the closing jaws sweep.

    The jaw volume is a box in the ee frame: `width` across the closing axis,
    FINGER_LENGTH along the approach, and generous across the third axis (the
    fingers are wide there and it is not what decides a grasp).
    """
    R = T.quat2mat(quat)
    approach = R @ approach_axis
    tips = pos + finger_offset * approach          # the proposer commands the site
    local = (points - tips) @ R                    # world -> ee frame

    approach_idx = int(np.argmax(np.abs(approach_axis)))
    third_idx = ({0, 1, 2} - {approach_idx, closing_idx}).pop()

    # along the approach the pads span from the tips BACKWARD into the gripper
    along = local[:, approach_idx] * np.sign(approach_axis[approach_idx])
    inside = (along > -FINGER_LENGTH) & (along < 0.005)
    inside &= np.abs(local[:, closing_idx]) < max(width, 0.02) / 2 + 0.005
    inside &= np.abs(local[:, third_idx]) < 0.02
    return int(inside.sum())


def main():
    if not grasp_cgn.available():
        print("no Contact-GraspNet checkpoint")
        return 1

    config = load_config()
    ec = dict(config["env"])
    ec["bounds_min"], ec["bounds_max"] = config["workspace"]["bounds_min"], config["workspace"]["bounds_max"]
    ec["interpolate_pos_step_size"] = config["main"]["interpolate_pos_step_size"]
    ec["interpolate_rot_step_size"] = config["main"]["interpolate_rot_step_size"]
    env = ReKepLiberoEnv(ec, task_id=config["libero"]["task_id"])

    off = env.finger_offset()
    closing_idx = env.gripper_closing_axis_idx()
    home = env.get_ee_pos()
    cgn = grasp_cgn.ContactGraspNetProposer(finger_offset=off)
    analytic = AnalyticGraspProposer(finger_offset=off)
    grasp_cgn.estimator()

    print(f"ee home {np.round(home, 3)}   closing axis local {'XYZ'[closing_idx]}\n")
    print(f"{'object':32s} {'proposer':9s} {'pts in jaws':>11s} {'reach dy':>9s} {'score':>6s}")

    for name in env._object_geom_ids:
        points = env.object_points(name)
        if len(points) < 10:
            continue

        result = cgn.propose_for_object(env, name)
        if result is None:
            print(f"{name:32s} {'cgn':9s} {'no grasp':>11s}")
        else:
            pos, quat, width = result
            n = jaw_occupancy(points, pos, quat, width, env.GRASP_APPROACH_AXIS,
                              closing_idx, off)
            score = cgn.last_candidates[0]["score"] if cgn.last_candidates else float("nan")
            print(f"{name:32s} {'cgn':9s} {n:8d}/{len(points):<4d} "
                  f"{pos[1] - home[1]:+8.3f}m {score:6.3f}")

        pos, quat, width = analytic.propose(points, env.GRASP_APPROACH_AXIS, closing_idx)
        if analytic.fits(width):
            n = jaw_occupancy(points, pos, quat, width, env.GRASP_APPROACH_AXIS,
                              closing_idx, off)
            print(f"{'':32s} {'analytic':9s} {n:8d}/{len(points):<4d} "
                  f"{pos[1] - home[1]:+8.3f}m")
        else:
            print(f"{'':32s} {'analytic':9s} {'too wide':>11s}")

    print("\npts in jaws > 0 means the grasp CONTAINS object geometry, so a FAIL in")
    print("the physical test is an execution problem, not a grasp-quality problem.")
    print("'reach dy' is how far the grasp sits from the ee home in y; LIBERO objects")
    print("near y=+0.25 were not reached at all.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
