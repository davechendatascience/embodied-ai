"""Where the learned proposer earns its dependency: objects PCA cannot grasp.

On the robosuite cube the two are close and PCA is slightly more accurate --
a box is exactly the geometry a principal axis describes. That comparison
cannot justify a 2 GB model.

This runs the case that can. LIBERO's tableware is 107-178 mm across, well past
the Panda's 68 mm opening, so `AnalyticGraspProposer` measures the full span,
reports it does not fit, and returns nothing. But a bowl does not have to be
grasped across its diameter -- the RIM fits in 68 mm easily. PCA cannot express
that grasp, because "narrowest horizontal axis of the whole cloud" is a
statement about the object's extent, not about where on it to pinch.

So the question here is not "who is more accurate" but "who proposes a grasp at
all", and then whether the one it proposes is reachable and fits the gripper.

    MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/compare_grasp_proposers.py
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

from rekep_libero.environment_libero import ReKepLiberoEnv  # noqa: E402
from rekep_libero.grasp import AnalyticGraspProposer  # noqa: E402
from rekep_libero import grasp_cgn  # noqa: E402
import transform_utils as T  # noqa: E402


def build_env(config, task_id):
    ec = dict(config["env"])
    ec["bounds_min"], ec["bounds_max"] = config["main"]["bounds_min"], config["main"]["bounds_max"]
    ec["interpolate_pos_step_size"] = config["main"]["interpolate_pos_step_size"]
    ec["interpolate_rot_step_size"] = config["main"]["interpolate_rot_step_size"]
    lib = config["libero"]
    return ReKepLiberoEnv(ec, task_suite=lib["task_suite"], task_id=task_id,
                          robot=lib["robot"], resolution=lib["resolution"])


def main():
    if not grasp_cgn.available():
        print(f"no Contact-GraspNet checkpoint at {grasp_cgn.CHECKPOINT_DIR}")
        return 1

    config = load_config()
    task_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    env = build_env(config, task_id)
    finger_offset = env.finger_offset()
    analytic = AnalyticGraspProposer(finger_offset=finger_offset)
    cgn = grasp_cgn.ContactGraspNetProposer(finger_offset=finger_offset)
    grasp_cgn.estimator()   # keep the load out of the per-object timing

    print(f"{config['libero']['task_suite']} task {task_id}: {env.instruction}")
    print(f"gripper opening {analytic.max_opening * 1000:.0f} mm "
          f"(usable {analytic.max_opening * analytic.clearance * 1000:.0f} mm)\n")
    print(f"{'object':30s} {'PCA span':>9s} {'PCA':>6s} {'CGN best':>9s} {'CGN':>6s} "
          f"{'cands':>6s} {'s':>5s}")

    wins = losses = 0
    for name in env._object_geom_ids:
        points = env.object_points(name)
        if len(points) < 10:
            continue
        _pos, _quat, span = analytic.propose(points, env.GRASP_APPROACH_AXIS,
                                             env.gripper_closing_axis_idx())
        pca_ok = analytic.fits(span)

        t0 = time.time()
        result = cgn.propose_for_object(env, name)
        elapsed = time.time() - t0
        cgn_width = result[2] if result else float("nan")

        print(f"{name:30s} {span * 1000:8.1f}mm {'fits' if pca_ok else '  --':>6s} "
              f"{cgn_width * 1000:8.1f}mm {'fits' if result else '  --':>6s} "
              f"{len(cgn.last_candidates):6d} {elapsed:5.1f}")
        if result and not pca_ok:
            wins += 1
        elif pca_ok and not result:
            losses += 1

    print(f"\nobjects CGN grasps that PCA cannot : {wins}")
    print(f"objects PCA grasps that CGN cannot : {losses}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
