"""Measure the real LIBERO workspace so configs/rekep_libero.yaml isn't guesswork.

The solver bounds decide where subgoals may be placed and how big the SDF grid
is, so getting them from the running sim beats copying OmniGibson's numbers.

    MUJOCO_GL=egl .venv/bin/python scripts/measure_workspace.py
"""

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from rekep_libero import add_rekep_to_path  # noqa: E402

add_rekep_to_path()

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402


def main(suite_name="libero_spatial", n_tasks=3):
    suite = benchmark.get_benchmark_dict()[suite_name]()
    ee_pts, obj_pts = [], []

    for task_id in range(min(n_tasks, suite.n_tasks)):
        task = suite.get_task(task_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(
            bddl_file_name=bddl, robots=["Panda"], camera_heights=128, camera_widths=128,
            controller="OSC_POSE", camera_depths=False, camera_names=["agentview"],
        )
        obs = env.reset()
        ee_pts.append(np.asarray(obs["robot0_eef_pos"]))
        for key in obs:
            # `<obj>_to_robot0_eef_pos` is an ee-relative offset, not a world
            # position — including it drags the measured bounds far off.
            if (key.endswith("_pos") and not key.startswith("robot")
                    and "_to_robot0" not in key and np.asarray(obs[key]).shape == (3,)):
                obj_pts.append(np.asarray(obs[key]))
        print(f"  task {task_id}: {task.language[:60]}")
        env.close()

    ee_pts, obj_pts = np.array(ee_pts), np.array(obj_pts)
    all_pts = np.vstack([ee_pts, obj_pts])
    lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
    # pad so the arm can reach above and around everything it must manipulate
    pad = np.array([0.15, 0.15, 0.20])

    print(f"\nee positions   : {np.round(ee_pts.min(0), 3)} .. {np.round(ee_pts.max(0), 3)}")
    print(f"object positions: {np.round(obj_pts.min(0), 3)} .. {np.round(obj_pts.max(0), 3)}")
    print("\nSuggested configs/rekep_libero.yaml workspace:")
    print(f"  bounds_min: {[round(float(v), 2) for v in lo - pad]}")
    print(f"  bounds_max: {[round(float(v), 2) for v in hi + pad]}")


if __name__ == "__main__":
    main()
