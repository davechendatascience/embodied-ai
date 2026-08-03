"""Drive ReKep's optimizers with no simulator attached.

The point is to separate two failure modes that are otherwise entangled once
MuJoCo is wired in: "ReKep's optimization core is broken" vs "our LIBERO
adapter feeds it bad data". Everything the solvers touch is stubbed here —
keypoints, SDF, IK — so a pass means the core works and any later failure is
ours.

Run:  .venv/bin/python third_party/ReKep/smoke_test_solvers.py [task_dir]
"""

import os
import sys
import glob
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from rekep_libero import add_rekep_to_path, REKEP_DIR
from rekep_libero.config import load_config

add_rekep_to_path()

import transform_utils as T  # noqa: E402  (upstream ReKep, flat imports)
from utils import load_functions_from_txt  # noqa: E402
from subgoal_solver import SubgoalSolver  # noqa: E402
from path_solver import PathSolver  # noqa: E402


class StubIKResult:
    """Mirrors the four attributes ReKep reads off a Lula IK result.

    subgoal_solver.py:54-60 and path_solver.py:78-82 read exactly these; the
    LIBERO IK wrapper has to expose the same set.
    """

    def __init__(self, success, position_error, num_descents, cspace_position):
        self.success = success
        self.position_error = position_error
        self.num_descents = num_descents
        self.cspace_position = cspace_position


class StubIKSolver:
    """Reachability proxy: anything inside a radius of the base is solvable.

    Deliberately not real kinematics — the LIBERO adapter will swap in the DLS
    solver. This exists only so the optimizer sees a smooth, plausible
    reachability gradient instead of a constant.
    """

    def __init__(self, reset_joint_pos, base_pos=np.array([0.0, 0.0, 0.7]), reach=1.2):
        self.reset_joint_pos = reset_joint_pos
        self.base_pos = base_pos
        self.reach = reach

    def solve(self, target_pose_homo, max_iterations=20, initial_joint_pos=None, **kw):
        dist = np.linalg.norm(target_pose_homo[:3, 3] - self.base_pos)
        frac = np.clip(dist / self.reach, 0.0, 1.0)
        seed = self.reset_joint_pos if initial_joint_pos is None else np.asarray(initial_joint_pos)
        # joints drift from the seed the further out the target is, so the
        # solver's reset-posture regularization sees a real gradient
        cspace = np.asarray(seed, dtype=float) + frac * 0.5
        # more descents the closer to the reachability limit
        return StubIKResult(
            success=bool(dist <= self.reach),
            position_error=float(max(0.0, dist - self.reach)),
            num_descents=int(1 + frac * (max_iterations - 1)),
            cspace_position=cspace,
        )


def build_sdf(bounds_min, bounds_max, voxel_size, obstacle=None):
    """Signed distance grid. Positive = free space (ReKep's convention).

    With `obstacle=(center, half_extent)` a box is carved out so the collision
    term has something to actually push against.
    """
    shape = np.ceil((bounds_max - bounds_min) / voxel_size).astype(int)
    axes = [np.linspace(bounds_min[i], bounds_max[i], shape[i]) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    if obstacle is None:
        return np.full(shape, 1.0, dtype=np.float32)
    center, half = obstacle
    # distance to box surface, negative inside
    delta = np.abs(grid - center) - half
    outside = np.linalg.norm(np.maximum(delta, 0.0), axis=-1)
    inside = np.minimum(delta.max(axis=-1), 0.0)
    return (outside + inside).astype(np.float32)


def main(task_dir=None):
    config = load_config()
    bounds_min = np.array(config["main"]["bounds_min"])
    bounds_max = np.array(config["main"]["bounds_max"])

    if task_dir is None:
        candidates = sorted(glob.glob(os.path.join(REKEP_DIR, "vlm_query", "2*")))
        assert candidates, "no generated task dir found; run constraint generation first"
        task_dir = candidates[-1]
    print(f"constraints from: {os.path.basename(task_dir)}")

    # stub grasping cost: 0 when grasping, 1 otherwise
    get_grasping_cost_fn = lambda keypoint_idx: 1

    subgoal_c = load_functions_from_txt(os.path.join(task_dir, "stage1_subgoal_constraints.txt"), get_grasping_cost_fn)
    path_path = os.path.join(task_dir, "stage2_path_constraints.txt")
    path_c = load_functions_from_txt(path_path, get_grasping_cost_fn) if os.path.exists(path_path) else []
    print(f"loaded {len(subgoal_c)} subgoal constraint(s), {len(path_c)} path constraint(s)")

    # synthetic scene: 4 keypoints spread through the workspace
    rng = np.random.default_rng(0)
    n_keypoints = 4
    scene_keypoints = bounds_min + rng.random((n_keypoints, 3)) * (bounds_max - bounds_min)
    ee_pos = (bounds_min + bounds_max) / 2.0
    ee_pose = np.concatenate([ee_pos, np.array([0.0, 0.0, 0.0, 1.0])])  # xyzw identity
    keypoints = np.concatenate([[ee_pos], scene_keypoints], axis=0)
    keypoint_movable_mask = np.zeros(n_keypoints + 1, dtype=bool)
    keypoint_movable_mask[0] = True  # index 0 is always the ee

    sdf = build_sdf(bounds_min, bounds_max, config["main"]["sdf_voxel_size"])
    collision_points = bounds_min + rng.random((200, 3)) * (bounds_max - bounds_min)
    reset_joint_pos = np.zeros(7)
    ik = StubIKSolver(reset_joint_pos)

    print(f"sdf grid {sdf.shape}, {sdf.size} voxels | {len(collision_points)} collision points")

    subgoal_solver = SubgoalSolver(config["subgoal_solver"], ik, reset_joint_pos)
    path_solver = PathSolver(config["path_solver"], ik, reset_joint_pos)

    print("\n--- subgoal solve ---")
    t = time.time()
    subgoal_pose, dbg = subgoal_solver.solve(
        ee_pose, keypoints, keypoint_movable_mask, subgoal_c, path_c,
        sdf, collision_points, True, reset_joint_pos, from_scratch=True,
    )
    subgoal_dt = time.time() - t
    print(f"solved in {subgoal_dt:.2f}s -> pos {np.round(subgoal_pose[:3], 4)} quat {np.round(subgoal_pose[3:], 3)}")
    print(f"  status={dbg.get('status')} cost={dbg.get('final_cost')}")
    assert subgoal_pose.shape == (7,), f"expected a 7-vector pose, got {subgoal_pose.shape}"
    assert np.all(np.isfinite(subgoal_pose)), "solver returned non-finite pose"
    in_bounds = np.all(subgoal_pose[:3] >= bounds_min - 1e-6) and np.all(subgoal_pose[:3] <= bounds_max + 1e-6)
    print(f"  within workspace bounds: {in_bounds}")

    print("\n--- path solve ---")
    t = time.time()
    path, dbg = path_solver.solve(
        ee_pose, subgoal_pose, keypoints, keypoint_movable_mask, path_c,
        sdf, collision_points, reset_joint_pos, from_scratch=True,
    )
    path_dt = time.time() - t
    print(f"solved in {path_dt:.2f}s -> path {np.asarray(path).shape}")
    print(f"  status={dbg.get('status')}")
    assert np.all(np.isfinite(np.asarray(path))), "path contains non-finite values"

    print(f"\nPASS  subgoal {subgoal_dt:.2f}s + path {path_dt:.2f}s = {subgoal_dt + path_dt:.2f}s per replan")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
