"""Run ReKep's own pen demo through OUR solver stack and render the plan.

The full demo needs OmniGibson on Isaac Sim, which does not build here (see
notes in the session log: OmniGibson targets the removed `omni.isaac.*`
namespace, Isaac 5.1 ships `isaacsim.*`). But the repo ships the cached query
for that demo -- real keypoints, real GPT-4o constraints -- so the *planning*
half can be replayed exactly.

This is the control experiment for "are we only bad on LIBERO?": identical
solver code, ReKep's own scene. Physics and grasping are NOT exercised.

    MUJOCO_GL=egl .venv/bin/python tests/replay_pen_demo.py
"""

import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from rekep_libero import add_rekep_to_path  # noqa: E402
from rekep_libero.config import load_config  # noqa: E402

add_rekep_to_path()

import transform_utils as T  # noqa: E402
from utils import load_functions_from_txt  # noqa: E402
from subgoal_solver import SubgoalSolver  # noqa: E402
from path_solver import PathSolver  # noqa: E402

PEN_DIR = os.path.join(ROOT, "reference", "ReKep-pristine", "vlm_query", "pen")


class StubIKResult:
    def __init__(self, success, position_error, num_descents, cspace_position):
        self.success, self.position_error = success, position_error
        self.num_descents, self.cspace_position = num_descents, cspace_position


class FetchStubIK:
    """Reachability proxy for the Fetch, whose real kinematics need Isaac."""

    def __init__(self, reset_joint_pos, base=np.array([-0.8, 0.0, 0.7]), reach=1.3):
        self.reset_joint_pos, self.base, self.reach = reset_joint_pos, base, reach

    def solve(self, target_pose_homo, max_iterations=20, initial_joint_pos=None, **kw):
        dist = np.linalg.norm(target_pose_homo[:3, 3] - self.base)
        frac = np.clip(dist / self.reach, 0.0, 1.0)
        seed = self.reset_joint_pos if initial_joint_pos is None else np.asarray(initial_joint_pos)
        return StubIKResult(bool(dist <= self.reach), float(max(0.0, dist - self.reach)),
                            int(1 + frac * (max_iterations - 1)),
                            np.asarray(seed, dtype=float) + frac * 0.3)


def main():
    config = load_config()
    meta = json.load(open(os.path.join(PEN_DIR, "metadata.json")))
    keypoints = np.array(meta["init_keypoint_positions"])
    n_stages = meta["num_stages"]

    # The pen scene lives in OmniGibson's Rs_int workspace, NOT our LIBERO one.
    # Use upstream's original bounds or every subgoal is clipped out of range.
    import yaml
    upstream = yaml.safe_load(open(os.path.join(ROOT, "third_party", "ReKep", "configs", "config.yaml")))
    bounds_min = np.array(upstream["main"]["bounds_min"])
    bounds_max = np.array(upstream["main"]["bounds_max"])
    for section in ("subgoal_solver", "path_solver"):
        config[section] = dict(config[section])
        config[section]["bounds_min"], config[section]["bounds_max"] = list(bounds_min), list(bounds_max)

    print(f"pen demo: {len(keypoints)} keypoints, {n_stages} stages")
    print(f"  grasp {meta['grasp_keypoints']}  release {meta['release_keypoints']}")
    print(f"  workspace {np.round(bounds_min,2)} .. {np.round(bounds_max,2)}")

    grasping = {"held": False}

    def grasp_cost(idx):
        return 0 if grasping["held"] else 1

    reset_q = np.zeros(8)
    ik = FetchStubIK(reset_q)
    subgoal_solver = SubgoalSolver(config["subgoal_solver"], ik, reset_q)
    path_solver = PathSolver(config["path_solver"], ik, reset_q)

    # free space: the pen demo's SDF came from open3d raycasting we cannot run
    shape = np.ceil((bounds_max - bounds_min) / 0.01).astype(int)
    sdf = np.full(shape, 1.0, dtype=np.float32)
    # solver dereferences collision_points, so give it a small gripper-shaped
    # cloud rather than None (upstream always has the Fetch's mesh points)
    rng = np.random.default_rng(0)
    gripper_local = rng.uniform(-0.03, 0.03, size=(24, 3))

    ee = np.concatenate([keypoints[0] + np.array([0.0, 0.0, 0.25]), [0, 0, 0, 1]])
    movable = np.zeros(len(keypoints) + 1, dtype=bool)
    movable[0] = True

    trajectory, stage_marks = [ee[:3].copy()], []
    for stage in range(1, n_stages + 1):
        sub = load_functions_from_txt(os.path.join(PEN_DIR, f"stage{stage}_subgoal_constraints.txt"), grasp_cost)
        pth_file = os.path.join(PEN_DIR, f"stage{stage}_path_constraints.txt")
        pth = load_functions_from_txt(pth_file, grasp_cost) if os.path.exists(pth_file) else []
        is_grasp = meta["grasp_keypoints"][stage - 1] != -1

        kp_all = np.concatenate([[ee[:3]], keypoints])
        collision_points = ee[:3] + gripper_local
        subgoal, dbg = subgoal_solver.solve(ee, kp_all, movable, sub, pth, sdf,
                                            collision_points, is_grasp, reset_q, from_scratch=True)
        path, _ = path_solver.solve(ee, subgoal, kp_all, movable, pth, sdf,
                                    collision_points, reset_q, from_scratch=True)
        path = np.asarray(path)
        print(f"  stage {stage}: {len(sub)} subgoal / {len(pth)} path constraints, "
              f"grasp={is_grasp} -> subgoal {np.round(subgoal[:3],3)}, path {path.shape}")

        trajectory.extend(path[:, :3])
        stage_marks.append(len(trajectory) - 1)
        ee = np.concatenate([path[-1][:3], path[-1][3:7]])
        if is_grasp:
            grasping["held"] = True
        if meta["release_keypoints"][stage - 1] != -1:
            grasping["held"] = False

    trajectory = np.array(trajectory)
    print(f"\nfull plan: {len(trajectory)} waypoints, "
          f"path length {np.linalg.norm(np.diff(trajectory,axis=0),axis=1).sum():.3f} m")

    render(trajectory, keypoints, stage_marks, meta)
    return 0


def render(trajectory, keypoints, stage_marks, meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio

    out = os.path.join(ROOT, "videos", "pen_demo_plan.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    grasp_idx = [i for i in meta["grasp_keypoints"] if i != -1]

    frames = []
    for step in range(1, len(trajectory) + 1):
        fig = plt.figure(figsize=(6, 5), dpi=110)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(*keypoints.T, c="tab:blue", s=45, label="keypoints")
        for i, kp in enumerate(keypoints):
            ax.text(*kp, f" {i}", fontsize=7)
        if grasp_idx:
            ax.scatter(*keypoints[grasp_idx].T, c="tab:red", s=140, marker="*", label="grasp target")
        ax.plot(*trajectory[:step].T, c="tab:orange", lw=2, label="planned ee path")
        ax.scatter(*trajectory[step - 1], c="k", s=50)
        stage = 1 + sum(1 for m in stage_marks if step - 1 > m)
        ax.set_title(f"ReKep pen demo — planning replay\nstage {min(stage, meta['num_stages'])}"
                     f" / {meta['num_stages']}   waypoint {step}/{len(trajectory)}", fontsize=10)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.legend(loc="upper left", fontsize=7)
        ax.view_init(elev=22, azim=-60 + step * 0.8)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)

    imageio.mimsave(out, frames, fps=10)
    print(f"video: {out}  ({len(frames)} frames)")


if __name__ == "__main__":
    sys.exit(main())
