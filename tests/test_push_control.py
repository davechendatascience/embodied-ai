"""Can the arm push the plate AT ALL? The control before any model claim.

The planner's first push run closed 0.0 mm and the model reported a flat cost
landscape (margins 0.0-1.4 mm across every candidate). That is consistent with
two very different stories:

    A. PointWorld cannot predict contact, so no candidate looks better.
    B. The gripper never reached the plate, so no candidate WAS better.

`NOTES.md` section 4 is explicit about this class of mistake -- "a counterfactual
needs a control that reproduces the factual", and without one "a broken
candidate generator is indistinguishable from a bad model, and it produces a
confident, plausible, completely wrong verdict." The first push video showed the
arm wandering over the stove while the plate sat untouched, which already points
at B, but pointing is not measuring.

So this script removes the planner and the model entirely. It stages the
gripper, drives a SCRIPTED straight line along the goal direction, and reports
where the plate actually went. If the plate moves, pushing is achievable and any
remaining failure is the planner's or the model's. If it does not, the staging
is wrong and nothing downstream can be interpreted.

    MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/test_push_control.py
"""

import argparse
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from rekep_libero import add_rekep_to_path  # noqa: E402
from rekep_libero.config import load_config  # noqa: E402

add_rekep_to_path()

from rekep_libero.environment_libero import ReKepLiberoEnv, EpisodeFinished  # noqa: E402
from rekep_libero import task_spec as specs  # noqa: E402
import transform_utils as T  # noqa: E402

SUITE, TASK = "libero_goal", 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--back", type=float, default=0.075)
    ap.add_argument("--lift", type=float, default=0.005)
    ap.add_argument("--rate-mm", type=float, default=20.0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--video", default="videos/push_control.mp4")
    cli = ap.parse_args()

    config = load_config()
    ec = dict(config["env"])
    ec["bounds_min"], ec["bounds_max"] = config["workspace"]["bounds_min"], config["workspace"]["bounds_max"]
    ec["interpolate_pos_step_size"] = config["main"]["interpolate_pos_step_size"]
    ec["interpolate_rot_step_size"] = config["main"]["interpolate_rot_step_size"]
    env = ReKepLiberoEnv(ec, task_suite=SUITE, task_id=TASK, robot="Panda",
                         resolution=config["libero"]["resolution"], reset_seed=0)
    spec = specs.for_task(SUITE, TASK)
    print(f"task    : {env.instruction}")
    print(f"spec    : {spec.name}")

    src0 = specs.object_center(env, spec.target)
    lo, hi = specs.object_aabb(env, spec.target)
    owed = spec.offset(env)
    d = owed / np.linalg.norm(owed)
    print(f"plate   : centre {np.round(src0, 3)}  aabb {np.round(lo,3)}..{np.round(hi,3)}")
    print(f"goal    : owes {np.linalg.norm(owed)*1000:.1f} mm along {np.round(d, 3)}")

    # Half-extent along the push direction, by PROJECTION rather than by summing
    # the axis extents -- the sum overestimates for a diagonal push and would
    # stage the gripper needlessly far back.
    half = float(np.sqrt((d[0] * (hi[0]-lo[0]) * 0.5) ** 2 + (d[1] * (hi[1]-lo[1]) * 0.5) ** 2))
    pos = src0 - d * (half + cli.back)
    pos[2] = lo[2] + cli.lift + env.finger_offset()
    print(f"stage   : half-extent {half*1000:.1f} mm, target ee {np.round(pos, 3)}")

    from rekep_libero.grasp import ee_rotation
    R = ee_rotation(np.array([0.0, 0.0, -1.0]), d,
                    env.GRASP_APPROACH_AXIS, env.gripper_closing_axis_idx())
    quat = T.mat2quat(R)

    env.close_gripper()
    try:
        env.execute_action(np.concatenate([pos + [0, 0, 0.10], quat,
                                           [env.get_gripper_null_action()]]), precise=True)
        env.execute_action(np.concatenate([pos, quat,
                                           [env.get_gripper_null_action()]]), precise=True)
    except EpisodeFinished:
        pass
    ee = env.get_ee_pos()
    err = np.linalg.norm(ee - pos) * 1000
    print(f"staged  : ee {np.round(ee, 3)}  (wanted {np.round(pos,3)}, off by {err:.1f} mm)")
    if err > 30:
        print("          ^ THE STAGING DID NOT REACH ITS TARGET. Everything below "
              "is measuring the controller, not the push.")

    # Scripted straight line along the goal direction. No model, no planner.
    moved = []
    for i in range(cli.steps):
        tgt = env.get_ee_pose().copy()
        tgt[:3] += d * cli.rate_mm / 1000.0
        try:
            env.execute_action(np.concatenate([tgt[:3], quat,
                                               [env.get_gripper_null_action()]]), precise=True)
        except EpisodeFinished:
            print(f"step {i:2d}: LIBERO ended the episode")
            break
        now = specs.object_center(env, spec.target)
        moved.append(np.linalg.norm(now - src0) * 1000)
        print(f"step {i:2d}: ee {np.round(env.get_ee_pos(),3)} | plate moved "
              f"{moved[-1]:6.1f} mm | owes {spec.remaining_mm(env):6.1f} mm | "
              f"success {spec.success(env)}")

    total = moved[-1] if moved else 0.0
    print(f"\nplate   : moved {total:.1f} mm total, owes "
          f"{spec.remaining_mm(env):.1f} mm of {np.linalg.norm(owed)*1000:.1f}")
    print(f"success : {spec.success(env)}   (LIBERO's own _check_success)")
    if cli.video:
        path = env.save_video(cli.video)
        print(f"video   : {path} ({len(env._frames)} frames)")
    print(f"verdict : pushing is {'ACHIEVABLE' if total > 20 else 'NOT achievable'} "
          f"with this staging — so a planner failure is "
          f"{'a real planner/model failure' if total > 20 else 'a SETUP failure'}")
    return 0 if total > 20 else 1


if __name__ == "__main__":
    sys.exit(main())
