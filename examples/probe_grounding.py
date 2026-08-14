"""Oracle boxes for a LIBERO scene, in the frame the POLICY sees. LIBERO host.

Increment 1 of the grounding probe: produce the boxes and prove they land on the
right pixels. No policy, no rollout, no detector. If this is wrong every number
downstream is wrong in a way that looks like a model finding, so it ships with
an overlay renderer and is meant to be checked by eye first.

The extraction itself lives in `xembody/boxes.py`, which imports no simulator --
RoboCasa runs the same code from a different venv on robosuite master, and the
two cannot share an interpreter. This file is only the LIBERO half: how to build
the environment and which observation keys to read.

WHY THE ENVIRONMENT IS BUILT HERE INSTEAD OF VIA `libero_ur5e.build`.
Segmentation sensors are created at CONSTRUCTION time from `camera_segmentations`
(robosuite `robot_env._create_segementation_sensor`), so it cannot be switched on
afterwards.

WHY `element` SEGMENTATION AND NOT `instance`.
LIBERO's own `SegmentationRenderEnv` hardcodes the robot instance as "Panda0",
which is wrong the moment a UR5e is mounted. `element` returns raw MuJoCo geom
ids with no naming assumptions, and geom -> body -> movable root is a mapping
that can be stated and checked.

Run:
  MUJOCO_GL=egl PYTHONPATH=. ./.venv/bin/python examples/probe_grounding.py \
      --suite libero_10 --task-id 0 --overlay pairs/diag/boxes.png
"""
import argparse
import os
import sys

import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R)
sys.path.insert(0, R + "/third_party/LIBERO")

from xembody.boxes import (draw_overlay, movable_bodies, oracle_boxes,
                           policy_view)

CAMS = ("agentview", "robot0_eye_in_hand")


def build_env(suite_name, task_id, robot="Panda", gripper="default", res=256,
              seed=0):
    """A single LIBERO env that also renders per-geom segmentation."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"),
                        task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl, robots=[robot], gripper_types=gripper,
        camera_heights=res, camera_widths=res, controller="OSC_POSE",
        camera_depths=True, camera_names=list(CAMS),
        camera_segmentations="element", horizon=10000)
    np.random.seed(seed)
    obs = env.reset()
    return env, suite, task, obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--robot", default="Panda")
    ap.add_argument("--gripper", default="default")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--cam", default="agentview", choices=list(CAMS))
    ap.add_argument("--min-px", type=int, default=12)
    ap.add_argument("--overlay", default=None,
                    help="write a PNG with the boxes drawn, in the POLICY frame")
    a = ap.parse_args()

    env, _, task, obs = build_env(a.suite, a.task_id, robot=a.robot,
                                  gripper=a.gripper, res=a.res)
    model = env.sim.model
    rgb = policy_view(obs[f"{a.cam}_image"])
    seg = policy_view(obs[f"{a.cam}_segmentation_element"])
    boxes = oracle_boxes(model, seg, movable_bodies(model), min_px=a.min_px)

    print(f"task   {task.language!r}")
    print(f"cam    {a.cam}  {a.res}x{a.res}  (policy frame: [::-1, ::-1])")
    print(f"objects {len(boxes)}")
    for n, (x0, y0, x1, y1) in sorted(boxes.items()):
        print(f"  {n:<34} x[{x0:>3},{x1:>3}] y[{y0:>3},{y1:>3}]"
              f"  {x1 - x0 + 1:>3}x{y1 - y0 + 1:<3} px")

    if a.overlay:
        import imageio
        os.makedirs(os.path.dirname(os.path.abspath(a.overlay)), exist_ok=True)
        imageio.imwrite(a.overlay, draw_overlay(rgb, boxes))
        print(f"overlay -> {a.overlay}")

    env.close()


if __name__ == "__main__":
    main()
