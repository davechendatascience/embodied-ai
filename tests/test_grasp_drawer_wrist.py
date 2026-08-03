"""Drawer handle via the wrist camera, on the real arm rather than a free camera.

`test_grasp_drawer_viewpoint.py` proved the point with MuJoCo free cameras
placed wherever was convenient: from the front, Contact-GraspNet lands 14-23 mm
from the handle with the correct pull direction, while the fixed agentview
yields zero grasps. That established the diagnosis but not a usable mechanism --
nothing on a robot can teleport a camera.

This does it the way the robot would: drive the arm so the WRIST camera faces
the drawer, capture from there, and ask the same question. It is strictly
harder than the free-camera version, because the viewpoint now has to be one
the arm can actually reach and hold.

    MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/test_grasp_drawer_wrist.py
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

from rekep_libero.environment_libero import ReKepLiberoEnv, AGENTVIEW, WRISTVIEW  # noqa: E402
from rekep_libero import grasp_cgn  # noqa: E402


def handle_geoms(env, drawer="cabinet_middle"):
    model, data = env.sim.model, env.sim.data
    body = [g for g in range(model.ngeom)
            if (model.body_id2name(model.geom_bodyid[g]) or "").endswith(drawer)]
    return [g for g in body if data.geom_xpos[g][1] > -0.160]


def ask(env, proposer, cam_id, geoms, truth, label):
    depth, K, cam2world = env.camera_view(cam_id)
    points = env.get_cam_obs()[cam_id]["points"]
    mask = env.points_in_geoms(points, geoms, margin=0.01)
    npx = int(mask.sum())
    if npx < 20:
        print(f"{label:26s} handle {npx:5d} px  -> too few to ask")
        return None
    cands = proposer._candidates(depth, mask, K, cam2world)
    if not cands:
        print(f"{label:26s} handle {npx:5d} px  -> 0 grasps")
        return None
    best = min(cands, key=lambda c: float(np.linalg.norm(c["position"] - truth)))
    d = float(np.linalg.norm(best["position"] - truth)) * 1000
    print(f"{label:26s} handle {npx:5d} px  -> {len(cands):2d} grasps, nearest "
          f"{d:5.1f} mm, approach.-Y {float(best['approach'] @ [0, -1, 0]):+.2f}, "
          f"score {best['score']:.3f}")
    return best


def main():
    if not grasp_cgn.available():
        print("no Contact-GraspNet checkpoint")
        return 1

    config = load_config()
    ec = dict(config["env"])
    ec["bounds_min"], ec["bounds_max"] = config["workspace"]["bounds_min"], config["workspace"]["bounds_max"]
    ec["interpolate_pos_step_size"] = config["main"]["interpolate_pos_step_size"]
    ec["interpolate_rot_step_size"] = config["main"]["interpolate_rot_step_size"]
    env = ReKepLiberoEnv(ec, task_suite="libero_goal", task_id=0,
                         robot="Panda", resolution=config["libero"]["resolution"])

    geoms = handle_geoms(env)
    truth = np.mean([env.sim.data.geom_xpos[g] for g in geoms], axis=0)
    print(f"task    : {env.instruction}")
    print(f"handle  : {np.round(truth, 4)}  ({len(geoms)} geoms)")
    print(f"ee home : {np.round(env.get_ee_pos(), 3)}\n")

    proposer = grasp_cgn.ContactGraspNetProposer(finger_offset=env.finger_offset(), top_k=64)
    grasp_cgn.estimator()

    # the two fixed cameras, from the home pose, as the baseline
    ask(env, proposer, AGENTVIEW, geoms, truth, "agentview @ home")
    ask(env, proposer, WRISTVIEW, geoms, truth, "wrist @ home")

    # now aim the wrist. The cabinet's drawers face +Y.
    standoff = env.look_at(truth, standoffs=(0.25, 0.30, 0.35, 0.40, 0.50),
                           direction=(0.0, 1.0, 0.0))
    if standoff is None:
        print("\nno IK-feasible viewpoint found")
        return 1
    err = float(np.linalg.norm(env.get_ee_pos() -
                               env.ee_pose_for_view(truth + [0, standoff, 0], truth)[:3]))
    print(f"\nmoved to {standoff:.2f} m standoff, ee tracking error {err * 1000:.1f} mm")
    print(f"ee now  : {np.round(env.get_ee_pos(), 3)}\n")

    best = ask(env, proposer, WRISTVIEW, geoms, truth, f"wrist @ {standoff:.2f}m front")
    if best is not None:
        print(f"\ngrasp   : {np.round(best['position'], 4)}  "
              f"opening {best['width'] * 1000:.1f} mm")
        print(f"handle  : {np.round(truth, 4)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
