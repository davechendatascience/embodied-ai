"""Is the drawer handle a viewpoint problem or a model problem?

Established so far: the handle mask is correct (462 px, cloud centroid 9-12 mm
from the handle's true pose) and Contact-GraspNet still returns ZERO grasps from
it. The agentview camera sits at (0.659, 0, 1.610) looking down -X/-Z while the
drawer face points +Y, so that face is seen at grazing incidence.

Two explanations remain, and they imply completely different work:

  VIEWPOINT   the network never sees a graspable surface. Fix by capturing from
              somewhere useful -- multi-view fusion or a wrist-camera shot --
              which the robot pipeline (cap_demo/grasp/cloud.py) already does.

  DISTRIBUTION  Contact-GraspNet is trained on ACRONYM, i.e. free-standing
              tabletop objects. A bar mounted flush to a large panel on a piece
              of furniture is out of distribution no matter how well you see it.
              Fixing that needs an articulated-object model (AO-Grasp,
              GAPart-style part affordances), not a better tabletop network.

This separates them by rendering the SAME scene from a camera placed in front of
the cabinet, facing the drawer, and asking the identical question. If handle
grasps appear, it is viewpoint and no new model is needed. If they still do not,
the view was never the binding constraint.

MuJoCo's own renderer is used rather than a robosuite camera because robosuite
cameras are declared in the scene XML; this needs an arbitrary pose.

    MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/test_grasp_drawer_viewpoint.py
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

from rekep_libero.environment_libero import ReKepLiberoEnv, AGENTVIEW  # noqa: E402
from rekep_libero import grasp_cgn  # noqa: E402

RES = 480


def render_from(env, eye, lookat, resolution=RES):
    """Depth, intrinsics and OpenCV cam2world for a free camera at `eye`.

    Returns metric depth. MuJoCo's renderer reports depth along the camera's
    view axis in metres, which is exactly what Contact-GraspNet's `depth2pc`
    expects, so no linearisation is needed here (unlike robosuite's normalised
    buffer, which is why `camera_view()` calls `get_real_depth_map`).
    """
    import mujoco

    model, data = env.sim.model._model, env.sim.data._data
    renderer = mujoco.Renderer(model, height=resolution, width=resolution)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    eye, lookat = np.asarray(eye, float), np.asarray(lookat, float)
    forward = lookat - eye
    distance = float(np.linalg.norm(forward))
    forward /= distance
    cam.lookat[:] = lookat
    cam.distance = distance
    # MuJoCo defines azimuth in the xy-plane and elevation off it, both degrees
    cam.azimuth = float(np.degrees(np.arctan2(forward[1], forward[0])))
    cam.elevation = float(np.degrees(np.arcsin(np.clip(forward[2], -1, 1))))

    renderer.enable_depth_rendering()
    renderer.update_scene(data, cam)
    depth = np.asarray(renderer.render(), dtype=np.float64).copy()
    renderer.disable_depth_rendering()

    # read the frame the renderer actually used rather than recomputing it
    scam = renderer.scene.camera[0]
    fwd = np.asarray(scam.forward, float); fwd /= np.linalg.norm(fwd)
    up = np.asarray(scam.up, float)
    up = up - fwd * float(up @ fwd); up /= np.linalg.norm(up)
    # OpenCV: +Z view direction, +Y down, +X right, and x = y cross z
    z_axis, y_axis = fwd, -up
    x_axis = np.cross(y_axis, z_axis)
    cam2world = np.eye(4)
    cam2world[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    cam2world[:3, 3] = np.asarray(scam.pos, float)

    fovy = float(model.vis.global_.fovy)
    f = (resolution / 2.0) / np.tan(np.radians(fovy) / 2.0)
    K = np.array([[f, 0, resolution / 2.0], [0, f, resolution / 2.0], [0, 0, 1.0]])
    renderer.close()
    return depth, K, cam2world


def points_from(depth, K, cam2world):
    h, w = depth.shape
    rows, cols = np.mgrid[0:h, 0:w]
    z = depth
    x = (cols - K[0, 2]) / K[0, 0] * z
    y = (rows - K[1, 2]) / K[1, 1] * z
    pts = np.stack([x, y, z, np.ones_like(z)], axis=-1)
    return (pts @ cam2world.T)[..., :3]


def try_view(label, env, proposer, eye, lookat, handle_geoms, truth):
    depth, K, cam2world = render_from(env, eye, lookat)
    points = points_from(depth, K, cam2world)
    mask = env.points_in_geoms(points, handle_geoms, margin=0.01)
    npx = int(mask.sum())
    if npx < 20:
        print(f"{label:26s} handle {npx:5d} px  -> too few to ask")
        return
    cands = proposer._candidates(depth, mask, K, cam2world)
    sel = points.reshape(-1, 3)[mask.reshape(-1)]
    centroid_err = float(np.linalg.norm(sel.mean(axis=0) - truth)) * 1000
    if not cands:
        print(f"{label:26s} handle {npx:5d} px, cloud {centroid_err:4.0f} mm off"
              f"  -> 0 grasps")
        return
    best = min(cands, key=lambda c: float(np.linalg.norm(c["position"] - truth)))
    print(f"{label:26s} handle {npx:5d} px, cloud {centroid_err:4.0f} mm off"
          f"  -> {len(cands):2d} grasps, nearest "
          f"{float(np.linalg.norm(best['position'] - truth)) * 1000:5.1f} mm, "
          f"approach.-Y {float(best['approach'] @ [0, -1, 0]):+.2f}, "
          f"score {best['score']:.3f}")


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

    model, data = env.sim.model, env.sim.data
    mid = [g for g in range(model.ngeom)
           if (model.body_id2name(model.geom_bodyid[g]) or "").endswith("cabinet_middle")]
    handle = [g for g in mid if data.geom_xpos[g][1] > -0.160]
    truth = np.mean([data.geom_xpos[g] for g in handle], axis=0)
    print(f"task   : {env.instruction}")
    print(f"handle : {np.round(truth, 4)}  ({len(handle)} geoms)\n")

    proposer = grasp_cgn.ContactGraspNetProposer(finger_offset=env.finger_offset(), top_k=64)
    grasp_cgn.estimator()

    # The drawer face points +Y, so a camera at larger y looks straight at it.
    # Sweep standoff and height to show this is a property of the viewpoint and
    # not one lucky pose.
    views = [
        ("front, level, 0.40m", truth + [0.0, 0.40, 0.00]),
        ("front, level, 0.60m", truth + [0.0, 0.60, 0.00]),
        ("front, above, 0.50m", truth + [0.0, 0.45, 0.20]),
        ("front-right, 0.50m", truth + [0.30, 0.40, 0.10]),
    ]
    for label, eye in views:
        try_view(label, env, proposer, eye, truth, handle, truth)

    # the existing agentview, as the baseline being explained
    depth, K, cam2world = env.camera_view()
    mask = env.points_in_geoms(env.get_cam_obs()[AGENTVIEW]["points"], handle, margin=0.01)
    cands = proposer._candidates(depth, mask, K, cam2world)
    print(f"\n{'agentview (baseline)':26s} handle {int(mask.sum()):5d} px"
          f"  -> {len(cands)} grasps")
    print(f"{'':26s} camera at {np.round(np.asarray(cam2world)[:3, 3], 3)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
