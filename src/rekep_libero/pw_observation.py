"""Turn the current LIBERO state into PointWorld's tensor layout.

Shared by the recorder (which writes episodes to disk) and the planner (which
sends observations over the socket), because a live observation and a recorded
one MUST reach the model by the same path. If they diverged, every offline
number we have measured would stop describing what the loop actually does --
and nothing would report the discrepancy.

Only `scene_*[0]` is ever read by the model (`base.py:440` takes
`data_dict["scene_flows"][:, 0]`), so the remaining timesteps are held
constant here. They exist because `gather_features` needs a T-length array to
derive `dist2robot` and the gripper channels from.
"""

import numpy as np

from .environment_libero import AGENTVIEW, WRISTVIEW

NS = 12000          # max_scene_points, from the checkpoint's own args
NR = 500            # max_robot_points
T_LEN = 11          # pred_horizon 10 + 1, since step 0 is the input
GRID = 0.015        # grid_size, metres


def voxel_downsample(points, grid, limit):
    """Indices of a grid-subsample to <= `limit` points, on PointWorld's 1.5 cm grid.

    Returns INDICES rather than points because colour and normal have to follow
    the same selection; returning points made that impossible to do correctly.
    """
    keys = np.floor(points / grid).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    keep = np.sort(keep)
    if len(keep) > limit:
        idx = np.random.default_rng(0).choice(len(keep), limit, replace=False)
        keep = keep[np.sort(idx)]
    return keep


def organized_normals(points_hw3, cam_pos):
    """Per-pixel normals from the organized cloud, oriented toward the camera.

    PointWorld's own normals come from the mesh the point was sampled on. We do
    not have meshes for the depth cloud, but we do have it on a pixel grid,
    where the surface tangents are just neighbour differences -- accurate on
    smooth surfaces and garbage across depth discontinuities, which is the same
    trade every depth-derived normal makes.
    """
    p = points_hw3
    du = np.zeros_like(p)
    dv = np.zeros_like(p)
    du[:, 1:-1] = p[:, 2:] - p[:, :-2]
    du[:, 0], du[:, -1] = p[:, 1] - p[:, 0], p[:, -1] - p[:, -2]
    dv[1:-1, :] = p[2:] - p[:-2]
    dv[0], dv[-1] = p[1] - p[0], p[-1] - p[-2]

    n = np.cross(du, dv)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.divide(n, norm, out=np.zeros_like(n), where=norm > 1e-12)
    # Orient toward the camera: a normal facing away is the same plane with the
    # wrong sign, and the model was trained on outward-facing mesh normals.
    flip = np.sum(n * (cam_pos - p), axis=-1) < 0
    n[flip] *= -1.0
    return n


def observe(env, cam_ids=(AGENTVIEW, WRISTVIEW)):
    """Per-camera images and matrices, plus a fused arm-free point cloud.

    The released checkpoint was trained with `min_num_cameras =
    max_num_cameras = 2`, so every training sample was a two-view fusion.
    Running one view is a silent change of experimental conditions.
    """
    obs = env.get_cam_obs()
    cams, pts, colors, normals = [], [], [], []
    n_robot_px = 0
    for cid in cam_ids:
        depth_img, K, cam2world = env.camera_view(cid)
        rgb_img = obs[cid]["rgb"]
        pts_hw3 = obs[cid]["points"]
        nrm_hw3 = organized_normals(pts_hw3, np.asarray(cam2world)[:3, 3])

        p = pts_hw3.reshape(-1, 3)
        # The arm is NOT part of the scene. PointWorld represents the robot
        # once, as `robot_flows`, and that IS the action; leaving the same
        # surface in the scene cloud gives those points two contradictory
        # roles. `robot_geom_mask` is exact -- the bounding-sphere version
        # would delete the drawer handle the gripper is holding.
        keep_pix = ~env.robot_geom_mask(cid).reshape(-1)
        in_bounds = np.all((p >= env.bounds_min) & (p <= env.bounds_max), axis=1)
        sel = in_bounds & keep_pix
        n_robot_px += int((~keep_pix & in_bounds).sum())

        pts.append(p[sel])
        colors.append(rgb_img.reshape(-1, 3)[sel])
        normals.append(nrm_hw3.reshape(-1, 3)[sel])
        cams.append({"rgb": rgb_img, "depth": depth_img, "intrinsic": K,
                     "extrinsic": np.asarray(cam2world)})
    return cams, np.concatenate(pts), np.concatenate(colors), np.concatenate(normals), n_robot_px


def live_observation(env, gripper, cam_ids=(AGENTVIEW, WRISTVIEW), steps=T_LEN):
    """The current state, in the layout `PointWorldClient.observe` expects.

    `gripper` is a bound `GripperPoints`. Its points are held constant across
    the T axis here: this call establishes the SCENE, and the candidate
    actions arrive separately through `rollout`. The only thing the robot
    entries do at this stage is fix the centring, which every candidate then
    shares.
    """
    cams, pts, colors, normals, n_robot_px = observe(env, cam_ids)
    keep = voxel_downsample(pts, GRID, NS)
    points0, colors0, normals0 = pts[keep], colors[keep], normals[keep]

    rp, rn = gripper(env)
    grip = 1.0 if env.last_gripper_action == -1.0 else 0.0

    def hold(a):
        return np.repeat(a[None], steps, axis=0)[None]

    return {
        "scene_flows": hold(points0).astype(np.float32),
        "scene_colors": hold(colors0).astype(np.uint8),
        "scene_normals": hold(normals0).astype(np.float32),
        "robot_flows": hold(rp).astype(np.float32),
        "robot_normals": hold(rn).astype(np.float32),
        "gripper_open": np.full((1, steps, 1), grip, dtype=np.float32),
        "rgb": np.stack([c["rgb"] for c in cams])[None].astype(np.uint8),
        "depth": np.stack([c["depth"] for c in cams])[None].astype(np.float32),
        "intrinsic": np.stack([c["intrinsic"] for c in cams])[None].astype(np.float32),
        "extrinsic": np.stack([c["extrinsic"] for c in cams])[None].astype(np.float32),
        "n_robot_px": np.array([n_robot_px]),
    }
