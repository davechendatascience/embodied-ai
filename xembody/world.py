"""Depth -> obstacle boxes. numpy only; no simulator, no privileged geometry."""

import numpy as np

VOXEL_M = 0.03
#: Emitted box edge as a fraction of the grid pitch. A box CENTRED on a surface
#: point inflates that surface by half a voxel in every direction -- 15 mm at
#: 3 cm -- which decides feasibility for a gripper working near a table.
#: The trade is gaps of (1-scale)*pitch between adjacent boxes; safe only while
#: the robot's collision spheres are much larger than the gaps.
DIMS_SCALE = 0.7
MAX_OBSTACLES = 480


def backproject(depth, K, T_world_cam):
    """(H, W) metres -> (N, 3) world points. Rows are +v downward."""
    depth = np.asarray(depth, dtype=np.float64)
    h, w = depth.shape
    rows, cols = np.mgrid[0:h, 0:w]
    z = depth
    x = (cols - K[0, 2]) / K[0, 0] * z
    y = (rows - K[1, 2]) / K[1, 1] * z
    pts = np.stack([x, y, z, np.ones_like(z)], -1).reshape(-1, 4)
    out = (pts @ np.asarray(T_world_cam, float).T)[:, :3]
    return out[np.isfinite(out).all(1) & (z.reshape(-1) > 1e-6)]


def drop_spheres(points, centres, radii):
    """Remove points inside any sphere. Use for the robot's OWN body (self-
    knowledge, not scene knowledge) and for the volume between the jaws when
    holding -- whatever is there rides the hand, and no object identity is
    needed to know that."""
    keep = np.ones(len(points), bool)
    for c, r in zip(np.asarray(centres), np.asarray(radii)):
        keep &= np.linalg.norm(points - c, axis=1) > r
    return points[keep]


def voxelise(points, bounds_min, bounds_max, voxel=VOXEL_M,
             dims_scale=DIMS_SCALE, max_obstacles=MAX_OBSTACLES, warn=print):
    """Occupied voxels as world-frame boxes: {name: {dims, pose}}."""
    pts = np.asarray(points, dtype=np.float64)
    if not len(pts):
        return {}, 0
    lo, hi = np.asarray(bounds_min, float), np.asarray(bounds_max, float)
    pts = pts[np.all((pts >= lo) & (pts < hi), axis=1)]
    if not len(pts):
        return {}, 0

    idx = np.floor((pts - lo) / voxel).astype(np.int64)
    uniq, counts = np.unique(idx, axis=0, return_counts=True)
    uniq = uniq[np.argsort(-counts)]      # densest first: truncation drops the
    total = len(uniq)                     # flimsiest evidence, not the best
    if total > max_obstacles:
        if warn:
            warn(f"world: {total} voxels over the {max_obstacles} budget; "
                 f"dropped ones read as FREE SPACE to the planner")
        uniq = uniq[:max_obstacles]

    centres = lo + (uniq + 0.5) * voxel
    return ({f"vox_{i}": {"dims": [voxel * dims_scale] * 3,
                          "pose": [*c.tolist(), 1.0, 0.0, 0.0, 0.0]}
             for i, c in enumerate(centres)}, total)


def from_depth(depth_maps, Ks, T_world_cams, bounds_min, bounds_max,
               robot_spheres=None, **kw):
    """The whole perception path: depth in, obstacle boxes out.

    Rebuild this EVERY replan. A drawer moves as it opens and a held object
    rides the hand, so a world built once is stale by the second keypose.
    """
    chunks = [backproject(d, K, T) for d, K, T in
              zip(depth_maps, Ks, T_world_cams)]
    pts = np.concatenate(chunks) if chunks else np.zeros((0, 3))
    if robot_spheres is not None and len(pts):
        c, r = robot_spheres
        pts = drop_spheres(pts, c, r)
    boxes, total = voxelise(pts, bounds_min, bounds_max, **kw)
    return boxes, {"points": int(len(pts)), "voxels": int(total)}
