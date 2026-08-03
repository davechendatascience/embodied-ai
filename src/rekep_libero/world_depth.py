"""The PERCEIVED obstacle world: depth in, cuRobo obstacles out.

`world_export.py` reads MuJoCo geometry and is a privileged input. This is its
honest counterpart, and the pair is the measurement: E3a runs on the oracle
world, E3b on this one, and **E3a − E3b is what the privileged geometry was
worth**. Building only the oracle leaves you unable to separate a perception
failure from a planning failure; building only this leaves you unable to say
how much perception cost.

Everything here comes from sensors the robot has:

  * **depth from the two LIBERO cameras** — agentview and the wrist, the same
    two the policy sees. `_depth_to_points` already backprojects them.
  * **its own body, removed by kinematics** — a robot knows where its links
    are. `NOTES.md` calls this out as the one legitimate use of model data;
    it is self-knowledge, not scene knowledge.
  * **the held object, removed the same way** — it rides the tool, so it is
    part of the robot for collision purposes and belongs in the attached set.

WHAT THIS CANNOT SEE, stated up front because it is the real limit and it will
show up as a planner that clips things:

  * anything **occluded**. The far side of the cabinet, the volume beneath an
    object, and whatever the arm itself hides are simply absent, and absent
    reads as FREE SPACE to a planner. The oracle world has no such holes.
  * thin structures thinner than a voxel.
  * anything outside the workspace bounds.

Occupied voxels are emitted as cuboids rather than a `VoxelGrid` because the
rest of the pipeline — `curobo_bridge`, `test_curobo_collision` — already
speaks cuboids, so the oracle and perceived worlds stay swappable and directly
comparable. That is the whole point of having both.
"""

import argparse

import numpy as np

#: Voxel edge in metres. 3 cm is a compromise measured against cuRobo's
#: `collision_cache` obb budget: finer resolution is better geometry and more
#: obstacles than the solver is configured to hold.
VOXEL_M = 0.03

#: Hard cap on emitted obstacles. Exceeding the collision cache silently
#: truncates inside cuRobo, which is the worst possible failure -- the planner
#: would treat dropped obstacles as free space. Truncating HERE is loud.
MAX_OBSTACLES = 480


def perceived_points(env, exclude_robot=True, exclude_held=True):
    """World-frame points from both cameras, robot and held object removed."""
    from .environment_libero import AGENTVIEW, CAM_NAMES, WRISTVIEW

    chunks = []
    for cam_id in (AGENTVIEW, WRISTVIEW):
        name = CAM_NAMES[cam_id]
        raw = env._last_obs.get(f"{name}_depth")
        if raw is None:
            continue
        pts = env._depth_to_points(raw, name).reshape(-1, 3)
        chunks.append(pts[np.isfinite(pts).all(axis=1)])
    if not chunks:
        return np.zeros((0, 3))
    pts = np.concatenate(chunks)

    data, model = env.sim.data, env.sim.model
    drop_ids = list(env._robot_geom_ids) + list(env._gripper_geom_ids)
    held_pts_removed = 0

    if exclude_robot and len(drop_ids):
        # Bounding-sphere rejection, as get_sdf_voxels does. Conservative: it
        # removes a little more than the body, which is the safe direction --
        # an over-removed robot never becomes an obstacle to itself.
        keep = np.ones(len(pts), dtype=bool)
        for gid in drop_ids:
            c = data.geom_xpos[gid]
            r = model.geom_rbound[gid]
            keep &= np.linalg.norm(pts - c, axis=1) > r
        pts = pts[keep]

    if exclude_held:
        # NO OBJECT IDENTITY. Whatever is between the jaws travels with the
        # hand, so for collision purposes it is part of the robot -- and the
        # robot knows where its own jaws are. Dropping points inside that
        # volume needs no lookup of WHICH object is held, which was the last
        # privileged thread in this path.
        from .grasp_detect import holding_by_width

        if holding_by_width(env) and len(pts):
            tips = np.stack([data.geom_xpos[g] for g in env._gripper_geom_ids])
            centre = tips.mean(axis=0)
            reach = float(np.max(np.linalg.norm(tips - centre, axis=1))) + 0.03
            before = len(pts)
            pts = pts[np.linalg.norm(pts - centre, axis=1) > reach]
            held_pts_removed = before - len(pts)
    return pts


def voxelise(points, bounds_min, bounds_max, voxel=VOXEL_M,
             max_obstacles=MAX_OBSTACLES, warn=print):
    """Occupied voxels as cuRobo cuboids, in the WORLD frame."""
    if not len(points):
        return {}, 0
    inside = np.all((points >= bounds_min) & (points < bounds_max), axis=1)
    pts = points[inside]
    if not len(pts):
        return {}, 0

    idx = np.floor((pts - bounds_min) / voxel).astype(np.int64)
    uniq, counts = np.unique(idx, axis=0, return_counts=True)
    order = np.argsort(-counts)          # densest first, so truncation drops
    uniq = uniq[order]                   # the flimsiest evidence, not the best
    n_total = len(uniq)
    if n_total > max_obstacles:
        if warn is not None:
            warn(f"world_depth: {n_total} occupied voxels exceeds the "
                 f"{max_obstacles} obstacle budget; keeping the densest "
                 f"{max_obstacles}. The dropped ones read as FREE SPACE to the "
                 f"planner -- raise the voxel size or the collision cache.")
        uniq = uniq[:max_obstacles]

    centres = bounds_min + (uniq + 0.5) * voxel
    cuboid = {
        f"vox_{i}": {"dims": [voxel] * 3,
                     "pose": [*c.tolist(), 1.0, 0.0, 0.0, 0.0]}
        for i, c in enumerate(centres)
    }
    return {"cuboid": cuboid}, n_total


def perceived_world(env, voxel=VOXEL_M, max_obstacles=MAX_OBSTACLES,
                    warn=print):
    """Depth -> cuRobo world, the non-privileged counterpart of world_export."""
    pts = perceived_points(env)
    world, n_total = voxelise(pts, np.asarray(env.bounds_min),
                              np.asarray(env.bounds_max), voxel,
                              max_obstacles, warn)
    return world, {"points": int(len(pts)), "voxels_total": int(n_total),
                   "voxel_m": voxel}


def main():
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--robot", default="Panda")
    ap.add_argument("--gripper", default="default")
    ap.add_argument("--voxel", type=float, default=VOXEL_M)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from .config import load_config
    from .environment_libero import ReKepLiberoEnv
    from .world_export import counts, export

    cfg = load_config()
    ec = dict(cfg["env"])
    ec["bounds_min"] = cfg["workspace"]["bounds_min"]
    ec["bounds_max"] = cfg["workspace"]["bounds_max"]
    ec["interpolate_pos_step_size"] = cfg["main"]["interpolate_pos_step_size"]
    ec["interpolate_rot_step_size"] = cfg["main"]["interpolate_rot_step_size"]
    env = ReKepLiberoEnv(ec, task_suite=args.suite, task_id=args.task_id,
                         robot=args.robot, gripper=args.gripper,
                         resolution=cfg["libero"]["resolution"])

    world, info = perceived_world(env, voxel=args.voxel)
    oracle, _ = export(env.sim)
    print(f"{args.suite}/{args.task_id}  {args.robot}")
    print(f"  perceived: {info['points']} points -> {info['voxels_total']} "
          f"voxels @ {info['voxel_m']} m -> {counts(world)}")
    print(f"  oracle   : {counts(oracle)}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(world, f)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
