"""Export a LIBERO scene as a cuRobo world — the ORACLE obstacle field.

This is E3a's world: obstacle geometry read straight out of MuJoCo, exact and
complete. It is a **privileged input** in the same sense as the mask, the goal
and the joint axis catalogued in `task_spec.py`, and it exists to unblock the
planner, not to be the headline. E3b replaces it with a depth-derived cloud
from the same two cameras, and the difference between the two IS the
measurement of what the oracle was worth.

Nothing here imports cuRobo. It runs in `.venv` beside MuJoCo and writes JSON;
`.venv-curobo` reads that JSON. Same rule as the PointWorld bridge — the file
on disk is the whole interface, and neither venv knows the other exists.

WHAT IS EXPORTED, and why each choice:

  * **Collision geoms only (group 0).** robosuite puts collision geometry in
    group 0 and visual meshes in group 1. Exporting both would double the
    obstacle count and hand the planner decorative geometry to avoid.

  * **Robot geoms are excluded** by body prefix -- a planner models its own
    body from its URDF, and feeding it back as world obstacles makes every
    configuration self-colliding.

  * **Meshes become their oriented bounding boxes by default.** A box that
    contains the mesh is CONSERVATIVE: it can refuse a path that was actually
    free, but it can never miss a collision. That is the correct direction to
    be wrong in, and it is far cheaper on an Orin/Thor than mesh collision
    queries. `--meshes exact` exports vertices/faces when a refusal matters.

    **On LIBERO this approximation never fires.** Measured on `libero_goal/0`:
    of 169 non-robot geoms, every mesh (18) and every cylinder (4) is group 1,
    i.e. visual only. The collision set is 138 boxes and nothing else, so the
    exported oracle world is EXACT rather than conservative. Do not carry that
    assumption to another benchmark without re-running the count.

  * **LIBERO's `floor` plane is group 1** — visual — so it is not exported.
    Harmless for a tabletop task mounted above the table, and wrong the moment
    something can fall to the ground or the base moves. If you need it, pass
    `--include-visual` and accept the 31 decorative geoms that come with it.

  * **Free-jointed objects are exported at their CURRENT pose.** The world is
    a snapshot, not a rig: it is only valid until something moves. A planner
    that opens a drawer must re-export after the drawer moves, and that is a
    property of the task, not a bug here.

Pose convention: cuRobo wants `[x, y, z, qw, qx, qy, qz]`. MuJoCo's native
quaternion order is already w-first, but this reads the world-frame rotation
matrix from `data.geom_xmat` rather than the model-frame `geom_quat`, because
the model frame is relative to the parent body and would be wrong for anything
that has moved.
"""

import argparse
import json

import numpy as np

ROBOT_BODY_PREFIXES = ("robot", "gripper", "mount")

# MuJoCo mjtGeom
PLANE, HFIELD, SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX, MESH = range(8)

# A plane is infinite; cuRobo wants a box. This is how thick and how wide the
# ground plane becomes. 4 m covers LIBERO's table scenes with room to spare.
PLANE_EXTENT = 4.0
PLANE_THICKNESS = 0.05


def _mat_to_quat_wxyz(mat):
    """Rotation matrix -> [w, x, y, z], the order cuRobo expects."""
    m = np.asarray(mat, dtype=np.float64).reshape(3, 3)
    t = np.trace(m)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def _mesh_obb(model, geom_id, mat):
    """Half-extents of the mesh's bounding box, in the geom's own frame.

    Taken from the mesh vertices rather than `geom_rbound`. `NOTES.md` §1
    already paid for that distinction once: `geom_rbound` is the radius of a
    bounding SPHERE, and using it swallowed the table in 517 of a bowl's 1541
    points. A sphere around a flat object is enormous; its box is not.
    """
    adr = int(model.mesh_vertadr[model.geom_dataid[geom_id]])
    num = int(model.mesh_vertnum[model.geom_dataid[geom_id]])
    verts = np.asarray(model.mesh_vert[adr: adr + num], dtype=np.float64).reshape(-1, 3)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    centre_local = (lo + hi) * 0.5
    half = (hi - lo) * 0.5
    return centre_local, half


def export(sim, meshes="obb", include_visual=False,
           robot_prefixes=ROBOT_BODY_PREFIXES, exclude_bodies=()):
    """Build a cuRobo WorldConfig dict from the live simulator state.

    `exclude_bodies` omits objects the robot is HOLDING. A grasped object is
    rigidly attached to the tool and travels with it; leaving it in the world
    means the planner is asked to avoid something moving with the gripper, and
    it will correctly report every subsequent motion infeasible. Pair this with
    `attached()`, which re-expresses the same geometry in the tool frame so it
    can be added to the robot model instead.
    """
    m, d = sim.model, sim.data
    cuboid, sphere, cylinder, mesh_out = {}, {}, {}, {}
    skipped = {"robot": 0, "visual": 0, "unsupported": 0}

    for g in range(m.ngeom):
        body = m.body_id2name(m.geom_bodyid[g]) or ""
        if any(body.startswith(p) for p in robot_prefixes):
            skipped["robot"] += 1
            continue
        if not include_visual and int(m.geom_group[g]) != 0:
            skipped["visual"] += 1
            continue
        if any(body.startswith(e) for e in exclude_bodies):
            skipped["held"] = skipped.get("held", 0) + 1
            continue

        name = m.geom_id2name(g) or f"geom{g}"
        gtype = int(m.geom_type[g])
        size = np.asarray(m.geom_size[g], dtype=np.float64)
        pos = np.asarray(d.geom_xpos[g], dtype=np.float64)
        mat = np.asarray(d.geom_xmat[g], dtype=np.float64).reshape(3, 3)
        quat = _mat_to_quat_wxyz(mat)
        pose = [*pos.tolist(), *quat.tolist()]

        if gtype == BOX:
            cuboid[name] = {"dims": (size * 2.0).tolist(), "pose": pose}
        elif gtype == SPHERE:
            sphere[name] = {"radius": float(size[0]), "position": pos.tolist()}
        elif gtype == CYLINDER:
            cylinder[name] = {"radius": float(size[0]),
                              "height": float(size[1] * 2.0), "pose": pose}
        elif gtype == CAPSULE:
            # A capsule's box is its cylinder plus a radius at each cap. Also
            # conservative, and cuRobo's capsule wants base/tip points which
            # buys nothing here.
            cuboid[name] = {
                "dims": [size[0] * 2.0, size[0] * 2.0, size[1] * 2.0 + size[0] * 2.0],
                "pose": pose,
            }
        elif gtype == PLANE:
            # Infinite in MuJoCo; a slab in cuRobo, pushed DOWN by half its
            # thickness so its top face stays where the plane was. Getting this
            # backwards raises the floor 25 mm and every reach looks short.
            centre = pos - mat[:, 2] * (PLANE_THICKNESS * 0.5)
            cuboid[name] = {
                "dims": [PLANE_EXTENT, PLANE_EXTENT, PLANE_THICKNESS],
                "pose": [*centre.tolist(), *quat.tolist()],
            }
        elif gtype == MESH:
            centre_local, half = _mesh_obb(m, g, mat)
            if meshes == "obb":
                centre = pos + mat @ centre_local
                cuboid[name] = {"dims": (half * 2.0).tolist(),
                                "pose": [*centre.tolist(), *quat.tolist()]}
            else:
                did = int(m.geom_dataid[g])
                va = int(m.mesh_vertadr[did])
                vn = int(m.mesh_vertnum[did])
                fa = int(m.mesh_faceadr[did])
                fn = int(m.mesh_facenum[did])
                mesh_out[name] = {
                    "vertices": np.asarray(
                        m.mesh_vert[va:va + vn], dtype=np.float64
                    ).reshape(-1, 3).tolist(),
                    "faces": np.asarray(
                        m.mesh_face[fa:fa + fn], dtype=np.int64
                    ).reshape(-1, 3).tolist(),
                    "pose": pose,
                }
        else:
            skipped["unsupported"] += 1

    world = {}
    if cuboid:
        world["cuboid"] = cuboid
    if sphere:
        world["sphere"] = sphere
    if cylinder:
        world["cylinder"] = cylinder
    if mesh_out:
        world["mesh"] = mesh_out
    return world, skipped


def attached(sim, object_names, tool_body="robot0_right_hand",
             meshes="obb", include_visual=False):
    """The held object's geometry, expressed in the TOOL frame.

    cuRobo needs a grasped object as part of the ROBOT, not the world: its
    spheres ride on the tool link and are exempt from collision with the
    gripper that holds them. Expressing the geometry in the tool frame is what
    makes that possible, and it is also the claim to verify -- while an object
    is genuinely held, this transform is CONSTANT. A drifting one means the
    grasp is slipping, which is worth knowing separately from a planning
    failure.
    """
    m, d = sim.model, sim.data
    tool = m.body_name2id(tool_body)
    R_wt = np.asarray(d.body_xmat[tool], dtype=np.float64).reshape(3, 3)
    p_wt = np.asarray(d.body_xpos[tool], dtype=np.float64)

    out = {}
    for g in range(m.ngeom):
        if not include_visual and int(m.geom_group[g]) != 0:
            continue
        body = m.body_id2name(m.geom_bodyid[g]) or ""
        if not any(body.startswith(o) for o in object_names):
            continue

        name = m.geom_id2name(g) or f"geom{g}"
        gtype = int(m.geom_type[g])
        size = np.asarray(m.geom_size[g], dtype=np.float64)
        # world -> tool
        R_wg = np.asarray(d.geom_xmat[g], dtype=np.float64).reshape(3, 3)
        p_wg = np.asarray(d.geom_xpos[g], dtype=np.float64)
        R_tg = R_wt.T @ R_wg
        p_tg = R_wt.T @ (p_wg - p_wt)

        if gtype == MESH:
            centre_local, half = _mesh_obb(m, g, R_wg)
            centre = p_tg + R_tg @ centre_local
            dims = (half * 2.0).tolist()
        elif gtype == BOX:
            centre, dims = p_tg, (size * 2.0).tolist()
        elif gtype == SPHERE:
            centre, dims = p_tg, [size[0] * 2.0] * 3
        elif gtype in (CAPSULE, CYLINDER):
            centre = p_tg
            dims = [size[0] * 2.0, size[0] * 2.0, size[1] * 2.0]
        else:
            continue
        out[name] = {"dims": dims,
                     "pose": [*centre.tolist(), *_mat_to_quat_wxyz(R_tg).tolist()]}
    return {"cuboid": out} if out else {}


def refresh(env, meshes="obb"):
    """Current world and held-object geometry, split correctly.

    Call this at every replan. `world_export` is a SNAPSHOT: the drawer moves
    as it opens and a grasped object moves with the hand, so a world computed
    once is stale by the second keypose.
    """
    held = tuple(env._contacting_objects()) if env.is_grasping() else ()
    world, skipped = export(env.sim, meshes=meshes, exclude_bodies=held)
    return world, attached(env.sim, held, meshes=meshes), held, skipped


def counts(world):
    return {k: len(v) for k, v in world.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--robot", default="Panda")
    ap.add_argument("--gripper", default="default")
    ap.add_argument("--init-state-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--meshes", choices=["obb", "exact"], default="obb")
    ap.add_argument("--include-visual", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from .config import load_config
    from .environment_libero import ReKepLiberoEnv

    # ReKepLiberoEnv takes the `env` section flattened with a few keys hoisted
    # out of `workspace`/`main`, not the whole config -- same shape
    # tests/test_drawer_open.py builds.
    cfg = load_config()
    ec = dict(cfg["env"])
    ec["bounds_min"] = cfg["workspace"]["bounds_min"]
    ec["bounds_max"] = cfg["workspace"]["bounds_max"]
    ec["interpolate_pos_step_size"] = cfg["main"]["interpolate_pos_step_size"]
    ec["interpolate_rot_step_size"] = cfg["main"]["interpolate_rot_step_size"]

    env = ReKepLiberoEnv(
        ec, task_suite=args.suite, task_id=args.task_id,
        robot=args.robot, gripper=args.gripper,
        init_state_id=args.init_state_id, reset_seed=args.seed, verbose=False,
    )
    world, skipped = export(env.sim, meshes=args.meshes,
                            include_visual=args.include_visual)
    print(f"{args.robot} / {args.gripper} — {args.suite}/{args.task_id}")
    print(f"  obstacles: {counts(world)}")
    print(f"  skipped:   {skipped}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(world, f)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
