"""Fit cuRobo collision spheres to the robot MuJoCo is actually simulating.

cuRobo models the robot body as spheres. This checkout ships sphere sets for
`ur10e` and `franka` and nothing for a UR5e, and certainly nothing for a UR5e
wearing a Panda hand, so they have to be fitted -- and fitted to the SAME model
the URDF came from, or the planner avoids collisions for a robot of the wrong
shape.

Measured on `libero_goal/0`: the robot's collision set (group 0) is 16 geoms
per arm, almost all MESHES -- one per link -- plus a few boxes and a cylinder
on the mount. So there are no primitives to read off; spheres are fitted to
mesh vertices.

METHOD, and why this one:

  Cluster each link's collision vertices with Lloyd's algorithm, then take each
  cluster's enclosing sphere (centroid, max vertex distance). That guarantees
  every VERTEX is covered by construction, which is checkable, and it degrades
  gracefully: more clusters means tighter cover at linear cost. `k` scales with
  the link's extent so a long forearm gets more spheres than a wrist.

  It is NOT a tight cover of the mesh SURFACE -- a triangle whose vertices sit
  in different clusters can bulge slightly outside every sphere. For collision
  avoidance the safe direction is to over-cover, so `--buffer` inflates every
  radius, and `tests/test_collision_spheres.py` measures the worst uncovered
  surface point rather than assuming there is none.

FRAMES. Sphere centres are expressed in the URDF LINK frame, which is the
MuJoCo body frame translated onto its joint (see `urdf_export.py`):

    centre_link = geom_R . centre_geom + geom_pos - jnt_pos(body)

THE MOUNT IS EXCLUDED BY DEFAULT, and the reason is measured. Folding the
pedestal and controller box into `robot0_base` produced 14 spheres of radius
0.185-0.294 m reaching to world z = 1.190 -- 0.28 m ABOVE the table top at
0.912. Those engulf the table and most of the workspace, so every
configuration reports a collision and planning returns nothing. The mount is
static, sits below the table, and is not reachable by the arm in these tasks,
so leaving it out is safe here and wrong in general -- `--include-mount`
restores it for a scene where the arm can actually reach its own pedestal.
"""

import argparse

import numpy as np

SPHERE, CAPSULE, CYLINDER, BOX, MESH = 2, 3, 5, 6, 7

#: metres of link extent per sphere; smaller = more spheres = tighter cover
EXTENT_PER_SPHERE = 0.025
MAX_SPHERES_PER_LINK = 48
MIN_SPHERES_PER_LINK = 1


def _quat_wxyz_to_mat(q):
    w, x, y, z = np.asarray(q, dtype=np.float64)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _geom_points(model, g, n_box=4):
    """Points sampling geom `g`, in the geom's own frame."""
    gtype = int(model.geom_type[g])
    size = np.asarray(model.geom_size[g], dtype=np.float64)
    if gtype == MESH:
        did = int(model.geom_dataid[g])
        adr = int(model.mesh_vertadr[did])
        num = int(model.mesh_vertnum[did])
        verts = np.asarray(model.mesh_vert[adr:adr + num],
                           dtype=np.float64).reshape(-1, 3)
        fa = int(model.mesh_faceadr[did])
        fn = int(model.mesh_facenum[did])
        faces = np.asarray(model.mesh_face[fa:fa + fn],
                           dtype=np.int64).reshape(-1, 3)
        # AREA-WEIGHTED SURFACE SAMPLES, not raw vertices. Clustering vertices
        # clusters by MESH DENSITY, and meshes pack vertices at the caps: on
        # the UR5e upper arm that left 159 mm of the shaft outside every
        # sphere -- a hole a planner drives a link straight through.
        # tests/test_collision_spheres.py measures exactly that.
        tri = verts[faces]
        area = 0.5 * np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        if area.sum() <= 0:
            return verts
        rng = np.random.default_rng(0)
        n = 4000
        pick = rng.choice(len(tri), size=n, p=area / area.sum())
        a = rng.random((n, 1))
        b = rng.random((n, 1))
        flip = (a + b) > 1.0
        a = np.where(flip, 1.0 - a, a)
        b = np.where(flip, 1.0 - b, b)
        t = tri[pick]
        pts = t[:, 0] + a * (t[:, 1] - t[:, 0]) + b * (t[:, 2] - t[:, 0])
        return np.concatenate([pts, verts])
    if gtype == SPHERE:
        return np.zeros((1, 3))
    if gtype == BOX:
        # a grid over the box, not just corners: corners alone put the enclosing
        # sphere's centre at the box centre and its radius at the diagonal,
        # which is a very loose cover for a flat box
        ax = [np.linspace(-s, s, n_box) for s in size[:3]]
        return np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    if gtype in (CAPSULE, CYLINDER):
        r, half = float(size[0]), float(size[1])
        zs = np.linspace(-half, half, max(int(2 * half / (r + 1e-9)) + 2, 3))
        ring = np.array([[np.cos(t) * r, np.sin(t) * r, 0.0]
                         for t in np.linspace(0, 2 * np.pi, 9)[:-1]])
        return np.concatenate([ring + [0, 0, z] for z in zs])
    return np.zeros((0, 3))


def _lloyd(points, k, iters=40, seed=0):
    """Minimal k-means. scipy is available but this keeps the seed explicit."""
    rng = np.random.default_rng(seed)
    if len(points) <= k:
        return points.copy(), np.arange(len(points))
    # k-means++ style seeding: first point random, rest farthest-from-chosen
    centres = [points[rng.integers(len(points))]]
    for _ in range(k - 1):
        d = np.min([np.linalg.norm(points - c, axis=1) for c in centres], axis=0)
        centres.append(points[int(np.argmax(d))])
    centres = np.stack(centres)
    labels = np.zeros(len(points), dtype=int)
    for _ in range(iters):
        d = np.linalg.norm(points[:, None, :] - centres[None, :, :], axis=2)
        new = np.argmin(d, axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for i in range(len(centres)):
            m = labels == i
            if m.any():
                centres[i] = points[m].mean(axis=0)
    return centres, labels


def _chain_transform(model, body, ancestors):
    """T from `body` frame into the nearest ancestor present in `ancestors`."""
    T = np.eye(4)
    b = body
    while b != -1 and model.body_id2name(b) not in ancestors:
        R = _quat_wxyz_to_mat(model.body_quat[b])
        step = np.eye(4)
        step[:3, :3] = R
        step[:3, 3] = np.asarray(model.body_pos[b], dtype=np.float64)
        T = step @ T
        b = int(model.body_parentid[b])
    if b == -1:
        return None, None
    return T, model.body_id2name(b)


def fit(sim, link_names, joint_offsets, buffer=0.0, seed=0,
        robot_prefixes=("robot", "gripper"), include_mount=False):
    """{link_name: [{center, radius}, ...]} in URDF link frames."""
    model = sim.model
    if include_mount:
        robot_prefixes = tuple(robot_prefixes) + ("mount",)
    link_set = set(link_names)
    per_link_points = {n: [] for n in link_names}

    for g in range(model.ngeom):
        if int(model.geom_group[g]) != 0:
            continue
        body = int(model.geom_bodyid[g])
        bname = model.body_id2name(body) or ""
        if not any(bname.startswith(p) for p in robot_prefixes):
            continue

        T_body_anc, anc = _chain_transform(model, body, link_set)
        if anc is None:
            continue

        pts = _geom_points(model, g)
        if not len(pts):
            continue
        R = _quat_wxyz_to_mat(model.geom_quat[g])
        pos = np.asarray(model.geom_pos[g], dtype=np.float64)
        in_body = pts @ R.T + pos                       # geom -> body
        in_anc = in_body @ T_body_anc[:3, :3].T + T_body_anc[:3, 3]
        per_link_points[anc].append(in_anc - joint_offsets[anc])

    out = {}
    for name in link_names:
        chunks = per_link_points[name]
        if not chunks:
            continue
        pts = np.concatenate(chunks)
        extent = float(np.linalg.norm(pts.max(0) - pts.min(0)))
        k = int(np.clip(round(extent / EXTENT_PER_SPHERE),
                        MIN_SPHERES_PER_LINK, MAX_SPHERES_PER_LINK))
        centres, labels = _lloyd(pts, k, seed=seed)
        spheres = []
        for i in range(len(centres)):
            m = labels == i if len(labels) == len(pts) else None
            member = pts[m] if m is not None and m.any() else centres[i:i + 1]
            c = member.mean(axis=0)
            r = float(np.max(np.linalg.norm(member - c, axis=1))) + buffer
            if r <= 1e-6:
                r = 0.005 + buffer
            spheres.append({"center": [float(v) for v in c], "radius": r})
        out[name] = spheres
    return out


def adjacency_ignore(model, link_names):
    """self_collision_ignore: each link ignores its parent and children.

    Adjacent links touch by construction, so without this every configuration
    is self-colliding and IK returns nothing -- which reads as "the arm cannot
    reach anything" rather than as a missing config block.
    """
    link_set = set(link_names)
    ignore = {n: [] for n in link_names}
    for name in link_names:
        b = model.body_name2id(name)
        p = int(model.body_parentid[b])
        pn = model.body_id2name(p) if p != -1 else None
        if pn in link_set:
            ignore[name].append(pn)
            ignore[pn].append(name)
    # the two fingers pass close to each other and to the hand at every opening
    fingers = [n for n in link_names if "finger" in n]
    for a in fingers:
        for c in fingers:
            if a != c:
                ignore[a].append(c)
    return {k: sorted(set(v)) for k, v in ignore.items() if v}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--robot", default="UR5e")
    ap.add_argument("--gripper", default="PandaGripper")
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--buffer", type=float, default=0.005)
    ap.add_argument("--include-mount", action="store_true",
                    help="fold the pedestal/controller box into the base")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import os
    import sys
    import xml.etree.ElementTree as ET

    import yaml

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "tests"))
    import test_ur5e_scene as T  # noqa: E402

    from . import fixtures as fx
    from .urdf_export import HINGE, SLIDE, _joints_of

    if args.robot == "Panda":
        env = T.build("Panda", gripper=args.gripper)
    else:
        ref_env = T.build("Panda", gripper="default")
        ref = fx.snapshot(ref_env.sim)
        ref_env.close()
        env = T.build(args.robot, gripper=args.gripper, fixture_ref=ref)

    model = env.sim.model
    root = ET.parse(args.urdf).getroot()
    link_names = [ln.get("name") for ln in root.findall("link")]

    joint_offsets = {}
    for n in link_names:
        b = model.body_name2id(n)
        js = [j for j in _joints_of(model, b)
              if int(model.jnt_type[j]) in (HINGE, SLIDE)]
        joint_offsets[n] = (np.asarray(model.jnt_pos[js[0]], dtype=np.float64)
                            if js else np.zeros(3))

    spheres = fit(env.sim, link_names, joint_offsets, buffer=args.buffer,
                  include_mount=args.include_mount)
    ignore = adjacency_ignore(model, link_names)

    cfg = {"robot_cfg": {"kinematics": {
        "urdf_path": os.path.abspath(args.urdf),
        "base_link": "robot0_base",
        "tool_frames": ["robot0_right_hand"],
        "collision_link_names": sorted(spheres),
        "collision_spheres": spheres,
        "collision_sphere_buffer": 0.0,
        "self_collision_ignore": ignore,
        "self_collision_buffer": {n: 0.0 for n in spheres},
    }}}
    with open(args.out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    total = sum(len(v) for v in spheres.values())
    print(f"{args.robot} + {args.gripper}: {len(spheres)} collision links, "
          f"{total} spheres -> {args.out}")
    for n in sorted(spheres):
        rs = [s["radius"] for s in spheres[n]]
        print(f"  {n:<34} {len(rs):>2} spheres  r {min(rs):.3f}-{max(rs):.3f} m")
    env.close()


if __name__ == "__main__":
    main()
