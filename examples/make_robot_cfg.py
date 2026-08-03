"""URDF + collision spheres for whatever arm the simulator is running.

Two lessons are baked in, both paid for:

URDF. A MuJoCo body frame is NOT a URDF link frame when the joint sits off the
body origin. The link frame is the body translated onto its joint:
    T(L_parent -> L_child) = Tr(-jnt_pos(p)) . T(p -> c) . Tr(jnt_pos(c))
Skip it and FK is exact at the zero configuration and drifts everywhere else.

SPHERES. Fitting spheres to ENCLOSE clusters of surface points inflates a
non-spherical link, and an inflated robot collides with a world it is actually
clear of -- measured: configurations the arm genuinely occupied reported up to
55 mm of penetration, and cuRobo then refused every goal. NVIDIA's own UR10e
set is 20 spheres INSCRIBED in the links. So: spheres are placed along each
link's principal axis with a radius taken from a PERCENTILE of the
perpendicular cross-section, not its maximum.
"""

import numpy as np

HINGE, SLIDE, MESH = 3, 2, 7
#: Cross-section percentile for the sphere radius. 100 would enclose the link
#: and is what broke it before; ~65 sits inside the surface.
RADIUS_PCT = 65.0
SPHERES_PER_LINK = 5


def quat_wxyz_to_mat(q):
    w, x, y, z = np.asarray(q, float)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_rpy(R):
    sy = np.clip(-R[2, 0], -1, 1)
    p = np.arcsin(sy)
    if abs(abs(sy) - 1) < 1e-9:
        return float(np.arctan2(-R[1, 2], R[1, 1])), float(p), 0.0
    return (float(np.arctan2(R[2, 1], R[2, 2])), float(p),
            float(np.arctan2(R[1, 0], R[0, 0])))


def _joint_of(m, b):
    n, adr = int(m.body_jntnum[b]), int(m.body_jntadr[b])
    js = [j for j in range(adr, adr + n) if int(m.jnt_type[j]) in (HINGE, SLIDE)]
    return js[0] if js else None


def chain(m, root, tip):
    b, out = m.body_name2id(tip), []
    while b != -1:
        out.append(b)
        if b == m.body_name2id(root):
            break
        b = m.body_parentid[b]
    return list(reversed(out))


def build(sim, root="robot0_base", tip="robot0_right_hand", extra=("gripper0",),
          name="arm"):
    import xml.etree.ElementTree as ET
    m = sim.model
    bodies = chain(m, root, tip)
    for b in range(m.nbody):
        nm = m.body_id2name(b) or ""
        if b not in bodies and nm.startswith(extra):
            bodies.append(b)
    jpos = {b: (np.asarray(m.jnt_pos[_joint_of(m, b)], float)
                if _joint_of(m, b) is not None else np.zeros(3)) for b in bodies}

    robot = ET.Element("robot", {"name": name})
    for b in bodies:
        ln = ET.SubElement(robot, "link", {"name": m.body_id2name(b)})
        it = ET.SubElement(ln, "inertial")
        ET.SubElement(it, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        ET.SubElement(it, "mass", {"value": f"{float(m.body_mass[b]):.9g}"})
        ET.SubElement(it, "inertia", dict(zip(
            ("ixx", "iyy", "izz"), (f"{float(v):.9g}" for v in m.body_inertia[b])),
            ixy="0", ixz="0", iyz="0"))
    for b in bodies:
        p = int(m.body_parentid[b])
        if b == bodies[0] or p not in bodies:
            continue
        T = np.eye(4)
        T[:3, :3] = quat_wxyz_to_mat(m.body_quat[b])
        T[:3, 3] = m.body_pos[b]
        L = np.eye(4); L[:3, 3] = -jpos[p]
        C = np.eye(4); C[:3, 3] = jpos[b]
        T = L @ T @ C
        j = _joint_of(m, b)
        jt = ET.SubElement(robot, "joint", {
            "name": m.joint_id2name(j) if j is not None
            else f"{m.body_id2name(p)}_to_{m.body_id2name(b)}",
            "type": "fixed" if j is None else
            ("revolute" if int(m.jnt_type[j]) == HINGE else "prismatic")})
        r, pi, y = mat_to_rpy(T[:3, :3])
        ET.SubElement(jt, "origin", {"xyz": " ".join(f"{v:.9g}" for v in T[:3, 3]),
                                     "rpy": f"{r:.9g} {pi:.9g} {y:.9g}"})
        ET.SubElement(jt, "parent", {"link": m.body_id2name(p)})
        ET.SubElement(jt, "child", {"link": m.body_id2name(b)})
        if j is not None:
            ET.SubElement(jt, "axis", {"xyz": " ".join(
                f"{v:.9g}" for v in m.jnt_axis[j])})
            lo, hi = (m.jnt_range[j] if m.jnt_limited[j] else (-6.283, 6.283))
            ET.SubElement(jt, "limit", {"lower": f"{lo:.9g}", "upper": f"{hi:.9g}",
                                        "effort": "1000", "velocity": "3.14"})
    ET.indent(robot, space="  ")
    return ET.tostring(robot, encoding="unicode"), bodies, jpos


def spheres(sim, bodies, jpos, per_link=SPHERES_PER_LINK, pct=RADIUS_PCT):
    """Inscribed-ish spheres along each link's principal axis."""
    m = sim.model
    out = {}
    for b in bodies:
        pts = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b or int(m.geom_group[g]) != 0:
                continue
            if int(m.geom_type[g]) != MESH:
                continue
            did = int(m.geom_dataid[g])
            a, n = int(m.mesh_vertadr[did]), int(m.mesh_vertnum[did])
            v = np.asarray(m.mesh_vert[a:a + n], float).reshape(-1, 3)
            pts.append(v @ quat_wxyz_to_mat(m.geom_quat[g]).T + m.geom_pos[g])
        if not pts:
            continue
        P = np.concatenate(pts) - jpos[b]
        c0 = P.mean(0)
        u, s, vt = np.linalg.svd(P - c0, full_matrices=False)
        axis = vt[0]
        t = (P - c0) @ axis
        ss = []
        for tc in np.linspace(t.min(), t.max(), per_link):
            sel = np.abs(t - tc) < max((t.max() - t.min()) / per_link, 1e-4)
            if sel.sum() < 4:
                continue
            perp = np.linalg.norm((P[sel] - c0) - np.outer(t[sel], axis), axis=1)
            ss.append({"center": (c0 + tc * axis).tolist(),
                       "radius": float(np.percentile(perp, pct))})
        if ss:
            out[m.body_id2name(b)] = ss
    return out


def geometric_ignore(sim, spheres, joint_offsets, samples=2000, seed=0,
                     robot_prefixes=("robot", "gripper")):
    """Self-collision exemptions derived from GEOMETRY, not from the tree.

    Adjacency is the obvious rule and the wrong one. NVIDIA's franka set
    exempts pairs that are non-adjacent but come close anywhere in the
    workspace -- link5 to the fingers, link1 to link4, link3 to link6. Emit
    only parent/child and EVERY configuration self-collides, so cuRobo reports
    infeasible for every goal on any robot. Measured: 26 pairs -> 0/7 feasible,
    37 pairs (NVIDIA's structure) -> 7/7, with the world identical.

    So: sample valid configurations the arm can actually hold, record which
    sphere pairs ever overlap, and exempt those. The arm is not colliding with
    itself in these poses -- MuJoCo would have said so -- therefore any overlap
    is an artefact of the sphere approximation and must be ignored.
    """
    import mujoco
    import numpy as np

    m, d = sim.model, sim.data
    robot = None
    for b in range(m.nbody):
        pass
    # actuated arm joints, in model order
    jidx = [j for j in range(m.njnt)
            if (m.body_id2name(m.jnt_bodyid[j]) or "").startswith(robot_prefixes)
            and int(m.jnt_type[j]) in (HINGE, SLIDE)]
    qadr = [int(m.jnt_qposadr[j]) for j in jidx]
    lo = np.array([m.jnt_range[j][0] if m.jnt_limited[j] else -np.pi for j in jidx])
    hi = np.array([m.jnt_range[j][1] if m.jnt_limited[j] else np.pi for j in jidx])

    names = list(spheres)
    centres = {n: np.array([s["center"] for s in spheres[n]]) for n in names}
    radii = {n: np.array([s["radius"] for s in spheres[n]]) for n in names}

    rng = np.random.default_rng(seed)
    backup = d.qpos.copy()
    overlap = {n: set() for n in names}
    for _ in range(samples):
        q = rng.uniform(lo, hi)
        for a, v in zip(qadr, q):
            d.qpos[a] = v
        mujoco.mj_forward(m._model, d._data)
        world = {}
        for n in names:
            b = m.body_name2id(n)
            R = np.asarray(d.body_xmat[b], float).reshape(3, 3)
            p = np.asarray(d.body_xpos[b], float)
            world[n] = (centres[n] - joint_offsets[n]) @ R.T + p
        for i, a in enumerate(names):
            for bnm in names[i + 1:]:
                dmat = np.linalg.norm(world[a][:, None, :] - world[bnm][None], axis=2)
                if (dmat - radii[a][:, None] - radii[bnm][None] < 0).any():
                    overlap[a].add(bnm)
                    overlap[bnm].add(a)
    d.qpos[:] = backup
    mujoco.mj_forward(m._model, d._data)
    return {k: sorted(v) for k, v in overlap.items() if v}
