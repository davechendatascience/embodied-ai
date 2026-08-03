"""Do the fitted spheres actually cover the robot? Surface, not just vertices.

`collision_spheres.py` clusters each link's collision VERTICES and takes each
cluster's enclosing sphere, so every vertex is inside a sphere by construction.
That is not the same as covering the mesh SURFACE: a triangle whose corners
land in different clusters can bulge outside every sphere, and a gap there is
a hole the planner will happily drive a link through.

This samples points across the triangles themselves and measures the worst
uncovered one. Under-coverage is a real safety hole; over-coverage only costs
refused paths, so the tolerance is deliberately asymmetric and small.

    MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/test_collision_spheres.py
"""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tests"))

from rekep_libero import add_rekep_to_path  # noqa: E402

add_rekep_to_path()

SAMPLES_PER_TRI = 6
# Every sphere already carries the 5 mm --buffer, so a surface point should be
# comfortably inside. Allow 1 mm of slack for the sampling itself.
UNCOVERED_TOL_MM = 1.0


def sample_triangles(verts, faces, per_tri=SAMPLES_PER_TRI, seed=0):
    """Barycentric samples across every triangle, plus the vertices."""
    rng = np.random.default_rng(seed)
    tri = verts[faces]                                    # (F, 3, 3)
    a = rng.random((len(tri), per_tri, 1))
    b = rng.random((len(tri), per_tri, 1))
    flip = (a + b) > 1.0
    a = np.where(flip, 1.0 - a, a)
    b = np.where(flip, 1.0 - b, b)
    pts = (tri[:, None, 0] + a * (tri[:, None, 1] - tri[:, None, 0])
           + b * (tri[:, None, 2] - tri[:, None, 0]))
    return np.concatenate([pts.reshape(-1, 3), verts])


def main():
    import yaml

    import test_ur5e_scene as T
    from rekep_libero import fixtures as fx
    from rekep_libero.collision_spheres import (
        MESH,
        _chain_transform,
        _quat_wxyz_to_mat,
    )
    from rekep_libero.urdf_export import HINGE, SLIDE, _joints_of

    failures = []
    for robot, gripper, yml in (
            ("Panda", "PandaGripper", "configs/curobo/panda_pandagripper.yml"),
            ("UR5e", "PandaGripper", "configs/curobo/ur5e_pandagripper.yml")):

        if robot == "Panda":
            env = T.build("Panda", gripper=gripper)
        else:
            ref_env = T.build("Panda", gripper="default")
            ref = fx.snapshot(ref_env.sim)
            ref_env.close()
            env = T.build(robot, gripper=gripper, fixture_ref=ref)

        cfg = yaml.safe_load(open(os.path.join(REPO, yml)))
        spheres = cfg["robot_cfg"]["kinematics"]["collision_spheres"]
        model = env.sim.model
        link_set = set(spheres)

        joint_offsets = {}
        for n in link_set:
            b = model.body_name2id(n)
            js = [j for j in _joints_of(model, b)
                  if int(model.jnt_type[j]) in (HINGE, SLIDE)]
            joint_offsets[n] = (np.asarray(model.jnt_pos[js[0]], dtype=np.float64)
                                if js else np.zeros(3))

        worst = 0.0
        worst_link = ""
        n_pts = 0
        for g in range(model.ngeom):
            if int(model.geom_group[g]) != 0 or int(model.geom_type[g]) != MESH:
                continue
            body = int(model.geom_bodyid[g])
            bname = model.body_id2name(body) or ""
            if not bname.startswith(("robot", "gripper")):
                continue
            T_body_anc, anc = _chain_transform(model, body, link_set)
            if anc is None or anc not in spheres:
                continue

            did = int(model.geom_dataid[g])
            va, vn = int(model.mesh_vertadr[did]), int(model.mesh_vertnum[did])
            fa, fn = int(model.mesh_faceadr[did]), int(model.mesh_facenum[did])
            verts = np.asarray(model.mesh_vert[va:va + vn],
                               dtype=np.float64).reshape(-1, 3)
            faces = np.asarray(model.mesh_face[fa:fa + fn],
                               dtype=np.int64).reshape(-1, 3)
            pts = sample_triangles(verts, faces)

            R = _quat_wxyz_to_mat(model.geom_quat[g])
            pos = np.asarray(model.geom_pos[g], dtype=np.float64)
            in_body = pts @ R.T + pos
            in_link = (in_body @ T_body_anc[:3, :3].T + T_body_anc[:3, 3]
                       - joint_offsets[anc])

            centres = np.array([s["center"] for s in spheres[anc]])
            radii = np.array([s["radius"] for s in spheres[anc]])
            d = np.linalg.norm(in_link[:, None, :] - centres[None], axis=2) - radii
            outside = d.min(axis=1)                       # <=0 means covered
            n_pts += len(in_link)
            if outside.max() > worst:
                worst, worst_link = float(outside.max()), anc

        total = sum(len(v) for v in spheres.values())
        print(f"{robot} + {gripper}: {len(spheres)} links, {total} spheres, "
              f"{n_pts} surface points")
        print(f"  worst uncovered surface point: {worst * 1000:.3f} mm "
              f"({worst_link or 'none'})")
        if worst * 1000.0 > UNCOVERED_TOL_MM:
            failures.append(f"{robot}: {worst * 1000:.2f} mm of {worst_link} "
                            f"lies outside every sphere")
        env.close()

    print()
    if failures:
        print("COVERAGE FAILS")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("COVERAGE HOLDS — every sampled surface point is inside a sphere")
    return 0


if __name__ == "__main__":
    sys.exit(main())
