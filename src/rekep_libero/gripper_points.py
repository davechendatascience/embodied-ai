"""Gripper surface points, live from the sim or for a hypothetical ee pose.

PointWorld's action IS the robot's point flow, so this module produces the
actions. Two callers need it and they need different things:

  the recorder   where the gripper's points ARE, at each step of a rollout
  the planner    where they WOULD BE, for a candidate ee pose, without
                 stepping physics -- because MPPI scores hundreds of
                 candidates per tick and cannot simulate any of them

It lives here rather than in the recording script because both callers must
agree exactly. A second copy of this sampler would be a second definition of
"the action", and the last time a predicate was duplicated in this project the
copy silently kept measuring contaminated clouds after the original was fixed
(`NOTES.md` section 2).

Do NOT sample `geom_rbound`. It is the CIRCUMSCRIBED radius, so the Franka
hand's 119.6 mm turns a 63 x 93 x 206 mm gripper into a 234 mm blob --
measured. Inflating the cloud misstates the action, `dist2robot` for every
scene point, and the model's idea of what is about to be touched.
"""

import numpy as np

from . import add_rekep_to_path

add_rekep_to_path()

import transform_utils as T  # noqa: E402  (upstream ReKep, flat imports)

MJ_GEOM_BOX = 6
MJ_GEOM_MESH = 7


def _mesh_triangles(model, gid):
    """(A, B, C) vertex triples of one mesh geom, in the geom's own frame.

    MuJoCo bakes `mesh_pos`/`mesh_quat` into the compiled geom frame, so
    `geom_xpos`/`geom_xmat` map these vertices straight to the world. Verified:
    the Franka hand mesh comes out 63.0 x 93.4 x 206.5 mm against a declared
    `geom_size * 2` of 63.1 x 99.3 x 208.2 mm.
    """
    did = int(model.geom_dataid[gid])
    va, vn = int(model.mesh_vertadr[did]), int(model.mesh_vertnum[did])
    fa, fn = int(model.mesh_faceadr[did]), int(model.mesh_facenum[did])
    verts = np.asarray(model.mesh_vert).reshape(-1, 3)[va:va + vn].astype(np.float64)
    faces = np.asarray(model.mesh_face).reshape(-1, 3)[fa:fa + fn]
    assert faces.max() < vn, "mesh face indices are not mesh-local"
    return verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]


def _box_triangles(model, gid):
    """A box's 6 faces as 12 triangles, in the geom's own frame."""
    sx, sy, sz = np.asarray(model.geom_size[gid][:3], dtype=np.float64)
    c = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
    quads = [(0, 1, 3, 2), (4, 6, 7, 5),      # -x, +x
             (0, 4, 5, 1), (2, 3, 7, 6),      # -y, +y
             (0, 2, 6, 4), (1, 5, 7, 3)]      # -z, +z
    tris = np.array([(q[0], q[1], q[2]) for q in quads]
                    + [(q[0], q[2], q[3]) for q in quads])
    return c[tris[:, 0]], c[tris[:, 1]], c[tris[:, 2]]


def _pose_matrix(pos, quat_xyzw):
    m = np.eye(4)
    m[:3, :3] = T.quat2mat(np.asarray(quat_xyzw))
    m[:3, 3] = np.asarray(pos)
    return m


class GripperPoints:
    """`n` gripper points with the SAME identity at every timestep.

    PointWorld samples 300-500 points per gripper off the URDF mesh, by face
    area, and propagates them by forward kinematics. MuJoCo carries the same
    meshes and gives us the geom poses, so this is that method rather than an
    approximation of it -- and, critically, it carries no dependence on WHICH
    arm this is, which is the property the whole representation rests on.

    Per body, the visual geoms win over the collision hulls, matching "sample
    off the URDF mesh"; bodies with no visual geom (the finger-tip pads) fall
    back to their collision geometry so every part is still represented.

    The barycentric coordinates are drawn ONCE, in each geom's local frame, and
    then carried through that geom's pose. Point i is the same material point
    for all t, so `robot_flows[t+1] - robot_flows[t]` is real motion. Drawing
    fresh samples per step instead makes the velocity and acceleration
    channels -- 6 of the 16 robot features -- pure sampling noise.
    """

    def __init__(self, env, n=500, rng=None):
        rng = rng or np.random.default_rng(0)
        model = env.sim.model

        by_body = {}
        for gid in (int(g) for g in env._gripper_geom_ids):
            by_body.setdefault(int(model.geom_bodyid[gid]), []).append(gid)
        chosen = []
        for gids in by_body.values():
            visual = [g for g in gids
                      if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0]
            chosen.extend(visual or gids)

        tris = {}
        for gid in chosen:
            t = int(model.geom_type[gid])
            if t == MJ_GEOM_MESH:
                tris[gid] = _mesh_triangles(model, gid)
            elif t == MJ_GEOM_BOX:
                tris[gid] = _box_triangles(model, gid)
            else:
                raise NotImplementedError(
                    f"gripper geom {gid} has MuJoCo type {t}; add a surface "
                    "sampler for it rather than falling back to a bounding sphere"
                )

        # Points are shared out by surface area, so density is uniform over the
        # gripper instead of uniform over geoms.
        areas = {g: np.linalg.norm(np.cross(B - A, C - A), axis=1) / 2.0
                 for g, (A, B, C) in tris.items()}
        total = sum(a.sum() for a in areas.values())
        counts = {g: int(round(n * areas[g].sum() / total)) for g in chosen}
        counts[chosen[0]] += n - sum(counts.values())    # absorb rounding

        which, local, local_n = [], [], []
        for gid in chosen:
            k = counts[gid]
            if k <= 0:
                continue
            A, B, C = tris[gid]
            face = rng.choice(len(A), size=k, p=areas[gid] / areas[gid].sum())
            # sqrt trick: uniform over the triangle, not clustered at a corner
            u, v = rng.random(k), rng.random(k)
            su = np.sqrt(u)
            w = np.stack([1 - su, su * (1 - v), su * v], axis=1)[:, :, None]
            a, b, c = A[face], B[face], C[face]
            nrm = np.cross(b - a, c - a)
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
            which.append(np.full(k, gid))
            local.append(w[:, 0] * a + w[:, 1] * b + w[:, 2] * c)
            local_n.append(nrm)

        self.gids = np.array(chosen)
        self.which = np.concatenate(which)
        self.local = np.concatenate(local, 0)
        self.local_n = np.concatenate(local_n, 0)
        self.n = len(self.which)
        assert self.n == n, f"{self.n} != {n} gripper points"
        self._rel = None
        self._bound_jaw = None

    # ------------------------------------------------------------- live path
    def __call__(self, env):
        """(points, normals) in the world frame at the CURRENT sim state."""
        data = env.sim.data
        pts = np.empty_like(self.local)
        nrm = np.empty_like(self.local_n)
        for gid in self.gids:
            m = self.which == gid
            R = np.asarray(data.geom_xmat[gid]).reshape(3, 3)
            pts[m] = self.local[m] @ R.T + np.asarray(data.geom_xpos[gid])
            nrm[m] = self.local_n[m] @ R.T
        return pts, nrm

    # ---------------------------------------------------- hypothetical poses
    def bind(self, env):
        """Freeze each geom's pose RELATIVE to the ee, for FK without physics.

        Every gripper geom is rigidly attached to the ee frame once the finger
        joints stop moving, so one capture of `ee^-1 * geom` is enough to place
        the whole gripper at any candidate ee pose. That is what makes MPPI
        affordable: a candidate trajectory costs a few matrix products instead
        of a simulator step.

        The rigidity assumption is exactly as good as "the jaw does not move",
        so the jaw opening is recorded here and checked on use. Re-`bind()`
        after opening or closing the gripper.
        """
        data = env.sim.data
        ee = env.get_ee_pose()
        world2ee = np.linalg.inv(_pose_matrix(ee[:3], ee[3:]))

        self._rel = {}
        for gid in self.gids:
            geom = np.eye(4)
            geom[:3, :3] = np.asarray(data.geom_xmat[gid]).reshape(3, 3)
            geom[:3, 3] = np.asarray(data.geom_xpos[gid])
            self._rel[int(gid)] = world2ee @ geom
        self._bound_jaw = self._jaw(env)
        return self

    @staticmethod
    def _jaw(env):
        q = env._last_obs.get("robot0_gripper_qpos")
        return None if q is None else np.asarray(q).copy()

    def check_binding(self, env, tol=2e-3):
        """How far the jaw has moved since `bind`. Returns metres, or None."""
        if self._bound_jaw is None:
            return None
        now = self._jaw(env)
        if now is None:
            return None
        return float(np.abs(now - self._bound_jaw).max())

    def at_ee_pose(self, ee_pos, ee_quat):
        """(points, normals) for a HYPOTHETICAL ee pose. No physics stepped."""
        if self._rel is None:
            raise RuntimeError("call bind(env) before at_ee_pose()")
        ee = _pose_matrix(ee_pos, ee_quat)
        pts = np.empty_like(self.local)
        nrm = np.empty_like(self.local_n)
        for gid in self.gids:
            m = self.which == gid
            g = ee @ self._rel[int(gid)]
            pts[m] = self.local[m] @ g[:3, :3].T + g[:3, 3]
            nrm[m] = self.local_n[m] @ g[:3, :3].T
        return pts, nrm

    def trajectory(self, ee_poses):
        """(T, n, 3) robot flows for a sequence of ee poses [(x,y,z,qx,qy,qz,qw)].

        This is a candidate action in exactly the layout the bridge wants --
        WORLD coordinates, uncentred. Centring happens inside the service;
        pre-centring double-shifts and puts the gripper a metre off
        (`NOTES.md` section 4).
        """
        out = np.empty((len(ee_poses), self.n, 3), dtype=np.float32)
        for i, pose in enumerate(ee_poses):
            pose = np.asarray(pose, dtype=np.float64)
            out[i] = self.at_ee_pose(pose[:3], pose[3:])[0]
        return out
