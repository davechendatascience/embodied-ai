"""Joint-space robot points from MuJoCo, without stepping physics.

Reuses the area-weighted mesh sampling already verified in
`rekep_libero.gripper_points` -- do NOT resample here, and do NOT fall back to
`geom_rbound`, which is the CIRCUMSCRIBED radius and turns a 63 x 93 x 206 mm
gripper into a 234 mm blob (`NOTES.md` section 4).

What is new is the driving variable. `GripperPoints.at_ee_pose()` binds every
gripper geom rigidly to one end-effector frame, which is exact only while the
jaw does not move -- hence `check_binding()` and the re-binds the drawer run
needed when the jaw crept 23 mm under load. Driving joint values instead makes
the finger motion part of the kinematics rather than an error term, and it is
the same variable the URDF backend and a real controller take.

Configurations are evaluated by writing `qpos`, running MuJoCo's own forward
kinematics, reading the geoms, and restoring. No dynamics, no contact, no
integration -- so a candidate costs kinematics, not a simulator step.
"""

import numpy as np

from rekep_libero.gripper_points import GripperPoints

from .base import RobotPoints


class MujocoRobotPoints(RobotPoints):
    def __init__(self, env, n=500, rng=None):
        self.env = env
        self._sampler = GripperPoints(env, n, rng)
        self.n = self._sampler.n
        self.gids = self._sampler.gids

        model = env.sim.model
        # The joints that actually move these geoms: walk each gripper body up
        # to the world root and collect the joints on the way. Derived from the
        # kinematic tree rather than from a name pattern, so it is right for
        # whichever arm the scene contains.
        adr, names = [], []
        seen = set()
        for gid in self.gids:
            body = int(model.geom_bodyid[gid])
            while body > 0:
                for k in range(int(model.body_jntnum[body])):
                    j = int(model.body_jntadr[body]) + k
                    if j not in seen:
                        seen.add(j)
                        adr.append(int(model.jnt_qposadr[j]))
                        names.append(model.joint_id2name(j))
                body = int(model.body_parentid[body])
        order = np.argsort(adr)
        self.qadr = np.array(adr)[order]
        self.joint_names = [names[i] for i in order]

    def current_config(self):
        return np.asarray(self.env.sim.data.qpos[self.qadr]).copy()

    def at_config(self, q):
        """(points, normals) at joint vector `q`, leaving the sim as it was.

        `qpos` is written, forward kinematics run, geoms read, and the original
        state put back. Restoring is not optional: the planner evaluates
        hundreds of hypothetical configurations per tick against a simulator
        that must still be at the real state when it executes one.
        """
        data = self.env.sim.data
        saved = np.asarray(data.qpos).copy()
        try:
            data.qpos[self.qadr] = np.asarray(q, dtype=np.float64)
            self.env.sim.forward()
            pts = np.empty((self.n, 3))
            nrm = np.empty((self.n, 3))
            for gid in self.gids:
                m = self._sampler.which == gid
                R = np.asarray(data.geom_xmat[gid]).reshape(3, 3)
                pts[m] = self._sampler.local[m] @ R.T + np.asarray(data.geom_xpos[gid])
                nrm[m] = self._sampler.local_n[m] @ R.T
        finally:
            data.qpos[:] = saved
            self.env.sim.forward()
        return pts, nrm

    def at_configs(self, qs):
        """(..., n, 3) world points for a batch of configs (..., J).

        `at_config` saves and restores `qpos` around EVERY evaluation, which
        costs two `sim.forward()` calls each. A planning tick evaluates 73
        candidates x 11 frames = 803 configurations, so that bookkeeping was
        1606 forwards where 804 do. The save/restore is still exact -- it just
        brackets the whole batch instead of each element.

        Identical arithmetic to `at_config`, so points agree bit for bit.
        """
        qs = np.asarray(qs, dtype=np.float64)
        flat = qs.reshape(-1, qs.shape[-1])
        data = self.env.sim.data
        saved = np.asarray(data.qpos).copy()
        out = np.empty((len(flat), self.n, 3), dtype=np.float32)
        masks = [(gid, self._sampler.which == gid) for gid in self.gids]
        try:
            for i, q in enumerate(flat):
                data.qpos[self.qadr] = q
                self.env.sim.forward()
                for gid, m in masks:
                    R = np.asarray(data.geom_xmat[gid]).reshape(3, 3)
                    out[i][m] = self._sampler.local[m] @ R.T + \
                        np.asarray(data.geom_xpos[gid])
        finally:
            data.qpos[:] = saved
            self.env.sim.forward()
        return out.reshape(qs.shape[:-1] + (self.n, 3))

    def ee_pose_at(self, q):
        """The end-effector pose a joint vector produces, (7,) pos + quat xyzw.

        SIMULATOR EXECUTION ADAPTER ONLY -- never part of planning or cost.
        Planning is joint space because that is PointWorld's actual action
        representation; this exists solely because LIBERO is driven by an OSC
        POSE controller. On the robot, joints are commanded directly and this
        disappears. Nothing else may depend on it.

        It reads the grip SITE from live `sim.data`, not `env.get_ee_pose()`,
        which returns a cached observation dict and would report the same pose
        for every hypothetical configuration.
        """
        model, data = self.env.sim.model, self.env.sim.data
        if not hasattr(self, "_site"):
            names = [model.site_id2name(i) for i in range(model.nsite)]
            cand = [n for n in names if n and "grip_site" in n] or \
                   [n for n in names if n and "eef" in n]
            self._site = model.site_name2id(cand[0]) if cand else None
        saved = np.asarray(data.qpos).copy()
        try:
            data.qpos[self.qadr] = np.asarray(q, dtype=np.float64)
            self.env.sim.forward()
            if self._site is None:
                pose = np.asarray(self.env.get_ee_pose()).copy()
            else:
                import transform_utils as T
                R = np.asarray(data.site_xmat[self._site]).reshape(3, 3)
                pose = np.concatenate([np.asarray(data.site_xpos[self._site]).copy(),
                                       T.mat2quat(R)])
        finally:
            data.qpos[:] = saved
            self.env.sim.forward()
        return pose

    def point_sensitivity(self, eps=1e-3):
        """Metres of ROBOT-POINT motion per radian, per joint. MEASURED.

        Deliberately NOT an end-effector Jacobian. PointWorld's action is the
        robot's point flow -- `RobotSampler.compute_points(joint_values)` takes
        named joint values and returns points, and no end effector appears
        anywhere in its contract. "The end effector" is robot-specific: a
        dual-arm UR7e has two and no canonical one, and our own
        `get_ee_pose()` reads a CACHED observation dict, so an ee-based
        sensitivity silently returned zero for every joint and collapsed the
        action scale to nothing.

        Measuring the mean displacement of the sampled points is faithful to
        the utility, needs no notion of an end effector, works unchanged on a
        bimanual robot, and reads live geometry. It is also the quantity the
        model actually sees.
        """
        q0 = self.current_config()
        p0 = self.at_config(q0)[0]
        out = np.zeros(len(q0))
        for i in range(len(q0)):
            q = q0.copy()
            q[i] += eps
            out[i] = np.linalg.norm(self.at_config(q)[0] - p0, axis=1).mean() / eps
        return out

    def live(self):
        """(points, normals) at the CURRENT state, with no save/restore.

        The recorder's path: no hypothetical to evaluate, so skip the write and
        the two forward passes.
        """
        return self._sampler(self.env)

    def __call__(self, env=None):
        """Alias for `live()`, so this drops into `live_observation` unchanged.

        The recorder and the planner share one observation builder; keeping the
        call signature means the sim backend can be swapped in without touching
        it, and the URDF backend inherits the same courtesy.
        """
        return self.live()
