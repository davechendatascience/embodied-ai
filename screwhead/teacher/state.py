"""The optimization teacher's state: what pi_theta reads and what a search's labels are stored
against (BRN-optimization-teacher).

It is the privileged state, per task, with nothing that carries the episode's history except what
a verdict needs: the joints, fingers and tool pose; the execution state the servo and gripper
carry between periods; every free object's pose, box and velocities; every fixture joint and the
fixture body poses LIBERO re-samples at each reset; the task loss's terms; the release watch's
contact record and open losses; and each unnamed movable body's support and face at the episode
start, which the disturbance verdict is judged against. Deliberately absent: MuJoCo's clock and
the step counter (a label must not depend on when it was asked), and any record of the last
commanded level (the finger target already carries it).
"""
from __future__ import annotations

import numpy as np



def _log_se3(T: np.ndarray) -> np.ndarray:
    from ..geometry import kin_np
    return np.asarray(kin_np.log_se3(T[None])[0], float)


class TeacherState:
    """Builds the state vector for one task's environment; `dim` is fixed after construction."""

    def __init__(self, env, verdicts):
        self.env, self.v = env, verdicts
        self.m, self.d = env.scene.raw()
        self.free = sorted(verdicts.free)
        self.fixture_joints = [j for j in range(self.m.njnt)
                               if int(self.m.jnt_type[j]) in (2, 3)
                               and not (self.m.joint(j).name or "").startswith(("robot", "gripper"))]
        self.fixture_bodies = sorted({int(self.m.body_rootid[b]) for b in range(1, self.m.nbody)
                                      if int(self.m.body_parentid[b]) == 0} - set(verdicts.free.values()))
        self.supports = [-1, *sorted({int(self.m.body_rootid[b]) for b in range(1, self.m.nbody)})]
        self.dim = len(self.vector(verdicts.watch(), verdicts.reference()))

    def vector(self, watch, start_reference: dict) -> np.ndarray:
        env, v, m, d = self.env, self.v, self.m, self.d
        r = env.robot
        q = np.asarray(d.qpos[r._ref_joint_pos_indexes], float)
        qd = np.asarray(d.qvel[r._ref_joint_vel_indexes], float)
        fq = np.asarray(d.qpos[r._ref_gripper_joint_pos_indexes], float)
        fqd = np.asarray(d.qvel[r._ref_gripper_joint_vel_indexes], float)
        ts = env.tool_state({"robot0_joint_pos": q, "robot0_gripper_qpos": fq, "robot0_gripper_qvel": fqd})
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = ts["R_tool"], ts["p_tool"]
        servo = env.servo
        parts = [q, qd, fq, fqd, ts["p_tool"], ts["R_tool"][:, :2].ravel("F"), servo.V_prev,
                 np.zeros(6) if servo.T_ref is None else _log_se3(np.linalg.inv(T) @ servo.T_ref),
                 np.zeros(len(q)) if servo.ref is None else servo.ref - q,
                 np.asarray(r.gripper.current_action, float)]
        for n in self.free:
            body = v.free[n]
            R = np.asarray(d.xmat[body], float).reshape(3, 3)
            parts += [np.asarray(d.xpos[body], float), R[:, :2].ravel("F"),
                      np.asarray(d.cvel[body][3:], float), np.asarray(d.cvel[body][:3], float),
                      np.asarray(env.scene.object_box(n).half, float)]
        for j in self.fixture_joints:
            parts.append([float(d.qpos[m.jnt_qposadr[j]]), float(d.qvel[m.jnt_dofadr[j]])])
        for b in self.fixture_bodies:                       # LIBERO re-samples these at every reset
            parts += [np.asarray(m.body_pos[b], float), np.asarray(m.body_quat[b], float)]
        for t in v.loss.terms():
            parts.append([t.distance, t.reach, float(t.satisfied), float(t.held)])
        for n in v.moved:                                   # the release watch, as the verdict reads it
            age, gap, speed = watch.open.get(n, (watch.count, 0.0, 0.0))
            parts.append([float(watch.touching.get(n, False)), float(watch.count - age if n in watch.open else -1),
                          gap, speed])
        for n in v.others:                                  # where each bystander started
            support, face = start_reference[n]
            parts += [np.eye(len(self.supports))[self.supports.index(support)], np.eye(6)[face]]
        return np.concatenate([np.asarray(p, float).ravel() for p in parts]).astype(np.float32)
