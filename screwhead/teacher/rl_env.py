"""A LIBERO task as an environment for training the teacher (BRN-rl-teacher-equilibrium-settle,
work in progress -- the pilot tests it).

The teacher minimises time to a settled success. Reward -1 per control step plus the
potential difference Phi(s) - Phi(s') at gamma = 1, Phi the task loss (task_loss.py) plus the
tool's distance to what must move, and Phi = 0 at every absorbing state, so an episode's
return is Phi(s_0) minus its length, less H if it ends in a violation: the potential's scale
cannot change which behaviour is best. The episode ends

  with success   when settled (settle.py): LIBERO accepts, no robot geom touches what was
                 moved, the object is in static equilibrium with too little energy to tip or
                 slide out of acceptance, a goal joint's predicted rest still satisfies it;
  as a failure   at the first violation of
                   gentle placement -- when the moved object loses its last robot contact
                     (checked at every 2 ms substep), its lowest point is more than 5 mm above
                     the surface beneath or it falls faster than 0.05 m/s (DEF-gentle-placement);
                   the scene's state -- another movable body changes what it rests on or the
                     face it rests on;
                   reachability -- the moved object's centre of mass below the arena's surface.

Execution is the new path (Execution lean, anchor, scale_lead): the state every step reads is
forwarded, execution memory is cleared at placements, the servo scales its lead. The action is
the student's: a normalised body twist and one of the student's gripper snap levels. Smoothness
is not in the reward; demonstrations are certified by CTR-teacher-smooth.

Not yet as the claim words it: the scene and reachability checks run at the end of each control
period, not each substep; reachability omits the arm's-reach bound; libero_goal 3's drawer region
is not covered by the loss.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from . import settle
from .task_loss import TaskLoss

GRIPPER_LEVELS = (0.0, 0.026, 0.08)        # m: the student's snap levels (tools/distill.py)
HORIZON = 600                              # LIBERO's evaluation horizon (configs/eval/default.yaml)
GENTLE_GAP = 0.005                         # DEF-gentle-placement
GENTLE_SPEED = 0.05


@dataclass
class StepInfo:
    success: bool = False
    violation: str = ""
    truncated: bool = False


class RLTaskEnv:
    def __init__(self, suite: str, task: int, seed: int = 0, horizon: int = HORIZON, phi_scale: float = 1.0):
        from ..sim.sim_arm import Execution
        from ..sim.task_env import TaskEnv
        self.env = TaskEnv(suite, task, seed=seed, horizon=horizon, render=False,
                           execution=Execution(lean=True, anchor=True, scale_lead=True))
        self.horizon = horizon
        # steps of time cost per unit of potential: the telescoping sum removes it from every
        # return, so it cannot change which behaviour is optimal -- only how fast it is learned
        self.phi_scale = phi_scale
        self.loss = TaskLoss(self.env)
        self.lib = self.env.env.env
        self.m, self.d = self.env.scene.raw()
        self.base = self.env.scene.base
        goal_objects = {g[1] for g in self.loss.goals if g[0] in ("in", "on")}
        goal_regions = {g[2] for g in self.loss.goals if g[0] in ("in", "on")} | {g[1] for g in self.loss.goals}
        self.moved = sorted(goal_objects)
        self.free = self._free_bodies()                              # name -> (root body, qposadr, dofadr)
        named = goal_objects | {n for n in goal_regions if n in self.free}
        self.others = sorted(set(self.free) - named)
        self.fixture_joints = [j for j in range(self.m.njnt) if int(self.m.jnt_type[j]) in (2, 3)
                               and not mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, j).startswith("robot")
                               and not mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, j).startswith("gripper")]
        self.goal_joints = self._goal_joints()
        self.floor = self._arena_surface()
        self.level = 2
        self.t = 0

    # -- episode --------------------------------------------------------------------------
    def reset(self, init_index: int | None = None) -> np.ndarray:
        self.env.reset(init_index)
        self.t, self.level = 0, 2
        self.start = {n: self._support_and_face(n) for n in self.others}
        self.start_pose = {n: self._pose(n) for n in self.free}
        self.phi = self._phi()
        return self.observe()

    def step(self, twist: np.ndarray, level: int) -> tuple[np.ndarray, float, bool, StepInfo]:
        from ..sim.gripper_servo import target_to_channel
        self.level = int(level)
        action = np.concatenate([np.clip(twist, -1, 1), [float(target_to_channel(GRIPPER_LEVELS[self.level]))]])
        released = self._release_watch()
        self.env.execute(action, substep=released.substep)
        self.t += 1
        info = StepInfo()
        info.violation = released.violation or self._scene_violation() or self._reach_violation()
        if info.violation:
            reward, done = -1.0 + self.phi - self.horizon, True
            self.phi = 0.0
        elif self.settled():
            info.success, done = True, True
            reward, self.phi = -1.0 + self.phi, 0.0
        else:
            phi = self._phi()
            reward, self.phi, done = -1.0 + self.phi - phi, phi, False
            info.truncated = self.t >= self.horizon
        return self.observe(), float(reward), done, info

    # -- potential and settling -------------------------------------------------------------
    def _phi(self) -> float:
        return self.phi_scale * float(sum(t.loss + t.reach for t in self.loss.terms()))

    def settled(self) -> bool:
        if not self.env.success():
            return False
        for name in self.moved:
            root = self.free[name][0]
            if self._robot_touches(root):
                return False
            if not settle.in_equilibrium(self.m, self.d, root):
                return False
            energy = settle.kinetic_energy(self.m, self.d, root)
            if energy >= settle.tipping_barrier(self.m, self.d, root):
                return False
            g = next(c for c in self.loss.goals if c[0] in ("in", "on") and c[1] == name)
            slide = settle.min_friction(self.m, self.d, root) * float(self.m.body_subtreemass[root]) \
                * float(np.linalg.norm(self.m.opt.gravity)) * self.loss.slide_margin(g)
            if energy >= slide:
                return False
        for g, joints in self.goal_joints.items():
            probe = self.loss._joint[g]
            for j, pj in zip(joints, probe["joints"], strict=True):
                if pj["side"] * (settle.joint_rest(self.m, self.d, j) - pj["theta"]) <= 0:
                    return False
        return True

    # -- violations -------------------------------------------------------------------------
    def _release_watch(self):
        """A per-substep monitor: when a moved object loses its last robot contact, its gap
        beneath and downward speed must meet DEF-gentle-placement."""
        env = self
        touching = {n: self._robot_touches(self.free[n][0]) for n in self.moved}

        class Watch:
            violation = ""

            def substep(self, _i):
                if self.violation:
                    return
                for n in env.moved:
                    now = env._robot_touches(env.free[n][0])
                    if touching[n] and not now:
                        gap, down = env._gap_below(n), -env._com_velocity(n)[2]
                        if gap > GENTLE_GAP or down > GENTLE_SPEED:
                            self.violation = f"release {n}: gap {gap * 1000:.1f} mm, falling {down:.3f} m/s"
                    touching[n] = now
        return Watch()

    def _scene_violation(self) -> str:
        for n in self.others:
            if self._support_and_face(n) != self.start[n]:
                return f"disturbed {n}"
        return ""

    def _reach_violation(self) -> str:
        for n in self.moved:
            if float(self.d.subtree_com[self.free[n][0]][2]) < self.floor:
                return f"lost {n} below the arena surface"
        return ""

    # -- observation ------------------------------------------------------------------------
    def observe(self) -> np.ndarray:
        r = self.env.robot
        q = self.d.qpos[r._ref_joint_pos_indexes]
        qd = self.d.qvel[r._ref_joint_vel_indexes]
        fq = self.d.qpos[r._ref_gripper_joint_pos_indexes]
        fqd = self.d.qvel[r._ref_gripper_joint_vel_indexes]
        ts = self.env.tool_state({"robot0_joint_pos": q, "robot0_gripper_qpos": fq, "robot0_gripper_qvel": fqd})
        servo = self.env.servo
        ref_err = np.zeros(6) if servo.T_ref is None else _log_se3(np.linalg.inv(_T(ts)) @ servo.T_ref)
        joint_err = np.zeros(7) if servo.ref is None else servo.ref - q
        parts = [q, qd, fq, fqd, ts["p_tool"], ts["R_tool"][:, :2].ravel("F"), servo.V_prev, ref_err, joint_err,
                 np.asarray(r.gripper.current_action, float), np.eye(3)[self.level]]
        for n in sorted(self.free):
            p, R = self._pose(n)
            v = self._com_velocity(n)
            w = np.asarray(self.d.cvel[self.free[n][0]][:3], float)
            sp, sR = self.start_pose[n]
            half = self.env.scene.object_box(n).half
            parts += [p, R[:, :2].ravel("F"), v, w, half, sp, sR[:, :2].ravel("F")]
        for j in self.fixture_joints:
            parts.append([float(self.d.qpos[self.m.jnt_qposadr[j]]), float(self.d.qvel[self.m.jnt_dofadr[j]])])
        for t in self.loss.terms():
            parts.append([t.distance, t.reach, float(t.satisfied), float(t.held)])
        return np.concatenate([np.asarray(x, float).ravel() for x in parts]).astype(np.float32)

    # -- scene queries --------------------------------------------------------------------
    def _free_bodies(self) -> dict:
        out = {}
        for name, obj_body in self.lib.obj_body_id.items():
            j = int(self.m.body_jntadr[obj_body])
            if j >= 0 and int(self.m.jnt_type[j]) == 0:
                out[name] = (int(obj_body), int(self.m.jnt_qposadr[j]), int(self.m.jnt_dofadr[j]))
        return out

    def _goal_joints(self) -> dict:
        out = {}
        for g in self.loss.goals:
            if g in self.loss._joint:
                out[g] = [self.m.joint(p["joint"]).id for p in self.loss._joint[g]["joints"]]
        return out

    def _arena_surface(self) -> float:
        """The table's collision top when the arena has a table, else the floor."""
        body = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "table")
        if body < 0:
            return 0.0
        tops = [float(self.d.geom_xpos[g][2] + np.abs(self.d.geom_xmat[g].reshape(3, 3) @ self.m.geom_size[g])[2])
                for g in range(self.m.ngeom)
                if int(self.m.geom_bodyid[g]) == body and int(self.m.geom_type[g]) == mujoco.mjtGeom.mjGEOM_BOX
                and (self.m.geom_contype[g] or self.m.geom_conaffinity[g])]
        return max(tops) if tops else 0.0

    def _pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        b = self.free[name][0]
        return (np.asarray(self.d.xpos[b], float) - self.base, np.asarray(self.d.xmat[b], float).reshape(3, 3))

    def _com_velocity(self, name: str) -> np.ndarray:
        mujoco.mj_subtreeVel(self.m, self.d)
        return np.asarray(self.d.subtree_linvel[self.free[name][0]], float)

    def _robot_touches(self, root: int) -> bool:
        bodies = settle.subtree_bodies(self.m, root)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            b1, b2 = int(self.m.geom_bodyid[c.geom1]), int(self.m.geom_bodyid[c.geom2])
            other = b2 if b1 in bodies else (b1 if b2 in bodies else -1)
            if other >= 0 and self.m.body(other).name.startswith(("robot", "gripper")):
                return True
        return False

    def _gap_below(self, name: str) -> float:
        """Height of the object's lowest point above the first surface beneath that point."""
        loss_boxes = self.loss._boxes.get(name)
        if loss_boxes is None:
            ids = [self.m.geom(n).id for n in self.lib.get_object(name).contact_geoms]
            from ..geometry.box_distance import BoxSet
            loss_boxes = self.loss._boxes[name] = BoxSet(self.m, ids)
        c, R = loss_boxes.pose(self.d)
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
        verts = (c[:, None, :] + np.einsum("pij,pkj->pki", R, corners[None] * loss_boxes.h[:, None, :])).reshape(-1, 3)
        low = verts[np.argmin(verts[:, 2])]
        # cast from inside the object's own hull (the ray skips the object), not from its lowest
        # point: a resting object sits a few microns into its support, and a ray started there
        # begins inside the support and reports its far side
        start = np.array([low[0], low[1], float(self.d.subtree_com[self.free[name][0]][2])])
        gid = np.zeros(1, np.int32)
        dist = mujoco.mj_ray(self.m, self.d, start, np.array([0.0, 0.0, -1.0]), None, 1, self.free[name][0], gid)
        return max(float(low[2] - (start[2] - dist)), 0.0) if dist >= 0 else np.inf

    def _support_and_face(self, name: str) -> tuple[int, int]:
        """The body beneath the object's centre of mass (a downward ray), and which of its six
        body-frame faces points down."""
        root = self.free[name][0]
        com = np.asarray(self.d.subtree_com[root], float)
        gid = np.zeros(1, np.int32)
        mujoco.mj_ray(self.m, self.d, com, np.array([0.0, 0.0, -1.0]), None, 1, root, gid)
        below = int(self.m.geom_bodyid[gid[0]]) if gid[0] >= 0 else -1
        while below > 0 and int(self.m.body_parentid[below]) != 0:
            below = int(self.m.body_parentid[below])                  # the object, not its part
        R = np.asarray(self.d.xmat[root], float).reshape(3, 3)
        down = R.T @ np.array([0.0, 0.0, -1.0])                      # world down, in the body frame
        axis = int(np.argmax(np.abs(down)))
        return below, axis * 2 + int(down[axis] > 0)


def _T(ts: dict) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = ts["R_tool"], ts["p_tool"]
    return T


def _log_se3(T: np.ndarray) -> np.ndarray:
    from ..geometry.kin_np import log_se3
    return np.asarray(log_se3(T[None]))[0]
