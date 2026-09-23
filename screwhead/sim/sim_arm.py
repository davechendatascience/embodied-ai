"""What both LIBERO environments do the same way: open a task, execute an action, reset a
scene, hold the arm, find a start.

PrivilegedEnv (teacher_env.py, the libero_spatial pipeline) and TaskEnv (task_env.py,
any task) grew the same settling hold, object-speed check, contact test and start-pose
sampler by copying -- and then the same action decode, LIBERO loading and snapshot, which
drifted: the student's env reset LIBERO's fixtures unseeded, computed the tool pose with
a different forward-kinematics implementation, and kept its own contact loop. The
execution path lives here once (AXM-one-execution-path). A subclass provides `rng`,
`start`, `_scene_seed`, `horizon` and a `label` for error messages, and calls `_open`.

It is also the one place that reaches into robosuite's private attributes
(`_ref_joint_pos_indexes`, `_get_observations`), so that when robosuite changes, one
file does.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

from . import contacts
from ..geometry.frames import axis_rot
from ..geometry.kinematics import fk

START_MIN_TOOL_Z = 0.08       # m above the base: keep a randomised start clear of the table
START_MIN_SIGMA = 0.02        # a start this close to singular is rejected
CAMERA_PX = 128
SEED_MOD = 2**32 - 1          # numpy's legacy seed range
REST_SPEED = 0.005            # m/s: the scene is at rest when no free object moves faster
HOLD_BAND = 0.003             # m: jaws asked this much narrower than they are are driven closed
INITIAL_SETTLE = 10           # control steps held after loading an initial state
SETTLE_CHUNK = 5              # then in chunks of this many, until at rest
SETTLE_ROUNDS = 10            # at most this many chunks


@dataclass(frozen=True)
class Execution:
    """How actions are executed -- the part that must be identical for teacher and student."""
    kp: float = 4000.0
    servo_iters: int = 1
    settle_steps: int = 5
    gripper_mode: str = "target"      # "target": a target aperture; "command": robosuite's -1/0/+1
    hard_reset: bool = False
    max_lin_acc: float = 2.0          # m/s^2 the commanded twist may change by (servo.py)
    max_ang_acc: float = 5.0          # rad/s^2
    max_lin_acc_holding: float | None = 0.5   # m/s^2 while the jaws hold something (servo.grip):
    #                                   the rack's bottle 1 of 50 at 2.0, 25 of 25 at 0.5 while held
    joint_ramp: float = 1.0           # fraction of each control period the joint goal is ramped over
    #                                   (joint_ramp.py); 0 steps it, as LIBERO does
    joint_step: float = 0.1           # rad the joint goal may move per period: robosuite's output_max
    #                                   and the servo's lead limit. LIBERO's 0.05 was sized for a stepped
    #                                   goal; a ramped arm trails by a period more, the 0.05 clips
    #                                   saturated, and the tool sank 52 mm below its transit plane and
    #                                   surged at 0.64 m/s against a 0.25 m/s command
    # Off by default: the certified skill teacher runs without them until its grasp switches are
    # re-validated (BRN-teacher-switches-dwell); the RL teacher runs with them.
    lean: bool = False                # a control period without env.step's bookkeeping, then one
    #                                   mj_forward and a forced observation of that state
    #                                   (BRN-lean-step-keeps-controller-cache, BRN-policies-read-one-
    #                                   forwarded-state)
    anchor: bool = False              # reset execution memory at every placement (BRN-reset-anchors-
    #                                   all-execution)
    scale_lead: bool = False          # scale the servo's whole lead instead of clipping per joint
    #                                   (BRN-servo-lead-scaled-uniformly)


class SimArm:
    label = "env"

    # -- opening a task -------------------------------------------------------------------
    def _open(self, suite: str, task_index: int, render: bool, ex: Execution) -> str:
        """Load a LIBERO task and set up the execution path; returns its bddl path."""
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        from .gripper_servo import GripperServo
        from ..geometry.interface import ActionSpec
        from .libero_env import build_chain, check_loaded_model, gripper_geom, register_ur5e
        from .servo import TwistServo
        register_ur5e()
        assert ex.gripper_mode in ("command", "target"), ex.gripper_mode
        self.execution = ex                 # kept whole: an episode record replays through it
        self.kp, self.gripper_mode, self.settle_steps = ex.kp, ex.gripper_mode, ex.settle_steps
        self.gripper_servo = GripperServo()
        self.spec = ActionSpec()
        self.scale = np.array([self.spec.max_angular_speed] * 3 + [self.spec.max_linear_speed] * 3)
        bm = benchmark.get_benchmark_dict()[suite]()
        task = bm.get_task(task_index)
        self.language = task.language
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        kw = (dict(camera_heights=CAMERA_PX, camera_widths=CAMERA_PX) if render else
              dict(use_camera_obs=False, has_offscreen_renderer=False))
        # a hard reset reloads the MuJoCo model (911 ms of a 1058 ms reset, measured), and
        # set_init_state overwrites the full state afterwards anyway
        self.env = OffScreenRenderEnv(bddl_file_name=bddl, robots=["Panda"], gripper_types="PandaGripper",
                                      controller="JOINT_POSITION", hard_reset=ex.hard_reset, **kw)
        self.joint_step = ex.joint_step      # read by robosuite when a reset rebuilds the controller
        self.robot.controller_config.update(output_max=ex.joint_step, output_min=-ex.joint_step)
        _load = torch.load        # LIBERO pickles its init states; torch>=2.6 refuses unless told
        torch.load = lambda *a, **k: _load(*a, **{**k, "weights_only": False})
        try:
            self.init_states = bm.get_task_init_states(task_index)
        finally:
            torch.load = _load
        self.env.reset()
        flange, _ = gripper_geom(self.env)
        self.chain = build_chain("panda", flange)
        check_loaded_model(self.robot.robot_model.file, "panda")
        self.servo = TwistServo(self.chain, self.spec, ex.joint_step, iters=ex.servo_iters, max_lag=ex.joint_step)
        self.servo.max_lin_acc, self.servo.max_ang_acc = ex.max_lin_acc, ex.max_ang_acc
        self.servo.max_lin_acc_holding = ex.max_lin_acc_holding
        self.servo.scale_lead = ex.scale_lead
        self.joint_ramp = ex.joint_ramp
        self.lean, self.anchoring = ex.lean, ex.anchor
        self.t = 0
        return bddl

    # -- scenes -------------------------------------------------------------------------
    def _reset_scene(self, k: int) -> None:
        """env.reset(), with LIBERO's fixture placement for init state k made repeatable.

        Every reset re-samples fixture poses (a cabinet within +-1 cm of its region) from
        numpy's global RNG, and writes them to model.body_pos -- which set_init_state does
        not restore. Unseeded, two resets moved the cabinet 12 mm apart, so identical runs
        diverged, and a state settled with the fixture in one place was replayed with it in
        another: an object resting ON the fixture started embedded in it or floating.
        """
        np.random.seed((self._scene_seed * 1009 + k) % SEED_MOD)
        self.env.reset()

    def _settled_init_state(self, k: int) -> np.ndarray:
        """LIBERO's initial state k, held until the objects in it are at rest -- not for a
        fixed count: in some init states an object is still sliding off a neighbour after
        10 steps (a bowl at 48.6 deg tilt moving 0.10 m/s, measured)."""
        from .libero_env import remap_init_state
        self._reset_scene(k)
        self.env.set_init_state(remap_init_state(self.init_states[k], self.env.sim))
        self._anchor()
        self._settle(INITIAL_SETTLE)
        for _ in range(SETTLE_ROUNDS):
            if self._max_object_speed() < REST_SPEED:
                break
            self._settle(SETTLE_CHUNK)
        return np.asarray(self.env.sim.get_state().flatten()).copy()

    # -- acting -------------------------------------------------------------------------
    def execute(self, action: np.ndarray, substep=None) -> tuple[dict, bool]:
        """action: 6 normalised body-twist + 1 gripper channel, each in [-1, 1].

        The twist is executed through TwistServo onto an absolute joint reference (per-step
        deltas lose 18% of every step to controller lag, and it compounds -- servo.py); the
        gripper channel is robosuite's command, or a target aperture run by GripperServo.
        `substep(i)`, lean execution only, is called after each 2 ms physics substep."""
        a = np.clip(np.asarray(action, np.float64), -1, 1)
        o = self.observe()
        cmd = np.zeros(self.env.env.action_dim)
        gq, gv = o["robot0_gripper_qpos"], o.get("robot0_gripper_qvel", np.zeros(2))
        # holding (the servo's latch): from the period both fingers touch one body that is not the
        # robot until the jaws are commanded open -- read from contacts and this period's action,
        # the same whatever policy acts. "Asked narrower than the jaws are" alone is true of every
        # free close (80 mm down to a 30 mm pre-grasp), and bounding those overran the approach.
        from . import contacts
        from .gripper_servo import channel_to_target
        if self.gripper_mode == "target":
            asked = float(channel_to_target(a[6])) < float(gq[0] - gq[1]) - HOLD_BAND
        else:
            asked = float(a[6]) > 0.0
        self.servo.grip(asked, contacts.pinched(self.env.sim.model, self.env.sim.data))
        cmd[:7] = self.servo.command(np.asarray(o["robot0_joint_pos"]), a[:6] * self.scale)
        if self.gripper_mode == "target":
            cmd[-1] = self.gripper_servo.command(float(channel_to_target(a[6])),
                                                 float(gq[0] - gq[1]), float(gv[0] - gv[1]))
        else:
            cmd[-1] = a[6]
        self._gains()
        if self.lean:
            self._advance(cmd, substep)
            raw, done = self.observe(), self.success()
        else:
            raw, _, done, _ = self.env.step(cmd)
        self.t += 1
        self.raw = raw
        return raw, bool(done)

    def _advance(self, cmd: np.ndarray, substep=None) -> None:
        """One control period of env.step's physics without its bookkeeping, then one forward.

        Per 2 ms substep: mj_step1; robosuite's own joint controller, its cache refreshed only
        when its new_update flag says robosuite would refresh it (a reset leaves it stale on
        purpose for the first period); set_goal on the first substep; clipped torques and the
        gripper action into ctrl; mj_step2. What env.step does besides -- two more mj_forward
        per substep, observables, the robot's recent-value buffers, LIBERO's visual flags,
        reward -- writes nothing the next substep's physics reads, so the integration state and
        the controller's joint_pos, joint_vel, mass matrix and ramp stay bit-identical. It is
        not a drop-in for env.step's outputs: the observation is the forwarded end-of-period
        one (env.step's comes from a mid-period substep, up to 0.11 rad behind), success read
        after the forward can fire on a different step (11 of 120 teacher episodes), LIBERO's
        visual flags (the stove burner) are not updated for renders, and robosuite's horizon
        never ends the episode. Measured 2.7-2.8x faster for the period alone, 2.1-2.2x for
        the whole execute() without rendering. The closing forward makes every quantity
        derived from positions current for whoever reads the state next. `substep(i)` is
        called after each substep."""
        import mujoco
        env = self.env.env
        m, d = env.sim.model._model, env.sim.data._data
        robot = self.robot
        c = robot.controller
        arm, grip = cmd[:c.control_dim], cmd[c.control_dim:]
        low, high = robot.torque_limits
        mass = np.empty((m.nv, m.nv))
        for i in range(round(env.control_timestep / env.model_timestep)):
            mujoco.mj_step1(m, d)
            if c.new_update:
                c.joint_pos = np.array(d.qpos[c.qpos_index])
                c.joint_vel = np.array(d.qvel[c.qvel_index])
                mujoco.mj_fullM(m, mass, d.qM)
                c.mass_matrix = mass[c.qvel_index, :][:, c.qvel_index]
                c.new_update = False
            if i == 0:
                c.set_goal(arm)
            torques = np.clip(c.run_controller(), low, high)
            robot.torques = torques
            robot.grip_action(gripper=robot.gripper, gripper_action=grip)
            d.ctrl[robot._ref_joint_actuator_indexes] = torques
            mujoco.mj_step2(m, d)
            if substep is not None:
                substep(i)
        env.timestep += 1
        env.cur_time += env.control_timestep
        mujoco.mj_forward(m, d)

    def _anchor(self) -> None:
        """Clear execution memory left from before a placement: robosuite's finger target to
        fully open (where every settle pass ends), the joint controller's cache refreshed from
        the placed state, the joint ramp restarted there (BRN-reset-anchors-all-execution)."""
        if not self.anchoring:
            return
        from .joint_ramp import restart
        robot = self.robot
        robot.gripper.current_action = np.array([1.0, -1.0])
        idx = [self.env.sim.model.actuator_name2id(a) for a in robot.gripper.actuators]  # as grip_action maps it
        lo, hi = self.env.sim.model.actuator_ctrlrange[idx].T
        self.env.sim.data.ctrl[idx] = 0.5 * (hi + lo) + 0.5 * (hi - lo) * robot.gripper.current_action
        robot.controller.update(force=True)
        e = self.env.env
        restart(robot.controller, self.joint_ramp, round(e.control_timestep / e.model_timestep))

    def tool_state(self, raw: dict | None = None) -> dict:
        """Tool pose (base frame) and jaw aperture and its rate, from the joint readings."""
        from ..geometry.kin_np import NpChain
        from ..geometry.kin_np import fk as fk_np
        raw = raw or getattr(self, "raw", None) or self.observe()
        T = fk_np(NpChain.of(self.chain), np.asarray(raw["robot0_joint_pos"], float))[0]
        gq, gv = raw["robot0_gripper_qpos"], raw.get("robot0_gripper_qvel", np.zeros(2))
        return dict(R_tool=T[:3, :3], p_tool=T[:3, 3], aperture=float(gq[0] - gq[1]),
                    aperture_rate=float(gv[0] - gv[1]))

    def success(self) -> bool:
        """LIBERO's own goal check."""
        return bool(self.env.env._check_success())

    # -- robosuite internals, touched only here ---------------------------------------
    @property
    def robot(self):
        return self.env.env.robots[0]

    @property
    def joint_indexes(self):
        return self.robot._ref_joint_pos_indexes

    @property
    def gripper_indexes(self):
        return self.robot._ref_gripper_joint_pos_indexes

    def render(self, camera: str, px: int) -> np.ndarray:
        """An upright px x px image from a named camera -- for videos; the observation
        cameras are CAMERA_PX."""
        return np.ascontiguousarray(self.env.sim.render(width=px, height=px, camera_name=camera)[::-1])

    def observe(self) -> dict:
        return self.env.env._get_observations(force_update=True)

    # -- holding still ----------------------------------------------------------------
    def _gains(self):
        from .joint_ramp import install
        from .libero_env import set_joint_gains
        set_joint_gains(self.env, self.kp)
        e = self.env.env
        install(self.robot.controller, self.joint_ramp, round(e.control_timestep / e.model_timestep))

    def _settle(self, n: int = 3, gripper: float = -1.0):
        """Hold the arm where it is, as an ABSOLUTE joint target.

        A zero action on robosuite's joint controller is a zero DELTA, and right after
        set_init_state it does not hold: measured, the tool moves 16 mm on the first step
        and 22 mm by the third, while an absolute hold keeps it within 0.0 mm. A servo
        armed after a zero-delta settle integrates every later twist from a start 22 mm
        off the demonstrations'.
        """
        hold = np.asarray(self.observe()["robot0_joint_pos"])
        for _ in range(n):
            meas = np.asarray(self.observe()["robot0_joint_pos"])
            cmd = np.zeros(self.env.env.action_dim)
            cmd[:7] = np.clip((hold - meas) / self.joint_step, -1, 1)
            cmd[-1] = gripper
            self._gains()
            if self.lean:
                self._advance(cmd)
            else:
                self.env.step(cmd)

    def _max_object_speed(self) -> float:
        """Fastest linear speed of any free object -- the scene is at rest when it is small."""
        m, d = self.env.sim.model, self.env.sim.data
        v = 0.0
        for j in range(m.njnt):
            if int(m.jnt_type[j]) == 0 and not contacts.is_robot(m.joint_id2name(j) or ""):
                a = int(m.jnt_dofadr[j])
                v = max(v, float(np.linalg.norm(d.qvel[a:a + 3])))
        return v

    def _robot_contact(self) -> bool:
        return contacts.robot_in_contact(self.env.sim.model, self.env.sim.data)

    # -- a random start -----------------------------------------------------------------
    def _randomize_start(self, batch: int = 16, max_rounds: int = 10) -> None:
        """Move the arm to a random TOOL pose near LIBERO's start, via IK.

        Sampled in task space -- position box, yaw about vertical, tilt about a random
        horizontal axis -- because that is what the approach depends on; jittering joints
        gives a tool distribution nobody chose. The IK seed is perturbed so the elbow (the
        null space of a 7-DoF arm) varies too. A candidate is kept only if Newton
        converged, it is away from singularity, and the arm penetrates nothing.
        """
        from ..geometry.ik import sigma_min, solve_ik
        sim = self.env.sim
        idx = self.joint_indexes
        q0 = sim.data.qpos[idx].copy()
        T0 = fk(self.chain, torch.tensor(q0)[None])[0].numpy()
        for attempt in range(max_rounds):
            # BRN-reset-anchors-all-execution: an episode's draws come from a generator seeded by
            # (seed, episode index), its redraws from (seed, episode, attempt). Drawing from the
            # shared stream instead made a start pose depend on every reset before it, so the same
            # episode replayed on its own began somewhere else.
            rng = np.random.default_rng((self.seed, self.episode, attempt))
            targets, seeds = self._start_candidates(T0, q0, batch, rng)
            res = solve_ik(self.chain, torch.tensor(np.stack(targets)), torch.tensor(np.stack(seeds)),
                           lam=0.02, max_iters=200, trust=0.2)
            ok = res["converged"] & (sigma_min(self.chain, res["theta"]) > START_MIN_SIGMA)
            for k in torch.nonzero(ok).flatten().tolist():
                sim.data.qpos[idx] = res["theta"][k].numpy()
                sim.data.qvel[:] = 0.0
                sim.forward()
                if not self._robot_contact():
                    self._anchor()
                    self._settle(self.settle_steps)
                    return
            sim.data.qpos[idx] = q0
            sim.forward()
        raise RuntimeError(f"{self.label}: no collision-free reachable start pose found")

    def _start_candidates(self, T0: np.ndarray, q0: np.ndarray, batch: int, rng=None):
        st = self.start
        rng = self.rng if rng is None else rng
        targets, seeds = [], []
        for _ in range(batch):
            dp = np.array([rng.uniform(-st["xy"], st["xy"]), rng.uniform(-st["xy"], st["xy"]),
                           rng.uniform(-st["z"], st["z"])])
            yaw = rng.uniform(-st["yaw"], st["yaw"])
            ax = rng.normal(size=2)
            ax = np.array([*ax / (np.linalg.norm(ax) + 1e-12), 0.0])
            tilt = rng.uniform(-st["tilt"], st["tilt"])
            T = T0.copy()
            T[:3, :3] = axis_rot(np.array([0, 0, 1.0]), yaw) @ axis_rot(ax, tilt) @ T0[:3, :3]
            T[:3, 3] = T0[:3, 3] + dp
            T[2, 3] = max(T[2, 3], START_MIN_TOOL_Z)
            targets.append(T)
            seeds.append(q0 + rng.normal(0, st["null"], size=len(q0)))
        return targets, seeds
