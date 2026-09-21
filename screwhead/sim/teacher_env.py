"""A LIBERO task the teacher can see into.

The teacher is allowed what the deployed policy is not: the simulator's own
object poses. Everything privileged is confined to `obs()`, so the boundary
between teacher and student is one function, and nothing downstream can mistake
one for the other.

Two things make this environment a test of perception rather than of memory:

  PLACEMENT. The goal object is displaced from LIBERO's recorded start by a
  uniform draw over a disc. LIBERO-spatial alone varies it by 12 mm (sd), inside
  the grasp basin of a trajectory aimed at the mean, which is why a blind policy
  solves it. A draw that is ejected on settling -- placed intersecting a
  neighbour -- is rejected and redrawn, since it tests a collision.

  ACTION SPACE. The teacher emits exactly what the student will: a body twist
  normalised by the action-space scale, and a gripper command, decoded to joints
  by damped least squares on the arm's own chain. A teacher that acted in joint
  space could not be distilled into a twist head without a second, unverified
  conversion.

Frames: positions are in the robot base frame. FK on the chain gives the tool
there directly; objects arrive in world and are shifted by the base pose, which
is measured from the simulator rather than assumed (residual against the grip
site: 0.00 mm).
"""
from __future__ import annotations


import numpy as np
import torch

from . import contacts
from ..geometry.kinematics import fk
from .sim_arm import Execution, SimArm

TARGET = "akita_black_bowl_1"
RECEPTACLE = "plate_1"
N_TASKS = 10
EJECT_MM = 10.0


class PrivilegedEnv(SimArm):
    # these keywords are the constructor API of distill.py, collect_scripted.py, scripted_eval.py and
    # record_rollout.py, so they are not regrouped into a config
    def __init__(self, task_index: int, suite: str = "libero_spatial",  # noqa: PLR0913 -- public keyword API
                 radius_m: float = 0.0, horizon: int = 300, seed: int = 0, kp: float = 4000.0,
                 render: bool = False, hard_reset: bool = False, servo_iters: int = 1, settle_steps: int = 5,
                 start_xy_m: float = 0.0, start_z_m: float = 0.0, start_yaw_deg: float = 0.0,
                 start_tilt_deg: float = 0.0, start_null_rad: float = 0.0,
                 shaping: bool = False, gamma: float = 0.99, success_bonus: float = 10.0,
                 rich_obs: bool = False, shaping_gamma: float = 1.0,
                 layout_radius: float = 0.0, layout_check=None, layout_tries: int = 20,
                 gripper_mode: str = "command"):
        self.ti, self.radius, self.horizon = task_index, radius_m, horizon
        self.rng = np.random.default_rng(seed)
        self._scene_seed = seed
        # the scripted teacher emits robosuite's -1/0/+1, hence the "command" default here;
        # everything else about execution is SimArm's, shared with TaskEnv
        self._open(suite, task_index, render, Execution(kp=kp, servo_iters=servo_iters, settle_steps=settle_steps,
                                                         gripper_mode=gripper_mode, hard_reset=hard_reset))
        self.onehot = np.eye(N_TASKS, dtype=np.float32)[task_index]
        self.t = 0
        self.last_obs = None
        self._settled: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.start = dict(xy=start_xy_m, z=start_z_m, yaw=np.deg2rad(start_yaw_deg),
                          tilt=np.deg2rad(start_tilt_deg), null=start_null_rad)
        self.start_offset = None
        from ..scripted.progress import PickPlaceGeometry
        self.geom = PickPlaceGeometry()
        self.shaping, self.gamma, self.success_bonus = shaping, gamma, success_bonus
        self.rich_obs = rich_obs
        self.shaping_gamma = shaping_gamma
        self.layout_radius, self.layout_check, self.layout_tries = layout_radius, layout_check, layout_tries
        self.layout = None
        self.layout_rejected = 0
        self.ref = None            # rest_z, d0_reach, d0_carry: fixed at the start of an episode
        self.ever_lifted = False
        self.phi = 0.0

    # -- simulator handles (re-fetched: a reset can rebuild the sim) -----------------
    def _ids(self):
        m = self.env.sim.model
        j = m.joint_name2id(f"{TARGET}_joint0")
        return (int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j]), int(m.jnt_bodyid[j]),
                int(m.jnt_bodyid[m.joint_name2id(f"{RECEPTACLE}_joint0")]),
                m.body_name2id("robot0_base"))

    @property
    def label(self) -> str:
        return f"task {self.ti}"

    # -- episode --------------------------------------------------------------------
    def reset(self, init_index: int | None = None, max_tries: int = 20) -> np.ndarray:
        """Start a LIBERO init state, then displace the goal object from where it RESTS.

        LIBERO's bundled init states spawn the bowl 72 mm above the table; it
        falls to rest (898.4 mm) in the first few steps. Displacing it before
        that and comparing against the spawn height rejects every valid draw as
        an ejection. So: settle, record the resting pose, displace from it,
        settle again, and accept only if the bowl stayed on the table surface
        and landed within EJECT_MM of where it was put.
        """
        self.episode = getattr(self, "episode", 0) + 1     # consumers key per-episode caches on this
        k = int(self.rng.integers(len(self.init_states))) if init_index is None else init_index
        self._settle_init_state(k)
        settled, rest = self._settled[k]
        if self.layout_radius > 0 and init_index is None:
            found = self._compatible_init_state(k, settled, rest)
            if found is None:
                return self.reset(init_index=None, max_tries=max_tries)
            k, settled, rest = found
        now, rest = self._place(k, settled, rest, max_tries)
        if any(v > 0 for v in self.start.values()):
            self._randomize_start()
        self.begin()
        self.placement = now - rest           # `now` is a view: the bowl after the start settled too
        self.ever_lifted = False
        self.last_obs = self.obs()            # refreshes self.raw, which snapshot() reads
        self.set_reference()
        if self.layout_check is not None and not self.layout_check(self):
            # a layout the demonstration program cannot solve (no verified grasp, or
            # the plate out of reach) is redrawn rather than kept as an impossible episode
            self.layout_rejected += 1
            if self.layout_rejected < self.layout_tries:
                return self.reset(init_index=init_index, max_tries=max_tries)
        self.layout_rejected = 0
        if self.rich_obs:
            self.last_obs = np.concatenate([self.last_obs, self._rich(self.snapshot())])
        return self.last_obs

    def _settle_init_state(self, k: int) -> None:
        """Cache init state k settled to rest, with the goal object's resting position."""
        if k in self._settled:
            return
        # The drop from LIBERO's spawn height is the same every time for a
        # given init state: pay for it once. Resets were 145 ms, 124 of them
        # settling, and a synchronous vector env waits on its slowest reset.
        state = self._settled_init_state(k)
        qadr, *_ = self._ids()
        self._settled[k] = (state, self.env.sim.data.qpos[qadr:qadr + 3].copy())

    def _layout_sampler(self):
        if self.layout is None:
            from ..scripted.layouts import LayoutSampler
            self.layout = LayoutSampler(self, radius=self.layout_radius)
        return self.layout

    def _compatible_init_state(self, k: int, settled, rest):
        """Redraw settled init states until one the layout grouping can move. Returns
        (k, settled, rest), or None when the draw lands on an init state not settled yet
        (the caller then starts the reset over)."""
        for _ in range(len(self.init_states)):
            self._reset_scene(k)
            self.env.set_init_state(settled)
            if self._layout_sampler().compatible():
                break
            k = int(self.rng.integers(len(self.init_states)))
            if k not in self._settled:
                return None
            settled, rest = self._settled[k]
        return k, settled, rest

    def _place(self, k: int, settled, rest, max_tries: int):
        """Lay out the objects and displace the goal object, redrawing a placement that
        was ejected on settling. Returns the goal object's position (a live view of qpos)
        and the rest position it was displaced from."""
        for _ in range(max_tries):
            self._reset_scene(k)                  # the fixtures init state k was settled with
            self.env.set_init_state(settled)
            sim = self.env.sim
            qadr, vadr, *_ = self._ids()
            if self.layout_radius > 0:
                if not self._layout_sampler().sample(self.rng, settle=self._settle):
                    continue
                rest = sim.data.qpos[qadr:qadr + 3].copy()      # the bowl rests somewhere new
            want = np.zeros(2)
            if self.radius > 0:
                r = self.radius * np.sqrt(self.rng.random())
                th = 2 * np.pi * self.rng.random()
                want = r * np.array([np.cos(th), np.sin(th)])
                sim.data.qpos[qadr:qadr + 2] = rest[:2] + want
                sim.data.qvel[vadr:vadr + 6] = 0.0
                sim.forward()
            self._settle(self.settle_steps)
            now = self.env.sim.data.qpos[qadr:qadr + 3]
            dz = abs(now[2] - rest[2]) * 1000
            dxy = np.linalg.norm(now[:2] - rest[:2] - want) * 1000
            if dz <= EJECT_MM and dxy <= EJECT_MM:
                return now, rest
        raise RuntimeError(f"task {self.ti}: no valid placement in {max_tries} draws "
                           f"at radius {self.radius} m")

    def obs(self, raw: dict | None = None) -> np.ndarray:
        """PRIVILEGED. The only place simulator object state enters the teacher.

        `raw` is the observation dict the environment just returned. Without it
        observables are refreshed with force_update -- they are cached on a
        sampling interval, and a state set without stepping (BC extraction, a
        reset) would otherwise read the previous frame -- which on a rendering
        environment renders both cameras a second time.
        """
        sim = self.env.sim
        o = raw if raw is not None else self.observe()
        self.raw = o
        q = np.asarray(o["robot0_joint_pos"], np.float64)
        gq = o["robot0_gripper_qpos"]
        T = fk(self.chain, torch.tensor(q)[None])[0].numpy()
        _, _, bowl, plate, base = self._ids()
        tb = sim.data.body_xpos[base]
        p_tool = T[:3, 3]
        p_bowl = sim.data.body_xpos[bowl] - tb
        p_plate = sim.data.body_xpos[plate] - tb
        R_bowl = sim.data.body_xmat[bowl].reshape(3, 3)
        vec = np.concatenate([
            q,                                              # 7
            p_tool, T[:3, :2].T.reshape(-1), [gq[0] - gq[1]],  # 10
            p_bowl, R_bowl[:3, :2].T.reshape(-1),           # 9
            p_plate,                                        # 3
            p_bowl - p_tool,                                # 3
            p_plate - p_bowl,                               # 3
            self.onehot,                                    # 10
        ])
        return vec.astype(np.float32)

    def step(self, action: np.ndarray):
        """action: 6 normalised body-twist + 1 gripper, each in [-1, 1].

        Executed through TwistServo: integrated onto an absolute joint reference,
        because per-step deltas lose 18% of every step to controller lag and the
        loss compounds (see servo.py)."""
        raw, success = self.execute(action)
        self.last_obs = self.obs(raw)
        truncated = self.t >= self.horizon
        info = {"success": success, "truncated": truncated}
        reward = float(success)
        if self.shaping:
            snap = self.snapshot()
            phi, stage, _ = self.progress(snap)
            if stage >= 3 and snap["p_bowl"][2] - self.ref["rest_z"] > self.geom.lifted_dz:
                self.ever_lifted = True
            # task reward only for a bowl that was actually picked up: pushing or
            # flipping it onto the plate satisfies On(bowl, plate) without a grasp
            task = self.success_bonus * float(success and self.ever_lifted)
            # Undiscounted difference. gamma*phi' - phi = (phi' - phi) - (1-gamma)*phi':
            # the second term is a per-step drag proportional to progress already
            # made. Measured, it collapsed a from-scratch policy's peak progress
            # from 0.82 (random) to 0.04 in 50 iterations -- random motion's
            # progress changes average to zero, the drag never does, so the cheapest
            # state was phi = 0. For an episodic task the undiscounted difference
            # telescopes to phi(s_T) - phi(s_0) and carries no drag.
            reward = task + self.shaping_gamma * phi - self.phi
            info.update(phi=phi, stage=stage, ever_lifted=self.ever_lifted,
                        success_lifted=bool(success and self.ever_lifted))
            self.phi = phi
            if self.rich_obs:
                self.last_obs = np.concatenate([self.last_obs, self._rich(snap, phi, stage)])
        return self.last_obs, reward, success or truncated, info

    def begin(self) -> None:
        """Arm the servo at the current joints. Called by reset(); call it yourself
        after placing the sim in a state by any other route."""
        o = self.observe()
        self.servo.reset(np.asarray(o["robot0_joint_pos"]))
        self.t = 0

    # -- task progress (privileged) ---------------------------------------------------
    def snapshot(self) -> dict:
        """Everything progress() needs, read from the simulator, base frame."""
        sim = self.env.sim; m, d = sim.model, sim.data
        raw = getattr(self, "raw", None) or self.observe()
        tool = self.tool_state(raw)
        tb = d.body_xpos[m.body_name2id("robot0_base")]
        _, _, bowl, plate, _ = self._ids()
        # contacts.py is the one definition of "touching" (the finger bodies' only colliding
        # geoms are the four this loop used to name, and every margin in the scene is 0)
        sides, any_grip, supported = contacts.touch_summary(m, d, bowl)
        return dict(tool,
                    R_bowl=d.body_xmat[bowl].reshape(3, 3).copy(), p_bowl=(d.body_xpos[bowl] - tb).copy(),
                    p_plate=(d.body_xpos[plate] - tb).copy(), side1=0 in sides, side2=1 in sides,
                    any_grip=any_grip, supported=supported, success=self.success())

    def _rich(self, snap: dict, phi: float | None = None, stage: int | None = None) -> np.ndarray:
        """PRIVILEGED progress features: phi/6, stage one-hot, tool->grasp waypoint error."""
        from ..scripted.progress import grasp_error
        if phi is None:
            phi, stage, _ = self.progress(snap)
        oh = np.zeros(7); oh[int(stage)] = 1.0
        return np.concatenate([[phi / 6.0], oh, grasp_error(snap, self.geom)]).astype(np.float32)

    def set_reference(self, snap: dict | None = None) -> None:
        """Episode constants: bowl rest height and the two stage normalisers that depend on layout."""
        from ..scripted.progress import reach_distance
        s = snap or self.snapshot(); g = self.geom
        d0_reach, _, _ = reach_distance(dict(s, rest_z=float(s["p_bowl"][2])), g)
        from ..scripted.progress import transport_remaining
        ref0 = dict(s, rest_z=float(s["p_bowl"][2]), d0_reach=float(d0_reach), d0_carry=1.0)
        d0_carry, _ = transport_remaining(ref0, g)       # the whole transport, from grasped at rest
        self.ref = dict(rest_z=float(s["p_bowl"][2]), d0_reach=float(d0_reach), d0_carry=float(d0_carry),
                        rest_xy=s["p_bowl"][:2].copy())
        self.phi = self.progress(s)[0]

    def progress(self, snap: dict | None = None):
        from ..scripted.progress import progress
        s = dict(snap or self.snapshot(), **self.ref)
        return progress(s, self.geom)

    # -- what the STUDENT is allowed to see ------------------------------------------
    def images(self) -> tuple[np.ndarray, np.ndarray]:
        """Agentview and wrist frames of the current state (render=True only)."""
        return self.raw["agentview_image"], self.raw["robot0_eye_in_hand_image"]

    def student_state(self) -> np.ndarray:
        """Tool pose + gripper aperture via tool_state, the student's proprioception.
        No object state, and not the teacher's layout of the same quantities."""
        from ..geometry.state import tool_state
        q = torch.tensor(np.asarray(self.raw["robot0_joint_pos"]), dtype=torch.float64)[None]
        g = self.raw["robot0_gripper_qpos"]
        return tool_state(self.chain, q, torch.tensor([float(g[0] - g[1])], dtype=torch.float64)
                          )[0].float().numpy()

    def close(self):
        self.env.env.close()
