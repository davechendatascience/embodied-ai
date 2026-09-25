"""Put a non-Panda arm in LIBERO. Several things break; this fixes the first two.

Recovered from stash@{0} of the cross-gripper-transfer work (examples/libero_ur5e.py)
rather than rewritten. Every item below was established by measurement then, and
each is silent if skipped -- rediscovering them would have cost the same effort
a second time. Two numbers in here are independently confirmed by the current
work: flange-to-TCP for PandaGripper is 0.0970 m, which the LIBERO
demonstrations put at 0.0972 m by least squares over 6079 frames.

LIBERO is a Panda benchmark. Everything below was measured, and each item is
silent if you skip it:

 1. LIBERO's problem classes prefix robot names with `Mounted` and resolve them
    through LIBERO's OWN registry, which contains two Pandas. `robots=["UR5e"]`
    dies with KeyError: 'MountedUR5e' before a frame renders. robosuite ships a
    perfectly good UR5e that LIBERO never consults.

 2. The pinned init states are FLATTENED MuJoCo states recorded against the
    Panda, read POSITIONALLY. A UR5e's robot block is a different width, so
    every object address shifts and `set_init_state` writes drawer positions
    into a bottle's quaternion.

 3. robosuite draws `randn(len(init_qpos))` of initialization noise EVEN AT
    ZERO MAGNITUDE. The draw length is the arm's DOF, so a 6-DOF arm leaves the
    RNG one number ahead of a 7-DOF one and every sampled fixture lands
    elsewhere -- ~7 mm on libero_goal/0, which reads as nothing at all. Not
    corrected here: a non-Panda scene keeps the fixtures its own draw placed.

Also: two live LIBERO envs means two EGL contexts, and destroying the second
corrupts the first -- the symptom is `get_real_depth_map` asserting on a garbage
depth buffer many calls later.
"""

import os
import sys
from pathlib import Path

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(REPO, "third_party", "LIBERO") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "third_party", "LIBERO"))

_JNT_W = {0: (7, 6), 1: (4, 3), 2: (1, 1), 3: (1, 1)}   # free/ball/slide/hinge
PANDA_ROBOT_NQ = PANDA_ROBOT_NV = 9
ROBOT_PREFIX = ("robot", "gripper", "mount")


def register_ur5e():
    """Both halves: the MODEL self-registers via metaclass, the CONTROL class
    does not. Base offsets are copied from MountedPanda so the two arms stand
    in the SAME place -- the comparison varies kinematics and nothing else."""
    from robosuite.models.robots.manipulators.manipulator_model import (
        ManipulatorModel,
    )
    from robosuite.robots import ROBOT_CLASS_MAPPING
    from robosuite.robots.single_arm import SingleArm
    from robosuite.utils.mjcf_utils import xml_path_completion

    class MountedUR5e(ManipulatorModel):
        def __init__(self, idn=0):
            super().__init__(xml_path_completion("robots/ur5e/robot.xml"), idn=idn)

        @property
        def default_mount(self):
            return "RethinkMount"

        @property
        def default_gripper(self):
            return "Robotiq85Gripper"

        @property
        def default_controller_config(self):
            return "default_ur5e"

        @property
        def init_qpos(self):
            return np.array([-0.470, -1.735, 2.480, -2.275, -1.590, -1.991])

        @property
        def top_offset(self):
            return np.array((0, 0, 1.0))

        @property  # noqa: V106  read by robosuite's RobotModel.horizontal_radius
        def _horizontal_radius(self):
            return 0.5

        @property
        def arm_type(self):
            return "single"

        @property
        def base_xpos_offset(self):
            return {"bins": (-0.5, -0.1, 0), "empty": (-0.6, 0, 0),
                    "table": lambda L: (-0.16 - L / 2, 0, 0),
                    "study_table": lambda L: (-0.25 - L / 2, 0, 0),
                    "kitchen_table": lambda L: (-0.16 - L / 2, 0, 0),
                    # libero_10 uses a LIVING ROOM arena and libero_90 a coffee
                    # table. MountedPanda declares neither, so LIBERO's own
                    # Panda would KeyError there too -- these come from
                    # bddl_base_domain.py:337,347 which index them by name.
                    "coffee_table": lambda L: (-0.16 - L / 2, 0, 0),
                    "living_room_table": lambda L: (-0.16 - L / 2, 0, 0)}

    # LIBERO prefixes by ARENA, not just by robot: table scenes ask for
    # `MountedX`, floor scenes for `OnTheGroundX`. libero_object is a floor
    # scene, so registering only MountedUR5e means the UR5e is never built
    # there -- KeyError: 'OnTheGroundUR5e'. Base positions differ by 0.912 m
    # between the two arenas, so goals are not portable across them either.
    class OnTheGroundUR5e(MountedUR5e):  # noqa: V102  robosuite's metaclass registers it by name
        """Floor arena. Offsets come from OnTheGroundPanda, NOT the table
        mount: subclassing MountedUR5e inherits `table` offsets and puts the
        arm 912 mm above where the Panda stands in the same scene, which the
        planner then reports as 'unreachable'."""

        @property
        def default_mount(self):
            # THE 912 mm. RethinkMount is a pedestal; floor arenas stand the
            # arm on the ground (LIBERO uses NullMount there).
            return None

        @property
        def base_xpos_offset(self):
            # Delegate to LIBERO's OWN floor-arena Panda rather than copying
            # the table values. The living-room and coffee-table arenas carry a
            # z offset (0.42 / 0.41) that the table arena does not; guessing
            # z=0 put the arm ~800 mm out and LIBERO then reported `done` after
            # ONE step -- a spurious success, not a result.
            import libero.libero.envs.robots.on_the_ground_panda as G
            return G.OnTheGroundPanda.base_xpos_offset.fget(self)

    ROBOT_CLASS_MAPPING.update({"MountedUR5e": SingleArm,
                                "OnTheGroundUR5e": SingleArm})
    return MountedUR5e


_REGISTERED: set[str] = set()


def register_arm(name: str) -> None:
    """Register robosuite's own arm `name` (IIWA, Jaco, Kinova3) with LIBERO, as register_ur5e does the UR5e:
    a table-mounted variant standing where MountedPanda stands, and a floor variant whose base offsets are
    LIBERO's own floor Panda's. The arm's model, joint start and controller stay robosuite's; only its mount and
    base placement change. The UR5e keeps its own registration."""
    if name == "UR5e":
        register_ur5e()
        return
    if name == "Panda" or name in _REGISTERED:      # LIBERO's own Pandas stay as LIBERO registers them
        return
    import robosuite.models.robots as models
    from robosuite.robots import ROBOT_CLASS_MAPPING
    from robosuite.robots.single_arm import SingleArm
    base = getattr(models, name)
    table = register_ur5e().base_xpos_offset.fget(None)     # the same table offsets as the mounted UR5e

    def _floor(self):
        import libero.libero.envs.robots.on_the_ground_panda as G
        return G.OnTheGroundPanda.base_xpos_offset.fget(self)

    def _const(value):
        return property(lambda _self: value)
    mounted = type(f"Mounted{name}", (base,), {"default_mount": _const("RethinkMount"), "base_xpos_offset": _const(table)})
    floor = type(f"OnTheGround{name}", (mounted,), {"default_mount": _const(None), "base_xpos_offset": property(_floor)})
    ROBOT_CLASS_MAPPING.update({mounted.__name__: SingleArm, floor.__name__: SingleArm})
    _REGISTERED.add(name)


def _joint_blocks(sim):
    m = sim.model
    robot, objs = [], []
    for j in range(m.njnt):
        body = m.body_id2name(m.jnt_bodyid[j]) or ""
        qw, vw = _JNT_W[int(m.jnt_type[j])]
        e = (int(m.jnt_qposadr[j]), qw, vw)
        (robot if body.startswith(ROBOT_PREFIX) else objs).append(e)
    return robot, sorted(objs)


def remap_init_state(state, sim, panda: bool = True):
    """Panda-recorded flattened state -> one this model can accept.

    Identity on a Panda, so Panda runs stay bit-for-bit what LIBERO recorded. On any other arm only the objects
    are copied and the arm keeps its own start pose: a 7-joint arm with the Panda gripper has the Panda's robot
    width, and recognized by width alone it was started in the Panda's recorded joint angles -- the Kinova3 with
    its sixth joint pinned at its limit and the Panda hand folded into its upper arm (11 of 40)."""
    state = np.asarray(state, float).ravel()
    m = sim.model
    robot, objs = _joint_blocks(sim)
    if panda and len(state) == 1 + m.nq + m.nv and sum(e[1] for e in robot) == PANDA_ROBOT_NQ:
        return state
    nq_o = sum(e[1] for e in objs)
    nv_o = sum(e[2] for e in objs)
    want = 1 + PANDA_ROBOT_NQ + nq_o + PANDA_ROBOT_NV + nv_o
    if len(state) != want:
        raise ValueError(f"init state len {len(state)}, expected {want} for "
                         f"{len(objs)} object joints")
    src = state[1 + PANDA_ROBOT_NQ: 1 + PANDA_ROBOT_NQ + nq_o]
    qpos = np.array(sim.get_state().qpos, float, copy=True)
    k = 0
    for adr, qw, _ in objs:
        qpos[adr:adr + qw] = src[k:k + qw]
        k += qw
    return np.concatenate([[0.0], qpos, np.zeros(m.nv)])


def gripper_geom(env):
    """[flange-to-TCP, wristcam-to-TCP] in metres, measured from the live model.

    Both are gripper-dependent and NEITHER appears in any observation, so a
    corrector cannot infer them -- they have to be handed to it. PandaGripper
    is (0.0970, 0.1091); Robotiq85Gripper is (0.1450, 0.1534).
    """
    m, d = env.sim.model, env.sim.data
    site = m.site_name2id(env.env.robots[0].controller.eef_name)
    return np.array([
        np.linalg.norm(d.site_xpos[site] - d.body_xpos[m.body_name2id("robot0_right_hand")]),
        np.linalg.norm(d.site_xpos[site] - d.cam_xpos[m.camera_name2id("robot0_eye_in_hand")]),
    ], float)




def build_chain(mjcf_name: str, tool_z: float):
    """Chain with the tool frame at THIS arm's grip site.

    tool_z is measured from the live model, never assumed. The offset is a
    property of the GRIPPER, not the arm: PandaGripper puts the grip site
    0.0970 m beyond the flange, Robotiq85Gripper 0.1450 m. LIBERO gives the
    Panda the former and the UR5e the latter, so a single hardcoded constant is
    48 mm wrong on the held-out arm -- and a 48 mm tool-frame error makes every
    decoded twist reference the wrong point while looking like a kinematic
    transfer failure, which is the opposite of what it is.
    """
    import torch
    from .libero import ROBOT_MJCF
    from .mjcf import from_mjcf
    base = from_mjcf(ROBOT_MJCF / mjcf_name / "robot.xml", angle="radian", name=mjcf_name)
    off = torch.eye(4, dtype=base.M.dtype)
    off[2, 3] = float(tool_z)
    return base.with_tool(off)


def check_loaded_model(loaded_file: str, mjcf_name: str) -> None:
    """The arm the simulator loaded is the model the chain is built from, byte for byte --
    the chain drives every servo period, so a different model would execute differently
    under an unchanged chain (a 0.1 mm link offset changed an episode from 187 to 144 steps)."""
    from .libero import ROBOT_MJCF
    ours = ROBOT_MJCF / mjcf_name / "robot.xml"
    if Path(loaded_file).read_bytes() != ours.read_bytes():
        raise RuntimeError(f"the simulator's {mjcf_name} model {loaded_file} differs from {ours}")


def set_joint_gains(env, kp: float) -> None:
    """Stiffen the joint controller. Re-fetched every call on purpose: reset()
    rebuilds the controller object, so a handle captured once goes stale and
    silently leaves the gain at its default."""
    c = env.env.robots[0].controller
    n = len(np.atleast_1d(c.kp))
    c.kp = np.ones(n) * kp
    c.kd = 2 * np.sqrt(c.kp)  # noqa: V101  read by robosuite's joint controller
