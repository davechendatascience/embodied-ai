"""Put a UR5e in LIBERO. Three things break; this fixes exactly those.

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
    elsewhere -- ~7 mm on libero_goal/0, which reads as nothing at all.

Also: build the reference Panda scene BEFORE the main env. Two live LIBERO envs
means two EGL contexts, and destroying the second corrupts the first -- the
symptom is `get_real_depth_map` asserting on a garbage depth buffer many calls
later.
"""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "third_party", "LIBERO"))

CAMS = ("agentview", "robot0_eye_in_hand")
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

        default_mount = property(lambda self: "RethinkMount")
        default_gripper = property(lambda self: "Robotiq85Gripper")
        default_controller_config = property(lambda self: "default_ur5e")
        init_qpos = property(lambda self: np.array(
            [-0.470, -1.735, 2.480, -2.275, -1.590, -1.991]))
        top_offset = property(lambda self: np.array((0, 0, 1.0)))
        _horizontal_radius = property(lambda self: 0.5)
        arm_type = property(lambda self: "single")

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
    class OnTheGroundUR5e(MountedUR5e):
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


def _joint_blocks(sim):
    m = sim.model
    robot, objs = [], []
    for j in range(m.njnt):
        body = m.body_id2name(m.jnt_bodyid[j]) or ""
        qw, vw = _JNT_W[int(m.jnt_type[j])]
        e = (int(m.jnt_qposadr[j]), qw, vw)
        (robot if body.startswith(ROBOT_PREFIX) else objs).append(e)
    return robot, sorted(objs)


def remap_init_state(state, sim):
    """Panda-recorded flattened state -> one this model can accept.

    Identity on a Panda, so Panda runs stay bit-for-bit what LIBERO recorded.
    """
    state = np.asarray(state, float).ravel()
    m = sim.model
    robot, objs = _joint_blocks(sim)
    if len(state) == 1 + m.nq + m.nv and sum(e[1] for e in robot) == PANDA_ROBOT_NQ:
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


def fixture_snapshot(sim):
    """World-child bodies with no joints: the furniture the sampler places."""
    m = sim.model
    return {m.body_id2name(b): (m.body_pos[b].copy(), m.body_quat[b].copy())
            for b in range(m.nbody)
            if (m.body_id2name(b) and m.body_id2name(b) != "world"
                and not m.body_id2name(b).startswith(ROBOT_PREFIX)
                and m.body_parentid[b] == 0 and m.body_jntnum[b] == 0)}


def pin_fixtures(sim, ref):
    m = sim.model
    worst = 0.0
    for name, (pos, quat) in ref.items():
        b = m.body_name2id(name)
        worst = max(worst, float(np.linalg.norm(m.body_pos[b] - pos)) * 1000)
        m.body_pos[b], m.body_quat[b] = pos, quat
    sim.forward()
    return worst


#: camera->TCP as a VECTOR in the camera frame, metres: [x_img, y_img, z_depth].
#: robosuite Panda + PandaGripper, and bit-for-bit identical on a UR5e wearing
#: the same gripper -- which is exactly why the arm swap transfers and the
#: gripper swap does not. A Robotiq85 gives [0, -0.050, -0.145]: it perturbs
#: DEPTH ONLY, leaving the 50 mm image-plane offset untouched.
PANDA_CAM_TO_TCP_VEC = np.array([0.0, -0.0500, -0.0970])
PANDA_CAM_TO_TCP = float(np.linalg.norm(PANDA_CAM_TO_TCP_VEC))   # 0.1091 m


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


def align_wrist_camera(sim, target=PANDA_CAM_TO_TCP_VEC):
    """Move the wrist camera so the camera->TCP VECTOR matches the training arm.

    The camera is mounted on the wrist LINK (pos="0.05 0 0"), identically for
    the Panda and the UR5e -- so swapping the gripper does not move the camera,
    it moves the TCP away from it. A policy that servos on that image has no
    access to the TCP: it learned to drive the CAMERA, and the TCP followed
    because camera->TCP was a rigid constant. That constant is the thing to
    preserve.

    Preserve the VECTOR, not its length. Measured in the camera frame
    [x_img, y_img, z_depth], the Panda is [0, -50.0, -97.0] mm and a Robotiq85
    is [0, -50.0, -145.0] mm -- the swap perturbs DEPTH ONLY. Rescaling along
    the whole vector to fix the length instead drags the image-plane offset from
    -50.0 to -35.6 mm, i.e. it repairs depth while moving the grasp point 14 mm
    across the field of view, which the policy reads as the object having
    shifted sideways.

    Every camera extrinsic is a free design choice on a real robot, so this
    restores training-time geometry rather than papering over it downstream.
    Returns (before, after) camera->TCP vectors in the camera frame, metres.
    """
    m, d = sim.model, sim.data
    cid = m.camera_name2id("robot0_eye_in_hand")
    # The grasp site is namespaced by the GRIPPER, not the robot.
    site = m.site_name2id(next(n for n in ("gripper0_grip_site",
                                           "robot0_grip_site")
                               if n in m.site_names))
    Rc = d.cam_xmat[cid].reshape(3, 3)
    before = Rc.T @ (d.site_xpos[site] - d.cam_xpos[cid])
    # Camera orientation is untouched, so matching the vector is a pure
    # translation of the camera by the difference, rotated into its parent body.
    world_shift = Rc @ (before - np.asarray(target, float))
    bid = m.cam_bodyid[cid]
    m.cam_pos[cid] += d.xmat[bid].reshape(3, 3).T @ world_shift
    sim.forward()
    Rc = d.cam_xmat[cid].reshape(3, 3)
    after = Rc.T @ (d.site_xpos[site] - d.cam_xpos[cid])
    return before, after


PANDA_TIP_FRICTION = (2.0, 0.05, 0.0)   # robosuite PandaGripper fingertip pads


def equalise_finger_friction(sim, friction=PANDA_TIP_FRICTION):
    """Give the target gripper the SAME fingertip pad as the policy's gripper.

    robosuite's PandaGripper carries a dedicated high-friction pad -- sliding
    friction 2.0 -- while every other gripper's fingers sit at 1.0 or below. The
    bowl is 0.95 with condim 3, and MuJoCo takes the element-wise MAX, so a
    Panda grips this object with TWICE the sliding friction of any replacement.

    That is a property of the benchmark's asset, not of the embodiment. Leaving
    it in place means a gripper-swap experiment silently measures friction and
    reports it as a transfer failure. Returns the number of geoms changed.
    """
    m = sim.model
    n = 0
    for gi in range(m.ngeom):
        bn = m.body_id2name(m.geom_bodyid[gi]) or ""
        if "finger" not in bn.lower() and "pad" not in bn.lower():
            continue
        if m.geom_contype[gi] == 0 and m.geom_conaffinity[gi] == 0:
            continue                      # visual-only geom
        m.geom_friction[gi] = friction
        n += 1
    sim.forward()
    return n


def build(suite_name, task_id, robot="Panda", gripper="default", res=256,
          fixture_ref=None, seed=0):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    if robot != "Panda":
        register_ur5e()
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"),
                        task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl, robots=[robot], gripper_types=gripper,
        camera_heights=res, camera_widths=res, controller="OSC_POSE",
        camera_depths=True, camera_names=list(CAMS), horizon=10000)
    np.random.seed(seed)
    env.reset()
    if fixture_ref:
        pin_fixtures(env.sim, fixture_ref)
    return env, suite, task
