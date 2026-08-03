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
                    "kitchen_table": lambda L: (-0.16 - L / 2, 0, 0)}

    ROBOT_CLASS_MAPPING.update({"MountedUR5e": SingleArm})
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
