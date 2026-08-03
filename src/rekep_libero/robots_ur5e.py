"""A UR5e for LIBERO — the second embodiment, registered from our side.

Why this file has to exist, measured rather than assumed:

    OffScreenRenderEnv(bddl_file_name=..., robots=["UR5e"])
    -> KeyError: 'MountedUR5e'

LIBERO's problem classes prefix every robot name with ``Mounted``
(``libero_tabletop_manipulation.py:23`` and its kitchen/study siblings), then
resolve the result through LIBERO's OWN registry:
``libero/libero/envs/robots/__init__.py`` registers exactly two models, both
Pandas. robosuite ships a perfectly good ``UR5e`` and LIBERO never consults it.

robosuite splits registration in two, and both are needed:

  * the MODEL, keyed by class name in ``REGISTERED_ROBOTS`` — automatic, via
    ``RobotModelMeta``, purely by defining the subclass below;
  * the CONTROL class, keyed by name in ``robosuite.robots.ROBOT_CLASS_MAPPING``
    — explicit, and LIBERO does its own ``.update()`` on the same dict.

Both happen at import, so ``import rekep_libero.robots_ur5e`` is the entire
API. Nothing under ``third_party/`` is touched.

**The base offsets are copied from MountedPanda deliberately.** Stock UR5e
declares the same ``table`` offset the Panda does — verified, both are
``(-0.16 - L/2, 0, 0)``, so the two bases coincide — but it omits the
``kitchen_table`` and ``study_table`` keys that LIBERO's kitchen and study
arenas index (``bddl_base_domain.py:317,357``), which would ``KeyError`` on
reset for those suites. Copying MountedPanda's dict wholesale fixes that AND
pins the UR5e base to exactly the Panda's world position, so a cross-embodiment
comparison varies arm kinematics and nothing else. Do not "improve" these
numbers: the experiment depends on them being identical.
"""

import numpy as np
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.robots.single_arm import SingleArm
from robosuite.utils.mjcf_utils import xml_path_completion


class MountedUR5e(ManipulatorModel):
    """UR5e on LIBERO's RethinkMount, at the Panda's base position.

    The XML is robosuite's stock ``robots/ur5e/robot.xml`` — unmodified, and
    deliberately so. MountedPanda overrides joint damping; we do not, because
    inventing damping for a second arm would silently become a confound in
    exactly the comparison this class exists to make. If the UR5e turns out to
    need it, measure it and say so here.
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/ur5e/robot.xml"), idn=idn)

    @property
    def default_mount(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        # Robotiq85 is the UR5e's own gripper and the E4 configuration. E3
        # overrides this with gripper_types="PandaGripper" so that arm transfer
        # is measured without the TCP-offset confound riding along.
        return "Robotiq85Gripper"

    @property
    def default_controller_config(self):
        return "default_ur5e"

    @property
    def init_qpos(self):
        # robosuite's stock UR5e rest pose. Whether this puts the tool over
        # LIBERO's table at a sane height is a QUESTION, not an assumption:
        # tests/test_ur5e_scene.py prints the achieved end-effector pose beside
        # the Panda's so the two can be compared before anything is planned.
        return np.array([-0.470, -1.735, 2.480, -2.275, -1.590, -1.991])

    @property
    def base_xpos_offset(self):
        # Verbatim from MountedPanda. See the module docstring.
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
            "study_table": lambda table_length: (-0.25 - table_length / 2, 0, 0),
            "kitchen_table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"


# The model half registers itself through RobotModelMeta when the class body
# above executes. The control half does not, and LIBERO's own registration runs
# on import of `libero.libero.envs.robots`, so this must follow it — importing
# LIBERO first is guaranteed because ManipulatorModel came from robosuite and
# the caller reaches us through rekep_libero, which puts LIBERO on sys.path.
ROBOT_CLASS_MAPPING.update({"MountedUR5e": SingleArm})


def registered():
    """True when both halves of the registration are in place.

    Cheap enough to assert on at env-construction time, and the failure it
    catches is otherwise a `KeyError` thrown from deep inside robosuite's model
    assembly, which reads like a robosuite bug rather than a missing import.
    """
    from robosuite.models.robots.robot_model import REGISTERED_ROBOTS

    return "MountedUR5e" in REGISTERED_ROBOTS and "MountedUR5e" in ROBOT_CLASS_MAPPING
