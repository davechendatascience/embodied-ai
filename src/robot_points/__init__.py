"""Where the robot's points are, for a given configuration. One contract, two backends.

PointWorld's action IS the robot's point flow, and its own sampler
(`third_party/PointWorld/robot_sampler.py`) is driven by **named joint values**
through URDF forward kinematics -- `compute_points({joint_name: values})`. It
works for any URDF, which is what makes the model embodiment-agnostic.

Our planner was built the other way round: it sampled end-effector position
deltas and placed a gripper rigidly bound to one ee frame. That is a MuJoCo
shortcut and a dead end for the real robot, because ee deltas

  * cannot express a DUAL-ARM action (the UR7e has two chains),
  * cannot open or close the jaw, though gripper state is an action channel
    PointWorld consumes, and
  * need IK before any controller can run them.

Joint values have none of those problems and are what a controller wants
anyway. So the action space is joint space, and the backends differ only in how
they turn a configuration into points:

| backend | source | for |
|---|---|---|
| `MujocoRobotPoints` | MuJoCo geoms + `mj_kinematics` | simulation |
| `UrdfRobotPoints` | upstream `RobotSampler` + pytorch_kinematics | deployment |

Both satisfy `RobotPoints`, so the planner never learns which it is holding.
That also makes them cross-checkable against each other on the same robot,
which is the control this project keeps needing.
"""

from .base import RobotPoints
from .mujoco_backend import MujocoRobotPoints

__all__ = ["RobotPoints", "MujocoRobotPoints"]
