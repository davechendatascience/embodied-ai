"""Is the gripper holding something? Proprioception only, no contact list.

`environment_libero._contacting_objects()` reads MuJoCo's CONTACT LIST. That is
a privileged input: a real robot has no such list, and building the
world/attached split on it quietly bakes a physics oracle into the pipeline
that E3b is supposed to be free of.

A gripper does, however, report **its own joint position**, on every real
robot. That is enough:

    commanded CLOSE  and  the jaws stopped short of empty  ->  holding

"Stopped short" is the whole test. Closing on air drives the fingers to their
mechanical limit; closing on an object stops at the object's width. Measured on
this stack, with the Panda gripper on both arms:

    closed on air (UR5e, failed grasp)   jaw [ 0.0008, -0.0016]  width 2.4 mm
    holding the drawer handle (Panda)    jaw [ 0.0118, -0.0128]  width 24.6 mm
    holding the drawer handle (UR5e)     jaw [ 0.0147, -0.0181]  width 32.8 mm

so the empty/holding boundary is wide, and 8 mm sits comfortably between them.

WHAT THIS DOES NOT GIVE YOU is the object's IDENTITY. Width says *that* you
hold something, never *what*. On a robot the identity comes from segmenting the
depth cloud between the fingers at the moment of grasp; in sim we still look it
up, and `world_export.refresh` labels that lookup as the remaining privileged
step rather than hiding it behind a detector that looks honest.
"""

import numpy as np

#: Jaw opening above which a commanded-closed gripper is judged to be holding
#: something. Measured boundary is 2.4 mm (empty) vs 24.6-32.8 mm (holding).
HOLDING_WIDTH_M = 0.008


def gripper_width(env):
    """Jaw opening in metres, from the gripper's own joint sensors."""
    sim = env.sim
    robot = env.env.robots[0] if hasattr(env, "env") else env.robots[0]
    m = sim.model
    qs = []
    for name in robot.gripper.joints:
        j = m.joint_name2id(name)
        qs.append(float(sim.data.qpos[m.jnt_qposadr[j]]))
    if len(qs) >= 2:
        return abs(qs[0] - qs[1])
    return abs(qs[0]) if qs else 0.0


def holding_by_width(env, threshold=HOLDING_WIDTH_M):
    """True when the gripper is commanded shut and did not fully close.

    Uses only what a real gripper publishes: its commanded state and its
    measured opening.
    """
    commanded_closed = getattr(env, "last_gripper_action", -1.0) > 0
    return bool(commanded_closed and gripper_width(env) > threshold)


def report(env):
    """(holding, width_m, commanded_closed) — for logging and for tests."""
    return (holding_by_width(env), gripper_width(env),
            bool(getattr(env, "last_gripper_action", -1.0) > 0))
