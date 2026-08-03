"""Keypose -> planner goal, in the frames robosuite actually publishes.

Everything here is measured in `tests/test_frame_math.py`, not assumed.

**The trap.** A recorded trace step carries `robot0_eef_pos` and
`robot0_eef_quat`, which look like one pose and are not:

    robot0_eef_pos   the grip SITE position          (verified exact)
    robot0_eef_quat  the hand BODY orientation       (verified; the site's
                     own quaternion does NOT match)

Those two frames differ by exactly **90 degrees about Z** -- measured on both
the Panda and the UR5e, identically:

    R_flange_tcp = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]

Read the pair as a single TCP pose and then offset along the site's own z and
every planner goal is rotated 90 degrees, which presents as the arm reaching
the right place with the wrong wrist and grasping nothing.

**The simplification.** Because the published quaternion is ALREADY the flange
orientation, the flange pose comes straight out with no rotation correction:

    R_flange = quat2mat(eef_quat)
    p_flange = eef_pos - R_flange . [0, 0, TCP_Z]
    goal     = (T_world_base)^-1 . homo(p_flange, R_flange)

`TCP_Z = 0.097` is the measured flange->grip-site offset, constant over 40
random configurations (0.00000 mm) and identical on both arms with the matched
Panda gripper. It is a property of the GRIPPER, so re-measure it for a
Robotiq85 before using this for E4.
"""

import numpy as np

#: Flange -> grip-site translation along the flange z, metres. Measured, not
#: from a datasheet: the real Franka hand's TCP is ~0.103, robosuite's grip
#: site is 0.097, and a 6 mm bias on every keypose reads as poor grasp quality.
TCP_Z = 0.097

#: Flange -> grip-site rotation. Recorded here because it is the thing that
#: makes the published pos/quat pair inconsistent; the functions below avoid
#: needing it, and that is the point.
R_FLANGE_TCP = np.array([[0.0, 1.0, 0.0],
                         [-1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0]])


def homo(pos, mat):
    T = np.eye(4)
    T[:3, :3] = mat
    T[:3, 3] = pos
    return T


def inv(T):
    R, p = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ p
    return out


def quat_xyzw_to_mat(q):
    """robosuite publishes xyzw; MuJoCo's own order is wxyz. Mixing them is a
    silent 180-degree class of error, so the order is in the name."""
    x, y, z, w = np.asarray(q, dtype=np.float64)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def flange_from_obs(eef_pos, eef_quat_xyzw, tcp_z=TCP_Z):
    """T_world_flange from the two fields a trace record actually carries."""
    R = quat_xyzw_to_mat(eef_quat_xyzw)
    return homo(np.asarray(eef_pos, dtype=np.float64) - R @ np.array([0.0, 0.0, tcp_z]), R)


def tcp_from_obs(eef_pos, eef_quat_xyzw):
    """T_world_tcp, with the 90-degree correction applied explicitly.

    Provided for completeness and for anything that genuinely wants the grip
    site frame. The planner path should use `flange_from_obs` instead -- fewer
    steps, and no chance of applying the correction twice.
    """
    R_flange = quat_xyzw_to_mat(eef_quat_xyzw)
    return homo(np.asarray(eef_pos, dtype=np.float64), R_flange @ R_FLANGE_TCP)


def goal_in_base(eef_pos, eef_quat_xyzw, T_world_base, tcp_z=TCP_Z):
    """The pose to hand a planner whose ee_link is the wrist flange."""
    return inv(np.asarray(T_world_base, dtype=np.float64)) @ flange_from_obs(
        eef_pos, eef_quat_xyzw, tcp_z)


def pose_to_curobo(T):
    """cuRobo wants [x, y, z, qw, qx, qy, qz] -- w FIRST, unlike robosuite."""
    R = T[:3, :3]
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    return [*T[:3, 3].tolist(), *(q / np.linalg.norm(q)).tolist()]
