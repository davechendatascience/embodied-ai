"""Pose conventions. Small, and the part most likely to be silently wrong."""

import numpy as np

#: Flange -> tool-centre offset along the flange z, metres. A PROPERTY OF THE
#: GRIPPER: measure it on your host, do not take it from a datasheet. Measured
#: 0.097 for robosuite's Panda hand where the real Franka TCP is ~0.103, and a
#: 6 mm bias on every keypose reads as poor grasp quality, not as a frame bug.
TCP_Z = 0.097


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
    """xyzw in, rotation matrix out. The order is in the name on purpose:
    mixing xyzw with wxyz is a silent 180-degree class of error."""
    x, y, z, w = np.asarray(q, dtype=np.float64)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_quat_wxyz(R):
    """wxyz, which is what cuRobo wants -- w FIRST, unlike most sim APIs."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s,
             (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s,
             (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
             0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
             (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def flange_from_obs(eef_pos, eef_quat_xyzw, tcp_z=TCP_Z):
    """T_world_flange from a recorded pose pair.

    THE TRAP: on robosuite `eef_pos` is the grip SITE and `eef_quat` is the
    hand BODY -- 90 degrees apart. Because the published quaternion is already
    the FLANGE orientation, the correction cancels and the flange pose comes
    out with no rotation term at all. Verify with `check_conventions` before
    trusting this on a new host.
    """
    R = quat_xyzw_to_mat(eef_quat_xyzw)
    p = np.asarray(eef_pos, dtype=np.float64) - R @ np.array([0.0, 0.0, tcp_z])
    return homo(p, R)


def goal_in_base(eef_pos, eef_quat_xyzw, T_world_base, tcp_z=TCP_Z):
    """The pose to hand a planner whose ee_link is the wrist flange."""
    return inv(np.asarray(T_world_base, float)) @ flange_from_obs(
        eef_pos, eef_quat_xyzw, tcp_z)


def to_curobo(T):
    """[x, y, z, qw, qx, qy, qz]."""
    return [*T[:3, 3].tolist(), *mat_to_quat_wxyz(T[:3, :3]).tolist()]


def check_conventions(eef_pos, eef_quat_xyzw, true_flange_T, tol_mm=0.1):
    """Does `flange_from_obs` reproduce the host's own flange pose?

    Run this once per host. It is four lines and it is the difference between
    a planner that works and one that is wrong by 90 degrees everywhere.
    """
    got = flange_from_obs(eef_pos, eef_quat_xyzw)
    d = inv(np.asarray(true_flange_T, float)) @ got
    mm = float(np.linalg.norm(d[:3, 3]) * 1000)
    deg = float(np.degrees(np.arccos(np.clip((np.trace(d[:3, :3]) - 1) / 2, -1, 1))))
    return mm <= tol_mm and deg <= 0.1, mm, deg
