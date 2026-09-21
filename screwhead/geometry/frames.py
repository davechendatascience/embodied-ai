"""Rotations and tool frames the skills are written in. Pure functions, no state."""
from __future__ import annotations

import numpy as np

Z = np.array([0.0, 0.0, 1.0])
EPS_NORM = 1e-12        # guard for normalising a vector that may be zero
EPS_DIR = 1e-6          # a direction shorter than this has no direction
EPS_ANGLE = 1e-8        # a rotation smaller than this is the identity
NEAR_PI = 1e-4          # rad from pi: the skew part of R vanishes, read the axis off the diagonal
MOSTLY_VERTICAL = 0.9   # |cos| to Z above which Z is a poor reference axis


def rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle) of a rotation matrix -- the so(3) log. The one
    implementation for both stacks: the teacher's copy divided by 2 sin(angle) all the way
    to pi, where the skew part vanishes and the axis is lost; the scripted teacher's read
    it off the diagonal there."""
    th = rot_angle(R)
    if th < EPS_ANGLE:
        return np.zeros(3)
    skew = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])     # 2 sin(th) axis
    if np.pi - th < NEAR_PI:
        # the symmetric part is (1 - cos th) axis axis^T + cos th I at any angle: magnitude
        # from its largest column, sign from the (small but signed) skew part
        B = (R + R.T) / 2 - np.cos(th) * np.eye(3)
        k = int(np.argmax(np.diag(B)))
        axis = B[:, k] / np.sqrt(max(float(B[k, k]) * (1 - np.cos(th)), EPS_NORM))
        return th * (axis if axis @ skew >= 0 else -axis)
    return th / (2 * np.sin(th)) * skew


def rot_angle(R: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))


def axis_rot(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation about `axis` by `angle`."""
    a = np.asarray(axis, float)
    a = a / (np.linalg.norm(a) + EPS_NORM)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def tool_frame(jaw_dir: np.ndarray, approach: np.ndarray) -> np.ndarray:
    """Tool frame that closes along `jaw_dir` and advances along `approach`.

    The Panda gripper closes along its own y axis and looks along +z, so z_tool is the
    approach direction and y_tool the jaw axis, squared against it.
    """
    z = np.asarray(approach, float)
    z = z / (np.linalg.norm(z) + EPS_NORM)
    y = np.asarray(jaw_dir, float)
    y = y - z * (y @ z)
    n = np.linalg.norm(y)
    if n < EPS_DIR:                               # jaw parallel to the approach: any square axis
        y = np.cross(z, Z if abs(z @ Z) < MOSTLY_VERTICAL else np.array([1.0, 0.0, 0.0]))
        n = np.linalg.norm(y)
    y = y / n
    return np.column_stack([np.cross(y, z), y, z])


def top_down(yaw_dir: np.ndarray) -> np.ndarray:
    """The common case: straight down on to an object lying on a support."""
    y = np.asarray(yaw_dir, float)
    y = y - Z * (y @ Z)
    return tool_frame(y if np.linalg.norm(y) > EPS_DIR else np.array([1.0, 0.0, 0.0]), -Z)


def pose(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """4x4 homogeneous transform."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T
