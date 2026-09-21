"""Rotations and tool frames the skills are written in. Pure functions, no state."""
from __future__ import annotations

import numpy as np

Z = np.array([0.0, 0.0, 1.0])
EPS_NORM = 1e-12        # guard for normalising a vector that may be zero
EPS_DIR = 1e-6          # a direction shorter than this has no direction
EPS_ANGLE = 1e-8        # a rotation smaller than this is the identity
MOSTLY_VERTICAL = 0.9   # |cos| to Z above which Z is a poor reference axis


def rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle) of a rotation matrix."""
    c = (np.trace(R) - 1) / 2
    th = float(np.arccos(np.clip(c, -1.0, 1.0)))
    if th < EPS_ANGLE:
        return np.zeros(3)
    return th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


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
