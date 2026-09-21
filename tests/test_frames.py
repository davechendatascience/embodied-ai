"""frames.py (the NumPy rotations both teachers use) held to se3.py (the torch reference).

  .venv-libero/bin/python -m pytest tests/test_frames.py -q

frames.rotvec replaced two copies that disagreed near a half turn: one divided by
2 sin(angle) all the way to pi, the other read the axis off the diagonal there.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from screwhead.geometry import frames, se3

RNG = np.random.default_rng(0)
TOL = 1e-9


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def ref_rot(axis: np.ndarray, angle: float) -> np.ndarray:
    return se3.exp_so3(torch.tensor(unit(axis))[None], torch.tensor([angle], dtype=torch.float64))[0].numpy()


@pytest.mark.parametrize("angle", [0.0, 1e-7, 1e-3, 0.5, 1.5, 3.0, np.pi - 1e-3, np.pi - 1e-5, np.pi])
def test_axis_rot_matches_reference(angle):
    for _ in range(20):
        axis = RNG.normal(size=3)
        np.testing.assert_allclose(frames.axis_rot(axis, angle), ref_rot(axis, angle), atol=TOL)


@pytest.mark.parametrize("angle", [1e-3, 0.5, 1.5, 3.0, np.pi - 1e-3, np.pi - 1e-5, np.pi])
def test_rotvec_inverts_axis_rot(angle):
    """Up to and including a half turn, where the axis sign is a free choice."""
    for _ in range(20):
        axis = unit(RNG.normal(size=3))
        w = frames.rotvec(ref_rot(axis, angle))
        assert abs(np.linalg.norm(w) - angle) < 1e-6
        np.testing.assert_allclose(ref_rot(w, np.linalg.norm(w)), ref_rot(axis, angle), atol=1e-6)


def test_rotvec_of_identity_is_zero():
    np.testing.assert_array_equal(frames.rotvec(np.eye(3)), np.zeros(3))


def test_rot_angle_matches_rotvec_norm():
    for _ in range(50):
        R = ref_rot(RNG.normal(size=3), RNG.uniform(0, np.pi))
        assert abs(frames.rot_angle(R) - np.linalg.norm(frames.rotvec(R))) < 1e-9
