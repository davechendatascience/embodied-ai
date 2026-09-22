"""screwhead/geometry/box_distance.py against distances known independently: analytic ones
for axis-aligned boxes, and a dense sampling of box surfaces for rotated ones.

  PYTHONPATH=third_party/LIBERO:. .venv-libero/bin/python -m pytest tests/test_box_distance.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from screwhead.geometry.box_distance import pair_distance

I3 = np.eye(3)


def _rot(rng) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _surface(c, r, h, n, rng) -> np.ndarray:
    """n points on the box surface: a random face, a uniform point on it."""
    u = rng.uniform(-1, 1, size=(n, 3))
    axis = rng.integers(3, size=n)
    u[np.arange(n), axis] = rng.choice([-1.0, 1.0], size=n)
    return c + (u * h) @ r.T


def _point_box(p, c, r, h) -> np.ndarray:
    q = (p - c) @ r
    return np.linalg.norm(np.maximum(np.abs(q) - h, 0.0), axis=1)


def test_overlap_is_zero():
    assert pair_distance(np.zeros(3), I3, np.ones(3), np.array([1.5, 0, 0]), I3, np.ones(3)) == 0.0


@pytest.mark.parametrize("offset,expected", [((3.0, 0, 0), 1.0), ((0, 2.5, 0), 0.5), ((3.0, 3.0, 0), np.sqrt(2.0)),
                                             ((3.0, 3.0, 3.0), np.sqrt(3.0))])
def test_axis_aligned_face_edge_and_corner(offset, expected):
    d = pair_distance(np.zeros(3), I3, np.ones(3), np.array(offset, float), I3, np.ones(3))
    assert d == pytest.approx(expected, abs=1e-12)


def test_rotated_pairs_match_dense_sampling():
    """The exact distance is below every sampled surface-to-box distance, and within the
    sampling's resolution of the smallest."""
    rng = np.random.default_rng(0)
    for _ in range(60):
        h1, h2 = rng.uniform(0.005, 0.08, 3), rng.uniform(0.005, 0.08, 3)
        r1, r2 = _rot(rng), _rot(rng)
        c1 = np.zeros(3)
        c2 = rng.normal(size=3) * 0.15
        d = pair_distance(c1, r1, h1, c2, r2, h2)
        sampled = min(_point_box(_surface(c1, r1, h1, 40000, rng), c2, r2, h2).min(),
                      _point_box(_surface(c2, r2, h2, 40000, rng), c1, r1, h1).min())
        assert d <= sampled + 1e-12
        if d > 0:
            assert sampled - d < 2e-3
