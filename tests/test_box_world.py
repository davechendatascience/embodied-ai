"""The box world's overlap test (LMA-separating-axis-boxes) and path sweep (LMA-swept-box-conservative)."""
import numpy as np

from screwhead.teacher.box_world import Boxes, overlap, sweep


def _rot(axis, ang):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def _boxes(*items):
    c = np.array([i[0] for i in items], float)
    R = np.array([i[1] for i in items], float)
    h = np.array([i[2] for i in items], float)
    n = len(items)
    return Boxes(c, R, h, np.arange(n), np.arange(n))


def test_face_separation_and_contact():
    A = _boxes(([0, 0, 0], np.eye(3), [1, 1, 1]))
    assert overlap(A, _boxes(([1.9, 0, 0], np.eye(3), [1, 1, 1])))[0, 0]
    assert not overlap(A, _boxes(([2.1, 0, 0], np.eye(3), [1, 1, 1])))[0, 0]


def test_rotated_corner_is_decided_by_face_axes():
    # a cube turned 45 deg about z reaches sqrt(2) along x
    A = _boxes(([0, 0, 0], np.eye(3), [1, 1, 1]))
    B = _rot([0, 0, 1], np.pi / 4)
    assert overlap(A, _boxes(([1 + np.sqrt(2) - 0.05, 0, 0], B, [1, 1, 1])))[0, 0]
    assert not overlap(A, _boxes(([1 + np.sqrt(2) + 0.05, 0, 0], B, [1, 1, 1])))[0, 0]


def test_edge_on_edge_needs_a_cross_axis():
    # two long thin bars crossed at right angles, one above the other: no face normal separates them
    # when their centres are within their thickness in z along a diagonal, but an edge-edge axis does
    A = _boxes(([0, 0, 0], _rot([0, 0, 1], np.pi / 4) @ _rot([1, 0, 0], np.pi / 4), [2.0, 0.1, 0.1]))
    B = _boxes(([0, 0, 0.25], _rot([0, 0, 1], -np.pi / 4) @ _rot([1, 0, 0], -np.pi / 4), [2.0, 0.1, 0.1]))
    ref = _monte_carlo_overlap(A, B)
    assert overlap(A, B)[0, 0] == ref


def _monte_carlo_overlap(A, B, n=200_000, seed=0):
    """Points sampled in A, tested against B: a lower bound on overlap (True only if some sample is in both)."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1, 1, (n, 3)) * A.h[0]
    pts = A.c[0] + u @ A.R[0].T
    local = (pts - B.c[0]) @ B.R[0]
    return bool(np.any(np.all(np.abs(local) <= B.h[0], axis=1)))


def test_random_pairs_agree_with_point_sampling_when_overlapping():
    rng = np.random.default_rng(1)
    for _ in range(300):
        A = _boxes((rng.uniform(-1, 1, 3), _rot(rng.normal(size=3), rng.uniform(0, np.pi)), rng.uniform(0.1, 1, 3)))
        B = _boxes((rng.uniform(-1, 1, 3), _rot(rng.normal(size=3), rng.uniform(0, np.pi)), rng.uniform(0.1, 1, 3)))
        if _monte_carlo_overlap(A, B, n=20_000):
            assert overlap(A, B)[0, 0], "a sampled common point, yet the test says disjoint"


def test_sweep_catches_a_thin_wall_between_samples():
    # a 2 mm wall across the path; samples every 50 mm would jump it, the grown boxes do not
    wall = _boxes(([0.5, 0, 0], np.eye(3), [0.001, 0.5, 0.5]))
    mover = _boxes(([0, 0, 0], np.eye(3), [0.01, 0.01, 0.01]))
    path = [(np.eye(3), np.array([0.0, 0, 0])), (np.eye(3), np.array([1.0, 0, 0]))]
    assert sweep(wall, mover, path, grow=0.0, step=0.05) is not None
    clear = [(np.eye(3), np.array([0.0, 0.8, 0])), (np.eye(3), np.array([1.0, 0.8, 0]))]
    assert sweep(wall, mover, clear, grow=0.0, step=0.05) is None


def _separated_along_some_direction(A, B, n=20_000, seed=2):
    """True if one of n random directions (or the fifteen axes) separates the boxes: proof of disjointness."""
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n, 3))
    Ra, Rb = A.R[0], B.R[0]
    axes = [Ra[:, i] for i in range(3)] + [Rb[:, j] for j in range(3)] + \
           [np.cross(Ra[:, i], Rb[:, j]) for i in range(3) for j in range(3)]
    dirs = np.vstack([dirs, [a for a in axes if np.linalg.norm(a) > 1e-9]])
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    ra = np.abs(dirs @ Ra) @ A.h[0]
    rb = np.abs(dirs @ Rb) @ B.h[0]
    return bool(np.any(np.abs(dirs @ (B.c[0] - A.c[0])) > ra + rb + 1e-12))


def test_random_pairs_never_claim_overlap_for_separated_boxes():
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(400):
        A = _boxes((rng.uniform(-1, 1, 3), _rot(rng.normal(size=3), rng.uniform(0, np.pi)), rng.uniform(0.05, 1, 3)))
        B = _boxes((rng.uniform(-1, 1, 3), _rot(rng.normal(size=3), rng.uniform(0, np.pi)), rng.uniform(0.05, 1, 3)))
        if overlap(A, B)[0, 0]:
            checked += 1
            assert not _separated_along_some_direction(A, B), "claimed overlap, but a separating direction exists"
    assert checked > 50
