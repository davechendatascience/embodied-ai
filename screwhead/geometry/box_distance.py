"""Exact distance between two bodies made of boxes (every LIBERO contact geom in the
libero_object, spatial and goal scenes is one).

MuJoCo's own mj_geomDistance was not usable for this: on the live LIBERO model its native
collider returned 0 for separated parallel faces -- exactly a resting placement -- at 10-44
of 48 teleported states, by up to 8 cm, and libccd reported spurious penetrations beyond a
5 cm distmax and saturated at it (review of BRN-task-loss-descends-to-scorer). This is the
exact convex-box distance instead: a separating-axis test decides overlap (distance 0), and
for a separated pair the distance is the least of the 16 vertex-to-box and 144 edge-to-edge
distances. Pairs are visited nearest-bound first and pruned against the best distance so
far, so a body pair costs 0.04-0.14 ms with the numba kernel.
"""
from __future__ import annotations

import mujoco
import numba
import numpy as np

PARALLEL_TOL = 1e-12   # |cross|^2 of edge axes counted as parallel: the face axes already cover that pair
_S = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
_E = np.array([(i, j) for i in range(8) for j in range(8)
               if np.sum(_S[i] != _S[j]) == 1 and np.all(_S[j] >= _S[i])])     # the 12 edges


class BoxSet:
    """The box contact geoms of one body; poses are read live from MjData."""

    def __init__(self, m, geom_ids):
        self.ids = np.asarray(geom_ids, int)
        if not np.all(m.geom_type[self.ids] == mujoco.mjtGeom.mjGEOM_BOX):
            raise ValueError("box contact geoms only")
        self.h = np.ascontiguousarray(m.geom_size[self.ids], float)

    def pose(self, d) -> tuple[np.ndarray, np.ndarray]:
        return (np.ascontiguousarray(d.geom_xpos[self.ids]),
                np.ascontiguousarray(d.geom_xmat[self.ids]).reshape(-1, 3, 3))


def separation(d, a: BoxSet, b: BoxSet) -> float:
    """Exact distance between two box sets (0 when any pair overlaps)."""
    ca, ra = a.pose(d)
    cb, rb = b.pose(d)
    return float(_body_distance(ca, ra, a.h, cb, rb, b.h, _E))


def pair_distance(c1, r1, h1, c2, r2, h2) -> float:
    """Exact distance between two boxes (centre, rotation, half-extents)."""
    args = [np.ascontiguousarray(x, float) for x in (c1, r1, h1, c2, r2, h2)]
    return float(_body_distance(args[0][None], args[1][None], args[2][None],
                                args[3][None], args[4][None], args[5][None], _E))


@numba.njit(cache=True)
def _point_box_sq(px, py, pz, c, r, h):
    dx, dy, dz = px - c[0], py - c[1], pz - c[2]
    s = 0.0
    for k in range(3):
        e = abs(r[0, k] * dx + r[1, k] * dy + r[2, k] * dz) - h[k]
        if e > 0.0:
            s += e * e
    return s


@numba.njit(cache=True)
def _segment_sq(p1, d1, p2, d2):
    """Squared distance between segments p1 + s d1 and p2 + t d2, s, t in [0, 1]."""
    r0, r1, r2 = p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]
    a = d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2]
    e = d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2]
    f = d2[0] * r0 + d2[1] * r1 + d2[2] * r2
    c = d1[0] * r0 + d1[1] * r1 + d1[2] * r2
    b = d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]
    den = a * e - b * b
    s = 0.0
    if den > 1e-12 * a * e:
        s = min(max((b * f - c * e) / den, 0.0), 1.0)
    t = (b * s + f) / e
    if t < 0.0:
        t = 0.0
        s = min(max(-c / a, 0.0), 1.0)
    elif t > 1.0:
        t = 1.0
        s = min(max((b - c) / a, 0.0), 1.0)
    x0 = r0 + d1[0] * s - d2[0] * t
    x1 = r1 + d1[1] * s - d2[1] * t
    x2 = r2 + d1[2] * s - d2[2] * t
    return x0 * x0 + x1 * x1 + x2 * x2


@numba.njit(cache=True)
def _sat_gap(c1, r1, h1, c2, r2, h2):
    """Largest projected gap over the 15 separating axes: > 0 exactly when the boxes are
    apart, and a lower bound on their distance."""
    cm = np.empty((3, 3))
    ac = np.empty((3, 3))
    t = np.empty(3)
    t0, t1, t2 = c2[0] - c1[0], c2[1] - c1[1], c2[2] - c1[2]
    for i in range(3):
        t[i] = r1[0, i] * t0 + r1[1, i] * t1 + r1[2, i] * t2
        for j in range(3):
            cm[i, j] = r1[0, i] * r2[0, j] + r1[1, i] * r2[1, j] + r1[2, i] * r2[2, j]
            ac[i, j] = abs(cm[i, j])
    g = -1e300
    for i in range(3):
        g = max(g, abs(t[i]) - h1[i] - (ac[i, 0] * h2[0] + ac[i, 1] * h2[1] + ac[i, 2] * h2[2]))
    for j in range(3):
        proj = abs(t[0] * cm[0, j] + t[1] * cm[1, j] + t[2] * cm[2, j])
        g = max(g, proj - h2[j] - (ac[0, j] * h1[0] + ac[1, j] * h1[1] + ac[2, j] * h1[2]))
    for i in range(3):
        i1, i2 = (i + 1) % 3, (i + 2) % 3
        for j in range(3):
            n2 = 1.0 - cm[i, j] * cm[i, j]
            if n2 <= PARALLEL_TOL:
                continue
            j1, j2 = (j + 1) % 3, (j + 2) % 3
            tp = t[i2] * cm[i1, j] - t[i1] * cm[i2, j]
            ra = h1[i1] * ac[i2, j] + h1[i2] * ac[i1, j]
            rb = h2[j1] * ac[i, j2] + h2[j2] * ac[i, j1]
            g = max(g, (abs(tp) - ra - rb) / np.sqrt(n2))
    return g


@numba.njit(cache=True)
def _verts(c, r, h, v):
    k = 0
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                for i in range(3):
                    v[k, i] = c[i] + r[i, 0] * sx * h[0] + r[i, 1] * sy * h[1] + r[i, 2] * sz * h[2]
                k += 1


@numba.njit(cache=True)
def _pair_sq(c1, r1, h1, c2, r2, h2, edges, best):
    """Squared distance of a separated pair, stopping early once it cannot beat `best`."""
    v1 = np.empty((8, 3))
    v2 = np.empty((8, 3))
    _verts(c1, r1, h1, v1)
    _verts(c2, r2, h2, v2)
    for k in range(8):
        best = min(best, _point_box_sq(v1[k, 0], v1[k, 1], v1[k, 2], c2, r2, h2))
        best = min(best, _point_box_sq(v2[k, 0], v2[k, 1], v2[k, 2], c1, r1, h1))
    d1 = np.empty(3)
    d2 = np.empty(3)
    for a in range(12):
        for i in range(3):
            d1[i] = v1[edges[a, 1], i] - v1[edges[a, 0], i]
        for b in range(12):
            for i in range(3):
                d2[i] = v2[edges[b, 1], i] - v2[edges[b, 0], i]
            best = min(best, _segment_sq(v1[edges[a, 0]], d1, v2[edges[b, 0]], d2))
    return best


@numba.njit(cache=True)
def _body_distance(ca, ra, ha, cb, rb, hb, edges):
    na, nb = ca.shape[0], cb.shape[0]
    bound = np.empty(na * nb)                       # squared world-AABB distance: a lower bound
    for i in range(na):
        ea = np.empty(3)
        for k in range(3):
            ea[k] = abs(ra[i, k, 0]) * ha[i, 0] + abs(ra[i, k, 1]) * ha[i, 1] + abs(ra[i, k, 2]) * ha[i, 2]
        for j in range(nb):
            s = 0.0
            for k in range(3):
                eb = abs(rb[j, k, 0]) * hb[j, 0] + abs(rb[j, k, 1]) * hb[j, 1] + abs(rb[j, k, 2]) * hb[j, 2]
                g = abs(ca[i, k] - cb[j, k]) - ea[k] - eb
                if g > 0.0:
                    s += g * g
            bound[i * nb + j] = s
    order = np.argsort(bound)
    best = 1e300
    for n in range(na * nb):
        p = order[n]
        if bound[p] >= best:
            break
        i, j = p // nb, p % nb
        g = _sat_gap(ca[i], ra[i], ha[i], cb[j], rb[j], hb[j])
        if g <= 0.0:
            return 0.0
        if g * g >= best:
            continue
        best = _pair_sq(ca[i], ra[i], ha[i], cb[j], rb[j], hb[j], edges, best)
    return np.sqrt(best)
