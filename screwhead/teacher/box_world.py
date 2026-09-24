"""The box world: the scene as oriented boxes, for testing a step's path before the simulator runs it.

DEF-box-world, BRN-plan-in-box-world. Every collision geom that is not the robot's becomes the box
aligned with its frame (a box geom exactly; a cylinder, capsule, ellipsoid, sphere or mesh the box
around it -- AXM-libero-collision-shapes). Perfect physics: nothing moves unless the plan moves it,
a held object rides rigidly with the tool. Overlap of two boxes is decided exactly by the fifteen
separating axes (LMA-separating-axis-boxes); a moving box is tested at samples of its path grown by
the farthest any of its points travels to the next sample, so nothing between samples is missed
(LMA-swept-box-conservative).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# MuJoCo geom types
PLANE, HFIELD, SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX, MESH = 0, 1, 2, 3, 4, 5, 6, 7
EPS = 1e-9


@dataclass
class Boxes:
    """N oriented boxes: centres (N, 3), axes as columns (N, 3, 3), half-extents (N, 3), and the
    body and geom each came from."""
    c: np.ndarray
    R: np.ndarray
    h: np.ndarray
    body: np.ndarray
    geom: np.ndarray

    def __len__(self) -> int:
        return len(self.c)

    def take(self, keep: np.ndarray) -> "Boxes":
        return Boxes(self.c[keep], self.R[keep], self.h[keep], self.body[keep], self.geom[keep])

    def grown(self, by: float) -> "Boxes":
        return Boxes(self.c, self.R, self.h + by, self.body, self.geom)

    def moved(self, R: np.ndarray, p: np.ndarray) -> "Boxes":
        """These boxes, given in a frame, placed by that frame's pose (R, p)."""
        return Boxes(self.c @ R.T + p, np.einsum("ij,njk->nik", R, self.R), self.h, self.body, self.geom)

    def relative_to(self, R: np.ndarray, p: np.ndarray) -> "Boxes":
        """These boxes, given in the world, expressed in the frame with pose (R, p)."""
        return Boxes((self.c - p) @ R, np.einsum("ji,njk->nik", R, self.R), self.h, self.body, self.geom)


def _collides(m, g: int) -> bool:
    return bool(m.geom_contype[g] or m.geom_conaffinity[g])


def geom_boxes(m, d, geoms) -> Boxes:
    """The frame-aligned box around each geom's collision shape, in the world (planes skipped)."""
    cs, Rs, hs, bs, gs = [], [], [], [], []
    for g in geoms:
        t, size = int(m.geom_type[g]), np.asarray(m.geom_size[g], float)
        off = np.zeros(3)
        if t == BOX or t == ELLIPSOID:
            h = size[:3].copy()
        elif t == SPHERE:
            h = np.full(3, size[0])
        elif t == CAPSULE:
            h = np.array([size[0], size[0], size[1] + size[0]])
        elif t == CYLINDER:
            h = np.array([size[0], size[0], size[1]])
        elif t == MESH:
            i = int(m.geom_dataid[g])
            v = np.asarray(m.mesh_vert[int(m.mesh_vertadr[i]): int(m.mesh_vertadr[i]) + int(m.mesh_vertnum[i])], float)
            lo, hi = v.min(0), v.max(0)
            off, h = (lo + hi) / 2.0, (hi - lo) / 2.0
        else:                                    # the floor plane, height fields: not boxes
            continue
        R = np.asarray(d.geom_xmat[g], float).reshape(3, 3)
        cs.append(np.asarray(d.geom_xpos[g], float) + R @ off)
        Rs.append(R)
        hs.append(h)
        bs.append(int(m.geom_bodyid[g]))
        gs.append(int(g))
    if not cs:
        z = np.zeros((0, 3))
        return Boxes(z, np.zeros((0, 3, 3)), z, np.zeros(0, int), np.zeros(0, int))
    return Boxes(np.array(cs), np.array(Rs), np.array(hs), np.array(bs), np.array(gs))


def overlap(A: Boxes, B: Boxes) -> np.ndarray:
    """(len(A), len(B)) bool: whether each pair of boxes overlaps -- the fifteen separating axes, after
    a bounding-sphere prefilter that only drops pairs too far apart to touch."""
    out = np.zeros((len(A), len(B)), bool)
    if not len(A) or not len(B):
        return out
    ra = np.linalg.norm(A.h, axis=1)
    rb = np.linalg.norm(B.h, axis=1)
    T = B.c[None, :, :] - A.c[:, None, :]
    near = np.linalg.norm(T, axis=2) <= ra[:, None] + rb[None, :]
    ia, ib = np.nonzero(near)
    if not len(ia):
        return out
    Ra, Rb, ha, hb, t = A.R[ia], B.R[ib], A.h[ia], B.h[ib], T[ia, ib]
    C = np.einsum("nki,nkj->nij", Ra, Rb)            # C[i, j] = A axis i . B axis j
    aC = np.abs(C) + EPS                             # the epsilon keeps parallel edges from false positives
    ta = np.einsum("nk,nki->ni", t, Ra)              # the centre offset in A's frame
    sep = np.zeros(len(ia), bool)
    # A's three face normals
    sep |= np.any(np.abs(ta) > ha + np.einsum("nij,nj->ni", aC, hb), axis=1)
    # B's three face normals
    tb = np.einsum("nk,nkj->nj", t, Rb)
    sep |= np.any(np.abs(tb) > hb + np.einsum("nij,ni->nj", aC, ha), axis=1)
    # the nine edge-edge axes A_i x B_j
    for i in range(3):
        i1, i2 = (i + 1) % 3, (i + 2) % 3
        for j in range(3):
            j1, j2 = (j + 1) % 3, (j + 2) % 3
            lhs = np.abs(ta[:, i2] * C[:, i1, j] - ta[:, i1] * C[:, i2, j])
            rhs = (ha[:, i1] * aC[:, i2, j] + ha[:, i2] * aC[:, i1, j]
                   + hb[:, j1] * aC[:, i, j2] + hb[:, j2] * aC[:, i, j1])
            sep |= lhs > rhs
    out[ia, ib] = ~sep
    return out


def _travel(R0, p0, R1, p1, reach: float) -> float:
    """The farthest a point within `reach` of the tool point moves between two tool poses: the
    translation plus the rotation angle times the reach."""
    c = float(np.clip((np.trace(R0.T @ R1) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.linalg.norm(np.asarray(p1) - np.asarray(p0))) + float(np.arccos(c)) * reach


def _slerp(R0, R1, f: float):
    """The rotation f of the way from R0 to R1 along the geodesic."""
    import math
    dR = R0.T @ R1
    c = float(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0))
    ang = math.acos(c)
    if ang < 1e-9:
        return R0
    w = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2.0 * math.sin(ang))
    a = ang * f
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    return R0 @ (np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K)


@dataclass
class Hit:
    sample: int              # the path sample at which it overlaps
    pose: tuple              # the tool pose there (R, p)
    mover_geom: int          # the moving box (hand, finger or held object geom)
    body: int                # the body it meets
    geom: int


def sweep(static: Boxes, moving: Boxes, poses: list, grow: float, step: float) -> Hit | None:
    """Test `moving` (boxes in the tool frame) along the tool path through `poses` (world (R, p)),
    against `static`. The path is the straight segments between poses, subdivided so that no point
    of the moving boxes travels farther than `step` between samples; each sample's boxes are grown by
    `step` (the lemma) and by `grow` (the servo's lead and the contact margins). The first overlap, or
    None: then the path overlaps nothing anywhere, not only at the samples."""
    if not len(moving) or len(poses) < 1:
        return None
    reach = float(np.max(np.linalg.norm(moving.c, axis=1) + np.linalg.norm(moving.h, axis=1)))
    samples = [poses[0]]
    for (R0, p0), (R1, p1) in zip(poses[:-1], poses[1:]):
        n = max(1, int(np.ceil(_travel(R0, p0, R1, p1, reach) / step)))
        for k in range(1, n + 1):
            f = k / n
            samples.append((_slerp(R0, R1, f), np.asarray(p0) + f * (np.asarray(p1) - np.asarray(p0))))
    grown = moving.grown(grow + step)
    for i, (R, p) in enumerate(samples):
        hit = overlap(grown.moved(R, np.asarray(p, float)), static)
        if hit.any():
            a, b = np.argwhere(hit)[0]
            return Hit(i, (R, p), int(moving.geom[a]), int(static.body[b]), int(static.geom[b]))
    return None


def scene_boxes(m, d, is_robot, exclude_bodies: set[int] = frozenset()) -> Boxes:
    """Every collision geom that is not the robot's and not of an excluded body, as boxes."""
    geoms = [g for g in range(m.ngeom) if _collides(m, g)
             and not is_robot(int(m.geom_bodyid[g])) and int(m.geom_bodyid[g]) not in exclude_bodies]
    return geom_boxes(m, d, geoms)


def subtree(m, root: int) -> set[int]:
    """A body and every body under it."""
    out = set()
    for b in range(m.nbody):
        x = b
        while x > 0 and x != root:
            x = int(m.body_parentid[x])
        if x == root:
            out.add(b)
    return out


def object_bodies(scene, obj: str) -> set[int]:
    """An object's root body and its children."""
    return subtree(scene.m, scene._root_id(obj))


def support_bodies(scene, region: str) -> set[int]:
    """The fixture a region's site hangs on, from its root down (a drawer's region counts its whole
    cabinet), or the object a region names (On(bowl, plate))."""
    m = scene.m
    try:
        b = int(m.site_bodyid[m.site_name2id(region)])
    except ValueError:
        b = scene._root_id(region)
    while int(m.body_parentid[b]) > 0:
        b = int(m.body_parentid[b])
    return subtree(m, b)
