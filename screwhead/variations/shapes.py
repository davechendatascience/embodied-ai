"""An object's shape in plan, measured from its collision geometry (AXM-object-geometry-known).

Used twice: once over the pool, at rest upright in a probe scene, to plan layouts (the catalog); and
again in every generated scene at rest, where the kept scene's checks and the task's fit are decided
(BRN-lv-scenes-valid, BRN-lv-tasks-doable). The planning numbers never decide anything by themselves.

Several of LIBERO's groceries are modelled lying down and stood up by their initial rotation (the
sauces, milk, juice, soup: their body z axis is horizontal at rest), so the upward axis is measured per
category, not assumed to be body z.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

FLOOR_FRACTION = 0.25     # a container's geom is floor if its top is in the lowest quarter of its height;
#                           the others are walls (bowl: floor tops 7 mm of 51; basket: 22 of 142)


@dataclass(frozen=True)
class Shape:
    up: tuple[float, float, float]         # the body axis that points up at rest, body frame
    axes: tuple[int, int]                  # the two body axes that lie horizontal at rest
    half: tuple[float, float]              # collision box half-extents along those axes, m
    height: float                          # collision box extent along the up axis, m
    walls: tuple[tuple[float, float, float, float], ...] = ()   # wall geoms' boxes in plan, (lo_a, lo_b, hi_a,
    #                                        hi_b) along `axes` about the collision box's centre (containers)

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Shape:
        return cls(up=tuple(d["up"]), axes=tuple(d["axes"]), half=tuple(d["half"]), height=float(d["height"]),
                   walls=tuple(tuple(w) for w in d.get("walls", ())))


def _geom_boxes(scene, name: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """(lo, hi) of each collision geom's box, in the object's body frame."""
    from ..sim.scene import _quat_to_R, geom_box
    m = scene.m
    bid = scene.body_id(name)
    out = []
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) != bid or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        R = _quat_to_R(m.geom_quat[g])
        centre, half = geom_box(m, g)
        c = m.geom_pos[g] + R @ centre
        h = np.abs(R @ np.diag(half)).sum(1)
        out.append((c - h, c + h))
    return out


def measure(scene, name: str, container: bool = False) -> Shape:
    """The object's shape as it now rests: its up axis is the body axis nearest the world's vertical."""
    R, _p = scene.body_pose(name)
    k = int(np.argmax(np.abs(R[2])))                   # body axis with the largest vertical component
    up = np.zeros(3)
    up[k] = np.sign(R[2, k])
    axes = tuple(i for i in range(3) if i != k)
    box = scene.object_box(name)
    walls: list[tuple[float, float, float, float]] = []
    if container:
        lo_up = box.centre[k] - box.half[k]
        for lo, hi in _geom_boxes(scene, name):
            top = hi[k] if up[k] > 0 else -lo[k]
            base = lo_up if up[k] > 0 else -(box.centre[k] + box.half[k])
            if top - base > FLOOR_FRACTION * 2 * box.half[k]:
                a, b = axes
                walls.append((float(lo[a] - box.centre[a]), float(lo[b] - box.centre[b]),
                              float(hi[a] - box.centre[a]), float(hi[b] - box.centre[b])))
    return Shape(up=tuple(float(v) for v in up), axes=axes, half=(float(box.half[axes[0]]), float(box.half[axes[1]])),
                 height=float(2 * box.half[k]), walls=tuple(walls))


def upright_angle(scene, name: str, shape: Shape) -> float:
    """Degrees between the category's up axis, as the object now stands, and the world's vertical."""
    R, _p = scene.body_pose(name)
    return float(np.degrees(np.arccos(np.clip(R[2] @ np.asarray(shape.up), -1.0, 1.0))))


def plan_bounds(scene, name: str, grow: float = 0.0, origin=(0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) in table coordinates -- world x, y less the table's `origin` -- of the plan box that holds the object's
    collision box as it now stands, grown by `grow` on every side. It contains the footprint, so tests on it are
    conservative."""
    box = scene.object_box(name)
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * box.half
    world = box.p + (box.centre + corners) @ box.R.T
    xy = world[:, :2] + scene.base[:2] - np.asarray(origin, float)
    return xy.min(0) - grow, xy.max(0) + grow


def fits_on(a: Shape, b: Shape, clearance: float) -> bool:
    """A's plan box, grown by the clearance on every side, fits inside B's plan box, as it is or turned a
    quarter turn."""
    wa, la = (2 * h + 2 * clearance for h in a.half)
    wb, lb = (2 * h for h in b.half)
    return (wa <= wb and la <= lb) or (la <= wb and wa <= lb)


def fits_in(a: Shape, c: Shape, clearance: float) -> bool:
    """A's plan box, grown by the clearance on every side and centred in C's collision box, meets none of C's
    walls in plan, as it is or turned a quarter turn."""
    if not c.walls:
        return False
    for ha, hb in (a.half, a.half[::-1]):
        ha, hb = ha + clearance, hb + clearance
        if all(lo_a >= ha or hi_a <= -ha or lo_b >= hb or hi_b <= -hb for lo_a, lo_b, hi_a, hi_b in c.walls):
            return True
    return False
