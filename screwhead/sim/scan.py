"""The scene as the simulator has it, measured rather than declared.

Hand-written geometry kept turning out wrong: the planner assumed the fingertips reach
5 mm past the tool point and the palm starts 40 mm behind it, but the Panda's finger
meshes reach 9.3 mm past it and its palm starts 31 mm behind; the box helper those numbers
came from treated every mesh as a cube of its first size about the geom's origin. With
the fingertips 4.3 mm into the floor at the planned height, every grasp candidate on the
butter and the cream cheese failed the collision screen (20 of 20, 39 of 40 forced).

Everything here reads the model and the current state (privileged, the teacher's input):
geom points are mesh vertices or the corners of MuJoCo's own exact geom_aabb.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .scene import geom_box

CORNERS = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
PROBE = 0.001            # m below an object's bottom the support ray starts


def geom_points(m, d, g: int) -> np.ndarray:
    """World points bounding a geom exactly: a mesh's vertices, otherwise its box's corners."""
    Rg = d.geom_xmat[g].reshape(3, 3)
    if int(m.geom_type[g]) == 7:
        mid = int(m.geom_dataid[g])
        v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        return d.geom_xpos[g] + np.asarray(v, float) @ Rg.T
    centre, half = geom_box(m, g)
    return d.geom_xpos[g] + (centre + CORNERS * half) @ Rg.T


def _collides(m, g: int) -> bool:
    return bool(m.geom_contype[g] or m.geom_conaffinity[g])


@dataclass(frozen=True)
class GripperScan:
    """Where the gripper's collision geometry is, along its approach axis."""
    reach: float        # m past the tool point, the deepest finger or pad point
    palm: float         # m behind the tool point, where the palm's shell begins

    @classmethod
    def measure(cls, scene, R_tool: np.ndarray, p_tool: np.ndarray) -> GripperScan:
        m, d = scene.m, scene.d
        z, p = R_tool[:, 2], p_tool + scene.base
        reach, palm = -np.inf, np.inf
        for g in range(m.ngeom):
            body = m.body_id2name(int(m.geom_bodyid[g])) or ""
            if not body.startswith("gripper0") or not _collides(m, g):
                continue
            along = (geom_points(m, d, g) - p) @ z
            if "finger" in body:
                reach = max(reach, float(along.max()))
            else:
                palm = min(palm, float(-along.max()))
        return cls(reach=reach, palm=palm)


def support_below(scene, ray, obj: str) -> float:
    """Height of the surface under an object's centre (the first thing a downward ray from
    just under its bottom meets, the object itself excluded)."""
    box = scene.object_box(obj)
    ext = np.abs(box.R) @ box.half
    c = box.world_centre
    start = np.array([c[0], c[1], c[2] - ext[2] - PROBE])
    _hit, dist = ray(start, np.array([0.0, 0.0, -1.0]), scene.body_id(obj))
    return float(start[2] - dist) if dist >= 0 else float(start[2] + PROBE)
