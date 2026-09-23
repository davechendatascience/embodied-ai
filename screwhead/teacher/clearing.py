"""Room to work: whether an opened container will crowd an object, and where to move it.

With one gripper, an object that the open container will overhang has to be moved before
the container is opened: once the container is open the object cannot be grasped, and while
the object is held the container cannot be opened. libero_goal 3: opened, the top drawer's
front panel overhangs the bowl by about 4 x 3.4 cm, and with the drawer set open at reset
the teacher picked and placed the bowl 0 times in 10. No partial opening serves: the
overhang clears only where the exposed interior is narrower than the bowl.

The test is geometric, on the container's footprint at its open target, not the grasp
screen: next to the open drawer the screen still passed 3-13 grasps, which then grazed the
drawer or never lifted the bowl.
"""
from __future__ import annotations

import itertools

import numpy as np

from ..geometry.frames import Z, axis_rot, pose
from .reach import SIGMA_WEIGHT
from ..sim.scene import geom_world_box

CROWD_XY = 0.03        # m: an open footprint this close to an object, in plan, crowds it
HAND_BAND = 0.15       # m: container parts this far above the object's top are in the hand's way
CLEAR_XY = 0.08        # m: a moved object keeps this far from the open footprint, in plan
SPOT_RADII = tuple(np.round(np.arange(0.08, 0.301, 0.02), 3))
SPOT_ANGLES = 32       # 5 radii x 16 angles left 1-2 free spots on crowded tables, none the arm was
#                        comfortable at, in 7 of 50 libero_goal 3 layouts: the bowl was then picked
#                        under the open drawer and lost. 12 x 32 found 10-15, best scores 0.86-0.94
SURFACE_TOL = 0.005    # m: every footprint ray lands within this of the surface it stood on
RAY_START = 0.15       # m above the surface the footprint rays start (below the arm at its start)
SPOT_MARGIN = 0.03     # m added to the object's half-extents for its footprint rays
FOOTPRINT_RAYS = 5     # per side: a 5 x 5 grid (5 rays missed a wine bottle between them)
REACH_ABOVE = (0.03, 0.15)   # m above the spot the tool must be comfortable: set down, and above it
MIN_SCORE = 0.2        # conditioning score (min joint margin, 8 x sigma) a spot must reach
SIGNS = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))


def _aabb_gap_xy(lo_a, hi_a, lo_b, hi_b) -> float:
    """Plan-view gap between two boxes; 0 when they overlap."""
    gap = np.maximum(0.0, np.maximum(lo_b[:2] - hi_a[:2], lo_a[:2] - hi_b[:2]))
    return float(np.linalg.norm(gap))


class Clearing:
    def __init__(self, scene, planner, reach):
        self.scene, self.planner, self.reach = scene, planner, reach

    def open_footprint(self, a: dict, dq: float) -> tuple[np.ndarray, np.ndarray]:
        """World box around the moving body's collision geoms with the joint moved by dq."""
        m, d = self.scene.m, self.scene.d
        pts = []
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) != a["body"] or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            c, R, half = geom_world_box(m, d, g, self.scene.base)
            corners = c + (SIGNS * half) @ R.T
            if a["jnt_type"] == 2:                                   # slide
                corners = corners + a["axis"] * dq
            else:                                                    # hinge, about its anchor
                corners = a["anchor"] + (corners - a["anchor"]) @ axis_rot(a["axis"], dq).T
            pts.append(corners)
        P = np.vstack(pts)
        return P.min(0), P.max(0)

    def _object_aabb(self, obj: str) -> tuple[np.ndarray, np.ndarray]:
        box = self.scene.object_box(obj)
        ext = np.abs(box.R) @ box.half
        return box.world_centre - ext, box.world_centre + ext

    def crowds(self, obj: str, a: dict, dq: float) -> bool:
        """Will the container, opened by dq, stand over or against the object where the
        hand has to work?"""
        lo_o, hi_o = self._object_aabb(obj)
        lo_c, hi_c = self.open_footprint(a, dq)
        return (_aabb_gap_xy(lo_o, hi_o, lo_c, hi_c) < CROWD_XY
                and lo_c[2] < hi_o[2] + HAND_BAND and hi_c[2] > lo_o[2])

    def spot(self, obj: str, a: dict, dq: float, R_tool: np.ndarray) -> np.ndarray | None:
        """The best-conditioned free place on the object's own surface, clear of the open
        container; None if the search finds none the arm is comfortable at. The nearest
        free place was beside the robot's base, where the set-down re-anchored and the
        bowl came down on a wine bottle."""
        lo_o, hi_o = self._object_aabb(obj)
        lo_c, hi_c = self.open_footprint(a, dq)
        half = (hi_o - lo_o)[:2] / 2 + SPOT_MARGIN
        centre, surface = (lo_o + hi_o)[:2] / 2, float(lo_o[2])
        exclude = self.scene.body_id(obj)
        cands = []
        for r, k in itertools.product(SPOT_RADII, range(SPOT_ANGLES)):
            th = 2 * np.pi * k / SPOT_ANGLES
            xy = centre + r * np.array([np.cos(th), np.sin(th)])
            gap = _aabb_gap_xy(np.r_[xy - half, 0.0], np.r_[xy + half, 0.0], lo_c, hi_c)
            if gap >= CLEAR_XY and self._free(xy, half, surface, exclude):
                cands.append(xy)
        if not cands:
            return None
        Ts = [pose(R_tool, np.array([xy[0], xy[1], surface + h])) for xy in cands for h in REACH_ABOVE]
        _th, conv, sig, margin = self.reach.solve(Ts)
        score = np.where(conv, np.minimum(margin, SIGMA_WEIGHT * sig), -np.inf)
        score = score.reshape(len(cands), len(REACH_ABOVE)).min(1)
        best = int(np.argmax(score))
        return cands[best] if score[best] >= MIN_SCORE else None

    def _free(self, xy, half, surface: float, exclude: int) -> bool:
        """Every ray of a grid over the footprint lands on the surface the object stood on."""
        down = -Z
        for fx, fy in itertools.product(np.linspace(-1, 1, FOOTPRINT_RAYS), repeat=2):
            p = np.array([xy[0] + fx * half[0], xy[1] + fy * half[1], surface + RAY_START])
            _g, dist = self.planner._ray(p, down, exclude)
            if dist < 0 or abs((p[2] - dist) - surface) > SURFACE_TOL:
                return False
        return True
