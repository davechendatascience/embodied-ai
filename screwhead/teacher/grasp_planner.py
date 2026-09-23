"""Grasp candidates, synthesised from an object's collision geometry.

A candidate is (tool rotation, tool position at the grasp, jaw opening needed, approach
direction). This module only PROPOSES; reach.py decides which one the arm can use.

Three tiers, tried in order (Skills.grasp_for):
  faces   the whole object, jaws across a face that fits (a can, a box)
  parts   one collision geom near the top, jaws across it (a bowl's 2.6 mm wall: the
          rim grasp, derived rather than hand-written)
  sides   the same geoms approached horizontally, for things under a shelf (a bowl in a
          drawer) -- the handle grasp, applied to objects
"""
from __future__ import annotations

import numpy as np

from ..geometry.frames import EPS_DIR, EPS_NORM, Z, tool_frame, top_down
from ..sim import contacts
from ..sim.scan import GripperScan, support_below
from ..sim.scene import VERTICAL_COS, geom_world_box

MIN_DEPTH = 0.02          # grasp at least this far below an object's top
DEPTH_FRACTION = 0.66     # ... or this fraction of its half-height, whichever is deeper
PALM_MARGIN = 0.003       # never deeper than the palm clearance less this. It was 10 mm against a
#                           palm measured at 40.6 mm; the palm is at 31 mm, so grasps had been
#                           running 0.4 mm clear of it, and 10 mm on the true value made tall
#                           grasps 10 mm shallower (orange juice slipped out mid-carry)
FLOOR_MARGIN = 0.005      # never lower than this above the object's bottom
FLOOR_CLEARANCE = 0.002   # the fingers' deepest point stays this far above the support
HEIGHT_TOL = 0.005        # a geom is "at the jaws' height" within this
PAD_HALF_WIDTH = 0.01     # material further than this off the jaw line misses the pads
MERGE_GAP = 0.002         # spans closer than this along the jaw axis are one piece
FROM_BELOW = 0.3          # an approach with z above this comes from underneath: never
RAY_TOL = 0.03            # a ray may stop this short of the handle on the same body
RIM_BAND = 0.02           # part geoms whose top is within this of the object's top
MAX_PARTS = 16
HANDLE_OUTER = 0.6        # a handle geom's centre is at least this fraction of the farthest geom
#                           centre's plan distance from the object's origin (pan: 113-259 of 259 mm
#                           against the dish's 100; moka pot: 42-75 of 75 against the body's 36)
HANDLE_ALONG = (0.25, 0.5, 0.75)   # points along a long handle geom, from its end nearer the origin


def _collides_geom(m, g: int) -> bool:
    return bool(m.geom_contype[g] or m.geom_conaffinity[g])


class GraspPlanner:
    def __init__(self, env, config):
        self.env, self.scene, self.k = env, env.scene, config
        self._gripper: GripperScan | None = None

    # -- the gripper, measured once -----------------------------------------------------
    def gripper(self) -> GripperScan:
        """The gripper's reach past the tool point and its palm clearance behind it, measured
        once from its exact collision geometry (sim/scan.py).

        A grasp deeper than the palm clearance under an object's top drives the palm into the
        object: a 146 mm bottle grasped 48 mm down stalled with the palm in contact and no
        finger contact at all. A grasp lower than the reach above the support puts the fingers
        into it: every candidate on the butter did. Measured along the tool's own axis, not
        world z: straight after a drawer the gripper is horizontal, and a world-z measurement
        put every bowl grasp 70 mm too high (0/20).
        """
        if self._gripper is None:
            s = self.env.snapshot()
            self._gripper = GripperScan.measure(self.scene, s["R_tool"], s["p_tool"])
        return self._gripper

    # -- tiers --------------------------------------------------------------------------
    def handles(self, obj: str, box, floor: float) -> list:
        """Top-down grasps on an object's handle: the collision geoms lying out from its origin
        (HANDLE_OUTER), jaws across each geom's thinnest horizontal axis, at points along a long one."""
        m, d = self.scene.m, self.scene.d
        bid = self.scene.body_id(obj)
        _R, q = self.scene.body_pose(obj)
        geoms = [g for g in self._body_geoms(bid) if _collides_geom(m, g)]
        if not geoms:
            return []
        boxes = {g: geom_world_box(m, d, g, self.scene.base) for g in geoms}
        dist = {g: float(np.linalg.norm((boxes[g][0] - q)[:2])) for g in geoms}
        far = max(dist.values())
        out = []
        for g in sorted(geoms, key=lambda g: dist[g]):
            if dist[g] < HANDLE_OUTER * far:
                continue
            c, Rg, hl = boxes[g]
            hw = np.abs(Rg @ np.diag(hl)).sum(1)
            radial = (c - q)[:2] / max(dist[g], EPS_NORM)
            horiz = [(2 * float(hl[i]), Rg[:, i]) for i in range(3) if abs(Rg[2, i]) < VERTICAL_COS]
            fits = [a for a in horiz if a[0] + self.k.grip_margin <= self.k.max_grip]
            if not fits:
                continue
            # the jaws close ACROSS the handle, not along the line back to the object: with a square
            # section the thinnest axis was a coin toss, the radial one ran the jaw line through the
            # moka pot's body (105 mm) and the descent stalled 50 mm up (libero_10 2, 9 -> 0 of 20)
            w, thin = min(fits, key=lambda a: abs(float(a[1][:2] @ radial)))
            long_len, long_dir = max(horiz, key=lambda a: abs(float(a[1][:2] @ radial)))
            points = [c]
            if long_len > 3 * w:                      # a bar: several points along it, nearest end first
                sgn = -1.0 if float((c - q)[:2] @ long_dir[:2]) > 0 else 1.0
                points = [c + sgn * long_dir * long_len * (0.5 - f) for f in HANDLE_ALONG]
            for pt in points:
                out += self.at(pt, float(hw[2]), thin, w, body=bid, floor=floor)
        return out

    def tiers(self, obj: str, box, held_by: str | None = None) -> list[tuple[str, list]]:
        """Candidate grasps in order of preference, each tier tried on its own.

        A tier that offers nothing FEASIBLE must hand over rather than force: forcing left
        the teacher reaching for a 107 mm grasp it could never close, 0/3 with the arm
        re-anchoring and touching nothing.
        """
        bid = self.scene.body_id(obj)
        half_h = float(np.abs(box.R @ np.diag(box.half)).sum(1)[2])
        floor = support_below(self.scene, self._ray, obj) + self.gripper().reach + FLOOR_CLEARANCE
        faces, parts, sides = [], [], []
        for w, d in box.width_axes():                     # narrow face first
            if w + self.k.grip_margin <= self.k.max_grip and self.clear_body(box.world_centre, -Z, bid):
                faces += self.at(box.world_centre, half_h, d, w, body=bid, floor=floor)
        deep, middle, shallow = [], [], []
        for c, hw, d, w in self.parts(obj, box):
            if self.clear_body(c, -Z, bid):
                deep += self.at(c, hw, d, w, body=bid, floor=floor, deep=1.0)
                middle += self.at(c, hw, d, w, body=bid, floor=floor, deep=0.5)
                shallow += self.at(c, hw, d, w, body=bid, floor=floor)
        # the palm's depth first, then halfway, then the usual depth: deep only, a bowl inside the top
        # drawer had no reachable pinch at all and was forced (libero_spatial 4, 49 -> 8 of 50); there
        # the palm's depth meets the cabinet in every layout, and the usual depth let the bowl pivot
        # out of the pads as it left the drawer floor (2 of 50)
        parts = deep + middle + shallow
        for g in self.part_geoms(obj, box):
            sides += self.handle_grasps(g)
        if not (faces or parts or sides):
            d0 = box.width_axes()[0][1] if box.width_axes() else np.array([1.0, 0.0, 0.0])
            faces = self.at(box.world_centre, half_h, d0, self.k.max_grip - self.k.grip_margin, floor=floor)
        handles = self.handles(obj, box, floor) if held_by == "handle" else []
        # a category held by its handle is offered the handle first (affordances.yaml held_by)
        return [(n, t) for n, t in (("handle", handles), ("faces", faces), ("parts", parts), ("sides", sides)) if t]

    def at(self, centre: np.ndarray, half_h: float, d: np.ndarray, w: float, body: int = -1,
           floor: float = -np.inf, deep: float = 0.0) -> list:
        """Top-down grasps on a body of this height, jaws both ways round, the tool point no
        lower than `floor` (the support plus the fingers' reach).

        `deep`: the fraction of the way from the usual depth to the deepest the palm allows (1: as
        deep as the palm allows, the preferred pinch on a thin wall). A rim pinch is a pivot
        the pads' vertical spread alone resists; LIBERO's humans pinch the bowl with the pads 36 mm
        apart up its wall where 20 mm below the rim put the teacher's 20 mm apart, and at the
        palm's depth the bowl's mid-carry drops fell 12 -> 1 (spatial 6), 14 -> 4 (spatial 7) and
        12 -> 5 (goal 3) episodes of 20."""
        deepest = self.gripper().palm - PALM_MARGIN
        usual = min(max(MIN_DEPTH, DEPTH_FRACTION * half_h), deepest)
        depth = usual + float(deep) * (deepest - usual)
        p = centre + Z * max(half_h - depth, -half_h + FLOOR_MARGIN)
        p[2] = max(float(p[2]), floor)
        h = np.array([d[0], d[1], 0.0])
        n = np.linalg.norm(h)
        if n < EPS_DIR:
            return []
        h = h / n
        if body >= 0:
            w = self.width_at(body, p, h) or w
        return [(top_down(h), p, w, -Z), (top_down(-h), p, w, -Z)]

    def width_at(self, body: int, p: np.ndarray, jaw: np.ndarray) -> float:
        """How wide the object is WHERE THE JAWS WILL BE, over the material connected to
        the grasp point.

        A wine bottle's box is 43 mm across and its neck, which a grasp 30 mm below the top
        holds, is far narrower. And only material the pads can meet counts: the far wall of
        a bowl is a separate piece 100 mm along the same axis, and neighbouring wall
        segments more than 10 mm off the jaw line never come between the ~20 mm pads.
        """
        spans = [s for g in self._body_geoms(body) if (s := self._span(g, p, jaw)) is not None]
        if not spans:
            return 0.0
        at = float(p @ jaw)
        lo = hi = None
        for a, b in sorted(spans):                     # merge, keep the piece under the jaws
            if lo is None or a > hi + MERGE_GAP:
                if lo is not None and lo - MERGE_GAP <= at <= hi + MERGE_GAP:
                    return float(hi - lo)
                lo, hi = a, b
            else:
                hi = max(hi, b)
        return float(hi - lo) if lo is not None and lo - MERGE_GAP <= at <= hi + MERGE_GAP else 0.0

    def _span(self, g: int, p: np.ndarray, jaw: np.ndarray) -> tuple[float, float] | None:
        """This geom's extent along the jaw axis, if it is where the pads will be."""
        m, d = self.scene.m, self.scene.d
        c, Rg, hl = geom_world_box(m, d, g, self.scene.base)
        hw = np.abs(Rg @ np.diag(hl)).sum(1)
        if not (c[2] - hw[2] - HEIGHT_TOL <= p[2] <= c[2] + hw[2] + HEIGHT_TOL):
            return None
        across = np.cross(Z, jaw)
        if abs(float((c - p) @ across)) > PAD_HALF_WIDTH + float(np.abs(across @ (Rg @ np.diag(hl))).sum()):
            return None
        extent = float(np.abs(jaw @ (Rg @ np.diag(hl))).sum())
        m_c = float(c @ jaw)
        return m_c - extent, m_c + extent

    def _body_geoms(self, body: int) -> list[int]:
        m = self.scene.m
        return [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == body and _collides_geom(m, g)]

    def handle_grasps(self, g: int) -> list:
        """Grasps on a handle geom, from every pairing of its own axes.

        A handle is not approached from above: the cabinet's pull bar has the next drawer
        directly over it, and a top-down funnel never converged. The bar is 15 x 16 mm
        across and 89 mm long, so the jaws take it from the front and close vertically --
        one pairing of the geom's axes, found the same way for any handle.
        """
        m, d = self.scene.m, self.scene.d
        c, Rg, hl = geom_world_box(m, d, g, self.scene.base)
        out, blocked = [], []
        for j in range(3):
            w = 2 * float(hl[j])
            if w + self.k.grip_margin > self.k.max_grip:
                continue
            for i in (i for i in range(3) if i != j):
                for sgn in (1.0, -1.0):
                    app = sgn * Rg[:, i]
                    if app[2] > FROM_BELOW:
                        continue
                    cands = [(tool_frame(jaw, app), c, w, app) for jaw in (Rg[:, j], -Rg[:, j])]
                    (out if self.clear_geom(c, app, g) else blocked).extend(cands)
        return out or blocked

    # -- is the way in clear? ---------------------------------------------------------------
    def _ray(self, start_base: np.ndarray, direction: np.ndarray, exclude: int = -1) -> tuple[int, float]:
        """(first geom hit, distance) along a ray from a base-frame point."""
        import mujoco
        mm, dd = self.scene.raw()
        gid = np.zeros(1, np.int32)
        dist = float(mujoco.mj_ray(mm, dd, start_base + self.scene.base, direction, None, 1, exclude, gid))
        return int(gid[0]), dist

    def ray_scene(self, start_base: np.ndarray, direction: np.ndarray, exclude: int = -1,
                  max_skips: int = 8) -> tuple[int, float]:
        """_ray through the robot's own geoms: a test of the scene must not see the arm over it."""
        m = self.scene.m
        v = np.asarray(direction, float)
        p, travelled = np.asarray(start_base, float).copy(), 0.0
        for _ in range(max_skips):
            g, dist = self._ray(p, v, exclude)
            if g < 0 or dist < 0:
                return -1, -1.0
            if not contacts.is_robot(contacts.body_name(m, int(m.geom_bodyid[g]))):
                return g, travelled + dist
            step = dist + 1e-4
            p, travelled = p + v * step, travelled + step
        return -1, -1.0

    def clear_geom(self, p: np.ndarray, app: np.ndarray, geom: int) -> bool:
        """Can the tool come in along `app` and reach this geom?

        IK does not know about the cabinet. Straight down on to the drawer's pull bar is a
        perfectly reachable pose, and the shelf 38 mm above the bar makes it impossible --
        the teacher hovered there for 300 steps. A ray down the approach answers it.
        """
        v = np.asarray(app, float)
        v = v / (np.linalg.norm(v) + EPS_NORM)
        hit, dist = self._ray(p - v * self.k.approach, v)
        if hit < 0 or dist < 0:
            return False
        if hit == geom:
            return True
        m = self.scene.m      # another geom of the same moving part, where the handle is
        return int(m.geom_bodyid[hit]) == int(m.geom_bodyid[geom]) and dist > self.k.approach - RAY_TOL

    def clear_body(self, p: np.ndarray, app: np.ndarray, body: int) -> bool:
        """The tool can come down this line and meet the object, not something else."""
        v = np.asarray(app, float)
        v = v / (np.linalg.norm(v) + EPS_NORM)
        hit, dist = self._ray(p - v * self.k.approach, v)
        return hit >= 0 and dist >= 0 and int(self.scene.m.geom_bodyid[hit]) == body

    # -- parts ------------------------------------------------------------------------------
    def geom_part(self, g: int) -> tuple[np.ndarray, float, np.ndarray, float] | None:
        """(centre, half height, thin horizontal direction, width) of one collision geom."""
        m, d = self.scene.m, self.scene.d
        c, Rg, hl = geom_world_box(m, d, g, self.scene.base)
        hw = np.abs(Rg @ np.diag(hl)).sum(1)
        axes = sorted(((2 * float(hl[i]), Rg[:, i]) for i in range(3) if abs(Rg[2, i]) < VERTICAL_COS),
                      key=lambda a: a[0])
        if not axes:
            return None
        return (c, float(hw[2]), axes[0][1], axes[0][0])

    def part_geoms(self, obj: str, box) -> list[int]:
        """Collision geoms near the object's top that the jaws could fit around."""
        m, d = self.scene.m, self.scene.d
        top = box.p[2] + (box.R @ box.centre)[2] + float(np.abs(box.R @ np.diag(box.half)).sum(1)[2])
        out = []
        for g in self._body_geoms(self.scene.body_id(obj)):
            c, Rg, hl = geom_world_box(m, d, g, self.scene.base)
            hw = np.abs(Rg @ np.diag(hl)).sum(1)
            if float(c[2] + hw[2]) < top - RIM_BAND:
                continue                               # the foot ring, not the rim
            if min(2 * float(hl[i]) for i in range(3)) + self.k.grip_margin > self.k.max_grip:
                continue
            out.append(g)
        return out[:MAX_PARTS]

    def parts(self, obj: str, box) -> list[tuple[np.ndarray, float, np.ndarray, float]]:
        """Top-down grasp sites on those geoms, the tallest wall first."""
        out = [part for g in self.part_geoms(obj, box)
               if (part := self.geom_part(g)) is not None and part[3] + self.k.grip_margin <= self.k.max_grip]
        out.sort(key=lambda o: -o[1])                  # a taller wall gives the pads more to hold
        return out[:MAX_PARTS]
