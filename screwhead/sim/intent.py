"""Which of the robot's contacts the state says are meant (DEF-intended-contact).

Per-intent state rules, decided 2026-09-26 over a single "finger zone": each contact the teacher means to make is its
own rule, computed from the simulator state alone, so a student executing through the same layer is judged the same
way. A contact between a robot geom and a body B is intended when

  held     both finger groups touch B, and B belongs to an object that moves on a free joint (the jaws hold it,
           or are closing on it); fingertips on either side of a table hold nothing;
  jaws     a finger touches B inside the jaws' closing volume: between the two fingers' inner faces, within the
           fingers' width across the jaw line, and along the approach from the palm's face to the finger tips (a rim
           pinched as deep as the palm touches the fingers above their pads), widened by CONTACT_TOL, where a
           contact point between two touching surfaces lies -- and gripping: its normal nearer the jaw line than the
           approach (a body wedged between the fingers and pressing back along the approach is not held by them;
           it pinned the Jaco's descent on the moka pot for 733 steps, libero_10 2);
  pushed   a finger touches B and B's category is moved by pushing (the affordance table), passed in as
           `pushed`;
  joint    the gripper touches B, and B is the body at the tool point among those that move on a joint of their
           own (a drawer, a door, a knob): the one whose collision geometry is nearest the tool point. The hand
           pushes a drawer shut (BRN-push-closes-sliding-drawer); a hand on the drawer above it is not driving it;
  support  a finger touches B while the jaws pinch another body that B touches (the support under a deep pinch);
  inside   the tool point lies inside a region site of B (a container the tool reaches into).

Every other robot-scene contact is unintended, and so is a contact between two robot bodies more than one joint
apart in the kinematic tree, unless both are the gripper's (bodies welded together, a parent and child, or the two
closed fingers touch by construction).

Calibrated on the Panda teacher, whose contacts must classify as intended wherever it relies on them.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .contacts import body_name, is_robot

CONTACT_TOL = 0.001    # m: a contact point lies half the penetration off each surface; pad contacts measured
#                        0.05-0.54 mm deep. Half a pad's thickness (4 mm) let fingertips pressing on top of
#                        a moka pot pass as 'between the jaws' (Jaco, libero_10 2)



@dataclass(frozen=True)
class Contact:
    robot: str          # the robot body touching
    other: str          # the body touched (a robot body for self-contact)
    rule: str           # held | jaws | joint | support | inside | UNINTENDED | SELF


class Intent:
    """The rules for one loaded model; call classify() on each forwarded state."""

    def __init__(self, m, d, pushed: frozenset[str] = frozenset()):
        self.m, self.d = getattr(m, "_model", m), getattr(d, "_data", d)
        mm = self.m
        self.names = [body_name(m, b) for b in range(mm.nbody)]
        self.robot = np.array([is_robot(n) for n in self.names])
        self.finger = np.array([is_robot(n) and "finger" in n for n in self.names])
        self.gripper = np.array([n.startswith("gripper") for n in self.names])
        self.left = np.array([self.finger[b] and ("left" in n or "joint1" in n) for b, n in enumerate(self.names)])
        collide = [g for g in range(mm.ngeom) if mm.geom_contype[g] or mm.geom_conaffinity[g]]
        self.pads = [g for g in collide if "pad" in (mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
        self.finger_geoms = [g for g in collide if self.finger[int(mm.geom_bodyid[g])]]
        self.palm_geoms = [g for g in collide if self.gripper_body(int(mm.geom_bodyid[g])) and not self.finger[int(mm.geom_bodyid[g])]]
        self.jointed = {b for b in range(mm.nbody) if not self.robot[b] and int(mm.body_jntnum[b]) > 0
                        and int(mm.jnt_type[int(mm.body_jntadr[b])]) in (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE)}
        self.free_roots = {b for b in range(mm.nbody) if int(mm.body_parentid[b]) == 0 and int(mm.body_jntnum[b]) > 0
                           and int(mm.jnt_type[int(mm.body_jntadr[b])]) == mujoco.mjtJoint.mjJNT_FREE}
        self.jointed_geoms = [(g, int(mm.geom_bodyid[g])) for g in collide if int(mm.geom_bodyid[g]) in self.jointed]
        self.regions = [s for s in range(mm.nsite) if "region" in (mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_SITE, s) or "")
                        and int(mm.site_type[s]) == mujoco.mjtGeom.mjGEOM_BOX and not self.robot[int(mm.site_bodyid[s])]]
        self._joints_between = {}
        self._xz = None
        self.pushed = {self._root(b) for b in range(mm.nbody) if not self.robot[b]
                       and any(self.names[self._root(b)] in (x, f"{x}_main") for x in pushed)}

    # -- the rules ------------------------------------------------------------------
    def _root(self, b: int) -> int:
        while int(self.m.body_parentid[b]) != 0:
            b = int(self.m.body_parentid[b])
        return b

    def _joint_count(self, a: int, b: int) -> int:
        """Joints on the kinematic-tree path between two bodies."""
        key = (min(a, b), max(a, b))
        if key not in self._joints_between:
            def up(x):
                path = [x]
                while x != 0:
                    x = int(self.m.body_parentid[x])
                    path.append(x)
                return path
            pa, pb = up(a), up(b)
            common = next(x for x in pa if x in set(pb))
            n = sum(int(self.m.body_jntnum[x]) for x in pa[:pa.index(common)]) + \
                sum(int(self.m.body_jntnum[x]) for x in pb[:pb.index(common)])
            self._joints_between[key] = n
        return self._joints_between[key]

    def gripper_body(self, b: int) -> bool:
        return bool(self.gripper[b])

    def _points(self, g: int, R_tool: np.ndarray, p_tool: np.ndarray) -> np.ndarray:
        """Geom g's surface points in the tool frame: a box's corners, a mesh's vertices."""
        m, d = self.m, self.d
        if int(m.geom_type[g]) == mujoco.mjtGeom.mjGEOM_MESH:
            mid = int(m.geom_dataid[g])
            local = m.mesh_vert[int(m.mesh_vertadr[mid]):int(m.mesh_vertadr[mid]) + int(m.mesh_vertnum[mid])]
        else:
            local = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * m.geom_size[g]
        return ((d.geom_xpos[g] + local @ d.geom_xmat[g].reshape(3, 3).T) - p_tool) @ R_tool

    def _jaws_volume(self, R_tool: np.ndarray, p_tool: np.ndarray):
        """The closing volume in the tool frame (lo, hi): between the pads' inner faces along the jaw axis (y), over
        the fingers' width (x), from the palm's face to the finger tips along the approach (z), widened by
        CONTACT_TOL. The fingers slide along y only, so their x and z extents are measured once."""
        if len(self.pads) != 2 or not self.palm_geoms:
            return None
        if self._xz is None:
            fingers = np.vstack([self._points(g, R_tool, p_tool) for g in self.finger_geoms])
            palm = max(float(self._points(g, R_tool, p_tool)[:, 2].max()) for g in self.palm_geoms)
            self._xz = (float(fingers[:, 0].min()), float(fingers[:, 0].max()), palm, float(fingers[:, 2].max()))
        a, b = sorted((self._points(g, R_tool, p_tool) for g in self.pads), key=lambda q: q[:, 1].mean())
        x0, x1, z0, z1 = self._xz
        return (np.array([x0, a[:, 1].max(), z0]) - CONTACT_TOL, np.array([x1, b[:, 1].min(), z1]) + CONTACT_TOL)

    def _at_tool(self, p_world: np.ndarray) -> int | None:
        """The body, of those moving on a joint of their own, whose collision geometry is nearest the point: distance
        to each geom's own bounding box, oriented with the geom."""
        m, d = self.m, self.d
        best, arg = np.inf, None
        for g, b in self.jointed_geoms:
            R = d.geom_xmat[g].reshape(3, 3)
            q = R.T @ (p_world - d.geom_xpos[g]) - m.geom_aabb[g, :3]
            dist = float(np.linalg.norm(np.maximum(np.abs(q) - m.geom_aabb[g, 3:], 0.0)))
            if dist < best:
                best, arg = dist, b
        return arg

    def _inside(self, p_world: np.ndarray) -> set[int]:
        """Roots of the bodies one of whose region sites contains the point."""
        out = set()
        for s in self.regions:
            R = self.d.site_xmat[s].reshape(3, 3)
            q = R.T @ (p_world - self.d.site_xpos[s])
            if np.all(np.abs(q) <= self.m.site_size[s]):
                out.add(self._root(int(self.m.site_bodyid[s])))
        return out

    def _contacts(self):
        """(contacts as (b1, b2, point, normal), bodies touching each body, bodies both finger groups touch)."""
        m, d = self.m, self.d
        found, sides, touching = [], {}, {}
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist > 0:
                continue
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            found.append((b1, b2, np.asarray(c.pos, float), np.asarray(c.frame[:3], float)))
            for x, y in ((b1, b2), (b2, b1)):
                touching.setdefault(x, set()).add(y)
                if self.finger[x] and not self.robot[y]:
                    sides.setdefault(y, set()).add("left" if self.left[x] else "right")
        return found, touching, {b for b, s in sides.items() if len(s) == 2 and self._root(b) in self.free_roots}

    def _is_self(self, b1: int, b2: int) -> bool:
        return b1 != b2 and not (self.gripper_body(b1) and self.gripper_body(b2)) and self._joint_count(b1, b2) > 1

    @staticmethod
    def _gripping(q: np.ndarray, normal: np.ndarray, vol, R_tool: np.ndarray) -> bool:
        """Inside the closing volume, pressing across the jaw line rather than along the approach."""
        return (vol is not None and bool(np.all((vol[0] <= q) & (q <= vol[1])))
                and abs(float(normal @ R_tool[:, 1])) >= abs(float(normal @ R_tool[:, 2])))

    def _rule(self, rb: int, ob: int, q: np.ndarray, normal: np.ndarray, at: dict) -> str:
        if ob in at["held"] or self._root(ob) in at["held_roots"]:
            return "held"
        if self.gripper_body(rb) and ob == at["at_tool"]:
            return "joint"
        if self.finger[rb]:
            if self._gripping(q, normal, at["vol"], at["R"]):
                return "jaws"
            if self._root(ob) in self.pushed:
                return "pushed"
            if any(ob in at["touching"].get(h, ()) for h in at["held"]):
                return "support"
        return "inside" if self._root(ob) in at["inside"] else "UNINTENDED"

    def classify(self, R_tool: np.ndarray, p_tool_world: np.ndarray) -> list[Contact]:
        """Every robot contact of negative or zero distance, with the rule that makes it intended, or none."""
        found, touching, held = self._contacts()
        at = dict(held=held, held_roots={self._root(b) for b in held}, touching=touching, R=R_tool,
                  vol=self._jaws_volume(R_tool, p_tool_world), inside=self._inside(p_tool_world),
                  at_tool=self._at_tool(p_tool_world))
        out, seen = [], set()
        for b1, b2, pt, normal in found:
            r1, r2 = self.robot[b1], self.robot[b2]
            if r1 and r2:
                key = (self.names[b1], self.names[b2], "SELF") if self._is_self(b1, b2) else None
            elif r1 or r2:
                rb, ob = (b1, b2) if r1 else (b2, b1)
                key = (self.names[rb], self.names[ob], self._rule(rb, ob, (pt - p_tool_world) @ R_tool, normal, at))
            else:
                key = None
            if key is not None and key not in seen:
                seen.add(key)
                out.append(Contact(*key))
        return out
