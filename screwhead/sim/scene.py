"""What a skill needs to read from a LIBERO scene, measured rather than tabulated.

Everything here comes from the live MuJoCo model: an object's extent from its collision
geoms, a region's pose and half-extents from its site, an articulated fixture's joint and
handle from the body that carries them. Nothing is keyed by task.

Two traps, both measured:
  - robosuite's `horizontal_radius_site` / `top_site` / `bottom_site` are boilerplate in
    every LIBERO object XML (alphabet_soup carries 0.025 / +-0.04 while its real footprint
    is 37 x 47 x 73 mm). Use the geoms.
  - handles have no sites. A drawer's pull bar is the collision geom furthest along the
    joint's slide axis; the microwave's is the capsule furthest along the door's swing.

Poses are returned in the ROBOT BASE frame, the frame the teacher and the servos work in.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VERTICAL_COS = 0.7      # an axis with |z| above this is not a jaw direction

# joint qpos thresholds LIBERO's own predicates use (libero/envs/objects/articulated_objects.py)
ARTICULATION = {
    "wooden_cabinet": dict(open=-0.14, close=0.0, sign=-1),
    "white_cabinet": dict(open=-0.14, close=0.0, sign=-1),
    "short_cabinet": dict(open=0.10, close=0.0, sign=+1),
    "microwave": dict(open=-1.3, close=-0.005, sign=-1),
    "short_fridge": dict(open=2.0, close=0.0, sign=+1),
    "flat_stove": dict(on=0.5, off=0.0, sign=+1),
}


@dataclass(frozen=True)
class Box:
    """An axis-aligned box in the body's own frame, plus that body's world pose."""
    R: np.ndarray           # body rotation, base frame
    p: np.ndarray           # body origin, base frame
    centre: np.ndarray      # box centre in the body frame
    half: np.ndarray        # half-extents in the body frame

    @property
    def world_centre(self) -> np.ndarray:
        return self.p + self.R @ self.centre

    def top(self) -> float:
        """Height of the box top above the body origin, along the world z axis."""
        return float((self.R @ self.centre)[2] + np.abs(self.R @ np.diag(self.half)).sum(1)[2])

    def width_axes(self) -> list[tuple[float, np.ndarray]]:
        """(width, world direction) for the two horizontal body axes, narrow one first."""
        out = []
        for i in (0, 1, 2):
            d = self.R[:, i]
            if abs(d[2]) > VERTICAL_COS:
                continue
            out.append((2 * float(self.half[i]), d))
        return sorted(out, key=lambda x: x[0])


class Scene:
    """Geometry queries against a live LIBERO environment."""

    def __init__(self, env):
        self.env = env                       # robosuite env (PrivilegedEnv.env.env)
        self.sim = env.sim
        self.m, self.d = self.sim.model, self.sim.data

    # -- frames ---------------------------------------------------------------------
    @property
    def base(self) -> np.ndarray:
        return self.d.body_xpos[self.m.body_name2id("robot0_base")].copy()

    def raw(self):
        """The mujoco.MjModel / MjData under robosuite's wrappers, for mujoco.* calls."""
        return getattr(self.m, "_model", self.m), getattr(self.d, "_data", self.d)

    def body_id(self, name: str) -> int:
        """robosuite names an object's root body `<instance>_main`; the bddl calls it `<instance>`."""
        names = self._body_names()
        for cand in (name, f"{name}_main"):
            if cand in names:
                return self.m.body_name2id(cand)
        raise ValueError(f"no body for {name!r}")

    def body_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        b = self.body_id(name)
        return self.d.body_xmat[b].reshape(3, 3).copy(), self.d.body_xpos[b] - self.base

    # -- objects --------------------------------------------------------------------
    def object_box(self, name: str) -> Box:
        """AABB over the object's COLLISION geoms, in its body frame."""
        bid = self.body_id(name)
        lo = np.full(3, np.inf); hi = np.full(3, -np.inf)
        for g in range(self.m.ngeom):
            if int(self.m.geom_bodyid[g]) != bid:
                continue
            if not (self.m.geom_contype[g] or self.m.geom_conaffinity[g]):
                continue                      # visual-only mesh
            R = _quat_to_R(self.m.geom_quat[g])
            centre, half = geom_box(self.m, g)
            c = self.m.geom_pos[g] + R @ centre
            h = np.abs(R @ np.diag(half)).sum(1)
            lo = np.minimum(lo, c - h); hi = np.maximum(hi, c + h)
        if not np.isfinite(lo).all():
            raise ValueError(f"{name}: no collision geoms")
        R, p = self.body_pose(self.m.body_id2name(bid))
        return Box(R=R, p=p, centre=(lo + hi) / 2, half=(hi - lo) / 2)

    def _body_names(self):
        return {self.m.body_id2name(i) for i in range(self.m.nbody)}

    # -- regions --------------------------------------------------------------------
    def region(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(R, p, half-extents) of a region site, base frame."""
        sid = self.m.site_name2id(name)
        R = self.d.site_xmat[sid].reshape(3, 3).copy()
        p = self.d.site_xpos[sid] - self.base
        half = np.asarray(self.m.site_size[sid], float).copy()
        sites = getattr(self.env, "object_sites_dict", {})
        if name in sites and getattr(sites[name], "size", None) is not None:
            half = np.asarray(sites[name].size, float).reshape(-1)[:3].copy()
        return R, p, half

    # -- articulation ---------------------------------------------------------------
    def articulation(self, region_or_fixture: str) -> dict:
        """The joint that opens a region (or switches a fixture) with its thresholds,
        pull direction and handle pose. Raises if the name is not articulated."""
        sites = getattr(self.env, "object_sites_dict", {})
        site = sites.get(region_or_fixture)
        joints = list(getattr(site, "joints", []) or []) if site is not None else []
        parent = getattr(site, "parent_name", None) if site is not None else None
        inst = parent or region_or_fixture
        if not joints:                        # a fixture named directly, e.g. flat_stove_1
            joints = [j for j in self._joint_names() if j.startswith(f"{inst}_")]
        if not joints:
            raise ValueError(f"{region_or_fixture}: no articulation joints")
        joint = joints[0]
        jid = self.m.joint_name2id(joint)
        cat = _category(inst)
        spec = ARTICULATION.get(cat)
        if spec is None:
            raise ValueError(f"{inst}: category {cat!r} has no declared articulation")
        axis_local = np.asarray(self.m.jnt_axis[jid], float)
        body = int(self.m.jnt_bodyid[jid])
        R = self.d.body_xmat[body].reshape(3, 3)
        axis = R @ axis_local
        qadr = int(self.m.jnt_qposadr[jid])
        handle, handle_geom = self._handle(body, axis_local, spec["sign"])
        anchor = self.d.body_xpos[body] + R @ np.asarray(self.m.jnt_pos[jid], float) - self.base
        lo, hi = (float(v) for v in self.m.jnt_range[jid])
        stop = lo if spec["sign"] < 0 else hi              # the joint's end in the opening direction
        open_room = ((stop - float(spec["open"])) * spec["sign"]
                     if "open" in spec and self.m.jnt_limited[jid] else np.inf)   # a knob has on/off, no open
        return dict(joint=joint, qposadr=qadr, qpos=float(self.d.qpos[qadr]), axis=axis,
                    sign=spec["sign"], thresholds=spec, category=cat, body=body,
                    handle=handle, handle_world=self.d.body_xpos[body] + R @ handle - self.base,
                    handle_geom=handle_geom, anchor=anchor, jnt_type=int(self.m.jnt_type[jid]),
                    R=R.copy(), open_room=open_room)

    def _handle(self, body: int, axis_local: np.ndarray, sign: int) -> tuple[np.ndarray, int]:
        """Grasp point on the moving part: the collision geom furthest along the opening
        direction (a drawer's pull bar, a door's handle), in that body's frame."""
        best, best_d, best_g = None, -np.inf, -1
        for g in range(self.m.ngeom):
            if int(self.m.geom_bodyid[g]) != body:
                continue
            if not (self.m.geom_contype[g] or self.m.geom_conaffinity[g]):
                continue
            if self.m.geom_type[g] == 7:      # mesh hull: the body shell, not the handle
                continue
            d = float(np.dot(self.m.geom_pos[g], axis_local) * sign)
            if d > best_d:
                best, best_d, best_g = np.asarray(self.m.geom_pos[g], float).copy(), d, g
        if best is None:
            raise ValueError("no graspable geom on the moving body")
        return best, best_g

    def _joint_names(self):
        return [self.m.joint_id2name(i) for i in range(self.m.njnt) if self.m.joint_id2name(i)]


def _category(instance: str) -> str:
    """wooden_cabinet_1 -> wooden_cabinet."""
    parts = instance.split("_")
    return "_".join(parts[:-1]) if parts[-1].isdigit() else instance


def geom_box(m, g) -> tuple[np.ndarray, np.ndarray]:
    """(centre, half-extents) of a geom's bounding box in the geom's own frame.

    MuJoCo's geom_aabb, exact for every type but the unbounded plane. The helper it replaces
    agreed with it on every primitive but treated a mesh as a cube of geom_size[0] about the
    geom's origin -- 21 to 118 mm off on each of the Panda's link and gripper meshes.
    """
    if int(m.geom_type[g]) == 0:               # a plane is unbounded; keep its drawn size
        return np.zeros(3), np.maximum(np.asarray(m.geom_size[g], float), 1e-4)
    a = np.asarray(m.geom_aabb[g], float)
    return a[:3].copy(), a[3:].copy()


def geom_world_box(m, d, g, base) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(centre relative to base, rotation, half-extents) of a geom's box, in the world."""
    centre, half = geom_box(m, g)
    Rg = d.geom_xmat[g].reshape(3, 3)
    return d.geom_xpos[g] - base + Rg @ centre, Rg, half


def _quat_to_R(q) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
