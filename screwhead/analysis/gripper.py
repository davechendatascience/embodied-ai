"""Grippers as trees of open chains.

A gripper is not a serial chain -- fingers branch from a shared palm -- but each
BRANCH is an ordinary open chain, so the same PoE machinery applies once the
tree is split into its root-to-leaf paths. Nothing new is needed for the
kinematics; only the tree walk changes.

What is NOT assumed: that the actuated command determines the pads. robosuite
couples the Robotiq85's fingers with a soft tendon spring whose length varies
0.26 to 3.06 over the command range, so pad pose is a function of the OBSERVED
joint vector and not of the command. Everything here takes joints as input.

Pad separation over the joint box is therefore an OUTER BOUND on what a gripper
can actually reach, exact for a parallel jaw (separation is affine in the slide
joints, so the extremes sit at the corners) and a superset for a linkage. An
outer bound is the right object for refusal: a feature wider than the derived
open span is provably ungraspable.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from ..geometry.kinematics import fk
from ..sim.mjcf import from_mjcf, leaf_paths
from ..geometry.poe import Chain

# Body origins closer than this give no closing axis (metres).
_COINCIDENT = 1e-12


@dataclass
class Gripper:
    name: str
    fingers: dict[str, Chain]           # tip body name -> chain from the palm

    @property
    def tips(self) -> list[str]:
        return sorted(self.fingers)


def _tip_bodies(path: str | Path) -> list[str]:
    root = ET.parse(path).getroot()
    world = root.find("worldbody")
    tips = []
    for body in world.findall("body"):
        for chain in leaf_paths(body):
            leaf = chain[-1]
            # A finger branch is one that carries at least one joint. `eef` and
            # the visualisation sites hang off the palm with none.
            if any(b.findall("joint") for b in chain):
                tips.append(leaf.get("name"))
    return tips


def load(path: str | Path, name: str | None = None) -> Gripper:
    """Parse a gripper MJCF into one chain per finger branch."""
    path = Path(path)
    root = ET.parse(path).getroot()
    fingers = {}
    for tip in _tip_bodies(path):
        try:
            fingers[tip] = from_mjcf(path, tool_body=tip, name=f"{path.stem}:{tip}",
                                     angle="radian")
        except (ValueError, NotImplementedError):
            # from_mjcf refuses a branch it cannot model as an open chain (a free or
            # ball joint, an orientation it does not implement); that branch is not a
            # finger, so it is left out rather than guessed at.
            continue
    return Gripper(name=name or root.get("model") or path.stem, fingers=fingers)


def pad_position(g: Gripper, q: dict[str, Tensor], tip: str) -> Tensor:
    """Tip origin of ONE finger, in the palm frame, from OBSERVED joints.

    Per-finger on purpose: a gripper's branches do not share joints, so
    computing every branch would demand joint values the caller has no reason
    to supply (the Robotiq85's knuckle branches, for instance).
    """
    chain = g.fingers[tip]
    theta = torch.stack([torch.as_tensor(q[j]) for j in chain.joint_names], -1)
    return fk(chain, theta.reshape(-1, chain.n))[:, :3, 3]


def separation(g: Gripper, q: dict[str, Tensor], a: str, b: str) -> Tensor:
    return torch.linalg.norm(pad_position(g, q, a) - pad_position(g, q, b), dim=-1)


def separation_bounds(g: Gripper, a: str, b: str, samples: int = 20000,
                      seed: int = 0) -> tuple[float, float]:
    """[min, max] pad separation over the joint box.

    Exact for a parallel jaw and a SUPERSET for a linkage, which is the useful
    direction: anything outside this interval is unreachable for certain.
    Corners are included explicitly because an affine separation attains its
    extremes there and random sampling would miss them.
    """
    ca, cb = g.fingers[a], g.fingers[b]
    joints = list(dict.fromkeys(ca.joint_names + cb.joint_names))
    lo, hi = {}, {}
    for chain in (ca, cb):
        for j, (l, h) in zip(chain.joint_names, chain.limits, strict=True):
            lo[j], hi[j] = float(l), float(h)

    gen = torch.Generator().manual_seed(seed)
    u = torch.rand(samples, len(joints), generator=gen)
    corners = torch.tensor([[float(bit) for bit in f"{i:0{len(joints)}b}"]
                            for i in range(min(2 ** len(joints), 4096))])
    u = torch.cat([u, corners])
    q = {j: torch.tensor(lo[j]) + u[:, k] * (hi[j] - lo[j])
         for k, j in enumerate(joints)}
    s = separation(g, q, a, b)
    return float(s.min()), float(s.max())


# ---- pad surfaces, measured from the live model -------------------------------
#
# Body origins are not pad surfaces, and the difference is the whole question
# when asking whether a feature fits between the jaws. A bounding sphere is far
# too loose: geom_rbound reads the PandaGripper's ~79 mm usable span as ~32 mm.
# What is needed is the SUPPORT FUNCTION of each geom along the closing axis --
# how far the solid actually reaches in that direction.

_BOX, _SPHERE, _CAPSULE, _CYLINDER, _MESH = 6, 2, 3, 5, 7


def geom_support(model, data, gid: int, axis) -> float:
    """Furthest extent of geom `gid` along world unit `axis`: c.a + h(a).

    h is the support half-extent, computed per primitive rather than bounded:
      box       |R^T a| . size          (size is half-extents)
      sphere    r
      capsule   r + |a_z| hz            (spherical caps reach r in every direction)
      cylinder  ||a_xy|| r + |a_z| hz
      mesh      exact, from the vertices
    """
    import numpy as np
    c = data.geom_xpos[gid]
    R = data.geom_xmat[gid].reshape(3, 3)
    a = R.T @ np.asarray(axis)
    s = model.geom_size[gid]
    t = int(model.geom_type[gid])
    if t == _BOX:
        h = float(np.abs(a) @ s)
    elif t == _SPHERE:
        h = float(s[0])
    elif t == _CAPSULE:
        h = float(s[0] + abs(a[2]) * s[1])
    elif t == _CYLINDER:
        h = float(np.linalg.norm(a[:2]) * s[0] + abs(a[2]) * s[1])
    elif t == _MESH:
        mid = int(model.geom_dataid[gid])
        start = int(model.mesh_vertadr[mid])
        num = int(model.mesh_vertnum[mid])
        v = np.asarray(model.mesh_vert[start:start + num]).reshape(-1, 3)
        h = float((v @ a).max())
    else:                                   # unknown primitive: be explicit
        raise NotImplementedError(f"no support function for geom type {t}")
    return float(c @ np.asarray(axis)) + h


def pad_gap(model, data, body_a: str, body_b: str, collision_group: int = 0) -> float:
    """Signed gap between the INNER surfaces of two pads, in metres.

    Positive is a clear opening; negative means the pads overlap along the
    closing axis, which is what a fully-closed jaw reports. Only collision geoms
    count -- robosuite puts visual meshes in group 1, and a visual shell is
    usually larger than the solid that actually blocks an object.
    """
    import numpy as np
    ia, ib = model.body_name2id(body_a), model.body_name2id(body_b)
    d = data.xpos[ib] - data.xpos[ia]
    n = float(np.linalg.norm(d))
    if n < _COINCIDENT:
        raise ValueError(f"{body_a} and {body_b} are coincident; no closing axis")
    axis = d / n                                    # points from a toward b

    ga = [g for g in range(model.ngeom)
          if model.geom_bodyid[g] == ia and model.geom_group[g] == collision_group]
    gb = [g for g in range(model.ngeom)
          if model.geom_bodyid[g] == ib and model.geom_group[g] == collision_group]
    if not ga or not gb:
        raise ValueError(f"no group-{collision_group} geoms on {body_a}/{body_b}")

    # a's furthest reach toward b, and b's furthest reach toward a
    inner_a = max(geom_support(model, data, g, axis) for g in ga)
    inner_b = -max(geom_support(model, data, g, -axis) for g in gb)
    return inner_b - inner_a
