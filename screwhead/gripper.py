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

from .kinematics import fk
from .mjcf import from_mjcf, leaf_paths
from .poe import Chain


@dataclass
class Gripper:
    name: str
    fingers: dict[str, Chain]           # tip body name -> chain from the palm
    linkage: bool                       # coupled (soft tendon) vs independent jaws

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
    coupled = root.find("tendon") is not None or root.find("equality") is not None
    fingers = {}
    for tip in _tip_bodies(path):
        try:
            fingers[tip] = from_mjcf(path, tool_body=tip, name=f"{path.stem}:{tip}",
                                     angle="radian")
        except Exception:
            continue
    return Gripper(name=name or root.get("model") or path.stem,
                   fingers=fingers, linkage=coupled)


def pad_position(g: Gripper, q: dict[str, Tensor], tip: str) -> Tensor:
    """Tip origin of ONE finger, in the palm frame, from OBSERVED joints.

    Per-finger on purpose: a gripper's branches do not share joints, so
    computing every branch would demand joint values the caller has no reason
    to supply (the Robotiq85's knuckle branches, for instance).
    """
    chain = g.fingers[tip]
    theta = torch.stack([torch.as_tensor(q[j]) for j in chain.joint_names], -1)
    return fk(chain, theta.reshape(-1, chain.n))[:, :3, 3]


def pad_positions(g: Gripper, q: dict[str, Tensor], tips=None) -> dict[str, Tensor]:
    return {t: pad_position(g, q, t) for t in (tips or g.tips)}


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
        for j, (l, h) in zip(chain.joint_names, chain.limits):
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
