"""MJCF -> screw axes.

robosuite and LIBERO ship MJCF, not URDF, so this is the path that reaches the
simulator we actually roll out in. The recipe is the URDF one of ch.4 sec.4.5
with MuJoCo's frame conventions substituted:

  1. Chain body pos/quat from the root to get each body's pose at qpos = 0.
  2. The joint's axis in reference coordinates is R_body @ axis.
  3. A point on the axis is p_body + R_body @ joint_pos.
  4. Revolute: v = -w x q.  Prismatic: w = 0, v = the slide direction.
  5. Carry on to the tool body to get M.

Orientations MuJoCo accepts that are not implemented raise rather than default,
because a silently-wrong orientation is the failure mode this whole component
exists to prevent (FM-rpy-order).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from torch import Tensor

from .poe import Chain

_DEG = torch.pi / 180.0


def _quat_to_R(q: list[float]) -> Tensor:
    """MuJoCo quaternion order is (w, x, y, z)."""
    w, x, y, z = q
    n = (w * w + x * x + y * y + z * z) ** 0.5
    w, x, y, z = w / n, x / n, y / n, z / n
    return torch.tensor([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=torch.float64)


def _axis_angle_to_R(a: list[float], degrees: bool) -> Tensor:
    ax = torch.tensor(a[:3], dtype=torch.float64)
    ang = torch.tensor(a[3], dtype=torch.float64) * (_DEG if degrees else 1.0)
    ax = ax / torch.linalg.norm(ax)
    K = torch.tensor([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]], dtype=torch.float64)
    return torch.eye(3, dtype=torch.float64) + torch.sin(ang) * K + (1 - torch.cos(ang)) * (K @ K)


def _euler_to_R(e: list[float], seq: str, degrees: bool) -> Tensor:
    """MuJoCo eulerseq: lowercase axes are intrinsic, applied left to right."""
    ang = [v * (_DEG if degrees else 1.0) for v in e]
    R = torch.eye(3, dtype=torch.float64)
    for axis, a in zip(seq, ang):
        c, s = torch.cos(torch.tensor(a, dtype=torch.float64)), torch.sin(torch.tensor(a, dtype=torch.float64))
        if axis.lower() == "x":
            Ri = torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float64)
        elif axis.lower() == "y":
            Ri = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float64)
        elif axis.lower() == "z":
            Ri = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float64)
        else:
            raise ValueError(f"bad euler axis {axis!r}")
        R = R @ Ri if axis.islower() else Ri @ R
    return R


def _floats(s: str | None, default: list[float]) -> list[float]:
    return default if s is None else [float(x) for x in s.split()]


def _body_pose(body: ET.Element, seq: str, degrees: bool) -> tuple[Tensor, Tensor]:
    p = torch.tensor(_floats(body.get("pos"), [0.0, 0.0, 0.0]), dtype=torch.float64)
    present = [k for k in ("quat", "euler", "axisangle", "xyaxes", "zaxis") if body.get(k) is not None]
    if len(present) > 1:
        raise ValueError(f"body {body.get('name')!r} sets {present}; ambiguous orientation")
    if not present:
        return torch.eye(3, dtype=torch.float64), p
    key = present[0]
    if key == "quat":
        return _quat_to_R(_floats(body.get("quat"), [1, 0, 0, 0])), p
    if key == "euler":
        return _euler_to_R(_floats(body.get("euler"), [0, 0, 0]), seq, degrees), p
    if key == "axisangle":
        return _axis_angle_to_R(_floats(body.get("axisangle"), [0, 0, 1, 0]), degrees), p
    raise NotImplementedError(
        f"body {body.get('name')!r} orients by {key!r}; implement it rather than guess"
    )


def _find_path(root: ET.Element, tool: str) -> list[ET.Element] | None:
    """Depth-first path of bodies from a worldbody child down to the tool body."""
    if root.get("name") == tool:
        return [root]
    for child in root.findall("body"):
        sub = _find_path(child, tool)
        if sub is not None:
            return [root] + sub
    return None


def from_mjcf(
    path: str | Path,
    tool_body: str = "right_hand",
    name: str | None = None,
    angle: str | None = None,
) -> Chain:
    """`angle` overrides the file's compiler setting.

    robosuite ships robot.xml as a *fragment*; the angle convention lives in
    base.xml (`<compiler angle="radian">`). Parsed standalone the file falls
    back to MuJoCo's default of degrees, which silently rescales every joint
    limit -- panda joint 1 becomes +/-0.05 rad instead of its real +/-2.8973
    (166 deg, the documented Franka limit). Geometry is unaffected because
    these files orient by quaternion, but limits are not.
    """
    tree = ET.parse(path)
    root = tree.getroot()

    compiler = root.find("compiler")
    # MuJoCo's default is degrees. Getting this wrong silently rescales every
    # euler angle and joint limit, so it is read, never assumed.
    degrees = True
    seq = "xyz"
    if compiler is not None:
        degrees = compiler.get("angle", "degree") == "degree"
        seq = compiler.get("eulerseq", "xyz")
    if angle is not None:
        if angle not in ("radian", "degree"):
            raise ValueError(f"angle must be 'radian' or 'degree', got {angle!r}")
        degrees = angle == "degree"

    world = root.find("worldbody")
    if world is None:
        raise ValueError(f"{path}: no <worldbody>")
    path_bodies = None
    for b in world.findall("body"):
        path_bodies = _find_path(b, tool_body)
        if path_bodies is not None:
            break
    if path_bodies is None:
        raise ValueError(f"{path}: no body named {tool_body!r}")

    R = torch.eye(3, dtype=torch.float64)
    p = torch.zeros(3, dtype=torch.float64)
    names, types, axes, limits = [], [], [], []

    for body in path_bodies:
        Rb, pb = _body_pose(body, seq, degrees)
        p = p + R @ pb
        R = R @ Rb
        for joint in body.findall("joint"):
            jtype = joint.get("type", "hinge")
            if jtype in ("free", "ball"):
                raise NotImplementedError(f"joint {joint.get('name')!r} is {jtype}; not a 1-DoF chain")
            axis = torch.tensor(_floats(joint.get("axis"), [0, 0, 1]), dtype=torch.float64)
            jpos = torch.tensor(_floats(joint.get("pos"), [0, 0, 0]), dtype=torch.float64)
            w_ref = R @ axis
            w_ref = w_ref / torch.linalg.norm(w_ref)
            q_ref = p + R @ jpos
            if jtype == "slide":
                S = torch.cat([torch.zeros(3, dtype=torch.float64), w_ref])
                types.append("prismatic")
            else:
                S = torch.cat([w_ref, -torch.linalg.cross(w_ref, q_ref)])
                types.append("revolute")
            names.append(joint.get("name") or f"joint{len(names) + 1}")
            axes.append(S)
            rng = _floats(joint.get("range"), [-torch.pi, torch.pi])
            scale = _DEG if (degrees and jtype != "slide") else 1.0
            limits.append([rng[0] * scale, rng[1] * scale])

    M = torch.eye(4, dtype=torch.float64)
    M[:3, :3] = R
    M[:3, 3] = p

    return Chain(
        name=name or root.get("model") or Path(path).stem,
        joint_names=names,
        joint_types=types,
        S=torch.stack(axes),
        M=M,
        limits=torch.tensor(limits, dtype=torch.float64),
        base_frame=path_bodies[0].get("name") or "root",
        tool_frame=tool_body,
        source=str(path),
    )
