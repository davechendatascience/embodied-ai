"""Task progress as distance on SE(3) to geometric waypoints.

A task is a chain of target configurations -- where the tool must be, then where
the object must be -- each defined by the scene's own geometry, never by a
demonstration. Progress is

    phi(s) = stage(s) + (1 - clip(d(s, waypoint_stage) / d_ref_stage, 0, 1))

where d is the SE(3) distance from the log map, ||p - p*|| + rho * ||log(R^T R*)||,
and the stage is a PURE FUNCTION OF THE CURRENT STATE (is the bowl held, lifted,
over the plate) -- no history -- so gamma*phi(s') - phi(s) is potential-based
shaping and cannot change which policy is optimal (Ng, Harada & Russell 1999).

Each d_ref is the distance between consecutive waypoints, so phi is continuous
across stage boundaries: arriving at a waypoint scores the same as starting the
next stage. A discontinuity would pay for crossing a boundary instead of for
motion.

Pick-and-place, bowl onto plate, from measured collision geometry:
  bowl wall   top 52.1 mm above the bowl origin, radius 50.8-53.9 mm
  plate       top surface 16.4 mm above the plate origin
  Panda pads  closing axis = tool y, pads at +-38.6..46.6 mm, z -11.6..+4.4 mm
so the grasp is a pinch across the wall: grip site at mid-wall radius, 12 mm below
the rim top (the whole pad on the wall), tool z down, closing axis radial.

Stages:
  0-1 reach  tool   -> grasp pose, penalised off-axis below the pre-grasp height
             (one continuous stage; the label is 1 once below the pre-grasp)
  2 close    aperture onto the wall + finger-side contact (closing on air scores half)
  3-5 transport  held bowl: remaining rise + cross + descend to resting on the plate
             (one continuous quantity; labels 3 lift / 4 carry / 5 place)
  6 success  (terminal)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PickPlaceGeometry:
    rim_radius: float = 0.0524       # mid-wall
    rim_top: float = 0.0521          # above bowl origin
    grasp_depth: float = 0.012       # grip site below rim top
    bowl_bottom: float = 0.0016
    plate_top: float = 0.0164
    open_aperture: float = 0.0784
    wall: float = 0.003
    approach: float = 0.08           # pre-grasp height above grasp
    lift: float = 0.08
    rho: float = 0.10                # metres per radian in the SE(3) distance
    grasp_ball: float = 0.012        # close stage: tool within this SE(3) distance of the grasp
    lateral_weight: float = 1.0      # off-axis penalty below the pre-grasp height, per metre
    over_plate_xy: float = 0.02
    lifted_dz: float = 0.02
    hold_min: float = 0.0025         # aperture pinching the wall: measured 3.8-15.6 mm
    hold_max: float = 0.020


def rot_angle(R: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))


def grasp_frames(p_tool: np.ndarray, p_bowl: np.ndarray, R_bowl: np.ndarray, g: PickPlaceGeometry):
    """Grasp and pre-grasp poses at the rim angle nearest the tool. Base frame."""
    z_b = R_bowl[:, 2]
    rel = p_tool - p_bowl
    rel_h = rel - z_b * (rel @ z_b)
    radial = rel_h / (np.linalg.norm(rel_h) + 1e-9) if np.linalg.norm(rel_h) > 1e-6 else R_bowl[:, 0]
    z_t = -z_b                                   # tool z toward the bowl base
    y_t = radial                                 # closing axis radial
    x_t = np.cross(y_t, z_t)
    R = np.stack([x_t, y_t, z_t], axis=1)
    p = p_bowl + radial * g.rim_radius + z_b * (g.rim_top - g.grasp_depth)
    pre = p + z_b * g.approach
    return R, p, pre


def tool_distance(R_tool, p_tool, R_goal, p_goal, g):
    """SE(3) distance with the gripper's 180-degree symmetry about its z axis."""
    flip = np.diag([-1.0, -1.0, 1.0])
    ang = min(rot_angle(R_tool.T @ R_goal), rot_angle(R_tool.T @ R_goal @ flip))
    return float(np.linalg.norm(p_tool - p_goal)) + g.rho * ang, ang


def bowl_distance(R_bowl, p_bowl, p_goal, g):
    """Position plus tilt from upright; yaw is free, the bowl is rotationally symmetric."""
    tilt = float(np.arccos(np.clip(R_bowl[2, 2], -1.0, 1.0)))
    return float(np.linalg.norm(p_bowl - p_goal)) + g.rho * tilt


def progress(s: dict, g: PickPlaceGeometry = PickPlaceGeometry()) -> tuple[float, int, dict]:
    """s: R_tool, p_tool, aperture, R_bowl, p_bowl, p_plate, rest_z, side1, side2 (a pad or
    finger of that side touches the bowl), any_grip (any gripper geom touches it),
    supported (the bowl touches something that is not the gripper), success,
    d0_reach, d0_carry. All base frame.

    HELD is defined physically, not by pad count: measured over 562 frames of carried
    bowls, both pads touch in only ~82% of frames while nothing but the gripper
    touches the bowl in 100%. So a bowl is held if both finger sides grip it with
    the gripper closed, or if it is raised and the gripper is its only support.
    """
    if s["success"]:
        return 6.0, 6, {}
    dz = s["p_bowl"][2] - s["rest_z"]
    # Measured on demonstrations: a bowl pinched across its 3 mm wall leaves the
    # gripper 3.8-15.6 mm open (p1-p99), well apart from open (78 mm) and from
    # closed on nothing (~0). Both pads touch in only ~82% of carried frames, and
    # nothing but the gripper touches a carried bowl in all of them.
    pinch = g.hold_min <= s["aperture"] <= g.hold_max
    # Contact flags flicker on the same grasp -- a bowl 99 mm in the air with the
    # gripper closed at 11.5 mm read zero contacts for a frame -- so geometry
    # decides: the wall lies between the pads. Contact is kept as a second route.
    held = pinch and (s["side1"] or s["side2"] or wall_between_pads(s, g)
                      or (not s["supported"] and dz > g.lifted_dz))
    if held:
        rem, label = transport_remaining(s, g)
        return 3 + 3 * (1 - min(rem / max(s["d0_carry"], 1e-3), 1.0)), label, dict(d=rem)
    d_reach, d_grasp, below = reach_distance(s, g)
    if d_grasp < g.grasp_ball:
        close = 1 - np.clip((s["aperture"] - g.wall) / (g.open_aperture - g.wall), 0, 1)
        return 2 + 0.5 * close + 0.25 * (int(s["side1"]) + int(s["side2"])), 2, dict(d=d_grasp)
    # Reach and descend are ONE continuous stage: a hard "column" boundary
    # flickered on every real approach (demonstrated grasps land 28-73 mm from the
    # bowl axis). Distance to the grasp pose, plus a penalty for being off the
    # grasp axis once below the pre-grasp height, so descending onto the rim from
    # above scores better than arriving sideways through the wall.
    v = 2 * (1 - min(max(d_reach - g.grasp_ball, 0.0) / max(s["d0_reach"] - g.grasp_ball, 1e-3), 1.0))
    return v, (1 if below else 0), dict(d=d_reach)


def wall_between_pads(s: dict, g: PickPlaceGeometry) -> bool:
    """The bowl wall sits inside the closed jaws, from poses alone.

    Tool in the bowl frame: its grip site within half an aperture (plus the wall
    and 2 mm) of the rim radius, its closing axis within 35 degrees of radial, and
    its pads -- which span 11.6 mm above to 4.4 mm below the grip site along the
    tool axis -- overlapping the top 20 mm of the wall, where the radius is the
    rim radius.
    """
    Rb = s["R_bowl"]
    rel = Rb.T @ (s["p_tool"] - s["p_bowl"])
    r = float(np.hypot(rel[0], rel[1]))
    if r < 1e-6:
        return False
    radial = np.array([rel[0] / r, rel[1] / r, 0.0])
    y_t = Rb.T @ s["R_tool"][:, 1]
    aligned = abs(float(y_t @ radial)) >= np.cos(np.deg2rad(35.0)) * np.linalg.norm(y_t[:2])
    radial_ok = abs(r - g.rim_radius) <= s["aperture"] / 2 + g.wall / 2 + 0.002
    height_ok = g.rim_top - 0.020 <= rel[2] <= g.rim_top + 0.0044
    return bool(aligned and radial_ok and height_ok)


def transport_remaining(s: dict, g: PickPlaceGeometry) -> tuple[float, int]:
    """Remaining path for a HELD bowl: rise to carry height, cross to the plate, descend.

    One continuous quantity for lift, carry and place. Hard boundaries between
    them flickered on real carries, which ran anywhere from 41 to 162 mm high
    (p10-p90). Carrying higher than the carry height costs nothing. Near the plate
    the far-field path cross-fades into straight descent over one tolerance
    width, so there is no edge anywhere.
    """
    h = float(np.linalg.norm((s["p_bowl"] - s["p_plate"])[:2]))
    z = float(s["p_bowl"][2])
    z_goal = float(s["p_plate"][2] + g.plate_top - g.bowl_bottom)
    z_carry = max(s["rest_z"], z_goal) + g.lift
    tol = g.over_plate_xy
    w = float(np.clip((h - tol) / tol, 0.0, 1.0))
    far = max(z_carry - z, 0.0) + max(h - tol, 0.0) + (z_carry - z_goal)
    near = abs(z - z_goal) + h
    tilt = float(np.arccos(np.clip(s["R_bowl"][2, 2], -1.0, 1.0)))
    rem = (1 - w) * near + w * far + g.rho * tilt
    label = 5 if w == 0.0 else (3 if z < z_carry - 0.01 else 4)
    return rem, label


def reach_distance(s: dict, g: PickPlaceGeometry) -> tuple[float, float, bool]:
    R_g, p_g, p_pre = grasp_frames(s["p_tool"], s["p_bowl"], s["R_bowl"], g)
    d_grasp, _ = tool_distance(s["R_tool"], s["p_tool"], R_g, p_g, g)
    z_b = s["R_bowl"][:, 2]
    below = float((s["p_tool"] - p_pre) @ z_b) < 0.0
    off = s["p_tool"] - p_g
    lateral = float(np.linalg.norm(off - z_b * (off @ z_b)))
    penalty = g.lateral_weight * max(lateral - g.grasp_ball, 0.0) if below else 0.0
    return d_grasp + penalty, d_grasp, below
