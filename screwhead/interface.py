"""Between the policy's action units and the decoder's twist units.

This is the piece that is not robotics and is therefore easy to leave implicit.
Three things live here, and each has a failure mode that looks like a modelling
problem when it goes wrong:

  RATE vs DISPLACEMENT. A twist is m/s and rad/s. A dataset action is a
  displacement per control step. They differ by the control period, and
  conflating them rescales every action by the control frequency.

  FRAME. robosuite's OSC_POSE delta is world-framed and DECOUPLED:
      R_new = R_delta @ R          (pre-multiply: a world-frame rotation)
      p_new = p + p_delta          (added in world, not rotated)
  That is not a single SE(3) group element pre-multiplying the pose -- a real
  pre-multiplication would give p_new = R_delta @ p + p_delta. Treating it as
  one silently corrupts every frame where the tool is far from the origin.

  SCALE. robosuite maps a [-1, 1] action onto +/-0.05 m and +/-0.5 rad. The two
  are scaled SEPARATELY, at a 10:1 ratio. One shared scale would swamp rotation.
  These constants are properties of the action space, not of the arm, which is
  what keeps normalisation embodiment-independent.

A useful consequence: because the action space is bounded, the twist it can
express is bounded too -- at 20 Hz, +/-1.0 m/s and +/-10 rad/s. That turns
"twist magnitude is reachable within one control period" from an assumption the
decoder makes into something the producer can actually guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .se3 import exp_twist, inverse, log_se3


@dataclass(frozen=True)
class ActionSpec:
    """Defaults are robosuite's osc_pose.json, which is what the demos live in."""
    control_hz: float = 20.0
    pos_scale: float = 0.05          # metres per unit action
    rot_scale: float = 0.5           # radians per unit action

    @property
    def dt(self) -> float:
        return 1.0 / self.control_hz

    @property
    def max_linear_speed(self) -> float:
        return self.pos_scale * self.control_hz

    @property
    def max_angular_speed(self) -> float:
        return self.rot_scale * self.control_hz


def compose_delta(T: Tensor, delta: Tensor) -> Tensor:
    """Apply a robosuite-style decoupled world delta to a pose. (B,4,4),(B,6)->(B,4,4)."""
    p_d, w_d = delta[..., :3], delta[..., 3:]
    R_d = exp_twist(torch.cat([w_d, torch.zeros_like(w_d)], dim=-1))[..., :3, :3]
    out = T.clone()
    out[..., :3, :3] = R_d @ T[..., :3, :3]
    out[..., :3, 3] = T[..., :3, 3] + p_d
    return out


def delta_to_twist(T: Tensor, delta: Tensor, spec: ActionSpec) -> Tensor:
    """Dataset action (unnormalised, world, decoupled) -> body twist RATE."""
    return log_se3(inverse(T) @ compose_delta(T, delta)) / spec.dt


def twist_to_delta(T: Tensor, twist: Tensor, spec: ActionSpec) -> Tensor:
    """Body twist RATE -> dataset action (unnormalised, world, decoupled)."""
    T_new = T @ exp_twist(twist * spec.dt)
    R_d = T_new[..., :3, :3] @ T[..., :3, :3].transpose(-1, -2)
    p_d = T_new[..., :3, 3] - T[..., :3, 3]
    return torch.cat([p_d, _rot_log(R_d)], dim=-1)


def _rot_log(R: Tensor) -> Tensor:
    """Axis-angle of a rotation, via the SE(3) log with zero translation."""
    T = torch.zeros(*R.shape[:-2], 4, 4, dtype=R.dtype, device=R.device)
    T[..., :3, :3] = R
    T[..., 3, 3] = 1
    return log_se3(T)[..., :3]


def normalize(delta: Tensor, spec: ActionSpec) -> Tensor:
    """Unnormalised world delta -> the [-1, 1] action the policy emits.

    Translation and rotation are scaled separately and by constants that belong
    to the action space, not to any robot -- normalising per embodiment would
    reintroduce the per-robot coupling this whole design exists to remove.
    """
    scale = torch.tensor([spec.pos_scale] * 3 + [spec.rot_scale] * 3,
                         dtype=delta.dtype, device=delta.device)
    return delta / scale


def denormalize(action: Tensor, spec: ActionSpec) -> Tensor:
    scale = torch.tensor([spec.pos_scale] * 3 + [spec.rot_scale] * 3,
                         dtype=action.dtype, device=action.device)
    return action * scale
