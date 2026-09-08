"""Proprioception the policy can read on any arm.

Raw joint angles are embodiment-specific by construction. Trained on a Panda,
j5 spans [+1.02, +3.44]; a UR5e sits at -1.991 on that slot, so the state vector
is out of distribution before the action representation is even consulted. The
policy then fails the swap for a reason that has nothing to do with what it
emits -- which is the confound the whole factorial exists to avoid.

Tool pose is the embodiment-free alternative, and it is the same quantity the
action is expressed in: the policy sees where the tool IS and says where it
should GO, both in the task space. What the arm's joints are doing is the
decoder's problem, and the spec tokens carry what geometry does not.

Rotation is the 6D continuous representation -- the first two columns of R.
Euler angles and quaternions are discontinuous as functions of rotation (a
quaternion double-covers, so q and -q are the same pose with opposite
coordinates), and a network regressing a discontinuous target has to spend
capacity papering over the seam.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .kinematics import fk
from .poe import Chain

STATE_DIM = 10          # 3 position + 6 rotation + 1 gripper aperture


def tool_state(chain: Chain, theta: Tensor, gripper: Tensor | None = None) -> Tensor:
    """(B, n) joint values -> (B, 10) embodiment-free proprioception."""
    T = fk(chain, torch.atleast_2d(theta))
    p = T[:, :3, 3]
    r6 = T[:, :3, :2].reshape(-1, 6)        # first two columns of R
    if gripper is None:
        gripper = torch.zeros(len(p), 1, dtype=p.dtype, device=p.device)
    else:
        gripper = torch.as_tensor(gripper, dtype=p.dtype, device=p.device).reshape(-1, 1)
    return torch.cat([p, r6, gripper], dim=-1)
