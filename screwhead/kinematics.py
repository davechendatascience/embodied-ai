"""Forward kinematics and the Jacobian, as products of exponentials.

Nothing here is learned and nothing is fitted. Given (M, {S_i}) the pose and the
Jacobian are determined, and column i of the Jacobian is joint i's screw axis
carried by the Adjoint of everything upstream (ch.5 sec.4.2).
"""
from __future__ import annotations

import torch
from torch import Tensor

from .poe import Chain
from .se3 import adjoint, exp_se3, inverse


def fk(chain: Chain, theta: Tensor) -> Tensor:
    """(B, n) joint values -> (B, 4, 4) tool pose. T = e^[S1]t1 ... e^[Sn]tn M."""
    theta = torch.atleast_2d(theta)
    b, n = theta.shape
    assert n == chain.n, f"{chain.name} has {chain.n} joints, got {n}"

    S = chain.S.to(theta.dtype)
    T = torch.eye(4, dtype=theta.dtype, device=theta.device).expand(b, 4, 4).clone()
    for i in range(n):
        T = T @ exp_se3(S[i].expand(b, 6), theta[:, i])
    return T @ chain.M.to(theta.dtype)


def space_jacobian(chain: Chain, theta: Tensor) -> Tensor:
    """(B, n) -> (B, 6, n). Column i is S_i pushed forward by everything before it."""
    theta = torch.atleast_2d(theta)
    b, n = theta.shape
    S = chain.S.to(theta.dtype)

    cols = []
    T = torch.eye(4, dtype=theta.dtype, device=theta.device).expand(b, 4, 4).clone()
    for i in range(n):
        # Column i uses the product of joints 1..i-1 -- nothing downstream of
        # joint i affects it, because those links ride along.
        cols.append((adjoint(T) @ S[i].expand(b, 6)[..., None])[..., 0])
        T = T @ exp_se3(S[i].expand(b, 6), theta[:, i])
    return torch.stack(cols, dim=-1)


def body_jacobian(chain: Chain, theta: Tensor) -> Tensor:
    """(B, n) -> (B, 6, n) in the tool frame. J_b = Ad(T^-1) J_s."""
    T = fk(chain, theta)
    return adjoint(inverse(T)) @ space_jacobian(chain, theta)
