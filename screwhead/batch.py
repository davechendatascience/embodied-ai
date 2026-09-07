"""Kinematics over a batch of DIFFERENT arms.

Training on mixed embodiments needs one forward pass covering arms with
different joint counts. Padding is safe here for a reason worth stating: a
padded joint carries a zero screw axis AND a zero joint value, and
exp([0] * 0) is exactly the identity, so a pad contributes nothing to the
product rather than contributing a small wrong transform. The mask still
travels alongside, because attention and the null-space projector both need to
know which columns are real -- a zero Jacobian column is trivially in the null
space, and an unmasked secondary objective would happily drive a joint that
does not exist.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .poe import Chain
from .se3 import adjoint, exp_se3, inverse


@dataclass
class BatchedChains:
    S: Tensor          # (B, max_n, 6)
    M: Tensor          # (B, 4, 4)
    limits: Tensor     # (B, max_n, 2)
    mask: Tensor       # (B, max_n) bool
    names: list[str]

    @property
    def max_n(self) -> int:
        return self.S.shape[1]


def stack(chains: list[Chain]) -> BatchedChains:
    max_n = max(c.n for c in chains)
    B = len(chains)
    dtype = chains[0].S.dtype
    S = torch.zeros(B, max_n, 6, dtype=dtype)
    M = torch.zeros(B, 4, 4, dtype=dtype)
    limits = torch.zeros(B, max_n, 2, dtype=dtype)
    mask = torch.zeros(B, max_n, dtype=torch.bool)
    for i, c in enumerate(chains):
        S[i, : c.n] = c.S
        M[i] = c.M
        limits[i, : c.n] = c.limits
        mask[i, : c.n] = True
    return BatchedChains(S=S, M=M, limits=limits, mask=mask,
                         names=[c.name for c in chains])


def fk_b(bc: BatchedChains, theta: Tensor) -> Tensor:
    """(B, max_n) -> (B, 4, 4). Padded joints are forced to zero, not trusted."""
    theta = theta * bc.mask.to(theta.dtype)
    T = torch.eye(4, dtype=theta.dtype, device=theta.device).expand(theta.shape[0], 4, 4).clone()
    for i in range(bc.max_n):
        T = T @ exp_se3(bc.S[:, i].to(theta.dtype), theta[:, i])
    return T @ bc.M.to(theta.dtype)


def body_jacobian_b(bc: BatchedChains, theta: Tensor) -> Tensor:
    """(B, max_n) -> (B, 6, max_n) in the tool frame; padded columns are zero."""
    theta = theta * bc.mask.to(theta.dtype)
    cols = []
    T = torch.eye(4, dtype=theta.dtype, device=theta.device).expand(theta.shape[0], 4, 4).clone()
    for i in range(bc.max_n):
        cols.append((adjoint(T) @ bc.S[:, i].to(theta.dtype)[..., None])[..., 0])
        T = T @ exp_se3(bc.S[:, i].to(theta.dtype), theta[:, i])
    Js = torch.stack(cols, dim=-1)
    Jb = adjoint(inverse(T @ bc.M.to(theta.dtype))) @ Js
    return Jb * bc.mask[:, None, :].to(Jb.dtype)


def decode_twist_b(
    bc: BatchedChains,
    theta: Tensor,
    twist: Tensor,
    dt: float = 1.0,
    lam: float = 0.05,
    secondary: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Batched one-step decode. Returns (theta_next, residual).

    This is the layer that sits inside the training graph: one linear solve,
    differentiable, bounded by ||e|| / (2 lambda), no data-dependent iteration.
    """
    J = body_jacobian_b(bc, theta)
    e = twist * dt
    A = J @ J.transpose(-1, -2) + (lam ** 2) * torch.eye(6, dtype=J.dtype, device=J.device)
    delta = (J.transpose(-1, -2) @ torch.linalg.solve(A, e[..., None]))[..., 0]

    if secondary is not None:
        Jp = torch.linalg.pinv(J)
        N = torch.eye(bc.max_n, dtype=J.dtype, device=J.device) - Jp @ J
        delta = delta + (N @ (secondary * bc.mask.to(J.dtype))[..., None])[..., 0]

    delta = delta * bc.mask.to(delta.dtype)
    lo, hi = bc.limits[..., 0].to(theta.dtype), bc.limits[..., 1].to(theta.dtype)
    nxt = torch.where(bc.mask, torch.clamp(theta + delta, lo, hi), theta)
    applied = nxt - theta
    return nxt, e - (J @ applied[..., None])[..., 0]
