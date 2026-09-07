"""Screw geometry on SE(3), in the Modern Robotics convention.

A screw axis is S = (w, v) with the angular part FIRST -- ch.3 sec.4.7. Every
matrix here is torch and differentiable; float64 is the default because the
conversion test asserts agreement with a simulator to 1e-6 and float32 cannot
carry that over a seven-factor product.
"""
from __future__ import annotations

import torch
from torch import Tensor


def hat3(w: Tensor) -> Tensor:
    """(..., 3) -> (..., 3, 3) skew-symmetric [w]."""
    z = torch.zeros_like(w[..., 0])
    return torch.stack([
        torch.stack([z, -w[..., 2], w[..., 1]], -1),
        torch.stack([w[..., 2], z, -w[..., 0]], -1),
        torch.stack([-w[..., 1], w[..., 0], z], -1),
    ], -2)


def exp_so3(w: Tensor, theta: Tensor) -> Tensor:
    """Rodrigues. w is a unit axis (..., 3); theta is (...,). -> (..., 3, 3)."""
    W = hat3(w)
    I = torch.eye(3, dtype=w.dtype, device=w.device).expand(W.shape)
    s = torch.sin(theta)[..., None, None]
    c = torch.cos(theta)[..., None, None]
    return I + s * W + (1 - c) * (W @ W)


def exp_se3(S: Tensor, theta: Tensor) -> Tensor:
    """exp([S] theta) for a screw axis S = (w, v). -> (..., 4, 4).

    Handles the prismatic case (w == 0) exactly rather than by a small-angle
    approximation: there the motion is a pure translation v * theta.
    """
    w, v = S[..., :3], S[..., 3:]
    wn = torch.linalg.norm(w, dim=-1)
    revolute = wn > 1e-12

    T = torch.eye(4, dtype=S.dtype, device=S.device).expand(*S.shape[:-1], 4, 4).clone()

    # Prismatic: R = I, p = v * theta.
    T[..., :3, 3] = v * theta[..., None]

    if revolute.any():
        wr = w[revolute]
        vr = v[revolute]
        tr = theta[revolute]
        R = exp_so3(wr, tr)
        W = hat3(wr)
        I = torch.eye(3, dtype=S.dtype, device=S.device).expand(W.shape)
        # G(theta) = I*theta + (1-cos)*[w] + (theta-sin)*[w]^2   (ch.3 sec.4.7)
        G = (tr[..., None, None] * I
             + (1 - torch.cos(tr))[..., None, None] * W
             + (tr - torch.sin(tr))[..., None, None] * (W @ W))
        T[revolute, :3, :3] = R
        T[revolute, :3, 3] = (G @ vr[..., None])[..., 0]
    return T


def adjoint(T: Tensor) -> Tensor:
    """Ad_T as a 6x6 acting on (w, v). -> (..., 6, 6)."""
    R, p = T[..., :3, :3], T[..., :3, 3]
    out = torch.zeros(*T.shape[:-2], 6, 6, dtype=T.dtype, device=T.device)
    out[..., :3, :3] = R
    out[..., 3:, 3:] = R
    out[..., 3:, :3] = hat3(p) @ R
    return out


def inverse(T: Tensor) -> Tensor:
    """Rigid inverse -- transpose the rotation, never a general solve."""
    R, p = T[..., :3, :3], T[..., :3, 3]
    Rt = R.transpose(-1, -2)
    out = torch.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -(Rt @ p[..., None])[..., 0]
    out[..., 3, 3] = 1
    return out


def log_se3(T: Tensor) -> Tensor:
    """Matrix log -> the twist V = (w, v) with |w| = the rotation angle.

    This is the only geometrically consistent pose error on SE(3): you cannot
    subtract two poses (ch.6 sec.4.1).
    """
    R, p = T[..., :3, :3], T[..., :3, 3]
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_t = torch.clamp((tr - 1) / 2, -1.0, 1.0)
    theta = torch.arccos(cos_t)

    small = theta < 1e-9
    st = torch.where(small, torch.ones_like(theta), torch.sin(theta))
    W = (R - R.transpose(-1, -2)) / (2 * st)[..., None, None]
    w_hat = torch.stack([W[..., 2, 1], W[..., 0, 2], W[..., 1, 0]], -1)
    w = w_hat * theta[..., None]

    I = torch.eye(3, dtype=T.dtype, device=T.device).expand(R.shape)
    th = torch.where(small, torch.ones_like(theta), theta)[..., None, None]
    # G^-1(theta) = I/theta - [w]/2 + (1/theta - cot(theta/2)/2) [w]^2, with
    # [w] the UNIT skew -- no theta factors on W here, that is the classic slip.
    Gi = I / th - W / 2 + (1 / th - 1 / (2 * torch.tan(th / 2))) * (W @ W)
    # S is a unit screw; the twist is S * theta, so v scales too.
    v = (Gi @ p[..., None])[..., 0] * theta[..., None]
    v = torch.where(small[..., None], p, v)
    w = torch.where(small[..., None], torch.zeros_like(w), w)
    return torch.cat([w, v], -1)
