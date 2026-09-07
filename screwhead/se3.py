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

    # theta from atan2, not arccos. arccos((tr-1)/2) is ill-conditioned where
    # its argument approaches +/-1: near theta = pi it returns only sqrt(eps)
    # accuracy, which caps the whole round-trip at ~1e-8 no matter how carefully
    # the axis is computed. atan2(sin, cos) is well conditioned everywhere.
    dual = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], -1) / 2
    sin_t = torch.linalg.norm(dual, dim=-1)
    cos_t = (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2
    theta = torch.atan2(sin_t, cos_t)

    small = theta < 1e-9
    # Near theta = pi the axis cannot come from `dual`, which vanishes there.
    # R + I = 2 a a^T instead: every column is parallel to the axis, so take the
    # best conditioned one. The sign of a is genuinely ambiguous at exactly pi
    # (a and -a give the same R) -- a property of SO(3), not of the code.
    # cos_t < 0 is essential: sin(theta) is small at theta ~ 0 too, and without
    # the cosine test a rotation of 1e-7 rad takes the pi branch and decodes
    # about an arbitrary axis.
    # Threshold at sqrt(2*eps) ~ 2e-8, where the two axis estimators cross.
    # R + I = 2 a a^T + delta*[a], so its axis error is O(delta); dual/sin has
    # error O(eps/delta). Switching at 1e-6 uses the R+I branch across three
    # decades where it is the WORSE of the two -- measured 3.6e-7 error at
    # delta = 9e-7, which is 0.4*delta exactly as that analysis predicts.
    near_pi = (sin_t < 2e-8) & (cos_t < 0)

    axis_gen = dual / torch.clamp(sin_t, min=1e-12)[..., None]
    I3 = torch.eye(3, dtype=T.dtype, device=T.device).expand(R.shape)
    A = R + I3
    col = torch.argmax(torch.diagonal(A, dim1=-2, dim2=-1), dim=-1)
    axis_pi = torch.gather(A, -1, col[..., None, None].expand(*A.shape[:-1], 1))[..., 0]
    axis_pi = axis_pi / torch.clamp(torch.linalg.norm(axis_pi, dim=-1, keepdim=True), min=1e-12)

    # Fix the sign against `dual`, which still carries orientation even when it
    # is small. Without this, a rotation of pi - 1e-6 can be decoded about the
    # OPPOSITE axis, giving an error of order 2*delta rather than machine eps.
    sign = torch.sign(torch.sum(axis_pi * dual, dim=-1))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    axis_pi = axis_pi * sign[..., None]
    axis = torch.where((near_pi & ~small)[..., None], axis_pi, axis_gen)
    W = hat3(axis)
    w = axis * theta[..., None]

    th = torch.where(small, torch.ones_like(theta), theta)[..., None, None]
    # G^-1(theta) = I/theta - [w]/2 + (1/theta - cot(theta/2)/2) [w]^2, with
    # [w] the UNIT skew -- no theta factors on W, that is the classic slip.
    # cot(theta/2) is finite at theta = pi, so this needs no guard of its own.
    # The W^2 coefficient is 1/theta - cot(theta/2)/2, a difference of two large
    # nearly equal numbers as theta -> 0: at theta = 1e-4 the operands are ~1e4
    # and the result ~8e-6, so nine digits cancel. Its series is theta/12 +
    # theta^3/720, which is exact to machine precision in that regime -- and
    # that regime is where retargeting lives (a 20 Hz step is ~0.01 rad).
    t_s = th.squeeze(-1).squeeze(-1)
    coef_series = t_s / 12 + t_s ** 3 / 720
    coef_exact = 1 / t_s - 1 / (2 * torch.tan(t_s / 2))
    coef = torch.where(t_s < 1e-3, coef_series, coef_exact)[..., None, None]
    Gi = I3 / th - W / 2 + coef * (W @ W)
    # S is a unit screw; the twist is S * theta, so v scales too.
    v = (Gi @ p[..., None])[..., 0] * theta[..., None]
    v = torch.where(small[..., None], p, v)
    w = torch.where(small[..., None], torch.zeros_like(w), w)
    return torch.cat([w, v], -1)


def exp_twist(V: Tensor) -> Tensor:
    """exp of a twist V = S*theta, recovering (S, theta) correctly.

    The scale factor is the norm of the ANGULAR part, not of the whole
    6-vector: a screw axis has ||w|| == 1, so theta = ||V[:3]||. Normalising by
    ||V|| instead yields a non-unit axis and the wrong angle, and the error only
    shows up once translation and rotation are both non-zero -- which is every
    real trajectory.
    """
    w = V[..., :3]
    theta = torch.linalg.norm(w, dim=-1)
    pure_translation = theta < 1e-12
    # A prismatic-only twist is scaled by its linear norm instead.
    lin = torch.linalg.norm(V[..., 3:], dim=-1)
    theta = torch.where(pure_translation, lin, theta)
    safe = torch.clamp(theta, min=1e-12)
    return exp_se3(V / safe[..., None], theta)
