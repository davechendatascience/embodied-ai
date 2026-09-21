"""The same kinematics as kinematics.py / ik.py / se3.py, in NumPy, for the control loop.

Profiled on one teacher episode: MuJoCo physics was 4% of the time and rendering 1%;
three-quarters went to PyTorch's per-op overhead on 4x4 and 6x7 matrices -- one forward
kinematics call on a seven-joint chain cost 3.9 ms. Tensors earn their keep in the
training graph, where the decoder must be differentiable and batched; a servo solving
one configuration twenty times a second needs neither, and a GPU would only add launch
latency to every one of those tiny ops.

This is a transcription, not a re-derivation: each function mirrors its torch original
line for line (same damping, same limit-clamping loop, same near-pi branch of the log),
and tests/test_kin_np.py holds the two to agreement at float64 precision. The torch
versions stay the reference; this is the fast path through the same maths.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np

I3 = np.eye(3)
# the same guards as se3.py, named: tests/test_kin_np.py holds the two to agreement
EPS_AXIS = 1e-12          # |w| below this is a prismatic joint / a pure translation
EPS_ANGLE = 1e-9          # a rotation smaller than this is the identity (log)
EPS_NEAR_PI = 2e-8        # sin(theta) below this with cos < 0 is a half turn
SERIES_BELOW = 1e-3       # use the series for the log's coefficient below this angle
EPS_DIV = 1e-12           # floor for a norm that is divided by


def hat3(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, float)
    out = np.zeros(w.shape[:-1] + (3, 3))
    out[..., 0, 1], out[..., 0, 2] = -w[..., 2], w[..., 1]
    out[..., 1, 0], out[..., 1, 2] = w[..., 2], -w[..., 0]
    out[..., 2, 0], out[..., 2, 1] = -w[..., 1], w[..., 0]
    return out


def inverse(T: np.ndarray) -> np.ndarray:
    R, p = T[..., :3, :3], T[..., :3, 3]
    Rt = np.swapaxes(R, -1, -2)
    out = np.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -(Rt @ p[..., None])[..., 0]
    out[..., 3, 3] = 1.0
    return out


def adjoint(T: np.ndarray) -> np.ndarray:
    R, p = T[..., :3, :3], T[..., :3, 3]
    out = np.zeros(T.shape[:-2] + (6, 6))
    out[..., :3, :3] = R
    out[..., 3:, 3:] = R
    out[..., 3:, :3] = hat3(p) @ R
    return out


def exp_se3(S: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """exp([S] theta), S = (w, v); prismatic (w == 0) handled exactly, as in se3.py."""
    S = np.asarray(S, float)
    theta = np.asarray(theta, float)
    w, v = S[..., :3], S[..., 3:]
    T = np.broadcast_to(np.eye(4), S.shape[:-1] + (4, 4)).copy()
    T[..., :3, 3] = v * theta[..., None]
    rev = np.linalg.norm(w, axis=-1) > EPS_AXIS
    if np.any(rev):
        W = hat3(w[rev])
        WW = W @ W
        t = theta[rev][..., None, None]
        s, c = np.sin(t), np.cos(t)
        T[rev, :3, :3] = I3 + s * W + (1 - c) * WW
        G = t * I3 + (1 - c) * W + (t - s) * WW
        T[rev, :3, 3] = (G @ v[rev][..., None])[..., 0]
    return T


def exp_twist(V: np.ndarray) -> np.ndarray:
    V = np.asarray(V, float)
    theta = np.linalg.norm(V[..., :3], axis=-1)
    pure = theta < EPS_AXIS
    theta = np.where(pure, np.linalg.norm(V[..., 3:], axis=-1), theta)
    safe = np.maximum(theta, EPS_DIV)
    return exp_se3(V / safe[..., None], theta)


def log_se3(T: np.ndarray) -> np.ndarray:
    """Matrix log -> (w, v), mirroring se3.log_se3 including its near-pi branch."""
    R, p = T[..., :3, :3], T[..., :3, 3]
    dual = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0],
                     R[..., 1, 0] - R[..., 0, 1]], -1) / 2
    sin_t = np.linalg.norm(dual, axis=-1)
    cos_t = (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2
    theta = np.arctan2(sin_t, cos_t)
    small = theta < EPS_ANGLE
    near_pi = (sin_t < EPS_NEAR_PI) & (cos_t < 0)

    axis_gen = dual / np.maximum(sin_t, EPS_DIV)[..., None]
    A = R + I3
    col = np.argmax(np.diagonal(A, axis1=-2, axis2=-1), axis=-1)
    axis_pi = np.take_along_axis(A, col[..., None, None].repeat(3, -2), -1)[..., 0]
    axis_pi = axis_pi / np.maximum(np.linalg.norm(axis_pi, axis=-1, keepdims=True), EPS_DIV)
    sign = np.sign(np.sum(axis_pi * dual, axis=-1))
    sign = np.where(sign == 0, 1.0, sign)
    axis_pi = axis_pi * sign[..., None]
    axis = np.where((near_pi & ~small)[..., None], axis_pi, axis_gen)
    W = hat3(axis)
    w = axis * theta[..., None]

    t_s = np.where(small, 1.0, theta)
    with np.errstate(divide="ignore", invalid="ignore"):
        coef_exact = 1 / t_s - 1 / (2 * np.tan(t_s / 2))
    coef = np.where(t_s < SERIES_BELOW, t_s / 12 + t_s ** 3 / 720, coef_exact)[..., None, None]
    th = t_s[..., None, None]
    Gi = I3 / th - W / 2 + coef * (W @ W)
    v = (Gi @ p[..., None])[..., 0] * theta[..., None]
    v = np.where(small[..., None], p, v)
    w = np.where(small[..., None], 0.0, w)
    return np.concatenate([w, v], -1)


@dataclass
class NpChain:
    """The (M, {S_i}, limits) of a Chain, as arrays."""
    S: np.ndarray           # (n, 6)
    M: np.ndarray           # (4, 4)
    limits: np.ndarray      # (n, 2)

    @classmethod
    def of(cls, chain) -> NpChain:
        c = getattr(chain, "_np", None)
        if c is None:
            c = cls(S=chain.S.detach().cpu().numpy().astype(float),
                    M=chain.M.detach().cpu().numpy().astype(float),
                    limits=chain.limits.detach().cpu().numpy().astype(float))
            with contextlib.suppress(AttributeError):     # a frozen Chain: just recompute
                chain._np = c
        return c

    @property
    def n(self) -> int:
        return len(self.S)


def fk_jac(c: NpChain, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(B, n) -> tool pose (B, 4, 4) and BODY Jacobian (B, 6, n), in one pass."""
    theta = np.atleast_2d(np.asarray(theta, float))
    b, n = theta.shape
    T = np.broadcast_to(np.eye(4), (b, 4, 4)).copy()
    Js = np.empty((b, 6, n))
    for i in range(n):
        Js[:, :, i] = adjoint(T) @ c.S[i]
        T = T @ exp_se3(np.broadcast_to(c.S[i], (b, 6)), theta[:, i])
    T = T @ c.M
    return T, adjoint(inverse(T)) @ Js


def fk(c: NpChain, theta: np.ndarray) -> np.ndarray:
    theta = np.atleast_2d(np.asarray(theta, float))
    b, n = theta.shape
    T = np.broadcast_to(np.eye(4), (b, 4, 4)).copy()
    for i in range(n):
        T = T @ exp_se3(np.broadcast_to(c.S[i], (b, 6)), theta[:, i])
    return T @ c.M


def dls(J: np.ndarray, e: np.ndarray, lam: float) -> np.ndarray:
    m = J.shape[-2]
    A = J @ np.swapaxes(J, -1, -2) + (lam ** 2) * np.eye(m)
    return (np.swapaxes(J, -1, -2) @ np.linalg.solve(A, e[..., None]))[..., 0]


def nullspace_projector(J: np.ndarray) -> np.ndarray:
    return np.eye(J.shape[-1]) - np.linalg.pinv(J) @ J


def decode_twist(c: NpChain, theta: np.ndarray, twist: np.ndarray, dt: float = 1.0,
                 lam: float = 0.05, secondary: np.ndarray | None = None,
                 respect_limits: bool = True, J: np.ndarray | None = None):
    """Mirror of ik.decode_twist. Returns (theta, delta, clamped)."""
    theta = np.atleast_2d(np.asarray(theta, float))
    twist = np.atleast_2d(np.asarray(twist, float))
    if J is None:
        _, J = fk_jac(c, theta)
    e = twist * dt
    delta = dls(J, e, lam)
    if secondary is not None:
        delta = delta + (nullspace_projector(J) @ np.atleast_2d(secondary)[..., None])[..., 0]
    clamped = np.zeros(theta.shape[0], bool)
    if respect_limits:
        lo, hi = c.limits[:, 0], c.limits[:, 1]
        frozen = np.zeros_like(theta)
        free = np.ones_like(theta, bool)
        for _ in range(c.n):
            proposed = theta + frozen + delta * free
            over = ((proposed < lo) | (proposed > hi)) & free
            if not over.any():
                break
            clamped = clamped | over.any(-1)
            frozen = np.where(over, np.clip(proposed, lo, hi) - theta, frozen)
            free = free & ~over
            residual_e = e - (J @ frozen[..., None])[..., 0]
            delta = dls(J * free[:, None, :], residual_e, lam)
        delta = frozen + delta * free
        delta = np.clip(theta + delta, lo, hi) - theta
    return theta + delta, delta, clamped


def solve_ik(c: NpChain, target: np.ndarray, theta0: np.ndarray, lam: float = 0.05,
             max_iters: int = 60, tol: float = 1e-5, respect_limits: bool = True,
             trust: float = 0.2) -> dict:
    """Mirror of ik.solve_ik, batched: rows that converge stop moving."""
    target = np.asarray(target, float)
    target = target if target.ndim == 3 else target[None]
    theta = np.atleast_2d(np.asarray(theta0, float)).copy()
    if theta.shape[0] == 1 and target.shape[0] > 1:
        theta = np.repeat(theta, target.shape[0], 0)
    b = theta.shape[0]
    iters = np.zeros(b, int)
    active = np.ones(b, bool)
    for _ in range(max_iters):
        T, J = fk_jac(c, theta)
        err = log_se3(inverse(T) @ target)
        nerr = np.linalg.norm(err, axis=-1)
        active = active & ~(nerr < tol)
        if not active.any():
            break
        scale = np.minimum(trust / np.maximum(nerr, EPS_DIV), 1.0)
        new, _, _ = decode_twist(c, theta, err * scale[:, None], dt=1.0, lam=lam,
                                 respect_limits=respect_limits, J=J)
        theta = np.where(active[:, None], new, theta)
        iters = iters + active
    T = fk(c, theta)
    err = log_se3(inverse(T) @ target)
    nerr = np.linalg.norm(err, axis=-1)
    return {"theta": theta, "residual": err, "residual_norm": nerr,
            "pos_err": np.linalg.norm(target[:, :3, 3] - T[:, :3, 3], axis=-1),
            "converged": nerr < tol, "iters": iters}


def sigma_min(c: NpChain, theta: np.ndarray) -> np.ndarray:
    _, J = fk_jac(c, theta)
    return np.linalg.svd(J, compute_uv=False)[..., -1]
