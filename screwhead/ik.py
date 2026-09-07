"""The decoder: a tool twist in, joint motion out.

Nothing here is learned. Given the Jacobian -- which is determined by the URDF
(ch.5 sec.4.2) -- the map from a commanded twist to joint motion is a linear
solve. That is the whole point of the design: the learned part emits an
embodiment-free twist, and this decodes it for whatever arm is present.

Two entry points, for two different jobs:

  decode_twist   one damped step. Differentiable, bounded, cheap. This is the
                 layer that sits inside the training graph.
  solve_ik       iterate to a pose. Used for retargeting demonstrations and for
                 asking whether a target is reachable at all.

The damping is not a tuning knob bolted on. It comes from a cost function
(ch.6 sec.4.4) and it buys a hard guarantee: ||dtheta|| <= ||e|| / (2*lambda),
because the singular values of J^T (J J^T + l^2 I)^-1 are s/(s^2+l^2), which is
maximised at s = l. An unbounded step is what makes a plain pseudo-inverse
unsafe to unroll through a network near a singularity.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .kinematics import body_jacobian, fk
from .poe import Chain
from .se3 import inverse, log_se3


@dataclass
class Decoded:
    """What the decoder achieved, and what it did not."""
    theta: Tensor           # (B, n) joint values after the step
    delta: Tensor           # (B, n) the step actually applied, post-clamp
    residual: Tensor        # (B, 6) commanded twist minus achieved twist
    clamped: Tensor         # (B,) bool -- a joint limit truncated the step

    @property
    def residual_norm(self) -> Tensor:
        return torch.linalg.norm(self.residual, dim=-1)


def dls(J: Tensor, e: Tensor, lam: float) -> Tensor:
    """Damped least squares: argmin ||J d - e||^2 + lam^2 ||d||^2.

    Solved in the 6x6 form J^T (J J^T + l^2 I)^-1 e rather than the n x n form.
    They are algebraically identical; the 6x6 is cheaper for a redundant arm and
    its size does not grow with the joint count.
    """
    b, m, _ = J.shape
    A = J @ J.transpose(-1, -2) + (lam ** 2) * torch.eye(m, dtype=J.dtype, device=J.device)
    return (J.transpose(-1, -2) @ torch.linalg.solve(A, e[..., None]))[..., 0]


def nullspace_projector(J: Tensor) -> Tensor:
    """I - J^+ J, with the TRUE pseudo-inverse.

    Deliberately not the damped inverse. (I - J_damped J) is only approximately
    a null-space projector, so a secondary objective pushed through it moves the
    tool by O(lambda^2). The primary step is damped for safety; the projector is
    exact so that the invariance the null space is *for* actually holds.
    """
    n = J.shape[-1]
    Jp = torch.linalg.pinv(J)
    return torch.eye(n, dtype=J.dtype, device=J.device) - Jp @ J


def decode_twist(
    chain: Chain,
    theta: Tensor,
    twist: Tensor,
    dt: float = 1.0,
    lam: float = 0.05,
    secondary: Tensor | None = None,
    respect_limits: bool = True,
) -> Decoded:
    """One control step: body twist -> joint values.

    `twist` is a RATE in the tool frame (m/s, rad/s); `dt` is the control period.
    The distinction matters -- a twist is not a displacement, and conflating them
    silently rescales every action by the control frequency.

    `secondary` is a joint-space preference (B, n). It is projected into the null
    space, so on a redundant arm it changes posture without moving the tool, and
    on a 6-DoF arm the projector annihilates it rather than fighting the task.
    """
    theta = torch.atleast_2d(theta)
    twist = torch.atleast_2d(twist)
    J = body_jacobian(chain, theta)

    e = twist * dt
    delta = dls(J, e, lam)

    if secondary is not None:
        delta = delta + (nullspace_projector(J) @ torch.atleast_2d(secondary)[..., None])[..., 0]

    clamped = torch.zeros(theta.shape[0], dtype=torch.bool, device=theta.device)
    if respect_limits:
        lo = chain.limits[:, 0].to(theta.dtype)
        hi = chain.limits[:, 1].to(theta.dtype)
        # Clamped least squares. Truncating a saturated joint throws away the
        # motion it was carrying and the free joints never pick it up. Instead:
        # pin the offender at its bound, subtract the twist it still delivers,
        # and re-solve for the free ones. `frozen` accumulates across passes --
        # recomputing it each pass silently releases joints pinned earlier.
        frozen = torch.zeros_like(theta)
        free = torch.ones_like(theta, dtype=torch.bool)
        for _ in range(chain.n):
            proposed = theta + frozen + delta * free
            over = ((proposed < lo) | (proposed > hi)) & free
            if not over.any():
                break
            clamped = clamped | over.any(dim=-1)
            frozen = torch.where(over, torch.clamp(proposed, lo, hi) - theta, frozen)
            free = free & ~over
            residual_e = e - (J @ frozen[..., None])[..., 0]
            delta = dls(J * free[:, None, :].to(J.dtype), residual_e, lam)
        delta = frozen + delta * free
        delta = torch.clamp(theta + delta, lo, hi) - theta

    proposed = theta + delta

    # The residual is what the arm did NOT do. It must be computed from the step
    # actually applied, or a clamp hides itself and the policy never learns that
    # the command was impossible.
    residual = e - (J @ delta[..., None])[..., 0]
    return Decoded(theta=proposed, delta=delta, residual=residual, clamped=clamped)


def solve_ik(
    chain: Chain,
    target: Tensor,
    theta0: Tensor,
    lam: float = 0.05,
    max_iters: int = 60,
    tol: float = 1e-5,
    respect_limits: bool = True,
    trust: float = 0.2,
) -> dict[str, Tensor]:
    """Newton on SE(3): error by matrix logarithm, step by body Jacobian.

    Returns the residual as well as the answer. A target outside the workspace
    has no solution at all and the iteration stalls with a non-zero error, which
    is a different thing from a singular but reachable target (ch.6 sec.3.4).
    Reporting the residual is what lets a caller tell them apart.

    `trust` caps the twist consumed per iteration. Newton on SE(3) converges
    quadratically NEAR the solution; far from it the damped step is bounded by
    ||e||/(2*lambda), which for a small lambda permits a step many times the
    error and simply overshoots. Capping the error per step is the globalisation
    that makes the local guarantee usable from an arbitrary start.
    """
    target = target if target.dim() == 3 else target[None]
    theta = torch.atleast_2d(theta0).clone()
    b = theta.shape[0]
    iters = torch.zeros(b, dtype=torch.long, device=theta.device)
    active = torch.ones(b, dtype=torch.bool, device=theta.device)

    for k in range(max_iters):
        T = fk(chain, theta)
        err = log_se3(inverse(T) @ target)          # body twist from T to target
        done = torch.linalg.norm(err, dim=-1) < tol
        active = active & ~done
        if not active.any():
            break
        scale = torch.clamp(trust / torch.clamp(torch.linalg.norm(err, dim=-1), min=1e-12), max=1.0)
        step = decode_twist(chain, theta, err * scale[:, None], dt=1.0, lam=lam,
                            respect_limits=respect_limits)
        theta = torch.where(active[:, None], step.theta, theta)
        iters = iters + active.long()

    T = fk(chain, theta)
    err = log_se3(inverse(T) @ target)
    pos_err = torch.linalg.norm((target[:, :3, 3] - T[:, :3, 3]), dim=-1)
    return {
        "theta": theta,
        "residual": err,
        "residual_norm": torch.linalg.norm(err, dim=-1),
        "pos_err": pos_err,
        "converged": torch.linalg.norm(err, dim=-1) < tol,
        "iters": iters,
    }


def manipulability(chain: Chain, theta: Tensor) -> Tensor:
    """Yoshikawa's measure, sqrt(det(J J^T)) -- zero at a singularity (ch.5 sec.4.6)."""
    J = body_jacobian(chain, theta)
    return torch.sqrt(torch.clamp(torch.linalg.det(J @ J.transpose(-1, -2)), min=0.0))


def sigma_min(chain: Chain, theta: Tensor) -> Tensor:
    """Smallest singular value. The condition the damping exists to survive."""
    return torch.linalg.svdvals(body_jacobian(chain, theta))[..., -1]
