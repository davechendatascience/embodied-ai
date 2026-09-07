"""Demonstrations -> embodiment-free tool twists.

This is the conversion that lets data from different arms pool. It is
arithmetic, not learning: run the demonstrating robot's own forward kinematics
over its joint trajectory, and read off the tool twist between consecutive
poses. Panda demonstrations and UR5e demonstrations become the same quantity.

Decoding back is NOT the mirror image, and the difference matters. On a
redundant arm a tool pose does not determine a configuration -- a whole
self-motion manifold maps to the same pose (ch.5 sec.4.8). So the fidelity of a
retarget is a claim about the TOOL trajectory, not about joint angles. Warm
starting keeps the solution on the demonstrated branch in practice, but nothing
guarantees it, and scoring joint agreement would fail a decode that reproduced
the task perfectly with a different elbow.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .ik import sigma_min, solve_ik
from .interface import ActionSpec
from .kinematics import fk
from .poe import Chain
from .se3 import inverse, log_se3


def to_twists(chain: Chain, q_traj: Tensor, spec: ActionSpec) -> Tensor:
    """(T, B, n) joint trajectories -> (T-1, B, 6) body twist rates.

    A (T, n) single trajectory is accepted and returns (T-1, 6).
    """
    single = q_traj.dim() == 2
    q = q_traj[:, None] if single else q_traj
    T, B, n = q.shape
    poses = fk(chain, q.reshape(T * B, n)).reshape(T, B, 4, 4)
    rel = inverse(poses[:-1]) @ poses[1:]
    out = log_se3(rel) / spec.dt
    return out[:, 0] if single else out


def usable(chain: Chain, q_traj: Tensor, sigma_thresh: float = 0.02) -> Tensor:
    """Frames far enough from a singularity to be decodable.

    Near-singular frames converge only ~92% of the time and the failures are
    stuck rather than slow, so retargeting drops them instead of expecting the
    solver to rescue them. Dropping is honest; a silently unconverged frame is
    a corrupted training target.
    """
    return sigma_min(chain, q_traj) >= sigma_thresh


def decode(
    chain: Chain,
    q0: Tensor,
    twists: Tensor,
    spec: ActionSpec,
    lam: float = 1e-3,
    max_iters: int = 60,
    tol: float = 1e-6,
) -> dict[str, Tensor]:
    """Replay a twist sequence on a chain, warm-starting each solve.

    Damping is far lighter here than in the training-graph step: retargeting is
    offline, so it can afford accuracy where the online decoder buys safety. A
    lambda of 0.02 leaves an O(lambda^2/sigma^2) error -- percent-level on a
    moderately conditioned Jacobian, which would corrupt the very targets this
    produces.
    """
    q = torch.atleast_2d(q0).clone()                 # (B, n)
    tw = twists if twists.dim() == 3 else twists[:, None]   # (T, B, 6)
    out_q, out_pose_err = [q.clone()], []
    for t in range(tw.shape[0]):
        target = fk(chain, q) @ _exp(tw[t] * spec.dt)
        res = solve_ik(chain, target, q, lam=lam, max_iters=max_iters, tol=tol)
        q = res["theta"]
        out_q.append(q.clone())
        out_pose_err.append(res["pos_err"])
    return {
        "q": torch.stack(out_q),                     # (T+1, B, n)
        "pose_err": torch.stack(out_pose_err),       # (T, B)
    }


def _exp(V: Tensor) -> Tensor:
    from .se3 import exp_twist
    return exp_twist(V)
