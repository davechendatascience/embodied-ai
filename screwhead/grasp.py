"""Grasping from the contact side: the grasp map and closure (ch.12).

The dual of the arm. J maps joint rates to a tool twist and J^T maps a wrench to
joint torques; G maps contact forces to an object wrench and G^T maps the object
twist to contact velocities. Both follow from the same power argument, so the
gripper admits the same decomposition the arm does: an intent expressed on the
OBJECT -- where to touch it, along which normals -- and a decoder that realises
it with whatever fingers are present.

Two properties of contacts make this not ordinary linear algebra:

  UNILATERAL. A finger pushes and never pulls, so f_n >= 0. Coefficients that
  must be non-negative behave very differently from coefficients of any sign,
  and "does G have full rank" is not enough.

  MOMENT-FIRST. A contact wrench is (p x n, n), moment first. Getting the order
  wrong scrambles every test built on it, so the order is asserted here rather
  than assumed.

Closure is DECIDED by exhaustive escape-direction search and only afterwards
EXHIBITED by alternating projection. Never the reverse: the projection proves
success and cannot prove failure, so a stalled iteration means "no certificate
yet", not "no closure" (ch.12 sec.4.8).
"""
from __future__ import annotations

from itertools import combinations

import torch
from torch import Tensor


def contact_wrench_planar(p: Tensor, n: Tensor) -> Tensor:
    """(..., 2), (..., 2) -> (..., 3) as (m_z, f_x, f_y).

    m_z = p_x n_y - p_y n_x, the only surviving component of p x n in the plane.
    """
    m = p[..., 0] * n[..., 1] - p[..., 1] * n[..., 0]
    return torch.stack([m, n[..., 0], n[..., 1]], -1)


def contact_wrench_spatial(p: Tensor, n: Tensor) -> Tensor:
    """(..., 3), (..., 3) -> (..., 6) as (p x n, n), moment first."""
    return torch.cat([torch.linalg.cross(p, n), n], -1)


def friction_edges_planar(n: Tensor, mu: float) -> Tensor:
    """The two edges of a planar friction cone, (..., 2, 2).

    Every force inside the cone is a non-negative combination of these, so a
    friction contact is replaced by two frictionless ones along the edges.
    Half-angle is arctan(mu).
    """
    t = torch.stack([-n[..., 1], n[..., 0]], -1)          # unit tangent
    s = (1 + mu ** 2) ** 0.5
    return torch.stack([(n + mu * t) / s, (n - mu * t) / s], dim=-2)


def grasp_matrix_planar(points: Tensor, normals: Tensor, mu: float = 0.0) -> Tensor:
    """(k,2),(k,2) -> (3, columns). With mu > 0 each contact contributes both
    cone edges, which is what turns form closure into force closure."""
    if mu <= 0:
        return contact_wrench_planar(points, normals).T
    cols = []
    for i in range(points.shape[0]):
        for e in friction_edges_planar(normals[i], mu):
            cols.append(contact_wrench_planar(points[i], e))
    return torch.stack(cols, dim=-1)


def has_closure(G: Tensor, tol: float = 1e-9) -> bool:
    """Decide closure. Exhaustive, no solver, no iteration limit to misread.

    Closure holds iff rank(G) = d and no non-zero y satisfies y^T F_i <= 0 for
    every column: such a y is an escape direction and is a PROOF of failure.
    The cone {y : G^T y <= 0} is pointed when rank(G) = d, so if it holds
    anything but 0 it holds an extreme ray, and an extreme ray in R^d lies on
    d-1 of the hyperplanes y^T F_i = 0. Enumerating (d-1)-subsets of columns and
    taking their common perpendicular is therefore an exhaustive search.
    """
    d, k = G.shape
    if torch.linalg.matrix_rank(G, atol=1e-9) < d:
        return False
    for subset in combinations(range(k), d - 1):
        A = G[:, list(subset)].T                       # (d-1, d)
        ns = torch.linalg.svd(A, full_matrices=True)[2][d - 1:]
        for y in ns:
            for cand in (y, -y):
                if torch.all(cand @ G <= tol):
                    return False
    return True


def closure_certificate(G: Tensor, iters: int = 3000, tol: float = 1e-9):
    """Exhibit k > 0 with Gk = 0, by alternating projection.

    Returns (k, converged). A False here means NO CERTIFICATE YET and never
    "no closure" -- the convergence theorem says nothing about how many
    iterations are needed, and on shallow intersections a few per cent of closed
    grasps still show residual ~1 after 3000 steps. Use has_closure to decide.
    """
    k = torch.ones(G.shape[1], dtype=G.dtype)
    GGt_inv = torch.linalg.inv(G @ G.T)
    for _ in range(iters):
        k = k - G.T @ (GGt_inv @ (G @ k))              # onto null(G)
        k = torch.clamp(k, min=1.0)                    # onto the box k_i >= 1
        if torch.linalg.norm(G @ k) < tol:
            return k, True
    return k, bool(torch.linalg.norm(G @ k) < tol)
