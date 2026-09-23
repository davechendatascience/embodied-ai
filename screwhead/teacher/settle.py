"""When a placed object has settled, decided from the model rather than from thresholds
(BRN-rl-teacher-equilibrium-settle, work in progress).

An object has settled when, with nothing of the robot touching it,
  - it is in static equilibrium: its weight can be balanced by forces inside each of its
    contacts' friction cones (the model's own friction coefficients). A projected-hull test
    rejected a real rest -- cream cheese wedged on the bowl's sloped wall, held by friction
    with its centre of mass 0.18 mm outside the contacts' triangle -- that this accepts;
  - its kinetic energy is below the least potential rise that would tip it over an edge of
    its support, so it cannot topple on its own;
  - and below mu_min m g times the distance its origin can slide before leaving LIBERO's
    acceptance set, so when it stops it is still accepted.
A goal joint (drawer, knob) has settled when its predicted rest position, q + qdot tau with
tau = its joint-space inertia over its damping (no spring in these fixtures), still
satisfies its predicate. The one tolerance, EQUILIBRIUM_TOL, is the least-squares solver's
numerical zero relative to the object's weight.
"""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.optimize import nnls

PYRAMID_EDGES = 8          # the friction cone approximated from inside: edges at mu cos(pi/8), so the faces
#                            reach mu cos^2(pi/8) = 0.854 mu and an equilibrium needing more may be rejected.
#                            Deliberately inside the inscribed pyramid (edges on the cone): MuJoCo's soft
#                            friction does not hold every Coulomb-feasible state near mu -- on an incline
#                            (elliptic cone, impratio 20, mu 0.95, 37 slope directions) the inscribed pyramid
#                            accepted states that then slid, 1 of 19 at 0.99 mu (331 mm in 2 s) and 10 of 18
#                            at 1.00 mu, while nothing this pyramid accepts slid
EQUILIBRIUM_TOL = 1e-6     # residual / weight counted as zero
COINCIDENT = 1e-9          # m: contact points closer than this give no tipping axis


def subtree_bodies(m, root: int) -> set[int]:
    return {b for b in range(m.nbody) if _is_under(m, b, root)}


def _is_under(m, b: int, root: int) -> bool:
    while b > 0:
        if b == root:
            return True
        b = int(m.body_parentid[b])
    return root == 0


def contacts_on(m, d, bodies: set[int]):
    """(position, force-direction frame rows (n, t1, t2) oriented so n pushes on the object,
    friction coefficient) for each contact of the object with anything else."""
    n = d.ncon
    if not n:
        return []
    b = m.geom_bodyid[d.contact.geom[:n]]           # (ncon, 2) body ids, in one pass
    mask = np.zeros(m.nbody, bool)
    mask[list(bodies)] = True
    inside = mask[b]
    out = []
    for i in np.flatnonzero(inside[:, 0] != inside[:, 1]):   # exactly one side is the object
        i = int(i)
        c = d.contact[i]
        frame = np.asarray(c.frame, float).reshape(3, 3)
        if inside[i, 0]:                    # the normal points from geom1 to geom2: reverse it
            frame = -frame
        out.append((np.asarray(c.pos, float).copy(), frame, float(c.friction[0])))
    return out


def in_equilibrium(m, d, root: int) -> bool:
    """Can the object's weight be balanced by forces inside its contacts' friction cones?"""
    bodies = subtree_bodies(m, root)
    cons = contacts_on(m, d, bodies)
    if not cons:
        return False
    mass = float(m.body_subtreemass[root])
    com = np.asarray(d.subtree_com[root], float)
    weight = mass * np.asarray(m.opt.gravity, float)
    cols = []
    for pos, frame, mu in cons:
        n, t1, t2 = frame
        slope = mu * np.cos(np.pi / PYRAMID_EDGES)
        for k in range(PYRAMID_EDGES):
            a = 2 * np.pi * k / PYRAMID_EDGES
            e = n + slope * (np.cos(a) * t1 + np.sin(a) * t2)
            cols.append(np.concatenate([e, np.cross(pos - com, e)]))
    A = np.array(cols).T
    b = -np.concatenate([weight, np.zeros(3)])
    _, residual = nnls(A, b)
    return residual <= EQUILIBRIUM_TOL * float(np.linalg.norm(weight))


def kinetic_energy(m, d, root: int) -> float:
    mujoco.mj_subtreeVel(m, d)
    mass = float(m.body_subtreemass[root])
    v = np.asarray(d.subtree_linvel[root], float)
    w = np.asarray(d.cvel[root][:3], float)                   # angular velocity, world frame
    return 0.5 * mass * float(v @ v) + 0.5 * float(w @ np.asarray(d.subtree_angmom[root], float))


def tipping_barrier(m, d, root: int) -> float:
    """Least potential energy the object must gain to tip over an edge of its support: the
    edges of the contact points' horizontal convex hull, in 3D, as rotation axes."""
    cons = contacts_on(m, d, subtree_bodies(m, root))
    pts = np.array([c[0] for c in cons])
    if len(pts) < 2:
        return 0.0
    com = np.asarray(d.subtree_com[root], float)
    g = float(np.linalg.norm(m.opt.gravity))
    mass = float(m.body_subtreemass[root])
    edges = _hull_edges(pts)
    if not edges:
        return 0.0
    a = np.array([e[0] for e in edges])                       # (k, 3) edge starts
    u = np.array([e[1] - e[0] for e in edges])                # (k, 3) edge directions
    length = np.linalg.norm(u, axis=1)
    keep = length >= COINCIDENT
    if not keep.any():
        return 0.0
    a, u = a[keep], u[keep] / length[keep, None]
    w = com - a                                               # (k, 3) edge start to centre of mass
    r = w - (np.einsum("ij,ij->i", w, u))[:, None] * u        # perpendicular from the axis
    rise = float(np.min(np.linalg.norm(r, axis=1) * np.sqrt(np.maximum(1.0 - u[:, 2] ** 2, 0.0)) - r[:, 2]))
    return mass * g * max(rise, 0.0)


def _hull_edges(pts: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Edges of the horizontal convex hull of 3D points (monotone chain on x, y).

    Plain floats rather than numpy rows: the hull runs once per object per physics substep of
    every rollout, over a handful of contacts, and at that size numpy's scalar indexing costs more
    than the arithmetic it does.
    """
    xy = sorted((float(p[0]), float(p[1]), i) for i, p in enumerate(pts))

    def turn(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float, int]] = []
    upper: list[tuple[float, float, int]] = []
    for q in xy:
        while len(lower) >= 2 and turn(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in reversed(xy):
        while len(upper) >= 2 and turn(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    hull = [q[2] for q in lower[:-1]] + [q[2] for q in upper[:-1]]
    if len(hull) < 2:
        hull = [xy[0][2], xy[-1][2]]
    return [(pts[hull[k]], pts[hull[(k + 1) % len(hull)]]) for k in range(len(hull))
            if not (len(hull) == 2 and k == 1)]


def min_friction(m, d, root: int) -> float:
    cons = contacts_on(m, d, subtree_bodies(m, root))
    return min((c[2] for c in cons), default=0.0)


def joint_rest(m, d, joint: int) -> float:
    """Where a damped, springless joint comes to rest from here: q + qdot * M_jj / damping."""
    dof = int(m.jnt_dofadr[joint])
    damping = float(m.dof_damping[dof])
    q, qd = float(d.qpos[m.jnt_qposadr[joint]]), float(d.qvel[dof])
    if damping <= 0:
        return q if abs(qd) == 0 else np.sign(qd) * np.inf
    inertia = float(d.qM[m.dof_Madr[dof]])
    return q + qd * inertia / damping
