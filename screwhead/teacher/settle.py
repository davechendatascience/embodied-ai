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

PYRAMID_EDGES = 8          # friction cone approximated from inside: a feasible pyramid is a feasible cone
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
    out = []
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if (b1 in bodies) == (b2 in bodies):
            continue
        frame = np.asarray(c.frame, float).reshape(3, 3)
        if b1 in bodies:                    # the normal points from geom1 to geom2: reverse it
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
    rise = np.inf
    for a, b in _hull_edges(pts):
        u = b - a
        if np.linalg.norm(u) < COINCIDENT:
            continue
        u /= np.linalg.norm(u)
        r = com - a - ((com - a) @ u) * u                     # axis to centre of mass, perpendicular
        rise = min(rise, float(np.linalg.norm(r)) * float(np.sqrt(max(1.0 - u[2] ** 2, 0.0))) - float(r[2]))
    return mass * g * max(rise, 0.0) if np.isfinite(rise) else 0.0


def _hull_edges(pts: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Edges of the horizontal convex hull of 3D points (monotone chain on x, y)."""
    order = sorted(range(len(pts)), key=lambda i: (pts[i][0], pts[i][1]))

    def turn(o, a, b):
        return (pts[a][0] - pts[o][0]) * (pts[b][1] - pts[o][1]) - (pts[a][1] - pts[o][1]) * (pts[b][0] - pts[o][0])
    lower, upper = [], []
    for i in order:
        while len(lower) >= 2 and turn(lower[-2], lower[-1], i) <= 0:
            lower.pop()
        lower.append(i)
    for i in reversed(order):
        while len(upper) >= 2 and turn(upper[-2], upper[-1], i) <= 0:
            upper.pop()
        upper.append(i)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 2:
        hull = [order[0], order[-1]]
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
