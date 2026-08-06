"""The PINCH POINT: where a gripper's pads actually converge, which is not the TCP.

Everything in this project until now aligned `grip_site` -- robosuite's tool
centre point. That is a MOUNTING CONVENTION chosen by whoever wrote the gripper
XML, not a statement about where the fingers meet. Aligning it across grippers
aligns nothing physical, which is why the Robotiq, driven to the Panda's grasp
pose, straddles the bowl wall (fingertips at 17.8 and 73.1 mm radius around a
wall at ~43 mm) and still records ZERO contacts. It is in the right place by the
wrong measure.

The pinch point is the physical quantity: the location an object gets clamped at.
For a parallel jaw it is a fixed point in the gripper frame. For an underactuated
linkage -- Robotiq85, 12 hinges tendon-coupled 1/-3/1 -- the pads travel an ARC,
so the pinch point MIGRATES with closure and is a curve, not a point.

If delta = pinch(source) - pinch(target), then the offset that made the Robotiq
hold 3/3 was never a fitted number: it is derivable from the two gripper models
for any pair of grippers, with no sweep. That claim is what diag_pinch.py tests.
"""
import numpy as np

PAD_HINTS = ("pad", "fingertip", "finger")


def pad_geoms(model):
    """Collision geoms belonging to the fingers, split into the two jaws.

    Split by sign along the direction of maximum spread, which IS the closing
    axis for every gripper here -- verified rather than assumed by checking that
    the spread along it collapses as the jaw closes.
    """
    ids = []
    for g in range(model.ngeom):
        bn = (model.body_id2name(model.geom_bodyid[g]) or "").lower()
        if not bn.startswith("gripper0"):
            continue
        if not any(h in bn for h in PAD_HINTS):
            continue
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        ids.append(g)
    return ids


def _split(model, data, ids):
    p = np.array([data.geom_xpos[g] for g in ids])
    c = p - p.mean(0)
    axis = np.linalg.svd(c, full_matrices=False)[2][0]
    s = c @ axis
    return [ids[i] for i in np.where(s < 0)[0]], [ids[i] for i in np.where(s >= 0)[0]], axis


def pinch_world(env):
    """World position where the two jaws would clamp, at the CURRENT closure.

    The midpoint of the two jaw centroids. For a thin wall -- this task's rim is
    ~2.6 mm -- the clamping location and the full-closure meeting point coincide
    to well under a millimetre, so no thickness correction is applied.
    """
    m, d = env.sim.model, env.sim.data
    ids = pad_geoms(m)
    if len(ids) < 2:
        return None
    lo, hi, _ = _split(m, d, ids)
    if not lo or not hi:
        return None
    a = np.array([d.geom_xpos[g] for g in lo]).mean(0)
    b = np.array([d.geom_xpos[g] for g in hi]).mean(0)
    return (a + b) / 2.0


def tcp_world(env):
    m = env.sim.model
    return env.sim.data.site_xpos[
        m.site_name2id(env.env.robots[0].controller.eef_name)].copy()


def wrist_frame(env):
    m, d = env.sim.model, env.sim.data
    b = m.body_name2id("robot0_right_hand")
    return d.body_xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()


def pinch_offset(env):
    """pinch - TCP, expressed in the WRIST frame so it is pose-independent.

    This is the per-gripper constant (parallel jaw) or configuration-dependent
    vector (underactuated) that the whole contact-frame argument turns on.
    """
    p = pinch_world(env)
    if p is None:
        return None
    _, Rw = wrist_frame(env)
    return Rw.T @ (p - tcp_world(env))


def sweep_closure(env, n=12, settle=18):
    """pinch_offset as a function of closure, from fully open to fully closed.

    robosuite's format_action is `current + speed*sign(action)` with speed 0.01,
    so closure is traversed by repeated sign commands and takes ~200 steps end to
    end; magnitude is discarded. Returns (fractions, offsets, spans).
    """
    a = env.env.action_dim - 1
    for _ in range(240):
        env.step([0.] * a + [-1.])
    out = []
    steps = max(int(240 / n), 1)
    for k in range(n + 1):
        if k:
            for _ in range(steps):
                env.step([0.] * a + [1.])
        for _ in range(settle):
            env.sim.forward()
        off = pinch_offset(env)
        m, d = env.sim.model, env.sim.data
        ids = pad_geoms(m)
        lo, hi, axis = _split(m, d, ids)
        span = float(abs(np.array([d.geom_xpos[g] for g in hi]).mean(0) @ axis
                         - np.array([d.geom_xpos[g] for g in lo]).mean(0) @ axis))
        out.append((k / n, None if off is None else off.copy(), span))
    return out
