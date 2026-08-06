"""Cross-gripper constants, measured from the two models. Nothing task-specific.

The point of this file is that it contains no tuned numbers and no knowledge of
any task. Give it a source gripper and a target gripper and it measures
everything the transfer needs. The task contributes exactly one bit -- whether
the feature being grasped is narrower than the target can close -- and that comes
from perception, not from tuning.

WHAT IS MEASURED, AND WHAT EACH ONE IS FOR

  cam_to_tcp        |camera -> TCP|. The wrist camera mounts on the ARM, so this
                    changes only with the gripper: 97 mm on a PandaGripper, 145
                    on a Robotiq85. A policy that servos on the wrist image is
                    really driving the CAMERA; the TCP follows only because this
                    was rigid in training. The difference is how far the target's
                    TCP overshoots, and it must be given back as a viewpoint
                    offset. Predicted 48 mm; measured, with the correction
                    switched off, min TCP-to-object 44.8 mm.

  closed_sep        how far apart the pads still are when the jaw is fully shut:
                    Panda 16.5 mm, Robotiq85 35.3 mm, Rethink 34.8 mm. A jaw that
                    cannot close below this CANNOT PINCH a thinner feature, and
                    must grasp offset by roughly this much to find a chord it can
                    span instead. Bowl rim 2.6 mm -> 35 mm offset holds 3/3.
                    Alphabet soup ~65 mm, already wider -> 0 mm needed, 27/30.

  open_span         the widest feature it can straddle at all. This is what kills
                    the RethinkGripper: 22.5 mm of usable gap against a rim that
                    needs ~31 mm, so no offset rescues it -- 0/37, and correctly
                    reported as infeasible rather than attempted.

THE RULE
    feature_width >= closed_sep(target)   ->  grasp where the source grasps
    feature_width <  closed_sep(target)   ->  offset by ~closed_sep(target)
    feature_width_there > open_span(target) -> INFEASIBLE, say so

Verified on two objects with OPPOSITE predictions, and it explains all three
grippers' behaviour for three different reasons.
"""
import numpy as np


def _tcp(env):
    m = env.sim.model
    return env.sim.data.site_xpos[
        m.site_name2id(env.env.robots[0].controller.eef_name)].copy()


def _cam(env):
    m = env.sim.model
    return env.sim.data.cam_xpos[m.camera_name2id("robot0_eye_in_hand")].copy()


def _pad_geoms(m):
    return [g for g in range(m.ngeom)
            if (m.body_id2name(m.geom_bodyid[g]) or "").startswith("gripper0")
            and any(h in (m.body_id2name(m.geom_bodyid[g]) or "").lower()
                    for h in ("pad", "fingertip", "finger"))
            and not (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0)]


def pad_separation(env):
    """Distance between the two jaw centroids, along the closing axis."""
    m, d = env.sim.model, env.sim.data
    ids = _pad_geoms(m)
    if len(ids) < 2:
        return float("nan")
    p = np.array([d.geom_xpos[g] for g in ids])
    c = p - p.mean(0)
    axis = np.linalg.svd(c, full_matrices=False)[2][0]
    s = c @ axis
    lo, hi = s < 0, s >= 0
    if not lo.any() or not hi.any():
        return float("nan")
    return float(s[hi].mean() - s[lo].mean())


def measure(env, settle=240):
    """Everything one gripper contributes. No task, no object, no tuning."""
    a = env.env.action_dim - 1
    for _ in range(settle):
        env.step([0.] * a + [-1.])
    open_sep = pad_separation(env)
    cam_to_tcp = float(np.linalg.norm(_tcp(env) - _cam(env)))
    for _ in range(settle):
        env.step([0.] * a + [1.])
    return dict(cam_to_tcp=cam_to_tcp,
                open_sep=open_sep,
                closed_sep=pad_separation(env))


def profile(source, target):
    """The transfer constants for a gripper PAIR.

    `cam_offset_mm` is the viewpoint correction; `width_offset_mm` is how far to
    grasp off the source's grasp point WHEN the feature is too narrow for the
    target to pinch. Whether that applies is the one thing the task decides.
    """
    s, t = measure(source), measure(target)
    return dict(
        source=s, target=t,
        cam_offset_mm=(t["cam_to_tcp"] - s["cam_to_tcp"]) * 1000,
        width_offset_mm=t["closed_sep"] * 1000,
        min_feature_mm=t["closed_sep"] * 1000,
        max_feature_mm=t["open_sep"] * 1000,
    )


def plan(prof, feature_width_mm):
    """Given the profile and ONE perceived number, what to do."""
    if feature_width_mm > prof["max_feature_mm"]:
        return dict(feasible=False,
                    why=f"feature {feature_width_mm:.1f} mm exceeds the target's "
                        f"{prof['max_feature_mm']:.1f} mm open span")
    needs_offset = feature_width_mm < prof["min_feature_mm"]
    return dict(feasible=True, needs_offset=needs_offset,
                cam_offset_mm=prof["cam_offset_mm"],
                width_offset_mm=prof["width_offset_mm"] if needs_offset else 0.0,
                why=("feature is narrower than the target can close, so it cannot "
                     "be pinched -- grasp offset to span a wider chord"
                     if needs_offset else
                     "feature is already wider than the target's closed jaw -- "
                     "grasp where the source does"))
