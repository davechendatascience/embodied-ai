"""Per-geom segmentation -> per-object 2D boxes. No simulator import.

Shared by every host. `examples/probe_grounding.py` calls it with a LIBERO env
on robosuite 1.4.1; `examples/probe_robocasa.py` calls it with a RoboCasa env on
robosuite master. Those two cannot share an interpreter, so this file must not
import mujoco, robosuite or either benchmark -- it takes a model object and an
array, and duck-types the handful of fields it needs.

THE FRAME TRAP, restated because it is the one that costs a whole experiment.
Boxes must be computed on the image AS THE POLICY RECEIVES IT. openpi rotates
LIBERO frames 180 degrees "to match train preprocessing" and this project's
harness does the same; a box computed on the raw render is point-reflected
about the centre with respect to what the model sees. It still looks like a
valid box on a real object -- just the wrong one whenever the scene is roughly
symmetric, and nothing in a scalar log will say so. `policy_view` is that
transform, stated once, applied to RGB and segmentation alike.

WHAT COUNTS AS AN OBJECT: a body that owns a joint and is not part of the robot.
Fixtures have no joint; a can has a free joint and a drawer a slide joint.
Geoms are walked UP the body tree to the nearest such body, so an articulated
object reports as the one thing a person would name.
"""

import numpy as np

#: Body-name prefixes that belong to the robot rather than the scene. RoboCasa
#: mounts more hardware than LIBERO (mobile base, second arm on some configs),
#: so this is a prefix list rather than the three names LIBERO needs.
ROBOT_PREFIXES = ("robot", "gripper", "mount", "base", "controller")


def policy_view(img):
    """(H, W, C) as rendered -> (H, W, C) as the policy is fed it.

    180 degree rotation, not a vertical flip. Verbatim from openpi's LIBERO
    example and from this project's own harness, which agree -- by two
    independent decisions, so if either moves this must move with it.
    """
    return np.ascontiguousarray(np.asarray(img)[::-1, ::-1])


def body_name(model, bid):
    """Body id -> name, across robosuite generations.

    1.4.1's MjModel wrapper exposes `body_id2name`; newer bindings drop it in
    favour of `mujoco.mj_id2name`. Resolved by capability rather than by
    version number, because the version that matters is whatever is installed.
    """
    fn = getattr(model, "body_id2name", None)
    if fn is not None:
        return fn(bid)
    import mujoco                                     # only on the new path
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(bid))


def movable_bodies(model, robot_prefixes=ROBOT_PREFIXES):
    """Body ids that own a joint and are not part of the robot."""
    out = set()
    for j in range(model.njnt):
        bid = int(model.jnt_bodyid[j])
        nm = body_name(model, bid)
        if nm and not nm.startswith(robot_prefixes):
            out.add(bid)
    return out


def movable_root(model, bid, movable):
    """Nearest ancestor of `bid` (inclusive) that is a movable object, or None."""
    seen = 0
    while bid > 0 and seen <= model.nbody:
        if bid in movable:
            return bid
        bid = int(model.body_parentid[bid])
        seen += 1
    return None


def oracle_boxes(model, seg, movable, min_px=12):
    """Per-object pixel boxes from a per-geom segmentation image.

    `seg` must ALREADY be in the policy's frame -- pass `policy_view(...)`.
    Returns {body_name: (x0, y0, x1, y1)}, inclusive integer bounds, origin at
    the top-left of the image as the policy receives it.

    `min_px` drops objects with fewer visible pixels than that. An object mostly
    behind the arm can leave three stray pixels, and a box around three pixels
    is a grounding claim the image does not support -- better absent than wrong.
    On a cluttered RoboCasa counter this removes a lot.
    """
    seg = np.asarray(seg).reshape(seg.shape[0], seg.shape[1])
    ngeom = int(model.ngeom)

    root_of = {}
    for gid in np.unique(seg):
        gid = int(gid)
        if gid < 0 or gid >= ngeom:
            continue                       # background, or a non-geom element
        root = movable_root(model, int(model.geom_bodyid[gid]), movable)
        if root is not None:
            root_of[gid] = root

    boxes, counts = {}, {}
    for gid, root in root_of.items():
        name = body_name(model, root)
        ys, xs = np.nonzero(seg == gid)
        if len(xs) == 0:
            continue
        b = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        if name in boxes:                  # union over all geoms of one object
            o = boxes[name]
            b = (min(o[0], b[0]), min(o[1], b[1]),
                 max(o[2], b[2]), max(o[3], b[3]))
        boxes[name] = b
        counts[name] = counts.get(name, 0) + int(len(xs))
    return {n: b for n, b in boxes.items() if counts[n] >= min_px}


def draw_overlay(rgb, boxes, width=2):
    """Boxes burned into a copy of `rgb`, for looking at. Both in policy frame.

    Deliberately not a plotting library: this runs next to a live EGL context
    and adding a GUI toolkit there has cost this project time before.
    """
    img = np.array(rgb, dtype=np.uint8, copy=True)
    h, w = img.shape[:2]
    palette = [(255, 64, 64), (64, 255, 64), (64, 160, 255), (255, 220, 64),
               (255, 64, 255), (64, 255, 255)]
    for i, (_, (x0, y0, x1, y1)) in enumerate(sorted(boxes.items())):
        c = palette[i % len(palette)]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w - 1, x1), min(h - 1, y1)
        img[y0:y0 + width, x0:x1 + 1] = c
        img[max(y0, y1 - width + 1):y1 + 1, x0:x1 + 1] = c
        img[y0:y1 + 1, x0:x0 + width] = c
        img[y0:y1 + 1, max(x0, x1 - width + 1):x1 + 1] = c
    return img
