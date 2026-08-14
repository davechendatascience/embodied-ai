"""Oracle 2D boxes for LIBERO objects, in the frame the POLICY actually sees.

Increment 1 of the grounding probe: produce the boxes and prove they land on the
right pixels. No policy, no rollout, no detector. If this part is wrong every
number downstream is wrong in a way that looks like a model finding, so it ships
with an overlay renderer and is meant to be checked by eye first.

THE FRAME TRAP, which is the whole reason this file exists separately.
`eval_transfer.py` hands the policy `obs["agentview_image"][::-1, ::-1]` -- a 180
degree rotation, not a plain vertical flip. A box computed on the raw
segmentation is therefore point-reflected about the image centre with respect to
what the policy sees: still a valid-looking box, still on an object, just the
wrong one whenever the scene is roughly symmetric. That failure is invisible in
every scalar we log. So the segmentation is rotated by the SAME expression, from
the same constant, and the overlay is drawn on the rotated RGB.

WHY THE ENVIRONMENT IS BUILT HERE INSTEAD OF VIA `libero_ur5e.build`.
Segmentation sensors are created at CONSTRUCTION time from `camera_segmentations`
(robosuite `robot_env._create_segementation_sensor`), and `build` does not pass
it. Rather than edit a shared helper for a probe, this constructs its own env
with the same camera settings. One environment, one process -- the follower is
not involved and nothing else here renders.

WHY `element` SEGMENTATION AND NOT `instance`.
LIBERO's own `SegmentationRenderEnv` hardcodes the robot instance as "Panda0",
which is wrong the moment a UR5e is mounted. `element` returns raw MuJoCo geom
ids with no naming assumptions, and geom -> body -> movable root is a mapping we
can state and check.

WHAT COUNTS AS AN OBJECT. A body that owns a joint and is not part of the robot,
which is the same filter `eval_transfer.leader_obj` uses and for the same reason:
without it, "place it on the plate" selects `flat_stove_1_burner_plate`, a
fixture. Geoms are walked UP the body tree to the nearest such body, so an
articulated object (a drawer's handle geom under a cabinet body with a slide
joint) reports as the one object a person would name.

Run:
  MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python examples/probe_grounding.py \
      --suite libero_10 --task-id 0 --overlay pairs/diag/boxes.png
"""
import argparse
import os
import sys

import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R)
sys.path.insert(0, R + "/examples")
sys.path.insert(0, R + "/third_party/LIBERO")

CAMS = ("agentview", "robot0_eye_in_hand")

#: The policy's view of an observation. Stated ONCE, imported by every later
#: increment, and applied to RGB and segmentation alike. Verbatim from
#: eval_transfer.py's `model.step(images=...)` call -- if that ever changes,
#: change it here in the same commit or the boxes silently stop matching.
def policy_view(img):
    """(H, W, C) as rendered -> (H, W, C) as the policy is fed it."""
    return np.ascontiguousarray(np.asarray(img)[::-1, ::-1])


def build_env(suite_name, task_id, robot="Panda", gripper="default", res=256,
              seed=0):
    """A single LIBERO env that also renders per-geom segmentation.

    Mirrors `libero_ur5e.build`'s camera settings so the RGB the probe measures
    on is the RGB the eval harness would have produced.
    """
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"),
                        task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl, robots=[robot], gripper_types=gripper,
        camera_heights=res, camera_widths=res, controller="OSC_POSE",
        camera_depths=True, camera_names=list(CAMS),
        camera_segmentations="element", horizon=10000)
    np.random.seed(seed)
    obs = env.reset()
    return env, suite, task, obs


def movable_bodies(model):
    """Body ids that own a joint and are not part of the robot.

    A fixture has no joint; every real manipulation target has one (a free joint
    on a can, a slide joint on a drawer). Robot links are excluded by name
    because they own joints too.
    """
    out = set()
    for j in range(model.njnt):
        bid = int(model.jnt_bodyid[j])
        nm = model.body_id2name(bid)
        if nm and not nm.startswith(("robot", "gripper", "mount")):
            out.add(bid)
    return out


def movable_root(model, bid, movable):
    """Nearest ancestor of `bid` (inclusive) that is a movable object, or None.

    Walks up `body_parentid` so a handle geom parented under a cabinet reports
    the cabinet. Stops at the world body (parent of 0 is 0).
    """
    seen = 0
    while bid > 0 and seen <= model.nbody:
        if bid in movable:
            return bid
        bid = int(model.body_parentid[bid])
        seen += 1
    return None


def oracle_boxes(model, seg, movable, min_px=12):
    """Per-object pixel bounding boxes from a per-geom segmentation image.

    `seg` must ALREADY be in the policy's frame -- pass `policy_view(...)`.
    Returns {body_name: (x0, y0, x1, y1)} with inclusive integer pixel bounds,
    origin top-left of the image as the policy receives it.

    `min_px` drops objects with fewer visible pixels than that. An object behind
    the arm can leave three stray pixels, and a box around three pixels is a
    grounding claim we cannot support -- better absent than wrong.
    """
    seg = np.asarray(seg).reshape(seg.shape[0], seg.shape[1])
    ngeom = int(model.ngeom)

    # geom id -> movable root body id, resolved once per distinct id present.
    root_of = {}
    for gid in np.unique(seg):
        gid = int(gid)
        if gid < 0 or gid >= ngeom:
            continue                       # background, or a non-geom element
        root = movable_root(model, int(model.geom_bodyid[gid]), movable)
        if root is not None:
            root_of[gid] = root

    boxes = {}
    for gid, root in root_of.items():
        name = model.body_id2name(root)
        ys, xs = np.nonzero(seg == gid)
        if len(xs) == 0:
            continue
        b = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        if name in boxes:                  # union over all geoms of one object
            o = boxes[name]
            b = (min(o[0], b[0]), min(o[1], b[1]),
                 max(o[2], b[2]), max(o[3], b[3]))
        boxes[name] = b

    counts = {n: 0 for n in boxes}
    for gid, root in root_of.items():
        counts[model.body_id2name(root)] += int((seg == gid).sum())
    return {n: b for n, b in boxes.items() if counts[n] >= min_px}


def draw_overlay(rgb, boxes, width=2):
    """Boxes burned into a copy of `rgb`, for looking at. Both in policy frame.

    Deliberately not a plotting library: this runs in the LIBERO venv next to a
    live EGL context and adding a GUI toolkit there has cost this project time
    before.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--robot", default="Panda")
    ap.add_argument("--gripper", default="default")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--cam", default="agentview", choices=list(CAMS))
    ap.add_argument("--min-px", type=int, default=12)
    ap.add_argument("--overlay", default=None,
                    help="write a PNG with the boxes drawn, in the POLICY frame")
    a = ap.parse_args()

    env, _, task, obs = build_env(a.suite, a.task_id, robot=a.robot,
                                  gripper=a.gripper, res=a.res)
    model = env.sim.model

    rgb = policy_view(obs[f"{a.cam}_image"])
    seg = policy_view(obs[f"{a.cam}_segmentation_element"])
    boxes = oracle_boxes(model, seg, movable_bodies(model), min_px=a.min_px)

    print(f"task   {task.language!r}")
    print(f"cam    {a.cam}  {a.res}x{a.res}  (policy frame: [::-1, ::-1])")
    print(f"objects {len(boxes)}")
    for n, (x0, y0, x1, y1) in sorted(boxes.items()):
        print(f"  {n:<34} x[{x0:>3},{x1:>3}] y[{y0:>3},{y1:>3}]"
              f"  {x1 - x0 + 1:>3}x{y1 - y0 + 1:<3} px")

    if a.overlay:
        import imageio
        os.makedirs(os.path.dirname(os.path.abspath(a.overlay)), exist_ok=True)
        imageio.imwrite(a.overlay, draw_overlay(rgb, boxes))
        print(f"overlay -> {a.overlay}")

    env.close()


if __name__ == "__main__":
    main()
