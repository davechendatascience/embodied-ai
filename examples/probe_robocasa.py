"""The grounding sweep on RoboCasa. Runs in .venv-robocasa, not .venv.

Same conditions, same noise floor, same verdict bands as the LIBERO probe --
`probe_conditions.sweep_and_report` is shared, deliberately, because two
benchmarks measured by two slightly different scripts cannot be compared.

WHY ROBOCASA IS THE INTERESTING CASE.
LIBERO's `libero_10/0` names its referents unambiguously against eight objects
on a clean table, so a correct box tells the policy nothing it did not already
have -- and the measured answer there was that the boxes are inert. RoboCasa is
the opposite regime by construction: 120 kitchen scenes, 2500+ objects across
153 categories, clutter, occlusion, and multiple instances of a category in one
scene. If appended grounding ever helps, it should help here first.

TWO THINGS THAT MAKE THIS AN UNFAIR TEST OF pi-0.5, BOTH LOAD-BEARING:

  1. pi05_libero IS FINE-TUNED ON LIBERO. On RoboCasa it is out of
     distribution -- different scenes, different robot (PandaOmron, mobile
     base), different camera rig. A null result could mean "grounding does not
     help" or "the policy is lost", and those are not the same finding.
     `probe_language.py` separates them: if a deliberately wrong INSTRUCTION
     still moves the policy, the prompt channel is live and a grounding null is
     about grounding. If nothing moves it, the measurement is void. RUN THAT
     FIRST, and do not report a sweep here without it.

  2. THE 180 DEGREE ROTATION IS A LIBERO QUIRK, NOT A POLICY REQUIREMENT.
     openpi rotates LIBERO frames "to match train preprocessing" because LIBERO
     renders inverted. Whether robosuite master + RoboCasa renders the same way
     is a separate empirical fact, and applying the rotation blindly would feed
     pi-0.5 an upside-down kitchen -- which is a great way to measure "the
     policy ignores everything" and call it a grounding result. So it is a
     FLAG, defaulted off, and `--dump` writes both orientations to be looked
     at before anything is believed.

Run:
  # 1. look at the frame, decide the orientation
  MUJOCO_GL=egl PYTHONPATH=. ./.venv-robocasa/bin/python \\
      examples/probe_robocasa.py --task PickPlaceCounterToSink --dump pairs/diag/rc
  # 2. language control (needs scripts/serve_pi05.sh)
  ... --policy pi05 --language
  # 3. the sweep
  ... --policy pi05 --style paligemma --repeats 8
"""
import argparse
import os
import sys

import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R)
sys.path.insert(0, R + "/examples")

from probe_conditions import (FakePolicy, Pi05Policy, quat2axisangle,
                              sweep_and_report)
from xembody.boxes import draw_overlay, movable_bodies, oracle_boxes
from xembody.probe import compare, noise_floor, verdict

#: RoboCasa's default rig. `left` stands in for LIBERO's `agentview` as the
#: base view; the wrist camera keeps its name across both benchmarks.
#: `right` measured as the view that actually frames the task objects on 7 of 8
#: PickPlace tasks; `left` and `center` often see neither the target nor its
#: distractors, which would make a grounding null meaningless.
BASE_CAM = "robot0_agentview_right"
WRIST_CAM = "robot0_eye_in_hand"


def build_env(task, res=256, seed=0, layout=None, style=None,
              cam=BASE_CAM):
    """A RoboCasa kitchen env.

    NO `camera_segmentations` HERE, UNLIKE LIBERO. RoboCasa's
    `Kitchen.__init__` enumerates the camera arguments it forwards to robosuite
    and that is not one of them, so the observable cannot be switched on through
    its API. Segmentation is rendered separately -- see `segmentation`.
    """
    from robocasa.utils.env_utils import create_env

    env = create_env(
        env_name=task,
        camera_names=[cam, WRIST_CAM],
        camera_widths=res, camera_heights=res,
        seed=seed, layout_ids=layout, style_ids=style,
    )
    obs = env.reset()
    return env, obs


def segmentation(env, obs, cam, res):
    """Per-geom segmentation for `cam`, in the SAME frame as `obs[cam_image]`.

    robosuite's own segmentation sensor is unreachable here (see `build_env`),
    so this calls MuJoCo directly -- `sim.render(..., segmentation=True)`
    returns (H, W, 2) whose second channel is the object id, which is exactly
    what the sensor would have used.

    THE ALIGNMENT IS ASSERTED, NOT ASSUMED. robosuite flips renders by
    `macros.IMAGE_CONVENTION`, and a segmentation map that disagrees with its
    image by a vertical flip still produces plausible boxes on real objects --
    wrong ones, silently. So the RGB is re-rendered through this same path and
    compared against the observation. If they match, the segmentation is in the
    observation's frame by construction; if they do not, this raises instead of
    quietly mislabelling every object in the scene.
    """
    import robosuite.macros as macros
    # Lives in mjcf_utils on robosuite master; camera_utils on 1.4.x. Import by
    # capability rather than by version, since both generations are in this repo.
    try:
        from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
    except ImportError:
        from robosuite.utils.camera_utils import IMAGE_CONVENTION_MAPPING

    conv = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
    sim = env.sim
    rgb = np.asarray(sim.render(camera_name=cam, width=res, height=res)[::conv],
                     dtype=np.int16)
    ref = np.asarray(obs[f"{cam}_image"], dtype=np.int16)

    # THE CHECK IS "NOT FLIPPED", NOT "BIT-IDENTICAL".
    # Exact equality was the first version and it was wrong: this renderer is
    # not bit-repeatable. Measured on PickPlaceCounterToStove, re-rendering the
    # same frame twice differs in ~1 pixel in 65536 by 1 LSB, which failed the
    # equality test on 3 of 6 tasks and looked like a frame bug. The failure
    # that actually matters is an ORIENTATION mismatch, so compare against the
    # flips explicitly and require the unflipped one to win by a wide margin.
    cands = {"as-is": ref, "vflip": ref[::-1], "hflip": ref[:, ::-1],
             "rot180": ref[::-1, ::-1]}
    errs = {k: float(np.abs(rgb - v).mean()) for k, v in cands.items()}
    best = min(errs, key=errs.get)
    runner = min((k for k in errs if k != "as-is"), key=errs.get)
    if best != "as-is" or errs["as-is"] > 0.1 * errs[runner]:
        raise RuntimeError(
            f"re-rendered RGB for {cam} is not in the observation's frame: "
            f"mean|diff| {errs}. The segmentation would be flipped relative to "
            "the image and every box would name the wrong object.")
    seg = sim.render(camera_name=cam, width=res, height=res,
                     segmentation=True)[::conv]
    return seg[:, :, 1]


def object_boxes(env, seg, min_px=12):
    """Boxes for RoboCasa's OWN registered objects, labelled in its own words.

    The joint-ownership heuristic that works on LIBERO is wrong here. A RoboCasa
    kitchen is full of articulated FIXTURES -- every cabinet door, drawer and
    sink handle owns a hinge or slide joint -- so "a body with a joint" returns
    a dozen doors, the robot's mobile base, and not the tomato. Measured on
    PickPlaceCounterToSink: 12 boxes, none of them the target.

    RoboCasa already knows which things are objects: `env.objects` maps a role
    to a MujocoObject, and `env.get_obj_lang(role)` gives the noun a human would
    use. Roles are:
        obj             the manipulation target named in the instruction
        distr_*         DISTRACTORS the benchmark places on purpose
    That last one is why this benchmark is worth probing at all -- the
    referential ambiguity is built in by the task designer, not injected by us.

    Returns {label: (x0, y0, x1, y1)} keyed by the LANGUAGE name, so the
    serialised prompt says "tomato" and not "obj_main".
    """
    model, data = env.sim.model, env.sim.data
    del data
    seg = np.asarray(seg)

    # role -> set of body ids in that object's subtree
    from xembody.boxes import body_name
    roots = {}
    for role, obj in env.objects.items():
        root = getattr(obj, "root_body", None)
        if root is None:
            continue
        for bid in range(model.nbody):
            nm = body_name(model, bid)
            if nm and (nm == root or nm.startswith(root)):
                roots.setdefault(role, set()).add(bid)

    body_to_role = {b: r for r, bs in roots.items() for b in bs}
    boxes, counts = {}, {}
    for gid in np.unique(seg):
        gid = int(gid)
        if gid < 0 or gid >= int(model.ngeom):
            continue
        role = body_to_role.get(int(model.geom_bodyid[gid]))
        if role is None:
            continue
        try:
            label = env.get_obj_lang(obj_name=role)
        except Exception:
            label = role
        ys, xs = np.nonzero(seg == gid)
        if len(xs) == 0:
            continue
        b = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        if label in boxes:
            o = boxes[label]
            b = (min(o[0], b[0]), min(o[1], b[1]),
                 max(o[2], b[2]), max(o[3], b[3]))
        boxes[label] = b
        counts[label] = counts.get(label, 0) + int(len(xs))
    return {n: b for n, b in boxes.items() if counts[n] >= min_px}


def view(img, rotate):
    """Apply the policy's frame transform, or not. See module docstring."""
    img = np.asarray(img)
    return np.ascontiguousarray(img[::-1, ::-1] if rotate else img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PickPlaceCounterToSink")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layout", type=int, default=None)
    ap.add_argument("--style-id", dest="style_id", type=int, default=None,
                    help="RoboCasa kitchen style; distinct from --style, "
                         "which is the box serialisation format")
    ap.add_argument("--no-rotate", dest="rotate", action="store_false",
                    default=True,
                    help="skip the 180 deg preprocessing. ON by default: "
                         "measured corr(world_z, image_y)=+0.686 on RoboCasa, "
                         "i.e. the raw frame is inverted, same as LIBERO")
    ap.add_argument("--cam", default=BASE_CAM)
    ap.add_argument("--min-px", type=int, default=12)
    ap.add_argument("--dump", default=None,
                    help="write <prefix>_raw.png and <prefix>_rot.png plus box "
                         "overlays, then exit without querying any policy")
    ap.add_argument("--policy", default="fake", choices=("fake", "blind", "pi05"))
    ap.add_argument("--style", default="paligemma", dest="fmt")
    ap.add_argument("--language", action="store_true",
                    help="run the language control instead of the sweep")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    env, obs = build_env(a.task, res=a.res, seed=a.seed,
                         layout=a.layout, style=a.style_id, cam=a.cam)
    model = env.sim.model
    instruction = env.get_ep_meta().get("lang", "") or a.task
    size = (a.res, a.res)

    rgb = view(obs[f"{a.cam}_image"], a.rotate)
    seg = view(segmentation(env, obs, a.cam, a.res), a.rotate)
    boxes = object_boxes(env, seg, min_px=a.min_px)

    print(f"task        {a.task}")
    print(f"instruction {instruction!r}")
    print(f"rotate      {a.rotate}  (measured: raw frame is inverted)")
    print(f"objects     {len(boxes)}")
    for n, (x0, y0, x1, y1) in sorted(boxes.items()):
        print(f"  {n:<40} x[{x0:>3},{x1:>3}] y[{y0:>3},{y1:>3}]")

    if a.dump:
        import imageio
        os.makedirs(os.path.dirname(os.path.abspath(a.dump)) or ".",
                    exist_ok=True)
        raw = np.ascontiguousarray(np.asarray(obs[f"{a.cam}_image"]))
        rot = np.ascontiguousarray(raw[::-1, ::-1])
        imageio.imwrite(f"{a.dump}_raw.png", raw)
        imageio.imwrite(f"{a.dump}_rot.png", rot)
        imageio.imwrite(f"{a.dump}_boxes.png", draw_overlay(rgb, boxes))
        print(f"\ndumped -> {a.dump}_raw.png / _rot.png / _boxes.png")
        print("LOOK AT THEM. Whichever is upright is the orientation pi-0.5 "
              "must be fed,\nand --rotate must be set to match before any "
              "number here means anything.")
        env.close()
        return

    images = [rgb, view(obs[f"{WRIST_CAM}_image"], a.rotate)]
    state = np.concatenate((obs["robot0_eef_pos"],
                            quat2axisangle(obs["robot0_eef_quat"]),
                            np.asarray(obs["robot0_gripper_qpos"]).ravel()[:2]))

    if a.policy in ("fake", "blind"):
        policy = FakePolicy(size, instruction, blind=(a.policy == "blind"))
    else:
        policy = Pi05Policy(host=a.host, port=a.port)

    if a.language:
        run_language(policy, images, state, instruction, boxes, a)
    else:
        sweep_and_report(policy, images, state, boxes, instruction, size,
                         a.fmt, a.repeats, policy_name=a.policy,
                         json_path=a.json,
                         extra={"benchmark": "robocasa", "task": a.task,
                                "rotate": a.rotate})
    env.close()


def run_language(policy, images, state, instruction, boxes, a):
    """The control from probe_language.py, on a RoboCasa frame.

    Reimplemented here rather than imported because probe_language builds its
    own LIBERO env at import time; only the logic is shared, and it is short.
    """
    from xembody.grounding import pretty

    words = set(instruction.lower().replace(",", " ").split())
    wrong_obj = next((pretty(n) for n in sorted(boxes)
                      if not any(t in words for t in pretty(n).split())), None)
    prompts = {
        "wrong": f"pick up the {wrong_obj} and put it in the sink" if wrong_obj
                 else None,
        "unrelated": "open the top drawer of the cabinet",
        "nonsense": "qwerty asdf zxcv plugh xyzzy",
    }
    base = []
    for _ in range(max(2, a.repeats)):
        policy.reset()
        base.append(np.asarray(policy.act(images, instruction, state), float))
    floor = noise_floor(base)
    print(f"\nnoise floor  n={floor['n']}  cos_dir>={floor['cos_dir']:.6f}  "
          f"d_dir<={floor['d_dir']:.6f}\n")
    print(f"{'prompt':<12} {'cos_dir':>9} {'deg':>7} {'d_dir':>9}  verdict")
    live = False
    for name, text in prompts.items():
        if not text:
            continue
        policy.reset()
        e = compare(base[0],
                    np.asarray(policy.act(images, text, state), float))
        v, deg = verdict(e, floor)
        live = live or v != "inert"
        print(f"{name:<12} {e['cos_dir']:>9.4f} {deg:>7.1f} {e['d_dir']:>9.4f}"
              f"  {v}")
    print("\nlanguage channel: " + ("OPEN -- a grounding null here would be "
                                    "about grounding."
                                    if live else
                                    "CLOSED -- any grounding result on this "
                                    "frame is VOID."))


if __name__ == "__main__":
    main()
