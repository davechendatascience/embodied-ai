"""The grounding probe on RoboCasa, driven by GR00T N1.6. Runs in robocasa_uv.

WHY THIS FILE EXISTS SEPARATELY FROM probe_robocasa.py.
That one drives pi-0.5 through openpi and builds its env with
`robocasa.utils.env_utils.create_env` on robocasa master. This one drives
GR00T through its zmq server and builds its env through GR00T's gymnasium
registration on the squarefk fork -- different env ids
(`robocasa_panda_omron/PnPCounterToSink_PandaOmron_Env`), different observation
schema, different venv. The MEASUREMENT is shared: same conditions, same noise
floor, same verdict bands out of xembody.

WHY GR00T AND NOT pi-0.5 HERE. pi05_libero is fine-tuned on LIBERO and is not
competent on RoboCasa -- measured, on three tasks, eight identical queries each,
the baseline disagreed with ITSELF by up to 110 degrees. Nothing can be
detected against that. GR00T-N1.6-3B is evaluated zero-shot on RoboCasa at
66.22% average over 24 tasks. Competence is a precondition for the measurement,
not a bonus.

RUN THE LANGUAGE CONTROL FIRST AND BELIEVE NOTHING WITHOUT IT. `--language`
checks that the prompt can move the policy at all on this frame. If a
deliberately wrong instruction lands inside the sampler's own spread, the
grounding sweep on that frame is void, and reporting its null as "grounding
does not help" would be false.

Run (needs `bash scripts/serve_groot.sh`):
  MUJOCO_GL=egl PYTHONPATH=. \\
    third_party/Isaac-GR00T/gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \\
    examples/probe_groot_robocasa.py --task PnPCounterToSink --language
"""
import argparse
import json
import os
import sys

import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R)
sys.path.insert(0, R + "/examples")

from xembody.groot import LANG_KEY, GrootPolicy
from xembody.boxes import overlay_conditions
from xembody.grounding import build_prompt, pretty
from xembody.probe import compare, noise_floor, verdict

UNRELATED = "open the top drawer of the cabinet"
NONSENSE = "qwerty asdf zxcv plugh xyzzy"


def build_env(task, robot="PandaOmron", seed=0, layout=1, style=1):
    """GR00T's gymnasium RoboCasa env, plus the robosuite env underneath it.

    The wrapper is what the policy talks to (it emits `video.*`, `state.*` and
    the instruction); the inner robosuite env is what the BOXES come from, since
    segmentation and the object registry live there. Both are returned so
    neither has to be dug out of the other at the call site.
    """
    import gymnasium as gym
    import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401  registers ids

    # SEED, LAYOUT AND STYLE MUST GO THROUGH gym.make, NOT reset().
    # RoboCasa samples the kitchen layout, style and object instances from its
    # own RNG at construction. `reset(seed=...)`, `np.random.seed` and
    # `random.seed` all leave it free-running: three consecutive builds of the
    # same task gave "condiment bottle", "corn" and "teapot". Every condition in
    # one run still saw one frozen frame, so a sweep was internally valid -- but
    # nothing was reproducible and no two tasks were matched scenes. Passing
    # these as kwargs pins it.
    env_id = f"robocasa_panda_omron/{task}_{robot}_Env"
    genv = gym.make(env_id, seed=seed, layout_ids=layout, style_ids=style)
    obs, _ = genv.reset(seed=seed)
    return genv, genv.unwrapped.env, obs


def camera_config(genv):
    """(mapped video keys, robosuite camera names) for this embodiment."""
    mapped, cams, _, _ = genv.unwrapped.key_converter.get_camera_config()
    return list(mapped), list(cams)


#: The four orientations a render can differ from its observation by. Named so
#: a calibration result can be stored and replayed instead of recomputed.
FLIPS = {"as-is": (lambda a: a), "vflip": (lambda a: a[::-1]),
         "hflip": (lambda a: a[:, ::-1]), "rot180": (lambda a: a[::-1, ::-1])}


def segmentation(inner, cam, h, w, reference, flip=None):
    """Per-geom segmentation, in the same frame as `reference`.

    THE ORIENTATION IS CALIBRATED, NOT LOOKED UP. The first version of this
    function trusted `IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]` the way
    the pi-0.5 host does. That is wrong here: on robosuite master "opengl" maps
    to +1 (no flip), yet GR00T's wrapper image is the VERTICAL FLIP of a raw
    render -- measured, mean|diff| 1.2 flipped versus 93 unflipped. So the
    segmentation was upside down relative to the image the policy sees, and
    every box named the wrong pixels while looking perfectly plausible.

    The constant differs between robosuite generations, so no constant is
    trusted. The RGB is re-rendered, compared against the wrapper's own image
    under each candidate flip, and whichever wins is applied to the
    segmentation. If none wins clearly this raises, because a silently
    misaligned box is worse than a crash.

    `reference` is the wrapper's image for this camera -- the actual array the
    policy is fed.
    """
    if flip is not None:
        # Orientation already calibrated for this camera. Re-checking per frame
        # is not just wasteful, it is FRAGILE: the residual between the cached
        # observable and a fresh render grows with how many other renders
        # happened in between (measured 3.4 to 25 on the same camera), so a
        # per-frame separation test eventually reports a false ambiguity on a
        # setup that was already proven correct.
        seg = inner.sim.render(camera_name=cam, width=w, height=h,
                               segmentation=True)
        return FLIPS[flip](seg)[:, :, 1]

    ref = np.asarray(reference, dtype=np.int16)
    rgb = np.asarray(inner.sim.render(camera_name=cam, width=w, height=h),
                     dtype=np.int16)
    cands = FLIPS
    errs = {k: float(np.abs(f(rgb) - ref).mean()) for k, f in cands.items()}
    best = min(errs, key=errs.get)
    runner = min((k for k in errs if k != best), key=errs.get)
    # SEPARATION, NOT IDENTITY. A fresh sim.render never matches the cached
    # observable exactly -- measured 3.4 mean against a byte-identical res512
    # key, and up to ~18 in other call contexts. Demanding near-identity
    # rejected correct alignments. What must be caught is a WRONG FLIP, and a
    # wrong flip costs 60-86 against 3-18, so a 2x margin separates them with
    # room to spare while still failing on a genuine ambiguity.
    if errs[best] > 0.5 * errs[runner]:
        raise RuntimeError(
            f"cannot align a re-render of {cam} with the wrapper's image: "
            f"mean|diff| {errs}. Boxes would name the wrong pixels.")
    seg = inner.sim.render(camera_name=cam, width=w, height=h,
                           segmentation=True)
    return cands[best](seg)[:, :, 1]


def calibrate_flip(inner, cam, h, w, reference):
    """Which flip maps a fresh render onto the wrapper's image. Do this ONCE."""
    ref = np.asarray(reference, dtype=np.int16)
    rgb = np.asarray(inner.sim.render(camera_name=cam, width=w, height=h),
                     dtype=np.int16)
    errs = {k: float(np.abs(f(rgb) - ref).mean()) for k, f in FLIPS.items()}
    best = min(errs, key=errs.get)
    runner = min((k for k in errs if k != best), key=errs.get)
    if errs[best] > 0.5 * errs[runner]:
        raise RuntimeError(f"ambiguous orientation for {cam}: {errs}")
    return best


def object_boxes(inner, seg, min_px=12):
    """RoboCasa's registered objects, labelled with `get_obj_lang`.

    A kitchen is full of articulated FIXTURES, so "a body with a joint" returns
    cabinet doors and the robot's base and misses the target. `env.objects`
    already distinguishes the manipulation target (`obj`) from the distractors
    the benchmark places on purpose (`distr_*`).
    """
    from xembody.boxes import body_name

    model = inner.sim.model
    seg = np.asarray(seg)
    roots = {}
    for role, obj in inner.objects.items():
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
            label = inner.get_obj_lang(obj_name=role)
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


def mean_chunk(policy, obs, prompt, draws):
    """Average `draws` samples for one prompt.

    GR00T's action head is stochastic: measured on PnPCounterToSink, eight
    IDENTICAL queries produced chunks up to 150 degrees apart (floor cos_dir
    -0.87). Comparing single draws against that is hopeless -- the sampler
    dominates any prompt effect. Averaging k draws shrinks the sampling term as
    1/sqrt(k) while leaving a real prompt effect untouched, so the comparison
    is between MEANS and the floor is measured the same way.
    """
    return np.mean([policy.act_obs(obs, prompt) for _ in range(draws)], axis=0)


def run_language(policy, obs, instruction, boxes, repeats, draws=8):
    """Can the prompt move this policy on this frame? The gate on everything."""
    words = set(instruction.lower().replace(",", " ").split())
    wrong_obj = next((n for n in sorted(boxes)
                      if not any(t in words for t in pretty(n).split())), None)
    prompts = {
        "wrong": f"pick up the {wrong_obj} and put it in the sink" if wrong_obj
                 else None,
        "unrelated": UNRELATED,
        "nonsense": NONSENSE,
    }
    # The floor is the spread between INDEPENDENT MEANS of the same prompt, so
    # it is built the same way every condition is. Comparing a mean-of-k against
    # a floor built from single draws would understate the noise by sqrt(k) and
    # manufacture effects.
    base = [mean_chunk(policy, obs, instruction, draws)
            for _ in range(max(2, repeats))]
    floor = noise_floor(base)
    print(f"\nnoise floor  n={floor['n']} means of {draws} draws  "
          f"cos_dir>={floor['cos_dir']:.6f}  d_dir<={floor['d_dir']:.6f}\n")
    print(f"{'prompt':<12} {'cos_dir':>9} {'deg':>7} {'d_dir':>9}  verdict")
    rows, live = {}, False
    for name, text in prompts.items():
        if not text:
            continue
        e = compare(base[0], mean_chunk(policy, obs, text, draws))
        v, deg = verdict(e, floor)
        e["verdict"], e["deg"], e["prompt"] = v, deg, text
        rows[name] = e
        live = live or v != "inert"
        print(f"{name:<12} {e['cos_dir']:>9.4f} {deg:>7.1f} {e['d_dir']:>9.4f}"
              f"  {v}")
    print("\nlanguage channel: " + (
        "OPEN -- a grounding null here would be about grounding."
        if live else
        "CLOSED -- any grounding result on this frame is VOID."))
    return {"floor": floor, "rows": rows, "open": live}


def run_sweep(policy, obs, instruction, boxes, size, style, repeats,
              draws=8):
    """The five conditions, on a frame whose language channel is known open."""
    from probe_conditions import conditions, pick_distractor, pick_target

    target = pick_target(boxes, instruction)
    distractor = pick_distractor(boxes, target, instruction)
    conds = conditions(boxes, target, distractor)
    print(f"target      {target}\ndistractor  {distractor}\n")

    # Means of `draws`, matching run_language: the action head is stochastic and
    # single draws put the floor at cos_dir -0.87, where nothing is detectable.
    base = [mean_chunk(policy, obs, instruction, draws)
            for _ in range(max(2, repeats))]
    floor = noise_floor(base)
    print(f"noise floor  n={floor['n']}  cos_dir>={floor['cos_dir']:.6f}  "
          f"d_dir<={floor['d_dir']:.6f}\n")
    chunks = {"none": base[0]}
    for name, bx in conds.items():
        if name == "none":
            continue
        chunks[name] = mean_chunk(
            policy, obs, build_prompt(instruction, bx, size, style), draws)
    print(f"{'condition':<12} {'cos_dir':>9} {'deg':>7} {'d_dir':>9}  verdict")
    rows = {}
    for name, chunk in chunks.items():
        if name == "none":
            continue
        e = compare(chunks["none"], chunk)
        v, deg = verdict(e, floor)
        e["verdict"], e["deg"] = v, deg
        rows[name] = e
        print(f"{name:<12} {e['cos_dir']:>9.4f} {deg:>7.1f} {e['d_dir']:>9.4f}"
              f"  {v}")
    causal = None
    if "mislabel" in chunks and "target" in chunks:
        causal = compare(chunks["target"], chunks["mislabel"])
        v, deg = verdict(causal, floor)
        causal["verdict"], causal["deg"] = v, deg
        print(f"\ncausal test   mislabel vs target: "
              f"cos_dir={causal['cos_dir']:.4f} ({deg:.1f} deg) -> {v}")
    return {"floor": floor, "effects": rows, "causal": causal,
            "target": target, "distractor": distractor}


def run_pixels(policy, obs, img_key, rgb, instruction, boxes, target,
               distractor, repeats, draws):
    """The pixel-injection channel: the mark goes in the IMAGE, not the prompt.

    The literature this was built to test says serialized coordinates are the
    weakest channel and pixel burn-in is where the causal steering evidence
    lives (Point-VLA on pi-0.5: box-as-text 70/37/83/73 vs box-as-overlay
    86.7/80/94.3/95.0; RoboGround reproduces the ordering; TraceVLA +2.4% text
    vs +6.4% pixels).

    THE CAUSAL TEST IS DIFFERENT HERE, and it is cleaner. An overlay carries
    LOCATION, not IDENTITY -- there is no label to falsify, so `mislabel` does
    not exist and "box on the distractor" IS the relocation. The test is
    `target` versus `distractor`: identical instruction, identical mark, moved
    to another object. If the policy follows the mark they diverge. Nothing
    about the prompt changes at all, which removes prompt length and token
    count as explanations in a way the serialized channel never could.

    CAVEAT THIS RUN CANNOT ESCAPE: every published overlay success FINE-TUNED on
    overlaid frames. This measures whether an untrained policy reacts to a mark
    it has never seen, which is a different and unanswered question. A null here
    is consistent with Point-VLA rather than a refutation of it.
    """
    variants = overlay_conditions(rgb, boxes, target, distractor)
    print(f"\npixel channel: {sorted(variants)}  (prompt held constant)")

    def query(image):
        o = dict(obs)
        o[img_key] = image
        return mean_chunk(policy, o, instruction, draws)

    base = [query(variants["none"]) for _ in range(max(2, repeats))]
    floor = noise_floor(base)
    print(f"noise floor  n={floor['n']} means of {draws} draws  "
          f"cos_dir>={floor['cos_dir']:.6f}  d_dir<={floor['d_dir']:.6f}\n")
    print(f"{'condition':<12} {'cos_dir':>9} {'deg':>7} {'d_dir':>9}  verdict")
    chunks, rows = {"none": base[0]}, {}
    for name in ("target", "all", "distractor"):
        if name not in variants:
            continue
        chunks[name] = query(variants[name])
        e = compare(chunks["none"], chunks[name])
        v, deg = verdict(e, floor)
        e["verdict"], e["deg"] = v, deg
        rows[name] = e
        print(f"{name:<12} {e['cos_dir']:>9.4f} {deg:>7.1f} {e['d_dir']:>9.4f}"
              f"  {v}")
    causal = None
    if "target" in chunks and "distractor" in chunks:
        causal = compare(chunks["target"], chunks["distractor"])
        v, deg = verdict(causal, floor)
        causal["verdict"], causal["deg"] = v, deg
        print(f"\ncausal test   mark on target vs mark on distractor: "
              f"cos_dir={causal['cos_dir']:.4f} ({deg:.1f} deg) -> {v}")
        print("              redirected/ambiguous => the mark steers; "
              "inert => the mark is noticed but not followed")
    return {"floor": floor, "effects": rows, "causal": causal,
            "channel": "pixels"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PnPCounterToSink")
    ap.add_argument("--robot", default="PandaOmron")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layout", type=int, default=1)
    ap.add_argument("--style-id", dest="style_id", type=int, default=1)
    ap.add_argument("--draws", type=int, default=8,
                    help="samples averaged per condition; the action head "
                         "is stochastic, so 1 is not enough")
    ap.add_argument("--cam-index", type=int, default=-1,
                    help="which of the embodiment's cameras to ground in")
    ap.add_argument("--style", default="paligemma")
    ap.add_argument("--language", action="store_true")
    ap.add_argument("--inject", default="prompt",
                    choices=("prompt", "pixels"))
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--min-px", type=int, default=12)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    genv, inner, obs = build_env(a.task, a.robot, a.seed, a.layout, a.style_id)
    mapped, cams = camera_config(genv)
    instruction = obs.get(LANG_KEY) or inner.get_ep_meta().get("lang", "")
    # PICK THE VIEW THAT ACTUALLY SEES THE OBJECTS.
    # Fixing a camera index blind cost two runs already: on both RoboCasa forks
    # the default side view frequently frames neither the target nor its
    # distractors, and grounding boxes for objects the policy cannot see is a
    # different experiment from the one intended. Score every camera by how many
    # registered objects are visible in it and take the best.
    idx = a.cam_index
    if idx < 0:
        scored = []
        for i, (mk, cm) in enumerate(zip(mapped, cams)):
            im = np.asarray(obs[mk])
            hh, ww = im.shape[:2]
            b = object_boxes(inner, segmentation(inner, cm, hh, ww, im), a.min_px)
            scored.append((len(b), i, cm, b))
        scored.sort(key=lambda t: -t[0])
        print("camera object counts: "
              + ", ".join(f"{c}={n}" for n, _, c, _ in scored))
        idx = scored[0][1]
    cam = cams[idx]
    img = np.asarray(obs[mapped[idx]])
    h, w = img.shape[:2]
    boxes = object_boxes(inner, segmentation(inner, cam, h, w, img), a.min_px)

    print(f"task        {a.task}  ({a.robot})")
    print(f"instruction {instruction!r}")
    print(f"cameras     {list(zip(mapped, cams))}")
    print(f"grounding in {cam}  {w}x{h}")
    print(f"objects     {len(boxes)}  {dict(boxes)}")

    policy = GrootPolicy(host=a.host, port=a.port)
    if a.language:
        out = run_language(policy, obs, instruction, boxes, a.repeats, a.draws)
    elif a.inject == "pixels":
        from probe_conditions import pick_distractor, pick_target
        tgt = pick_target(boxes, instruction)
        dis = pick_distractor(boxes, tgt, instruction)
        print(f"target      {tgt}\ndistractor  {dis}")
        out = run_pixels(policy, obs, mapped[idx], img, instruction, boxes,
                         tgt, dis, a.repeats, a.draws)
    else:
        out = run_sweep(policy, obs, instruction, boxes, (w, h), a.style,
                        a.repeats, a.draws)

    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        json.dump({"task": a.task, "instruction": instruction,
                   "boxes": {k: list(v) for k, v in boxes.items()},
                   "result": out}, open(a.json, "w"), indent=2, default=float)
        print(f"\njson -> {a.json}")
    genv.close()


if __name__ == "__main__":
    main()
