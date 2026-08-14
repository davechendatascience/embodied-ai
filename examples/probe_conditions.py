"""Does a box in the prompt change what the policy does? Five conditions.

Increment 3. No rollout: one observation, several prompts, the policy queried
once per prompt. Nothing is executed, so nothing can be confounded by the arm
having gone somewhere different.

THE CONDITIONS, and which question each answers.

  none        instruction alone. The baseline, byte-identical to normal use.
  target      a box on the object the instruction names.
              "Does correct grounding change anything?"
  all         boxes on every visible object, correctly labelled.
              What a real detector would emit. Correct grounding plus clutter.
  distractor  a box on some OTHER object, labelled as itself.
              "Does an irrelevant box perturb it?" -- separates 'reads the box'
              from 'reacts to a longer prompt'.
  mislabel    the DISTRACTOR's box carrying the TARGET's label.
              The causal test, and the only condition that can prove anything.

WHY `mislabel` IS THE EXPERIMENT AND THE OTHERS ARE CONTEXT.
A success bump under `target` is weak evidence: prompt length changed, token
count changed, and on a benchmark the policy already solves there is no headroom
for it to show up in anyway. But if the box is on the tomato sauce, the label
says alphabet soup, and the arm goes to the tomato sauce, then the box is being
read and acted on. That is a causal claim, it costs one extra condition, and it
is measurable on a single frame with no rollout at all.

A NULL RESULT HERE HAS A PREREQUISITE. If nothing moves the policy, check the
serialisation format against the checkpoint's training data BEFORE concluding
the channel is closed -- a format the backbone never saw produces exactly the
signature of a policy that ignores grounding. See xembody/grounding.py.

TWO TRAPS IN THE POLICY INTERFACE, both of which silently return stale numbers:

  1. Chunked inference. A policy that queries its server only when
     `step % chunk == 0` will REPLAY a cached chunk for every other step. A
     sweep that increments `step` measures the cache, not the prompt. Every
     query here passes step=0.
  2. Carried state. Action ensemblers, image history and sticky-gripper latches
     make query N depend on query N-1, so conditions would contaminate each
     other in the order they happened to run. `reset()` before every query, and
     the conditions are also evaluated in a fixed order so any residue is at
     least reproducible.

SELF-TEST BEFORE MEASUREMENT. `--policy fake` is a stand-in that provably reads
boxes; `--policy blind` provably ignores them. The harness must report a causal
channel for the first and an inert one for the second, and `--policy blind
--noise 0.02` must still report inert despite cosines near -0.6. Run all three
after touching anything in here: a harness that fails them cannot produce
evidence about a real policy.

Run:
  # self-test, no GPU and no server
  MUJOCO_GL=egl PYTHONPATH=. ./.venv/bin/python examples/probe_conditions.py \
      --suite libero_10 --task-id 0 --policy fake
  MUJOCO_GL=egl PYTHONPATH=. ./.venv/bin/python examples/probe_conditions.py \
      --suite libero_10 --task-id 0 --policy blind --noise 0.02 --repeats 6

  # the measurement -- needs `bash scripts/serve_pi05.sh` in another shell
  MUJOCO_GL=egl PYTHONPATH=. ./.venv/bin/python examples/probe_conditions.py \
      --suite libero_10 --task-id 0 --policy pi05 --style paligemma
"""
import argparse
import json
import os
import sys

import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R)
sys.path.insert(0, R + "/examples")
sys.path.insert(0, R + "/third_party/LIBERO")

from xembody.boxes import movable_bodies, oracle_boxes, policy_view
from xembody.grounding import (build_prompt, decode_paligemma_items,
                               pretty)
from xembody.probe import angle_deg, compare, exceeds, noise_floor, verdict

#: Fixed evaluation order. Not alphabetical, not dict order -- reproducible.
ORDER = ("none", "target", "all", "distractor", "mislabel")


def quat2axisangle(quat):
    """robosuite `[x, y, z, w]` quaternion -> axis-angle, matching openpi.

    Reproduced from openpi's `examples/libero/main.py`, which took it from
    robosuite. It is here rather than imported because the STATE VECTOR'S
    CONVENTION IS PART OF THE POLICY'S CONTRACT: robosuite publishes several
    rotation representations, pi-0.5 was trained against this one, and a state
    built with a w-first quaternion or a different branch at the singularity is
    a plausible-looking vector that means something else.
    """
    import math

    quat = np.asarray(quat, float).copy()
    quat[3] = float(np.clip(quat[3], -1.0, 1.0))
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)                 # (close to) zero rotation
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def pick_target(boxes, instruction):
    """Which visible object the instruction refers to.

    Scores each object by how many of its label's words appear in the
    instruction, so `alphabet soup` beats `basket` on "put both the alphabet
    soup and the tomato sauce in the basket" -- both match, the first matches
    more. Ties break on the longer label, which prefers the more specific
    referent.

    THIS IS A HEURISTIC OVER STRINGS AND IT IS THE THING THE GROUNDING LAYER IS
    MEANT TO REPLACE. It is used here only to CHOOSE WHICH BOX TO SHOW, never to
    score an outcome. Its known failure -- a word appearing in two body names --
    is why the chosen target is printed on every run: check it.
    """
    best, best_score = None, (0, 0)
    words = set(instruction.lower().replace(",", " ").split())
    for name in sorted(boxes):
        label = pretty(name)
        score = (sum(w in words for w in label.split()), len(label))
        if score > best_score:
            best, best_score = name, score
    return best


def pick_distractor(boxes, target, instruction):
    """A visible object that is NOT the target and IS NOT NAMED in the task.

    Two filters, both load-bearing:

    Unnamed. On `libero_10/0` -- "put both the alphabet soup and the tomato
    sauce in the basket" -- three of the eight objects are named. Using one of
    them as the distractor makes `mislabel` uninterpretable: moving toward the
    basket is what the task asks for anyway, so following the false box and
    ignoring it look the same. The distractor must be an object the policy has
    no instructed reason to approach.

    Farthest. If the two candidates sit side by side, a policy that went to the
    wrong one is indistinguishable from one that went to the right one with a
    small error. Maximising separation makes the metric decisive.

    Returns None when no unnamed object is visible -- in which case this task
    cannot support the causal condition and the sweep says so rather than
    quietly substituting a named one.
    """
    if target is None or len(boxes) < 2:
        return None
    words = set(instruction.lower().replace(",", " ").split())

    def centre(b):
        return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])

    unnamed = [n for n in sorted(boxes)
               if n != target
               and not any(t in words for t in pretty(n).split())]
    if not unnamed:
        return None
    ct = centre(boxes[target])
    return max(unnamed, key=lambda n: np.linalg.norm(centre(boxes[n]) - ct))


def conditions(boxes, target, distractor):
    """condition name -> the box dict shown under it. `None` means no grounding.

    `mislabel` keys the DISTRACTOR's box by the TARGET's name, so the serialised
    label reads as the target while the coordinates point elsewhere. That is the
    whole trick, and it is one line, because labels are derived from keys.
    """
    out = {"none": None}
    if target is not None:
        out["target"] = {target: boxes[target]}
    out["all"] = dict(boxes)
    if distractor is not None:
        out["distractor"] = {distractor: boxes[distractor]}
        if target is not None:
            out["mislabel"] = {target: boxes[distractor]}
    return {k: out[k] for k in ORDER if k in out}


class FakePolicy:
    """Deterministic stand-in that DOES read the boxes. For testing the harness.

    It aims at the centre of whichever box in the prompt carries a label the
    instruction mentions, and at a fixed point when the prompt has no boxes.
    That makes it a policy with a known, correct answer under every condition:

      none/target/all   aims at the target
      mislabel          aims at the distractor  <- the sweep MUST detect this

    So a run against this policy checks the thing that is otherwise impossible
    to check -- that the metric can see an effect that is definitely there, and
    reports no effect when there is definitely none. A harness that cannot pass
    this is not evidence about any real policy.
    """

    def __init__(self, size, instruction, chunk=8, noise=0.0, seed=0,
                 blind=False):
        self.size = size
        self.instruction = instruction
        #: `blind` ignores the prompt entirely. It is the NEGATIVE half of the
        #: self-test: a harness that cannot report "inert" for a policy that
        #: provably reads nothing will report a causal channel for any policy.
        self.blind = blind
        self.chunk = chunk
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def act(self, images, prompt, state):
        del images, state
        w, h = self.size
        aim = np.array([0.5, 0.5])
        # Match on the LABEL, not on position in the list. Taking the first box
        # would make `all` agree with `target` whenever sorting happened to put
        # the target first, and would make `distractor` and `mislabel`
        # numerically identical -- both artefacts of the stand-in, indis-
        # tinguishable in the output from findings about a real policy.
        words = set(self.instruction.lower().replace(",", " ").split())
        best, best_score = None, 0
        for box, label in ([] if self.blind
                           else decode_paligemma_items(prompt, self.size)):
            score = sum(t in words for t in label.split())
            if score > best_score:
                best, best_score = box, score
        if best is not None:
            x0, y0, x1, y1 = best
            aim = np.array([(x0 + x1) / 2.0 / w, (y0 + y1) / 2.0 / h])
        step = np.zeros(7)
        step[:2] = (aim - 0.5) / self.chunk
        step[2] = -0.05 / self.chunk
        step[6] = -1.0
        out = np.tile(step, (self.chunk, 1))
        if self.noise:
            out = out + self.rng.normal(0, self.noise, out.shape)
        return out


class Pi05Policy:
    """pi-0.5 behind openpi's websocket policy server.

    Payload taken verbatim from openpi's own `examples/libero/main.py`, which is
    the only description of this contract that cannot drift from the checkpoint:

        observation/image        agentview, rotated 180, resized-with-pad to 224
        observation/wrist_image  eye-in-hand, same treatment
        observation/state        eef_pos(3) + axis-angle(3) + gripper_qpos(2)
        prompt                   the task string -- and where grounding goes

    THE ROTATION IS NOT OURS. openpi applies `[::-1, ::-1]` with the comment
    "rotate 180 degrees to match train preprocessing". This project's harness
    applies the same expression for its own reasons, and `probe_grounding.
    policy_view` is that expression. They agree, which is why boxes measured in
    our frame are boxes in pi-0.5's frame -- but they agree by coincidence of
    two independent decisions, so if either moves, the boxes silently stop
    describing the image the model sees.

    RESIZING DOES NOT MOVE THE BOXES. Coordinates are serialised normalised (or
    binned), never in pixels, so a 256 -> 224 rescale of a SQUARE image is
    exactly the identity on them. `resize_with_pad` would letterbox a
    non-square source, which WOULD shift them; LIBERO renders square, and this
    asserts that rather than trusting it.

    NO STATE IS CARRIED. openpi's server is stateless per `infer` -- one call in,
    one full action chunk out, and `WebsocketClientPolicy.reset()` is a no-op.
    Both traps in the module docstring are therefore inapplicable to this policy
    specifically. They are still enforced, because the next policy may differ.
    """

    def __init__(self, host="0.0.0.0", port=8000, resize=224):
        from openpi_client import image_tools, websocket_client_policy
        self._tools = image_tools
        self.resize = resize
        self.client = websocket_client_policy.WebsocketClientPolicy(host, port)
        self.meta = self.client.get_server_metadata()

    def reset(self):
        self.client.reset()

    def _prep(self, img):
        img = np.asarray(img)
        assert img.shape[0] == img.shape[1], (
            f"non-square source {img.shape[:2]}: resize_with_pad will letterbox "
            "it and every serialised box coordinate shifts")
        return self._tools.convert_to_uint8(
            self._tools.resize_with_pad(img, self.resize, self.resize))

    def act(self, images, prompt, state):
        base, wrist = images[0], images[1]
        out = self.client.infer({
            "observation/image": self._prep(base),
            "observation/wrist_image": self._prep(wrist),
            "observation/state": np.asarray(state, np.float32),
            "prompt": str(prompt),
        })
        return np.asarray(out["actions"], float)


def run(policy, images, state, instruction, conds, size, style, repeats):
    """Query every condition, plus the baseline `repeats` times. Returns raw."""
    chunks, prompts = {}, {}
    base = []
    for _ in range(max(1, repeats)):
        policy.reset()
        p = build_prompt(instruction, None, size, None)
        base.append(np.asarray(policy.act(images, p, state), float))
    prompts["none"], chunks["none"] = p, base[0]

    for name, boxes in conds.items():
        if name == "none":
            continue
        policy.reset()
        p = build_prompt(instruction, boxes, size, style)
        prompts[name] = p
        chunks[name] = np.asarray(policy.act(images, p, state), float)
    return chunks, prompts, base


def sweep_and_report(policy, images, state, boxes, instruction, size, style,
                     repeats, policy_name="", json_path=None, extra=None):
    """The whole measurement, given a frame. Host-agnostic on purpose.

    LIBERO and RoboCasa cannot share an interpreter -- robosuite 1.4.1 versus
    master -- so everything downstream of "here is an image, a state and a set
    of boxes" has to live somewhere neither of them owns. Both hosts call this
    with their own env built in their own venv, and get the identical
    conditions, floor and verdict bands. A comparison between two benchmarks
    measured by two slightly different scripts is not a comparison.
    """
    target = pick_target(boxes, instruction)
    distractor = pick_distractor(boxes, target, instruction)
    conds = conditions(boxes, target, distractor)

    print(f"task        {instruction!r}")
    print(f"objects     {len(boxes)}  "
          f"({', '.join(pretty(n) for n in sorted(boxes))})")
    print(f"target      {target}   <- ORACLE, from a string heuristic. Check it.")
    print(f"distractor  {distractor}")
    print(f"style       {style}   policy={policy_name}\n")

    chunks, prompts, base = run(policy, images, state, instruction, conds,
                                size, style, repeats)
    floor = noise_floor(base)
    det = floor["deterministic"]
    print(f"noise floor  n={floor['n']}  cos_dir>={floor['cos_dir']:.6f}  "
          f"d_dir<={floor['d_dir']:.6f}"
          f"  {'(deterministic)' if det else '(stochastic)'}\n")

    print(f"{'condition':<12} {'cos_dir':>9} {'deg':>7} {'d_dir':>9} "
          f"{'grip':>6}  verdict")
    rows = {}
    for name in ORDER:
        if name not in chunks or name == "none":
            continue
        e = compare(chunks["none"], chunks[name])
        rows[name] = e
        v, deg = verdict(e, floor)
        e["verdict"], e["deg"] = v, deg
        print(f"{name:<12} {e['cos_dir']:>9.4f} {deg:>7.1f} {e['d_dir']:>9.4f} "
              f"{e['grip_agree']:>6.2f}  {v}")

    # THE CAUSAL TEST IS NOT `mislabel` VS `none`.
    # Against `none`, `mislabel` and `target` both differ simply because both
    # added a box. What separates "the false box was followed" from "the false
    # box was ignored" is `mislabel` vs `target`: coinciding with target means
    # the box was ignored and the policy grounded the referent itself; diverging
    # means it was read and followed to the wrong object. Only the second is
    # evidence for grounding.
    causal = None
    if "mislabel" in chunks and "target" in chunks:
        causal = compare(chunks["target"], chunks["mislabel"])
        v, deg = verdict(causal, floor)
        causal["verdict"], causal["deg"] = v, deg
        print(f"\ncausal test   mislabel vs target: "
              f"cos_dir={causal['cos_dir']:.4f} ({deg:.1f} deg)  "
              f"d_dir={causal['d_dir']:.4f}")
        say = {
            "inert": "false box IGNORED -- inside the noise floor",
            "perturbed": "false box NOT followed -- detectable, but the aim "
                         "barely moved",
            "ambiguous": "AMBIGUOUS -- outside the floor, too small to call a "
                         "redirection",
            "redirected": "false box FOLLOWED -- aim redirected toward the "
                          "distractor",
        }[v]
        print(f"              -> {say}")
    elif "mislabel" not in chunks:
        print("\ncausal test   not run: no unnamed distractor visible in this "
              "task.")

    if json_path:
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        out = {"task": instruction, "target": target, "distractor": distractor,
               "style": style, "policy": policy_name, "floor": floor,
               "effects": rows, "causal": causal, "prompts": prompts}
        out.update(extra or {})
        json.dump(out, open(json_path, "w"), indent=2)
        print(f"\njson -> {json_path}")
    return {"floor": floor, "effects": rows, "causal": causal}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--robot", default="Panda")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--cam", default="agentview")
    ap.add_argument("--style", default="paligemma")
    ap.add_argument("--policy", default="fake",
                    choices=("fake", "blind", "pi05"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--noise", type=float, default=0.0,
                    help="fake policy only: per-element action noise, to check "
                         "that the floor actually suppresses a spread")
    ap.add_argument("--repeats", type=int, default=4,
                    help="baseline re-queries; this IS the noise floor")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    # Imported HERE, not at module scope. probe_robocasa imports this module
    # from a venv that has no LIBERO and no robosuite 1.4.1, and a top-level
    # import would make the shared sweep unimportable there.
    sys.path.insert(0, R + "/third_party/LIBERO")
    from probe_grounding import build_env

    env, _, task, obs = build_env(a.suite, a.task_id, robot=a.robot, res=a.res)
    model = env.sim.model
    seg = policy_view(obs[f"{a.cam}_segmentation_element"])
    boxes = oracle_boxes(model, seg, movable_bodies(model))
    size = (a.res, a.res)
    images = [policy_view(obs[f"{c}_image"])
              for c in ("agentview", "robot0_eye_in_hand")]
    # Real proprioception, in openpi's LIBERO layout: eef_pos(3) +
    # axis-angle(3) + gripper_qpos(2). Zeros would be a silent out-of-
    # distribution input applied EQUALLY to every condition -- which biases
    # nothing in the comparison, but makes each individual chunk a reading of a
    # state the robot is not in. The comparison is the result; the chunks are
    # the evidence for it, so both have to be honest.
    state = np.concatenate((obs["robot0_eef_pos"],
                            quat2axisangle(obs["robot0_eef_quat"]),
                            np.asarray(obs["robot0_gripper_qpos"]).ravel()))
    if a.policy in ("fake", "blind"):
        policy = FakePolicy(size, task.language, noise=a.noise,
                            blind=(a.policy == "blind"))
    else:
        policy = Pi05Policy(host=a.host, port=a.port)

    sweep_and_report(policy, images, state, boxes, task.language, size,
                     a.style, a.repeats, policy_name=a.policy,
                     json_path=a.json, extra={"suite": a.suite,
                                              "task_id": a.task_id})
    env.close()


if __name__ == "__main__":
    main()
