"""Is the policy's language channel open at all? The control for every null.

A grounding probe that reports "no effect" has proven nothing until this has
run. Appending boxes to a prompt can only change behaviour if the PROMPT can
change behaviour, so if a deliberately wrong instruction also moves nothing,
the grounding null is a fact about the checkpoint's use of text, not about
grounding, and no serialisation format will rescue it.

This is the same measurement shape as the condition sweep -- same frame, same
noise floor, same verdict bands -- with the prompt varied instead of the boxes.

FOUR PROMPTS, INCREASING IN WRONGNESS:

  correct    the task's own instruction. The baseline, and the floor.
  wrong      names a DIFFERENT object that is present in the scene. The most
             informative of the three: it is in-distribution English about a
             real object, so a policy that follows language should retarget,
             and one that has collapsed onto "do the task this scene affords"
             will not.
  unrelated  a plausible instruction for a different task entirely.
  nonsense   not English. If even this does nothing, the text is being ignored
             rather than misread, and the two failures are worth separating.

WHAT A NULL HERE MEANS, AND WHAT IT DOES NOT. It does not mean the checkpoint is
broken -- a policy fine-tuned on one scene layout can solve it from vision alone
and score well while ignoring text. It does mean that any conclusion of the form
"pi-0.5 does not read grounding tokens" is unsupported by this setup, because
the channel those tokens travel on was never shown to carry anything.

Run (needs `bash scripts/serve_pi05.sh`):
  MUJOCO_GL=egl PYTHONPATH=. ./.venv/bin/python examples/probe_language.py \
      --suite libero_10 --task-id 0 --repeats 8
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

from probe_conditions import Pi05Policy, quat2axisangle
from probe_grounding import build_env, movable_bodies, oracle_boxes, policy_view
from xembody.grounding import pretty
from xembody.probe import compare, noise_floor, verdict

UNRELATED = "open the top drawer of the cabinet"
NONSENSE = "qwerty asdf zxcv plugh xyzzy"


def wrong_instruction(boxes, instruction):
    """An instruction naming a visible object the real one does not name.

    Built from the scene rather than hardcoded, so it stays valid on any task:
    a wrong instruction that names an ABSENT object tests something else
    entirely (whether the policy can be confused by an impossible request), and
    that is not the question here.
    """
    words = set(instruction.lower().replace(",", " ").split())
    for name in sorted(boxes):
        label = pretty(name)
        if not any(t in words for t in label.split()):
            return f"pick up the {label} and place it in the basket", label
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    env, _, task, obs = build_env(a.suite, a.task_id, res=a.res)
    boxes = oracle_boxes(env.sim.model,
                         policy_view(obs["agentview_segmentation_element"]),
                         movable_bodies(env.sim.model))
    images = [policy_view(obs[f"{c}_image"])
              for c in ("agentview", "robot0_eye_in_hand")]
    state = np.concatenate((obs["robot0_eef_pos"],
                            quat2axisangle(obs["robot0_eef_quat"]),
                            np.asarray(obs["robot0_gripper_qpos"]).ravel()))

    wrong, wrong_obj = wrong_instruction(boxes, task.language)
    prompts = {"correct": task.language, "wrong": wrong,
               "unrelated": UNRELATED, "nonsense": NONSENSE}
    prompts = {k: v for k, v in prompts.items() if v}

    policy = Pi05Policy(host=a.host, port=a.port)
    print(f"task       {task.language!r}")
    print(f"wrong      {wrong!r}   (names {wrong_obj!r}, present in scene)\n")

    base = []
    for _ in range(max(2, a.repeats)):
        policy.reset()
        base.append(np.asarray(policy.act(images, task.language, state), float))
    floor = noise_floor(base)
    print(f"noise floor  n={floor['n']}  cos_dir>={floor['cos_dir']:.6f}  "
          f"d_dir<={floor['d_dir']:.6f}\n")

    print(f"{'prompt':<12} {'cos_dir':>9} {'deg':>7} {'d_dir':>9} "
          f"{'grip':>6}  verdict")
    rows = {}
    for name, text in prompts.items():
        if name == "correct":
            continue
        policy.reset()
        chunk = np.asarray(policy.act(images, text, state), float)
        e = compare(base[0], chunk)
        v, deg = verdict(e, floor)
        e["verdict"], e["deg"], e["prompt"] = v, deg, text
        rows[name] = e
        print(f"{name:<12} {e['cos_dir']:>9.4f} {deg:>7.1f} {e['d_dir']:>9.4f} "
              f"{e['grip_agree']:>6.2f}  {v}")

    open_channel = any(r["verdict"] != "inert" for r in rows.values())
    print("\nlanguage channel: " + (
        "OPEN -- the prompt changes behaviour, so a grounding null is a fact "
        "about\n                  the FORMAT or the content, not about the "
        "channel."
        if open_channel else
        "CLOSED on this frame -- no instruction, however wrong, moved the\n"
        "                  policy outside its own sampling noise. Any grounding "
        "null\n                  measured here is uninformative about grounding."))

    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        json.dump({"task": task.language, "floor": floor, "rows": rows,
                   "open_channel": open_channel}, open(a.json, "w"), indent=2)
        print(f"\njson -> {a.json}")
    env.close()


if __name__ == "__main__":
    main()
