"""Can the policy actually do the task? Closed-loop rollouts, with and without boxes.

The condition sweep measures whether the prompt changes the ACTION. It cannot
say whether the task gets done, and the two are not the same question: a policy
that already succeeds has no headroom for grounding to help into, and a policy
that fails for motor reasons will not be rescued by a better referent.

So this is the denominator for everything in probe_conditions.py. Run it before
concluding that a grounding null matters.

    boxes off   the published setting. Does the policy solve this task at all?
    boxes on    the same rollout with the grounding segment appended to every
                prompt. Does it help, hurt, or do nothing?

WHY BOXES ARE RECOMPUTED EVERY REPLAN, not once at reset. The scene moves --
that is the point of the task -- and a box measured at t=0 describes where the
object WAS. A stale box is a different intervention from a live one, and the
interesting version is the live one because that is what a real detector would
produce.

ORACLE BOXES. Same as everywhere else in this repo: MuJoCo segmentation, not a
detector. This is the CEILING a real detector could reach, so a null here bounds
every detector rather than just the one we happened to pick.

THE LOOP IS openpi's. Query once per `replan` steps, replay the chunk prefix
open loop -- matching examples/libero/main.py, because a different cadence is a
different experiment. `num_steps_wait` exists because LIBERO objects are still
falling for the first few steps and a policy queried mid-drop is being asked
about a scene that is not settled yet.

Run (needs `bash scripts/serve_pi05.sh`):
  MUJOCO_GL=egl PYTHONPATH=. ./.venv/bin/python examples/rollout_pi05.py \
      --suite libero_10 --task-id 0 --episodes 3 --boxes off
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
from xembody.boxes import movable_bodies, oracle_boxes, policy_view
from xembody.grounding import build_prompt

#: LIBERO objects are dropped in at reset and take a few steps to settle.
#: Verbatim from openpi's LIBERO example.
NUM_STEPS_WAIT = 10
DUMMY = [0.0] * 6 + [-1.0]


def rollout(env, policy, task, init_state, boxes_on, style, replan, max_steps,
            res):
    obs = env.set_init_state(init_state)
    done, t = False, 0
    while t < max_steps + NUM_STEPS_WAIT:
        if t < NUM_STEPS_WAIT:
            obs, _, done, _ = env.step(DUMMY)
            t += 1
            continue

        img = policy_view(obs["agentview_image"])
        wrist = policy_view(obs["robot0_eye_in_hand_image"])
        state = np.concatenate((obs["robot0_eef_pos"],
                                quat2axisangle(obs["robot0_eef_quat"]),
                                np.asarray(obs["robot0_gripper_qpos"]).ravel()))

        prompt = task.language
        if boxes_on:
            # Recomputed HERE, inside the loop, from the current frame.
            seg = policy_view(obs["agentview_segmentation_element"])
            boxes = oracle_boxes(env.sim.model, seg, movable_bodies(env.sim.model))
            prompt = build_prompt(task.language, boxes, (res, res), style)

        chunk = policy.act([img, wrist], prompt, state)
        for a in np.asarray(chunk)[:replan]:
            obs, _, done, _ = env.step(a.tolist())
            t += 1
            if done or t >= max_steps + NUM_STEPS_WAIT:
                break
        if done:
            break
    return bool(done)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--boxes", default="off", choices=("off", "on", "both"))
    ap.add_argument("--style", default="paligemma")
    ap.add_argument("--replan", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=520)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    from probe_grounding import build_env
    import torch

    env, suite, task, _ = build_env(a.suite, a.task_id, res=a.res)
    # LIBERO ships init states as pickled tensors; torch 2.6+ defaults
    # weights_only=True and refuses them.
    _load = torch.load
    torch.load = lambda *x, **k: _load(*x, **{**k, "weights_only": False})
    inits = suite.get_task_init_states(a.task_id)
    torch.load = _load

    policy = Pi05Policy(host=a.host, port=a.port)
    modes = ("off", "on") if a.boxes == "both" else (a.boxes,)

    print(f"task     {task.language!r}")
    print(f"episodes {a.episodes}  replan {a.replan}  max_steps {a.max_steps}")
    print(f"boxes    {a.boxes}  (oracle, recomputed every replan)\n")

    results = {}
    for mode in modes:
        ok = []
        for i in range(a.episodes):
            s = rollout(env, policy, task, inits[i % len(inits)], mode == "on",
                        a.style, a.replan, a.max_steps, a.res)
            ok.append(s)
            print(f"  boxes={mode:<3} ep {i}  {'SUCCESS' if s else 'fail'}")
        results[mode] = ok
        print(f"  boxes={mode:<3} -> {sum(ok)}/{len(ok)}\n")

    if len(results) == 2:
        print(f"delta: {sum(results['on'])}/{a.episodes} with boxes vs "
              f"{sum(results['off'])}/{a.episodes} without")

    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        json.dump({"task": task.language, "suite": a.suite,
                   "task_id": a.task_id, "results": results}, open(a.json, "w"),
                  indent=2)
        print(f"json -> {a.json}")
    env.close()


if __name__ == "__main__":
    main()
