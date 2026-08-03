"""How good is the grounding, ALONE? In millimetres, against the oracle.

The planner is solved on two LIBERO tasks, but its target comes from
`task_spec.py` -- the simulator. Replacing that with perception is the whole
remaining gap, and the first question is not "does the loop still work" but
"how wrong is the grounder", because those are different questions and the
second one is answerable without running the loop at all.

WHY THE GOAL AND NOT THE MASK. Both are oracle inputs, and the intuition is
that segmentation is the hard part. Measured, it is not:

  * `--freeze-scene`: pin the entire visual scene to tick 0 for a whole
    episode -> the drawer still succeeds, 0.3 mm difference.
  * `--mask-jitter-mm 80`: hand the planner a mask with IoU 0.00, containing
    ZERO points of the target -> still succeeds.
  * a global DINOv3 feature-similarity mask tops out at 0.29-0.67 IoU even
    when the query prototype is derived from the answer.

So the mask is nearly free and the goal VECTOR is doing the work. This scores
the goal.

WHAT IS STILL PRIVILEGED HERE, and it must not be quietly forgotten: the
keypoints are farthest-point samples of the SCENE CLOUD, which a robot has, but
the ORACLE used to score is `spec.offset(env)`, which it does not. That is the
point -- the oracle is the ruler, not an input.

    MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/measure_grounding.py \
        --suite libero_goal --task-ids 0 1
"""

import argparse
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from rekep_libero import add_rekep_to_path  # noqa: E402
from rekep_libero.config import load_config  # noqa: E402

add_rekep_to_path()

from rekep_libero.environment_libero import ReKepLiberoEnv  # noqa: E402
from rekep_libero.pw_observation import NR, T_LEN, live_observation  # noqa: E402
from rekep_libero import task_spec as specs  # noqa: E402
from rekep_libero import keypoint_grounding as kg  # noqa: E402
from rekep_libero import scene_graph as sg  # noqa: E402
from robot_points import MujocoRobotPoints  # noqa: E402


def score(env, spec, points, goal_idx, goal_pos):
    """(on-target fraction, displacement error mm, cos) against the oracle.

    The grounder names points and where they should go. Two things can be
    wrong independently and they need different fixes, so they are scored
    separately -- a scalar would hide which one failed (`NOTES.md` section 3).
    """
    if len(goal_idx) == 0:
        return None
    on = env.points_in_geoms(points[goal_idx], spec.geoms(env), margin=0.02)
    truth = spec.offset(env)
    pred = (goal_pos - points[goal_idx]).mean(axis=0)
    err = float(np.linalg.norm(pred - truth)) * 1000.0
    cos = float(pred @ truth / max(np.linalg.norm(pred) * np.linalg.norm(truth), 1e-9))
    return float(on.mean()), err, cos, float(np.linalg.norm(truth)) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--task-ids", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--keypoints", type=int, default=24)
    ap.add_argument("--outdir", default="videos/grounding")
    ap.add_argument("--mode", choices=["objects", "keypoints"], default="objects",
                    help="objects = the guide's scene-graph + goal formalizer; "
                         "keypoints = KUDA's offset prompt (the old baseline)")
    cli = ap.parse_args()
    os.makedirs(cli.outdir, exist_ok=True)

    from rekep_libero.vlm_backends import make_backend
    backend = make_backend({"backend": "qwen_local", "model": cli.model,
                            "temperature": 0.0, "max_tokens": 512})

    config = load_config()
    ec = dict(config["env"])
    ec["bounds_min"], ec["bounds_max"] = config["workspace"]["bounds_min"], config["workspace"]["bounds_max"]
    ec["interpolate_pos_step_size"] = config["main"]["interpolate_pos_step_size"]
    ec["interpolate_rot_step_size"] = config["main"]["interpolate_rot_step_size"]

    rows = []
    for tid in cli.task_ids:
        env = ReKepLiberoEnv(ec, task_suite=cli.suite, task_id=tid, robot="Panda",
                             resolution=config["libero"]["resolution"], reset_seed=0)
        spec = specs.for_task(cli.suite, tid, env)
        rp = MujocoRobotPoints(env, NR)
        obs = live_observation(env, rp, steps=T_LEN)
        points = obs["scene_flows"][0, 0]
        rgb = obs["rgb"][0][0]
        K = obs["intrinsic"][0][0]
        cam2world = obs["extrinsic"][0][0]

        img_path = os.path.join(cli.outdir, f"{cli.suite}_{tid}.png")
        print(f"\n=== {cli.suite}/{tid} — {env.instruction}")
        print(f"    oracle: {spec.name}, owes {spec.remaining_mm(env):.1f} mm")
        try:
            if cli.mode == "objects":
                gi, gp, graph, _, reply = sg.ground(
                    env.instruction, rgb, points, K, cam2world, backend,
                    save_image=img_path)
                print(f"    {len(graph)} object proposals")
            else:
                gi, gp, _, reply = kg.ground(env.instruction, rgb, points, K,
                                             cam2world, backend,
                                             n_keypoints=cli.keypoints,
                                             save_image=img_path)
        except Exception as exc:  # noqa: BLE001 - a backend failure is a result
            print(f"    GROUNDING FAILED: {type(exc).__name__}: {str(exc)[:200]}")
            rows.append((tid, None))
            continue
        print(f"    reply: {reply.strip()[:300]}")
        s = score(env, spec, points, gi, gp)
        if s is None:
            print("    returned NO targets (or 'Done.') — nothing to score")
            rows.append((tid, None))
            continue
        on, err, cos, owed = s
        print(f"    on-target {on*100:5.1f}%  |  goal error {err:7.1f} mm  "
              f"cos {cos:+.2f}  (task owes {owed:.1f} mm)")
        rows.append((tid, (on, err, cos, owed)))

    print(f"\n{'task':>6} {'on-target':>10} {'goal err':>10} {'cos':>6} {'owed':>8}")
    for tid, r in rows:
        if r is None:
            print(f"{tid:>6} {'--':>10} {'FAILED':>10}")
        else:
            on, err, cos, owed = r
            print(f"{tid:>6} {on*100:9.1f}% {err:9.1f} {cos:+6.2f} {owed:8.1f}")
    print("\nA goal error much larger than the distance owed means the grounder "
          "is not usable yet; comparable means it is worth wiring in.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
