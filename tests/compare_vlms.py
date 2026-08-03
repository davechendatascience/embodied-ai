"""Compare VLM backends on the same ReKep constraint-generation prompt.

The layered diagnostic established that vision and control work, and that the
remaining failure on robosuite Lift is L2 — task decomposition. qwen3-vl
produced structurally correct constraints but only ONE stage for "pick up the
cube": approach-and-grasp with no lift, so robosuite's success check
(`cube_height > table_height + 0.04`) can never pass.

This runs the identical image and prompt through two backends and compares the
plan, not the prose. The question is narrow: does a stronger VLM decompose the
task into the two stages the task actually needs?

    MUJOCO_GL=egl .venv/bin/python tests/compare_vlms.py
"""

import contextlib
import io
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from rekep_libero import add_rekep_to_path  # noqa: E402
from rekep_libero.config import load_config  # noqa: E402

add_rekep_to_path()

from rekep_libero.environment_robosuite import ReKepRobosuiteEnv, AGENTVIEW  # noqa: E402
from keypoint_proposal import KeypointProposer  # noqa: E402
from constraint_generation import ConstraintGenerator  # noqa: E402

BACKENDS = {
    "qwen3-vl (local)": {
        "backend": "openai_compat",
        "model": "qwen3-vl:latest",
        "base_url": "http://localhost:11434/v1",
        "prefill": "<think>\n\n</think>\n\n",
        "temperature": 0.0,
    },
    "claude-opus-5": {
        "backend": "anthropic",
        "model": "claude-opus-5",
        "effort": "high",
    },
}


def build_scene(config):
    """Propose keypoints once so both backends see an identical prompt."""
    ws, ec = config["robosuite_workspace"], dict(config["env"])
    ec["bounds_min"], ec["bounds_max"] = ws["bounds_min"], ws["bounds_max"]
    ec["interpolate_pos_step_size"] = config["main"]["interpolate_pos_step_size"]
    ec["interpolate_rot_step_size"] = config["main"]["interpolate_rot_step_size"]
    env = ReKepRobosuiteEnv(ec)

    kp_config = dict(config["keypoint_proposer"])
    kp_config["bounds_min"], kp_config["bounds_max"] = ws["bounds_min"], ws["bounds_max"]
    cam = env.get_cam_obs()[AGENTVIEW]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        keypoints, projected = KeypointProposer(kp_config).get_keypoints(
            cam["rgb"], cam["points"], cam["seg"])
    return env, keypoints, projected


def run_backend(label, overrides, config, projected, keypoints, instruction):
    cfg = dict(config["constraint_generator"])
    cfg.update(overrides)
    print(f"\n=== {label} ===")
    try:
        gen = ConstraintGenerator(cfg)
        t0 = time.time()
        task_dir = gen.generate(projected, instruction,
                                {"init_keypoint_positions": keypoints, "num_keypoints": len(keypoints)})
        elapsed = time.time() - t0
    except Exception as exc:  # noqa: BLE001 - one backend failing shouldn't kill the comparison
        print(f"  FAILED: {type(exc).__name__}: {str(exc)[:200]}")
        return None

    meta = json.load(open(os.path.join(task_dir, "metadata.json")))
    n_files = len([f for f in os.listdir(task_dir) if "constraints" in f])
    print(f"  {elapsed:.1f}s | stages={meta['num_stages']} "
          f"grasp={meta['grasp_keypoints']} release={meta['release_keypoints']} "
          f"| {n_files} constraint files")
    return {"label": label, "elapsed": elapsed, "meta": meta, "dir": task_dir}


def main():
    config = load_config()
    env, keypoints, projected = build_scene(config)
    instruction = env.instruction
    print(f"task       : {instruction}")
    print(f"keypoints  : {len(keypoints)}")

    results = [r for r in (
        run_backend(label, overrides, config, projected, keypoints, instruction)
        for label, overrides in BACKENDS.items()
    ) if r]

    print("\n" + "=" * 68)
    print(f"{'backend':22s} {'wall':>7s} {'stages':>7s}  {'grasp':>12s}  lift stage?")
    for r in results:
        m = r["meta"]
        # "pick up the cube" needs approach-and-grasp THEN lift. A single-stage
        # plan grasps and stops, so robosuite's height check never passes.
        has_lift = m["num_stages"] >= 2
        print(f"{r['label']:22s} {r['elapsed']:6.1f}s {m['num_stages']:7d}  "
              f"{str(m['grasp_keypoints']):>12s}  {'YES' if has_lift else 'NO — grasps and stops'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
