"""Success-rate rollouts for grounded test-time action selection. Arms A/B/C.

Runs GR00T's own evaluation harness with a selector in place of the bare
policy client, so the only thing that differs between arms is HOW a candidate
is chosen -- not the environment, the wrappers, the scene, or the success
criterion.

    --mode none      K=1. Reproduces the published baseline (~46% on
                     PnPCounterToSink). If this does not reproduce, nothing
                     else measured here means anything.
    --mode mean      K candidates, pick the one closest to their mean.
                     Verifier-free control. THE ARM THAT MATTERS -- it
                     separates "selection helps" from "grounding helps".
    --mode grounded  K candidates, pick by cosine to the grounded target.

THE TARGET IS READ FROM THE LIVE SIM, ORACLE, AND LABELLED AS SUCH. It comes
from `env.objects["obj"]`'s body position -- the same referent RoboCasa names
in the instruction. That is an upper bound on any detector, which is the point:
measure the ceiling before paying for perception. The same `target_fn` seam
accepts a detector's mask later without touching anything else.

SCENES ARE PINNED across arms (`layout/style/seed`). RoboCasa samples layout,
style AND object instances from its own RNG at construction, so unpinned arms
would run in different kitchens and the between-scene variance would swamp the
effect at ~15 episodes.

Run (needs `bash scripts/serve_groot.sh`):
  MUJOCO_GL=egl PYTHONPATH=. <robocasa_uv python> examples/rollout_select.py \\
      --task PnPCounterToSink --episodes 15 --mode grounded --k 8
"""
import argparse
import json
import os
import sys

import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R)
sys.path.insert(0, R + "/examples")

from select_grounded import GroundedSelector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PnPCounterToSink")
    ap.add_argument("--robot", default="PandaOmron")
    ap.add_argument("--episodes", type=int, default=15)
    ap.add_argument("--mode", default="grounded",
                    choices=("none", "mean", "grounded"))
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--layout", type=int, default=1)
    ap.add_argument("--style-id", dest="style_id", type=int, default=1)
    ap.add_argument("--scene-seed", type=int, default=0)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    import gr00t.eval.rollout_policy as RP
    from gr00t.eval.rollout_policy import (WrapperConfigs,
                                           run_rollout_gymnasium_policy)
    from gr00t.policy.server_client import PolicyClient

    # Capture the env as it is built so the target can be read from the live
    # sim. The harness owns env construction, so this is the only seam.
    built = {}
    scenes = []   # C-0004: one entry per env reset

    def _make(env_name, env_idx, total_n_envs):
        import gymnasium as gym
        # Registration happens on import of robocasa's gym module, and
        # GR00T's own env_fn does these imports inside the factory. A
        # replacement factory must too, or gym.make raises
        # NamespaceNotFound for robocasa_panda_omron.
        import robocasa  # noqa: F401
        import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401
        import robosuite  # noqa: F401
        env = gym.make(env_name, enable_render=True, seed=a.scene_seed,
                       layout_ids=a.layout, style_ids=a.style_id)
        built["env"] = env

        # C-0004: record the scene ACTUALLY traversed on every reset. Passing
        # identical layout/style/seed is not evidence that pinning holds --
        # RoboCasa samples object instances from its own RNG at construction,
        # so only an identical instruction/target sequence across two
        # independent runs demonstrates it.
        _reset = env.reset

        def reset_recording(*args, **kw):
            out = _reset(*args, **kw)
            try:
                inner = env.unwrapped.env
                obj = inner.objects.get("obj")
                scenes.append({
                    "instruction": inner.get_ep_meta().get("lang", ""),
                    "target": getattr(obj, "root_body", None),
                })
            except Exception as exc:
                scenes.append({"error": repr(exc)})
            return out

        env.reset = reset_recording
        return env

    RP.get_gym_env = _make

    def target_world():
        """Target's world position, ORACLE, or None if unavailable.

        `obj` is RoboCasa's own name for the thing the instruction refers to,
        so this is the same referent the language names -- not a guess.
        """
        env = built.get("env")
        if env is None:
            return None
        inner = env.unwrapped.env
        obj = inner.objects.get("obj")
        root = getattr(obj, "root_body", None) if obj else None
        if not root:
            return None
        try:
            return np.asarray(
                inner.sim.data.body_xpos[inner.sim.model.body_name2id(root)],
                float).copy()
        except Exception:
            return None

    client = PolicyClient(host=a.host, port=a.port, timeout_ms=120000)
    if not client.ping():
        raise SystemExit(f"no policy server at {a.host}:{a.port}")

    sel = GroundedSelector(client, mode=a.mode, k=a.k,
                           target_fn=target_world if a.mode == "grounded"
                           else None)

    env_name = f"robocasa_panda_omron/{a.task}_{a.robot}_Env"
    print(f"{env_name}\n  mode={a.mode}  K={sel.k}  episodes={a.episodes}  "
          f"scene: layout={a.layout} style={a.style_id} seed={a.scene_seed}")

    results = run_rollout_gymnasium_policy(
        env_name=env_name, policy=sel, wrapper_configs=WrapperConfigs(),
        n_episodes=a.episodes, n_envs=1)

    # C-0002: print the harness's RAW return before any key extraction. The
    # smoke run reported 2/3 successes for a 1-episode request, so a
    # requested-vs-recorded mismatch must be visible rather than averaged into
    # a rate.
    print("\n--- raw harness return (C-0002) ---")
    print(f"  type: {type(results).__name__}")
    if isinstance(results, dict):
        for _k, _v in results.items():
            _n = len(_v) if isinstance(_v, (list, tuple)) else "-"
            print(f"  {_k}: type={type(_v).__name__} len={_n} value={str(_v)[:120]}")
    else:
        print(f"  {str(results)[:400]}")

    # The harness returns a TUPLE: (env_name, [bool per episode], info).
    # The previous fallback took `succ = results` for any tuple, so it measured
    # the truthiness of a string, a list and an empty dict -> "66.7% (2/3)" for
    # a 15-episode run. The number had nothing to do with the robot. Pull the
    # boolean sequence explicitly and fail loudly if it cannot be found.
    succ = None
    if isinstance(results, dict):
        for key in ("episode_successes", "successes", "success"):
            if key in results:
                succ = results[key]
                break
    elif isinstance(results, (list, tuple)):
        for item in results:
            if isinstance(item, (list, tuple)) and item and all(
                    isinstance(x, (bool,)) or hasattr(x, "item") for x in item):
                succ = list(item)
                break
    if succ is None:
        raise RuntimeError(
            f"could not locate the per-episode success sequence in a "
            f"{type(results).__name__} return: {str(results)[:200]}")
    succ = [bool(s) for s in succ]
    rate = float(np.mean(succ)) if succ else float("nan")
    accounting_ok = len(succ) == a.episodes
    print(f"  requested episodes={a.episodes}  recorded successes={len(succ)}  "
          f"-> accounting {'OK' if accounting_ok else 'MISMATCH'}")
    print(f"  scenes recorded (C-0004): {len(scenes)}")
    for i, sc in enumerate(scenes[:20]):
        print(f"    ep{i}: {sc}")

    # Candidate spread is the premise of the method: if the samples agree,
    # there is nothing to select between and a null is about the sampler, not
    # about grounding.
    spreads = [r["spread"] for r in sel.log if "spread" in r]
    ungrounded = sum(1 for r in sel.log if r.get("grounded") is False)
    print(f"\n{a.task}  mode={a.mode}  success {rate:.1%} "
          f"({sum(succ)}/{len(succ)})")
    if spreads:
        print(f"  candidate spread: mean {np.mean(spreads):.4f}  "
              f"min {np.min(spreads):.4f}  max {np.max(spreads):.4f}  "
              f"(1 - cos to the candidate mean; ~0 means no diversity)")
    if ungrounded:
        print(f"  steps with no visible target (fell back to mean): "
              f"{ungrounded}/{len(sel.log)}")

    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        json.dump({"task": a.task, "mode": a.mode, "k": sel.k,
                   "episodes": a.episodes, "success_rate": rate,
                   "successes": succ,
                   "accounting_ok": accounting_ok,
                   "recorded_episodes": len(succ),
                   "scenes": scenes,
                   "spread_mean": float(np.mean(spreads)) if spreads else None,
                   "ungrounded_steps": ungrounded,
                   "total_steps": len(sel.log)},
                  open(a.json, "w"), indent=2)
        print(f"json -> {a.json}")


if __name__ == "__main__":
    main()
