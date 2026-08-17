"""Does the overlay make the policy BETTER? Closed-loop success, 2x2.

THE PROBE CANNOT ANSWER THIS AND WAS NEVER GOING TO.
`probe_groot_robocasa.py` measures whether the mark CHANGES the action --
cosines, degrees, noise floors. That was the right instrument for "is the
grounding causal", which the literature reports carelessly. It says nothing
about whether the task gets done. Point-VLA's claim is 32.4% -> 92.5% SUCCESS,
and only rollouts measure that.

The earlier justification for avoiding success -- LIBERO saturated at 93-98%, no
headroom -- does not transfer. PnPCounterToSink is 46.0% zero-shot and
PnPCounterToMicrowave 19.0%, so there is room to improve and success is the
metric that matters.

THE 2x2, AND WHY BOTH ROWS ARE NEEDED:

                    no overlay              overlay
    base N1.6       sanity vs published     mark measured inert -> expect flat
    fine-tuned      PRESERVATION CHECK      the Point-VLA claim

The bottom-right cell is the headline. The bottom-LEFT is the guard: fine-tuning
on 55 demos of one task can easily wreck ungrounded behaviour, and an
improvement that only exists when a mark is present, bought by destroying the
policy without one, is not the result anyone wants. Point-VLA co-trains 1:1
precisely to avoid that; this measures whether it held.

HOW THE OVERLAY IS INJECTED. Their `create_eval_env` builds
    get_gym_env -> [VideoRecordingWrapper] -> MultiStepWrapper
so wrapping the result of `get_gym_env` puts the overlay UNDERNEATH their whole
stack and leaves every other detail of their evaluation untouched. Patching
further out would change observation stacking or video recording as a side
effect and confound the comparison.

THE MARK MATCHES TRAINING EXACTLY -- same oracle segmentation, same native-
resolution box scaled into the 256 frame, same two side cameras and a clean
wrist view. A model fine-tuned on one kind of mark and evaluated on another
measures the mismatch, not the method.

IT IS AN ORACLE MARK, at train and at eval. So the result reads "with PERFECT
grounding, does the fine-tune improve success" -- an upper bound on any real
detector, not a detector-in-the-loop number. Report it that way.

Run (needs a served checkpoint; see scripts/serve_groot.sh):
  MUJOCO_GL=egl PYTHONPATH=. <robocasa_uv python> examples/rollout_overlay.py \\
      --task PnPCounterToSink --episodes 10 --overlay on
"""
import argparse
import json
import os
import sys

import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R)
sys.path.insert(0, R + "/examples")

OVERLAY_KEYS = ("video.res256_image_side_0", "video.res256_image_side_1")


class OverlayWrapper:
    """Burns the target's oracle box into the side views, in place.

    A plain object rather than a gym.Wrapper subclass so it can forward
    everything by __getattr__ -- the surrounding GR00T wrappers reach for
    `unwrapped`, `key_converter` and other attributes that a strict wrapper
    would have to enumerate and would silently break on if it missed one.
    """

    def __init__(self, env):
        import gymnasium as gym  # noqa: F401
        self.env = env
        self._flips = {}
        inner = env.unwrapped.env
        self._inner = inner
        self._cams = env.unwrapped.key_converter.get_camera_config()[1]
        self._mapped = env.unwrapped.key_converter.get_camera_config()[0]

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _paint(self, obs):
        from probe_groot_robocasa import calibrate_flip, segmentation
        from build_overlay_dataset import target_box
        from xembody.boxes import draw_overlay

        for key in OVERLAY_KEYS:
            if key not in obs:
                continue
            img = np.asarray(obs[key], np.uint8)
            native_key = key.replace("res256", "res512")
            native = np.asarray(obs.get(native_key, img), np.uint8)
            cam = self._cams[self._mapped.index(key)]
            if cam not in self._flips:
                self._flips[cam] = calibrate_flip(
                    self._inner, cam, native.shape[0], native.shape[1], native)
            seg = segmentation(self._inner, cam, native.shape[0],
                               native.shape[1], native, flip=self._flips[cam])
            box = target_box(self._inner, seg)
            if box is None:          # occluded: leave the frame clean, exactly
                continue             # as the training set does
            sx = img.shape[1] / native.shape[1]
            sy = img.shape[0] / native.shape[0]
            obs[key] = draw_overlay(img, {"t": (int(box[0] * sx),
                                                int(box[1] * sy),
                                                int(box[2] * sx),
                                                int(box[3] * sy))})
        return obs

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        return self._paint(obs), info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        return self._paint(obs), rew, term, trunc, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PnPCounterToSink")
    ap.add_argument("--robot", default="PandaOmron")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--overlay", default="off", choices=("off", "on"))
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--pin-scenes", action="store_true", default=True)
    ap.add_argument("--no-pin-scenes", dest="pin_scenes",
                    action="store_false")
    ap.add_argument("--layout", type=int, default=1)
    ap.add_argument("--style-id", dest="style_id", type=int, default=1)
    ap.add_argument("--scene-seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    import gr00t.eval.rollout_policy as RP
    from gr00t.eval.rollout_policy import (WrapperConfigs,
                                           run_rollout_gymnasium_policy)
    from gr00t.policy.server_client import PolicyClient

    # PIN THE KITCHEN, OR THE COMPARISON IS NOISE.
    # `get_robocasa_env_fn` calls gym.make(env_name) with no layout/style, and
    # RoboCasa samples both -- plus the object instances -- from its own RNG at
    # construction. Two conditions would then run in different kitchens with
    # different objects, and at ~10 episodes that between-scene variance dwarfs
    # any effect of the overlay. Pinning makes both arms traverse the same
    # scene, so the only difference is the mark.
    _base_get = RP.get_gym_env

    def _pinned(env_name, env_idx, total_n_envs):
        import gymnasium as gym
        # Registration happens on import of robocasa's gym module, and
        # GR00T's own env_fn does these imports inside the factory. A
        # replacement factory must too, or gym.make raises
        # NamespaceNotFound for robocasa_panda_omron.
        import robocasa  # noqa: F401
        import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401
        import robosuite  # noqa: F401
        return gym.make(env_name, enable_render=True, seed=a.scene_seed,
                        layout_ids=a.layout, style_ids=a.style_id)

    if a.pin_scenes:
        RP.get_gym_env = _pinned
        print(f"scenes pinned: layout={a.layout} style={a.style_id} "
              f"seed={a.scene_seed}")

    if a.overlay == "on":
        # Wrap the BASE env, so VideoRecordingWrapper and MultiStepWrapper are
        # applied on top exactly as in an unmodified GR00T evaluation.
        original = RP.get_gym_env   # already pinned above, if requested

        def patched(env_name, env_idx, total_n_envs):
            return OverlayWrapper(original(env_name, env_idx, total_n_envs))

        RP.get_gym_env = patched
        print("overlay: ON (oracle box burned into the side views)")
    else:
        print("overlay: off")

    env_name = f"robocasa_panda_omron/{a.task}_{a.robot}_Env"
    policy = PolicyClient(host=a.host, port=a.port, timeout_ms=120000)
    if not policy.ping():
        raise SystemExit(f"no policy server at {a.host}:{a.port}")

    print(f"env {env_name}  episodes {a.episodes}")
    results = run_rollout_gymnasium_policy(
        env_name=env_name, policy=policy, wrapper_configs=WrapperConfigs(),
        n_episodes=a.episodes, n_envs=1)

    # The harness's return shape has moved between releases, so pull the
    # successes defensively rather than assuming a key.
    succ = None
    if isinstance(results, dict):
        for k in ("episode_successes", "successes", "success"):
            if k in results:
                succ = results[k]
                break
    if succ is None and isinstance(results, (list, tuple)):
        succ = results
    rate = float(np.mean([bool(s) for s in succ])) if succ else float("nan")
    print(f"\ntask {a.task}  overlay={a.overlay}  "
          f"success {rate:.1%}  ({sum(bool(s) for s in succ or [])}/"
          f"{len(succ or [])})")

    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        json.dump({"task": a.task, "overlay": a.overlay,
                   "episodes": a.episodes, "success_rate": rate,
                   "successes": [bool(s) for s in (succ or [])]},
                  open(a.json, "w"), indent=2)
        print(f"json -> {a.json}")


if __name__ == "__main__":
    main()
