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

        # V-002 ACCOUNTING. `_paint` runs on reset AND on step and loops over
        # both side cameras, so the identity the plan checks is
        #     painted + skipped == 2 * (env_steps + env_resets)
        # An earlier form of that criterion said `2 * env_steps` and was off by
        # exactly 34 on a 15-episode run -- 17 resets times two cameras -- so it
        # would have FAILED on a perfectly healthy run. Resets are counted
        # separately for that reason.
        #
        # The identity alone is close to a tautology: gymnasium_groot emits both
        # res256 side keys unconditionally, so the `key not in obs` skip cannot
        # fire and the only live skip is `box is None` (occlusion). The
        # substantive quantities are therefore the painted FRACTION and the mean
        # box area -- `painted_frames > 0` passes on one frame in thousands,
        # which is exactly how the 0/15 ckpt2000 run was indistinguishable from
        # "nothing was painted".
        self.painted_frames = 0
        self.skipped_frames = 0
        self.env_steps = 0
        self.env_resets = 0
        self._area_sum = 0.0

    @property
    def mean_box_area_fraction(self):
        """Mean box area as a fraction of the 256 frame, over PAINTED frames."""
        return (self._area_sum / self.painted_frames
                if self.painted_frames else 0.0)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _paint(self, obs):
        from probe_groot_robocasa import calibrate_flip, segmentation
        from build_overlay_dataset import target_box
        from xembody.boxes import draw_overlay

        for key in OVERLAY_KEYS:
            if key not in obs:
                self.skipped_frames += 1
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
                self.skipped_frames += 1   # as the training set does. THIS is
                continue                   # the only skip path that can fire.
            sx = img.shape[1] / native.shape[1]
            sy = img.shape[0] / native.shape[0]
            x0, y0 = int(box[0] * sx), int(box[1] * sy)
            x1, y1 = int(box[2] * sx), int(box[3] * sy)
            obs[key] = draw_overlay(img, {"t": (x0, y0, x1, y1)})
            self.painted_frames += 1
            self._area_sum += (abs(x1 - x0) * abs(y1 - y0)) / float(
                img.shape[0] * img.shape[1])
        return obs

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.env_resets += 1
        return self._paint(obs), info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        self.env_steps += 1
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

    # C-0004 / V-004: RECORD THE SCENE ACTUALLY TRAVERSED ON EVERY RESET.
    # Ported from rollout_select.py:90-105. The committed baseline artifacts
    # carry this sequence; without the same record on this side the two arms
    # cannot be SHOWN to be paired, and the plan's per-scene analysis would rest
    # on an assumption rather than a measurement. Passing identical
    # layout/style/seed is not evidence that pinning held -- RoboCasa samples
    # object instances from its own RNG at construction, so only an identical
    # instruction/target sequence demonstrates it.
    #
    # This wraps the FACTORY rather than living inside OverlayWrapper, so the
    # record exists for both arms and not only when the overlay is on.
    scenes = []
    _factory = RP.get_gym_env

    def _recording(env_name, env_idx, total_n_envs):
        env = _factory(env_name, env_idx, total_n_envs)
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

    RP.get_gym_env = _recording

    wrapper = {}
    if a.overlay == "on":
        # Wrap the BASE env, so VideoRecordingWrapper and MultiStepWrapper are
        # applied on top exactly as in an unmodified GR00T evaluation.
        original = RP.get_gym_env   # already pinned and recording, as requested

        def patched(env_name, env_idx, total_n_envs):
            w = OverlayWrapper(original(env_name, env_idx, total_n_envs))
            wrapper["w"] = w        # kept so V-002 can read its counters
            return w

        RP.get_gym_env = patched
        print("overlay: ON (oracle box burned into the side views)")
    else:
        print("overlay: off")

    env_name = f"robocasa_panda_omron/{a.task}_{a.robot}_Env"
    # A BARE PolicyClient DOES NOT WORK WITH THIS HARNESS.
    # rollout_policy hands out FLAT observation keys (video.res256_image_side_0)
    # while Gr00tPolicy._validate requires nested modality dicts and rejects
    # anything else with "Observation must contain a 'video' key". The harness
    # also unpacks `actions, _ = policy.get_action(obs)` and the env's action
    # space uses `action.`-prefixed keys while the server answers unprefixed.
    # GroundedSelector already handles all three; mode="none" means K=1 with no
    # selection, so it is a pure adapter here and the arms stay comparable to
    # the committed baseline, which was measured through the same code path.
    from select_grounded import GroundedSelector

    policy = GroundedSelector(
        PolicyClient(host=a.host, port=a.port, timeout_ms=120000), mode="none")
    if not policy.client.ping():
        raise SystemExit(f"no policy server at {a.host}:{a.port}")

    print(f"env {env_name}  episodes {a.episodes}")
    results = run_rollout_gymnasium_policy(
        env_name=env_name, policy=policy, wrapper_configs=WrapperConfigs(),
        n_episodes=a.episodes, n_envs=1)

    # The harness's return shape has moved between releases, so pull the
    # successes defensively rather than assuming a key.
    # The harness returns a TUPLE: (env_name, [bool per episode], info).
    # Taking `succ = results` for any tuple measured the truthiness of a
    # string, a list and an empty dict -- reporting "66.7% (2/3)" for a
    # 15-episode run. Pull the boolean sequence explicitly, and fail loudly
    # rather than silently scoring the wrong object.
    succ = None
    if isinstance(results, dict):
        for k in ("episode_successes", "successes", "success"):
            if k in results:
                succ = results[k]
                break
    elif isinstance(results, (list, tuple)):
        for item in results:
            if isinstance(item, (list, tuple)) and item and all(
                    isinstance(x, bool) or hasattr(x, "item") for x in item):
                succ = list(item)
                break
    if succ is None:
        raise RuntimeError(
            f"could not locate the per-episode success sequence in a "
            f"{type(results).__name__} return: {str(results)[:200]}")
    succ = [bool(s) for s in succ]
    rate = float(np.mean(succ)) if succ else float("nan")
    accounting_ok = len(succ) == a.episodes
    print(f"\ntask {a.task}  overlay={a.overlay}  "
          f"success {rate:.1%}  ({sum(succ)}/{len(succ)})")
    print(f"  requested episodes={a.episodes}  recorded={len(succ)}  "
          f"-> accounting {'OK' if accounting_ok else 'MISMATCH'}")
    print(f"  scenes recorded (C-0004): {len(scenes)}")
    for i, sc in enumerate(scenes[:20]):
        print(f"    ep{i}: {sc}")

    # V-002: PROVE THE MARK WAS ACTUALLY PAINTED.
    # The "overlay: ON" line above prints before any episode runs and survives
    # every silent no-op path, so it is not evidence of anything. These counters
    # are.
    paint = None
    w = wrapper.get("w")
    if w is not None:
        total = w.painted_frames + w.skipped_frames
        expected = 2 * (w.env_steps + w.env_resets)
        frac = (w.painted_frames / total) if total else 0.0
        paint = {"painted_frames": w.painted_frames,
                 "skipped_frames": w.skipped_frames,
                 "env_steps": w.env_steps,
                 "env_resets": w.env_resets,
                 "expected_frames": expected,
                 "identity_ok": total == expected,
                 "painted_fraction": frac,
                 "mean_box_area_fraction": w.mean_box_area_fraction}
        print("\n  V-002 paint accounting:")
        print(f"    painted={w.painted_frames}  skipped={w.skipped_frames}  "
              f"steps={w.env_steps}  resets={w.env_resets}")
        print(f"    painted+skipped={total}  2*(steps+resets)={expected}  "
              f"-> identity {'OK' if total == expected else 'MISMATCH'}")
        print(f"    painted fraction {frac:.3f} (>= 0.5 required)")
        print(f"    mean box area fraction {w.mean_box_area_fraction:.5f} "
              f"(> 0 required)")

    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        json.dump({"task": a.task, "overlay": a.overlay,
                   "episodes": a.episodes, "success_rate": rate,
                   "successes": succ,
                   "accounting_ok": accounting_ok,
                   "recorded_episodes": len(succ),
                   "scenes": scenes,
                   "paint": paint},
                  open(a.json, "w"), indent=2)
        print(f"json -> {a.json}")


if __name__ == "__main__":
    main()
