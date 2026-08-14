"""RoboCasa demos -> frames with the target's box burned in. Point-VLA's recipe.

The measurement that motivates this: neither prompt-serialised boxes nor pixel
overlays steer an UNTRAINED policy (see README). Every published overlay success
fine-tuned on overlaid frames first, so the intervention costs a training run.
This builds its input.

WHY REPLAY INSTEAD OF USING THE SHIPPED IMAGE DATASETS.
RoboCasa's `human_raw` demos store MuJoCo STATES, not pixels -- 72 MB for 55
episodes instead of several GB. Replaying them lets us re-render every frame
ourselves, which is the only way to get a segmentation mask aligned to the
frame. It also means the annotation is ORACLE: Point-VLA auto-labelled with
Gemini ER1.5 at 92% accuracy, and we get 100% from the simulator for free.

THE 1:1 CO-TRAINING SPLIT IS EMITTED IN THE SAME PASS. Point-VLA co-trains
overlaid examples against plain ones at 1:1 so the model does not collapse into
"always go to the mark" and lose language conditioning. Both variants come from
the same render here, so they are pixel-identical apart from the drawn box --
which is exactly the contrast the fine-tune needs to learn, and impossible to
get by rendering them separately.

ORIENTATION. GR00T's gym wrapper feeds the policy the VERTICAL FLIP of a raw
MuJoCo render -- measured, mean|diff| 1.2 flipped against 93 unflipped. Training
frames must match what the policy sees at inference, so the same flip is applied
here, to RGB and segmentation alike. Getting this wrong trains the model on
upside-down kitchens and is invisible in any loss curve.

Run (verify first, ALWAYS -- look at the PNGs before building anything):
  MUJOCO_GL=egl PYTHONPATH=. <robocasa_uv python> \\
      examples/build_overlay_dataset.py --dataset <demo.hdf5> --verify pairs/diag/ov
"""
import argparse
import json
import os
import sys

import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R)

from xembody.boxes import draw_overlay

#: GR00T's RoboCasa rig, at the resolution its wrapper feeds the policy.
CAMS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
RES = 256


def make_env(dataset_path):
    """Rebuild the environment the demos were collected in.

    `env_args` on the file carries the exact kwargs, so the env is reconstructed
    rather than guessed. Cameras are overridden to GR00T's rig because that is
    what the policy will be fed; everything else is left as recorded.
    """
    import h5py
    import robocasa  # noqa: F401  registers PnP* envs with robosuite
    import robosuite

    with h5py.File(dataset_path, "r") as f:
        env_meta = json.loads(f["data"].attrs["env_args"])
    kw = dict(env_meta["env_kwargs"])
    kw["env_name"] = env_meta["env_name"]
    kw.update(has_renderer=False, has_offscreen_renderer=True,
              use_camera_obs=False, renderer="mjviewer",
              camera_names=list(CAMS),
              camera_heights=RES, camera_widths=RES)
    return robosuite.make(**kw)


def reset_to(env, state):
    """Restore a recorded simulator state. Adapted from robocasa's playback."""
    import robosuite
    if "model" in state:
        meta = json.loads(state["ep_meta"]) if state.get("ep_meta") else {}
        if hasattr(env, "set_attrs_from_ep_meta"):
            env.set_attrs_from_ep_meta(meta)
        elif hasattr(env, "set_ep_meta"):
            env.set_ep_meta(meta)
        env.reset()
        xml = env.edit_model_xml(state["model"])
        env.reset_from_xml_string(xml)
        env.sim.reset()
    if "states" in state:
        env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()


def render(env, cam):
    """(rgb, segmentation) for `cam`, both in the frame the POLICY is fed.

    The vertical flip is not cosmetic -- see the module docstring. Both arrays
    get it, so a box measured on the segmentation lands on the same pixels in
    the RGB by construction.
    """
    rgb = np.asarray(env.sim.render(camera_name=cam, width=RES, height=RES))[::-1]
    seg = np.asarray(env.sim.render(camera_name=cam, width=RES, height=RES,
                                    segmentation=True))[::-1]
    return np.ascontiguousarray(rgb), seg[:, :, 1]


def target_box(env, seg, role="obj", min_px=8):
    """Pixel box of the manipulation target, or None if it is not visible.

    `role="obj"` is RoboCasa's name for the thing the instruction refers to;
    `distr_*` are the distractors it places on purpose. Returning None matters:
    a frame where the target is occluded must be DROPPED rather than annotated
    with a box around nothing, or the model learns that the mark is noise.
    """
    from xembody.boxes import body_name

    model = env.sim.model
    obj = env.objects.get(role)
    if obj is None or getattr(obj, "root_body", None) is None:
        return None
    root = obj.root_body
    bids = {b for b in range(model.nbody)
            if (body_name(model, b) or "").startswith(root)}
    ys, xs = [], []
    for gid in np.unique(seg):
        gid = int(gid)
        if gid < 0 or gid >= int(model.ngeom):
            continue
        if int(model.geom_bodyid[gid]) not in bids:
            continue
        y, x = np.nonzero(seg == gid)
        ys.append(y)
        xs.append(x)
    if not xs:
        return None
    xs, ys = np.concatenate(xs), np.concatenate(ys)
    if len(xs) < min_px:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def episode_frames(env, ep, stride, cam):
    """Yield (t, rgb, box, lang) for one episode, subsampled by `stride`."""
    states = np.asarray(ep["states"])
    meta = ep.attrs["ep_meta"]
    reset_to(env, {"model": ep.attrs["model_file"], "ep_meta": meta,
                   "states": states[0]})
    lang = json.loads(meta).get("lang", "")
    for t in range(0, len(states), stride):
        reset_to(env, {"states": states[t]})
        rgb, seg = render(env, cam)
        yield t, rgb, target_box(env, seg), lang


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cam", default="robot0_agentview_left", choices=list(CAMS))
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--verify", default=None,
                    help="write <prefix>_epE_tT_{plain,overlay}.png and stop. "
                         "LOOK AT THESE before building a dataset.")
    a = ap.parse_args()

    import h5py
    env = make_env(a.dataset)
    f = h5py.File(a.dataset, "r")
    demos = sorted(f["data"], key=lambda k: int(k.split("_")[-1]))

    n_box, n_none = 0, 0
    for e, key in enumerate(demos[:a.episodes]):
        for t, rgb, box, lang in episode_frames(env, f["data"][key], a.stride,
                                                a.cam):
            if box is None:
                n_none += 1
                continue
            n_box += 1
            if a.verify:
                import imageio
                os.makedirs(os.path.dirname(os.path.abspath(a.verify)) or ".",
                            exist_ok=True)
                imageio.imwrite(f"{a.verify}_ep{e}_t{t}_plain.png", rgb)
                imageio.imwrite(f"{a.verify}_ep{e}_t{t}_overlay.png",
                                draw_overlay(rgb, {"target": box}))
                print(f"ep{e} t{t:>4}  box={box}  {lang!r}")
    print(f"\nframes with a visible target: {n_box}   dropped (occluded): "
          f"{n_none}")
    if a.verify:
        print(f"wrote {a.verify}_*.png -- CHECK the box is on the target object "
              "and the kitchen is upright before trusting any of this.")
    f.close()
    env.close()


if __name__ == "__main__":
    main()
