"""RoboCasa demos -> a LeRobot dataset for GR00T, half of it box-overlaid.

Second half of Point-VLA's recipe. `build_overlay_dataset.py` proved the replay
and the box alignment; this turns it into something `examples/finetune.sh` can
train on.

THE STATE VECTOR IS NOT HAND-MAPPED, AND THAT IS THE WHOLE DESIGN.
The obvious approach is to read the demo's `datagen_info` (eef_pos, eef_rot,
base_pos, base_rot) and assemble GR00T's state layout from it. That mapping is
exactly the kind of thing that trains cleanly and evaluates as noise: the
policy's modality config asks for `base_position` and `base_rotation`, a live
observation exposes neither, and quietly guessing the convention for a
7-dimensional rotation is how a run gets thrown away after two days.

Instead, every recorded state is replayed THROUGH THE SAME GYM WRAPPER the
policy is served with, and `state.*` / `video.*` are read out of it. Whatever
the wrapper produces at inference is what lands in the training set, by
construction, without anyone needing to know the convention.

Actions come straight from the demo: the recorded `actions` are already 12-D
and match the environment's action space exactly.

THE 1:1 CO-TRAINING SPLIT IS TWO EPISODES PER DEMO -- one plain, one with the
target's box burned into the side cameras, identical actions and states. Point-
VLA co-trains at 1:1 so the policy does not collapse into "always go to the
mark" and lose language conditioning; emitting both from one replay means they
are pixel-identical apart from the box.

Frames where the target is occluded keep their PLAIN variant and are skipped for
the overlay variant -- annotating a box around nothing teaches that the mark is
noise.

Run:
  MUJOCO_GL=egl PYTHONPATH=. <robocasa_uv python> examples/emit_lerobot.py \\
      --dataset <demo.hdf5> --out data/robocasa_overlay --episodes 2
"""
import argparse
import json
import os
import sys

import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R)
sys.path.insert(0, R + "/examples")

from xembody.boxes import draw_overlay

#: Which wrapper video keys get the overlay. The wrist view is left CLEAN: a box
#: drawn on a close-up moving camera is mostly off-frame or covering everything,
#: and Point-VLA overlays a single fixed view for the same reason.
OVERLAY_KEYS = ("video.res256_image_side_0", "video.res256_image_side_1")
#: Emitted at the resolution the policy consumes. The res512_* duplicates the
#: wrapper also produces are dropped -- training on both wastes disk and teaches
#: nothing new.
KEEP_VIDEO = ("video.res256_image_side_0", "video.res256_image_side_1",
              "video.res256_image_wrist_0")
FPS = 20
#: Emitted frame size; must match what the wrapper feeds the policy.
RES = 256


def groot_obs(genv, inner, lang):
    """The observation the POLICY would see for the sim's current state.

    `get_groot_observation` reads `raw_obs["language"]`, which the RoboCasaEnv
    wrapper injects during its own reset/step and a bare `_get_observations()`
    does not carry. We are driving the sim directly from recorded states, so it
    is supplied here from the demo's own ep_meta.
    """
    raw = inner._get_observations()
    raw["language"] = lang
    return genv.unwrapped.get_groot_observation(raw)


def state_layout(obs):
    """Fixed (key, width) order for flattening `state.*` into one array.

    Sorted, so it is stable across episodes and reproducible from the file
    alone. modality.json records the resulting ranges, which is the only thing
    that has to agree with it.
    """
    keys = sorted(k for k in obs if k.startswith("state."))
    return [(k, int(np.asarray(obs[k]).reshape(-1).size)) for k in keys]


def flat_state(obs, layout):
    return np.concatenate([np.asarray(obs[k], np.float32).reshape(-1)
                           for k, _ in layout]).astype(np.float32)


def write_video(path, frames, fps=FPS):
    import imageio
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # imageio's PyAV plugin (imageio>=2.37 on this stack) does not accept
    # macro_block_size; ffmpeg-plugin kwargs are not portable across backends.
    # 256x256 is already a multiple of 16, so no padding hint is needed.
    # plugin="FFMPEG" is explicit: imageio's default PyAV backend on this
    # stack negotiates a 3-pixel-wide video and dies inside the encoder
    # ("could not broadcast (256,3) into (256,3,3)") on frames that are
    # verified (256, 256, 3) on the way in.
    with imageio.get_writer(path, format="FFMPEG", fps=fps,
                            codec="libx264", macro_block_size=1) as w:
        for i, f in enumerate(frames):
            arr = np.asarray(f, np.uint8)
            # A malformed frame here surfaces deep inside the encoder as an
            # unhelpful broadcast error, so it is caught at the source with the
            # path and index that produced it.
            if arr.ndim != 3 or arr.shape[-1] != 3:
                raise ValueError(f"{path} frame {i}: expected (H, W, 3), got "
                                 f"{arr.shape}")
            w.append_data(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--stride", type=int, default=2,
                    help="subsample recorded timesteps; demos run at 20 Hz")
    ap.add_argument("--task", default="PnPCounterToSink")
    a = ap.parse_args()

    import h5py
    import pandas as pd
    from build_overlay_dataset import reset_to, target_box
    from probe_groot_robocasa import (build_env, calibrate_flip,
                                      segmentation)

    genv, inner, _ = build_env(a.task)
    cams_for_keep = genv.unwrapped.key_converter.get_camera_config()[1]
    f = h5py.File(a.dataset, "r")
    demos = sorted(f["data"], key=lambda k: int(k.split("_")[-1]))[:a.episodes]

    os.makedirs(f"{a.out}/meta", exist_ok=True)
    layout, tasks, episodes, ep_idx, gidx = None, {}, [], 0, 0
    flips = {}  # camera -> orientation, calibrated once
    total_frames = 0

    for d, key in enumerate(demos):
        ep = f["data"][key]
        states = np.asarray(ep["states"])
        actions = np.asarray(ep["actions"], np.float32)
        meta = ep.attrs["ep_meta"]
        lang = json.loads(meta).get("lang", "")
        reset_to(inner, {"model": ep.attrs["model_file"], "ep_meta": meta,
                         "states": states[0]})

        rows, vids, ovids = [], {k: [] for k in KEEP_VIDEO}, {k: [] for k in KEEP_VIDEO}
        for t in range(0, len(states), a.stride):
            reset_to(inner, {"states": states[t]})
            obs = groot_obs(genv, inner, lang)
            if layout is None:
                layout = state_layout(obs)
            rows.append((actions[min(t, len(actions) - 1)],
                         flat_state(obs, layout)))
            # The box is measured on the SAME camera it is drawn on, from a
            # segmentation aligned to that camera's wrapper image.
            for k in KEEP_VIDEO:
                img = np.asarray(obs[k], np.uint8)
                vids[k].append(img)
                if k not in OVERLAY_KEYS:
                    ovids[k].append(img)
                    continue
                # THE CAMERAS RENDER AT 512; res256_* ARE DOWNSAMPLED COPIES.
                # Segmenting at 256 and comparing against the wrapper's
                # downsampled image leaves a mean|diff| of 15.6 (different
                # resampling), and the resulting boxes are at the wrong scale
                # entirely. So the mask is taken at the camera's NATIVE size,
                # aligned against the native-size image, and the box is scaled
                # into the frame that actually gets trained on.
                cam = cams_for_keep[KEEP_VIDEO.index(k)]
                native = np.asarray(obs[k.replace("res256", "res512")],
                                    np.uint8)
                if cam not in flips:
                    flips[cam] = calibrate_flip(inner, cam, native.shape[0],
                                                native.shape[1], native)
                    print(f"  orientation for {cam}: {flips[cam]}")
                seg = segmentation(inner, cam, native.shape[0],
                                   native.shape[1], native, flip=flips[cam])
                box = target_box(inner, seg)
                if box is None:
                    ovids[k].append(img)
                    continue
                sx = img.shape[1] / native.shape[1]
                sy = img.shape[0] / native.shape[0]
                b = (int(box[0] * sx), int(box[1] * sy),
                     int(box[2] * sx), int(box[3] * sy))
                ovids[k].append(draw_overlay(img, {"t": b}))

        task_id = tasks.setdefault(lang, len(tasks))
        for variant, frames in (("plain", vids), ("overlay", ovids)):
            n = len(rows)
            df = pd.DataFrame({
                "action": [r[0] for r in rows],
                "observation.state": [r[1] for r in rows],
                "timestamp": np.arange(n, dtype=np.float32) / FPS,
                "frame_index": np.arange(n, dtype=np.int64),
                "episode_index": np.full(n, ep_idx, dtype=np.int64),
                "index": np.arange(gidx, gidx + n, dtype=np.int64),
                "task_index": np.full(n, task_id, dtype=np.int64),
            })
            os.makedirs(f"{a.out}/data/chunk-000", exist_ok=True)
            df.to_parquet(f"{a.out}/data/chunk-000/"
                          f"episode_{ep_idx:06d}.parquet")
            for k in KEEP_VIDEO:
                write_video(f"{a.out}/videos/chunk-000/observation.images."
                            f"{k.split('.')[-1]}/episode_{ep_idx:06d}.mp4",
                            frames[k])
            episodes.append({"episode_index": ep_idx, "tasks": [lang],
                             "length": n})
            gidx += n
            total_frames += n
            ep_idx += 1
            print(f"demo {d} -> episode {ep_idx - 1} ({variant}) {n} frames")

    # ---- meta ----
    ranges, off = {}, 0
    for k, w in layout:
        ranges[k[len("state."):]] = {"start": off, "end": off + w}
        off += w
    modality = {
        "state": ranges,
        "action": {"end_effector_position": {"start": 0, "end": 3},
                   "end_effector_rotation": {"start": 3, "end": 6},
                   "gripper_close": {"start": 6, "end": 7},
                   "base_motion": {"start": 7, "end": 11},
                   "control_mode": {"start": 11, "end": 12}},
        "video": {k.split(".")[-1]: {"original_key":
                                     f"observation.images.{k.split('.')[-1]}"}
                  for k in KEEP_VIDEO},
        "annotation": {"human.action.task_description":
                       {"original_key": "task_index"}},
    }
    json.dump(modality, open(f"{a.out}/meta/modality.json", "w"), indent=2)
    info = {
        "codebase_version": "v2.1", "robot_type": "robocasa_panda_omron",
        "total_episodes": ep_idx, "total_frames": total_frames,
        "total_tasks": len(tasks), "chunks_size": 1000, "fps": FPS,
        "splits": {"train": f"0:{ep_idx}"},
        "data_path": "data/chunk-{episode_chunk:03d}/"
                     "episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/"
                      "episode_{episode_index:06d}.mp4",
        # VIDEO FEATURES ARE REQUIRED, not optional metadata. The loader
        # asserts that every `original_key` named in modality.json exists here:
        #   AssertionError: Original key observation.images.res256_image_side_0
        #   not found in feature config
        # and it fires inside a DataLoader worker after the model has loaded,
        # which reads like a training bug rather than a missing dict entry.
        "features": {
            "action": {"dtype": "float32", "shape": [int(actions.shape[1])]},
            "observation.state": {"dtype": "float32", "shape": [off]},
            **{f"observation.images.{k.split('.')[-1]}": {
                "dtype": "video", "shape": [RES, RES, 3],
                "names": ["height", "width", "channels"],
                "info": {"video.height": RES, "video.width": RES,
                         "video.codec": "h264", "video.pix_fmt": "yuv420p",
                         "video.is_depth_map": False, "video.fps": FPS,
                         "video.channels": 3, "has_audio": False},
            } for k in KEEP_VIDEO},
        },
    }
    json.dump(info, open(f"{a.out}/meta/info.json", "w"), indent=2)
    with open(f"{a.out}/meta/tasks.jsonl", "w") as fh:
        for lang, i in tasks.items():
            fh.write(json.dumps({"task_index": i, "task": lang}) + "\n")
    with open(f"{a.out}/meta/episodes.jsonl", "w") as fh:
        for e in episodes:
            fh.write(json.dumps(e) + "\n")

    print(f"\n{ep_idx} episodes ({ep_idx // 2} demos x plain+overlay), "
          f"{total_frames} frames, state dim {off}")
    print(f"-> {a.out}")
    f.close()
    genv.close()


if __name__ == "__main__":
    main()
