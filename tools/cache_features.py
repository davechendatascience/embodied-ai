#!/usr/bin/env python
"""Precompute frozen CLIP features and both action targets for libero_spatial.

The encoder is frozen, so its features are computed once and training reads
cached arrays. That is the whole reason a small-head experiment is affordable
here: the deciding question is whether the spec tokens are used, and answering
it does not require fine-tuning a backbone.

Both policies read the SAME cached features and differ only in what they
predict, which is what makes the comparison about the action representation
rather than about perception:

  baseline   delta-q, joint space, padded to MAX_DOF and sliced on a new arm.
             This is what pi0 and GR00T do -- pad to a fixed width, no
             kinematics. Zero-shot on a new arm is the charitable reading of
             what those systems can do without collecting data for it.

  screwhead   body twist in the tool frame, decoded through the arm's own
              Jacobian. Embodiment-free by construction.

Targets are ACHIEVED, from joint_states, not the recorded OSC setpoint. The
setpoint implies a twist ~3x larger than the motion the arm made (measured:
0.62 m/s median difference against 0.19 m/s achieved), because OSC is an
impedance controller that only partly reaches its goal in 50 ms. Our decoder
replaces OSC and does not under-track, so training on the command would move
the arm about three times too far.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from screwhead.interface import ActionSpec                      # noqa: E402
from screwhead.libero import demos, panda_chain, task_files     # noqa: E402
from screwhead.retarget import to_twists, usable                # noqa: E402

MODEL = "openai/clip-vit-base-patch32"
MAX_DOF = 7                       # padded width, panda's joint count


def encode_images(model, pixel, device, batch=256):
    out = []
    with torch.no_grad():
        for i in range(0, len(pixel), batch):
            x = pixel[i:i + batch].to(device, non_blocking=True)
            f = model.vision_model(pixel_values=x).pooler_output
            out.append(f.to(torch.float16).cpu())
    return torch.cat(out)


def prep(images: np.ndarray, mean, std) -> torch.Tensor:
    """(N,H,W,3) uint8 -> CLIP-normalised (N,3,224,224)."""
    x = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
    x = torch.nn.functional.interpolate(x, size=224, mode="bilinear", align_corners=False)
    return (x - mean) / std


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--out", default="cache/libero_spatial")
    ap.add_argument("--demos-per-task", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from transformers import CLIPModel, CLIPTokenizer
    torch.set_default_dtype(torch.float32)
    device = args.device
    model = CLIPModel.from_pretrained(MODEL).to(device).eval()
    tok = CLIPTokenizer.from_pretrained(MODEL)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

    chain = panda_chain()
    spec = ActionSpec()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import h5py
    files = task_files(args.suite)
    print(f"{len(files)} tasks -> {out_dir}")
    for ti, f in enumerate(files):
        instruction = f.stem.replace("_demo", "").replace("_", " ")
        with torch.no_grad():
            t_ids = tok([instruction], return_tensors="pt", padding=True).to(device)
            text_feat = model.get_text_features(**t_ids)[0].to(torch.float16).cpu().numpy()

        agent, wrist, twists, dq, grip, state, demo_id = [], [], [], [], [], [], []
        with h5py.File(f, "r") as h:
            keys = list(h["data"].keys())[: args.demos_per_task]
            for di, k in enumerate(keys):
                g = h["data"][k]
                q = torch.tensor(g["obs"]["joint_states"][:], dtype=torch.float64)
                a = torch.tensor(g["actions"][:], dtype=torch.float64)
                n = len(q) - 1
                if n < 2:
                    continue
                V = to_twists(chain, q, spec)                    # (n, 6) achieved
                ok = usable(chain, q[:-1])                       # drop near-singular frames
                idx = torch.nonzero(ok).flatten().numpy()
                if len(idx) == 0:
                    continue
                agent.append(prep(g["obs"]["agentview_rgb"][:-1][idx], mean, std))
                wrist.append(prep(g["obs"]["eye_in_hand_rgb"][:-1][idx], mean, std))
                twists.append(V[idx].numpy().astype(np.float32))
                dq.append((q[1:] - q[:-1])[idx].numpy().astype(np.float32))
                grip.append(a[:-1, 6][idx].numpy().astype(np.float32))
                state.append(q[:-1][idx].numpy().astype(np.float32))
                demo_id.append(np.full(len(idx), di, np.int16))

        if not agent:
            print(f"  [{ti}] {f.stem[:45]}: no usable frames")
            continue
        af = encode_images(model, torch.cat(agent), device)
        wf = encode_images(model, torch.cat(wrist), device)
        np.savez_compressed(
            out_dir / f"task{ti:02d}.npz",
            agent=af.numpy(), wrist=wf.numpy(), text=text_feat,
            twist=np.concatenate(twists), dq=np.concatenate(dq),
            gripper=np.concatenate(grip), qpos=np.concatenate(state),
            demo=np.concatenate(demo_id), instruction=instruction, task_index=ti,
        )
        print(f"  [{ti}] {f.stem[:45]:45s} {len(af):6d} frames", flush=True)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
