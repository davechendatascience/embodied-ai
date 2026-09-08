#!/usr/bin/env python
"""Cache raw frames for a fine-tuned backbone.

The CLIP cache holds features, which is useless once the encoder trains. This
stores the frames themselves as uint8 memmaps -- ~6 GB for libero_spatial, read
per batch rather than held in RAM.

Frame selection is IDENTICAL to tools/cache_features.py: the same near-singular
frames are dropped by the same predicate. A backbone comparison where the two
runs saw different data would measure the data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_default_dtype(torch.float64)

from screwhead.interface import ActionSpec          # noqa: E402
from screwhead.libero import demos, panda_chain, task_files  # noqa: E402
from screwhead.retarget import to_twists, usable    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--out", default="cache/libero_spatial_img")
    ap.add_argument("--demos-per-task", type=int, default=50)
    args = ap.parse_args()

    import h5py
    chain, spec = panda_chain(), ActionSpec()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    files = task_files(args.suite)
    manifest = []
    for ti, f in enumerate(files):
        instruction = f.stem.replace("_demo", "").replace("_", " ")
        ag, wr, tw, dq, gr, qp, gs, dm = [], [], [], [], [], [], [], []
        with h5py.File(f, "r") as h:
            for di, k in enumerate(list(h["data"].keys())[: args.demos_per_task]):
                g = h["data"][k]
                q = torch.tensor(g["obs"]["joint_states"][:], dtype=torch.float64)
                a = torch.tensor(g["actions"][:], dtype=torch.float64)
                if len(q) - 1 < 2:
                    continue
                V = to_twists(chain, q, spec)
                idx = torch.nonzero(usable(chain, q[:-1])).flatten().numpy()
                if len(idx) == 0:
                    continue
                ag.append(g["obs"]["agentview_rgb"][:-1][idx])
                wr.append(g["obs"]["eye_in_hand_rgb"][:-1][idx])
                tw.append(V[idx].numpy().astype(np.float32))
                dq.append((q[1:] - q[:-1])[idx].numpy().astype(np.float32))
                gr.append(a[:-1, 6][idx].numpy().astype(np.float32))
                qp.append(q[:-1][idx].numpy().astype(np.float32))
                gsr = np.asarray(g["obs"]["gripper_states"][:], np.float32)
                gs.append((gsr[:-1][idx][:, 0] - gsr[:-1][idx][:, 1]).astype(np.float32))
                dm.append(np.full(len(idx), di, np.int16))
        if not ag:
            print(f"  [{ti}] no usable frames"); continue
        np.save(out / f"task{ti:02d}_agent.npy", np.concatenate(ag))
        np.save(out / f"task{ti:02d}_wrist.npy", np.concatenate(wr))
        np.savez_compressed(out / f"task{ti:02d}_meta.npz",
                            twist=np.concatenate(tw), dq=np.concatenate(dq),
                            gripper=np.concatenate(gr), qpos=np.concatenate(qp),
                            gripper_state=np.concatenate(gs), demo=np.concatenate(dm),
                            instruction=instruction, task_index=ti)
        n = sum(len(x) for x in ag)
        manifest.append({"task": ti, "n": int(n), "instruction": instruction})
        print(f"  [{ti}] {f.stem[:44]:44s} {n:6d} frames", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("done ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
