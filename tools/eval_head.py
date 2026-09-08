#!/usr/bin/env python
"""Score a trained CLIP-feature head on the full validation split.

Reports per-channel correlation between predicted and demonstrated action at the
first step of the chunk -- the same quantity the SigLIP trainer prints, so the
two are comparable -- but over every held-out demonstration of every task rather
than a prefix of them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from screwhead.libero import panda_chain                       # noqa: E402
from screwhead.policy import MAX_DOF, BaselineHead, ScrewHead   # noqa: E402
from screwhead.spec import encode                               # noqa: E402

from train import load                                          # noqa: E402

LAB7 = ["wx", "wy", "wz", "vx", "vy", "vz", "grip"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["baseline", "screwhead"], default="screwhead")
    ap.add_argument("--ckpt", default="checkpoints/screwhead_libero_spatial.pt")
    ap.add_argument("--cache", default="cache/libero_spatial")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--val-demos", type=int, default=5)
    ap.add_argument("--center", action="store_true",
                    help="standardise the cached visual features per dimension using "
                         "training-split statistics. Frozen encoder embeddings are "
                         "anisotropic: >90%% of their energy sits in one shared "
                         "direction, leaving the frame-to-frame signal as a few "
                         "percent residual that the first linear layer has to dig out.")
    ap.add_argument("--zero", default="none",
                    choices=["none", "image", "state", "text", "image+text"],
                    help="ablate an input by zeroing it in train and val alike, to "
                         "measure what the head is actually using")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--per-task", action="store_true")
    args = ap.parse_args()

    ag, wr, tx, st, tgt, demo, task, std = load(Path(args.cache), args.chunk, args.policy)
    if args.zero in ("image", "image+text"):
        ag.zero_(); wr.zero_()
    if args.zero in ("text", "image+text"):
        tx.zero_()
    if args.zero == "state":
        st.zero_()
    if args.zero != "none":
        print(f"ABLATION: {args.zero} zeroed in train and val")

    val_ids = set()
    for t in task.unique().tolist():
        val_ids.update(sorted(demo[task == t].unique().tolist())[: args.val_demos])
    is_val = torch.tensor([d.item() in val_ids for d in demo])
    if args.center:
        for f in (ag, wr):
            m = f[~is_val].mean(0, keepdim=True)
            sd = f[~is_val].std(0, keepdim=True).clamp(min=1e-6)
            f.sub_(m).div_(sd)
        print("features standardised per dimension on the training split")
    va = torch.nonzero(is_val).flatten()
    print(f"val {len(va)} windows over {len(task[is_val].unique())} tasks")

    dev = args.device
    model = (BaselineHead(chunk=args.chunk) if args.policy == "baseline"
             else ScrewHead(chunk=args.chunk)).to(dev)
    sd = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(sd["state_dict"])
    model.eval()
    tok, mask = encode(panda_chain()).padded(MAX_DOF)
    tok, mask = tok.float().to(dev), mask.to(dev)

    P = []
    with torch.no_grad():
        for k in range(0, len(va), args.batch):
            i = va[k:k + args.batch]
            a, w, t, s = ag[i].to(dev), wr[i].to(dev), tx[i].to(dev), st[i].to(dev)
            p = (model(a, w, t, s, torch.zeros(len(i), dtype=torch.long, device=dev))
                 if args.policy == "baseline" else
                 model(a, w, t, s, tok.expand(len(i), -1, -1), mask.expand(len(i), -1)))
            P.append(p[:, 0].cpu().numpy())
    P = np.concatenate(P)
    Y = tgt[va][:, 0].numpy()
    lab = LAB7 if args.policy == "screwhead" else [f"q{j}" for j in range(MAX_DOF)] + ["grip"]

    def report(name, p, y):
        c = [float(np.corrcoef(p[:, j], y[:, j])[0, 1]) for j in range(y.shape[1])]
        print(f"{name:<12} " + "  ".join(f"{l}:{v:.2f}" for l, v in zip(lab, c)), flush=True)

    report("all tasks", P, Y)
    if args.per_task:
        tv = task[va].numpy()
        for ti in sorted(set(tv.tolist())):
            m = tv == ti
            report(f"task {ti} (n={int(m.sum())})", P[m], Y[m])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
