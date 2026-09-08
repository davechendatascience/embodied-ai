#!/usr/bin/env python
"""Score a trained SigLIP patch head on the full validation split.

Same metric as tools/eval_head.py, so a patch-token policy and a pooled-feature
policy can be compared on one table.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from screwhead.libero import panda_chain          # noqa: E402
from screwhead.policy import MAX_DOF              # noqa: E402
from screwhead.spec import encode                 # noqa: E402

from train_siglip import PatchHead, load_meta     # noqa: E402

LAB = ["wx", "wy", "wz", "vx", "vy", "vz", "grip"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default="cache/libero_spatial_siglip")
    ap.add_argument("--policy", choices=["baseline", "screwhead"], default="screwhead")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--val-demos", type=int, default=5)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--zero", default="none", choices=["none", "image", "text", "image+text"])
    ap.add_argument("--per-task", action="store_true")
    args = ap.parse_args()

    cache = Path(args.cache)
    items, tasks = load_meta(cache, args.chunk, args.policy)
    mm = {ti: (np.load(t["agent"], mmap_mode="r"), np.load(t["wrist"], mmap_mode="r"))
          for ti, t in tasks.items()}
    val_ids = {ti: set(sorted(np.unique(t["demo"]).tolist())[: args.val_demos])
               for ti, t in tasks.items()}
    va = [x for x in items if x[2] in val_ids[x[0]]]
    print(f"val {len(va)} windows over {len(tasks)} tasks")

    sd = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    out_dim = 7 if args.policy == "screwhead" else MAX_DOF + 1
    head = PatchHead(out_dim, chunk=args.chunk,
                     spec_tokens=(args.policy == "screwhead")).cuda()
    head.load_state_dict(sd["head"]); head.eval()
    tok, mask = encode(panda_chain()).padded(MAX_DOF)
    tok, mask = tok.float().cuda(), mask.cuda()

    P, Y, T = [], [], []
    with torch.no_grad():
        for k in range(0, len(va), args.batch):
            sel = va[k:k + args.batch]
            fa = torch.tensor(np.stack([mm[t][0][i] for t, i, _ in sel]),
                              dtype=torch.float32).cuda()
            fw = torch.tensor(np.stack([mm[t][1][i] for t, i, _ in sel]),
                              dtype=torch.float32).cuda()
            if args.zero in ("image", "image+text"):
                fa = torch.zeros_like(fa); fw = torch.zeros_like(fw)
            st = torch.tensor(np.stack([tasks[t]["state"][i] for t, i, _ in sel])).cuda()
            tx = torch.tensor(np.stack([tasks[t]["text"] for t, _, _ in sel])).cuda()
            if args.zero in ("text", "image+text"):
                tx = torch.zeros_like(tx)
            n = len(sel)
            p = (head(fa, fw, st, tx, tok.expand(n, -1, -1), mask.expand(n, -1))
                 if args.policy == "screwhead" else
                 head(fa, fw, st, tx, eid=torch.zeros(n, dtype=torch.long, device="cuda")))
            P.append(p[:, 0].cpu().numpy())
            Y.append(np.stack([tasks[t]["act"][i] for t, i, _ in sel]))
            T.append(np.array([t for t, _, _ in sel]))
    P, Y, T = np.concatenate(P), np.concatenate(Y), np.concatenate(T)
    # the head predicts standardised actions; correlation is scale-free either way
    def report(name, p, y):
        c = [float(np.corrcoef(p[:, j], y[:, j])[0, 1]) for j in range(y.shape[1])]
        print(f"{name:<16} " + "  ".join(f"{l}:{v:.2f}" for l, v in zip(LAB, c)), flush=True)
    report("all tasks", P, Y)
    if args.per_task:
        for ti in sorted(set(T.tolist())):
            m = T == ti
            report(f"task {ti} (n={int(m.sum())})", P[m], Y[m])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
