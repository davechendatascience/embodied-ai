#!/usr/bin/env python
"""Which frozen visual features can locate the grasp to the precision the task needs?

Same targets and split as tools/probe_localization.py -- the privileged grasp
position error (p_grasp - p_tool, analysis only), held-out episodes, errors near
the grasp -- but the images are re-encoded from high-resolution renders by each
candidate, and the probe has the same capacity for every variant:

  per-token linear (128 -> 32) + learned position embedding, flattened per camera,
  concatenated with language (512 -> 64) and tool pose, then a 2-layer MLP.

Pooled features are one token; patch features keep an 8x8 (or 7x7) grid, pooled
from the encoder's own. Each token is first reduced to 128-d by a fixed random
projection (seeded, identical across a variant's cameras) to bound memory.

Variants:
  clip_pool@128   the VLA's current input: 128 px, CLIP ViT-B/32, pooled
  clip_patch@224  CLIP ViT-B/32 patch tokens, 7x7
  siglip@224      SigLIP so400m patch tokens, 27x27 -> 8x8
  dinov2@224      DINOv2-base patch tokens, 16x16 -> 8x8
  dinov2@128      the same, from a 128 px image (resolution vs encoder)
  blind           language + tool pose only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
MEAN = dict(clip=(0.48145466, 0.4578275, 0.40821073), siglip=(0.5, 0.5, 0.5), dino=(0.485, 0.456, 0.406))
STD = dict(clip=(0.26862954, 0.26130258, 0.27577711), siglip=(0.5, 0.5, 0.5), dino=(0.229, 0.224, 0.225))


def load_encoder(kind, dev):
    from transformers import AutoModel
    mid = {"clip": "openai/clip-vit-base-patch32", "siglip": "google/siglip-so400m-patch14-384",
           "dino": "facebook/dinov2-base"}[kind]
    m = AutoModel.from_pretrained(mid, dtype=torch.float16).to(dev).eval()
    return m.vision_model if hasattr(m, "vision_model") else m


@torch.no_grad()
def encode(frames, kind, mode, in_res, dev, proj, batch=64):
    """frames: (N,H,W,3) uint8 memmap. Returns (N, T, 128) float16."""
    enc = load_encoder(kind, dev)
    mean = torch.tensor(MEAN[kind], device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD[kind], device=dev).view(1, 3, 1, 1)
    net_res = 384 if kind == "siglip" else 224
    out = []
    for k in range(0, len(frames), batch):
        x = torch.from_numpy(np.asarray(frames[k:k + batch])).to(dev).permute(0, 3, 1, 2).float() / 255.0
        if in_res != x.shape[-1]:
            x = nn.functional.interpolate(x, size=in_res, mode="area")
        x = nn.functional.interpolate(x, size=net_res, mode="bilinear", align_corners=False)
        x = ((x - mean) / std).half()
        o = enc(pixel_values=x)
        if mode == "pool":
            tok = o.pooler_output[:, None]
        else:
            h = o.last_hidden_state
            if kind in ("clip", "dino"):
                h = h[:, 1:]
            side = int(round(h.shape[1] ** 0.5))
            g = h.float().transpose(1, 2).reshape(len(h), -1, side, side)
            grid = 7 if side == 7 else 8
            g = nn.functional.adaptive_avg_pool2d(g, grid)
            tok = g.flatten(2).transpose(1, 2)
        t = tok.float() if proj is None else tok.float() @ proj[: tok.shape[-1]]   # (N, T, D) @ (D, 128)
        out.append(t.half().cpu())
    del enc; torch.cuda.empty_cache()
    return torch.cat(out).numpy()


class Probe(nn.Module):
    def __init__(self, n_tok, d=128):
        super().__init__()
        self.n_tok = n_tok
        self.tok = nn.Sequential(nn.Linear(d, 32), nn.GELU()) if n_tok else None
        self.pos = nn.Parameter(torch.zeros(2, max(n_tok, 1), 32)) if n_tok else None
        self.txt = nn.Linear(512, 64)
        width = 2 * n_tok * 32 + 64 + 10
        self.mlp = nn.Sequential(nn.Linear(width, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 3))

    def forward(self, a, w, t, s):
        parts = []
        if self.n_tok:
            parts += [(self.tok(a) + self.pos[0]).flatten(1), (self.tok(w) + self.pos[1]).flatten(1)]
        return self.mlp(torch.cat(parts + [self.txt(t), s], 1))


def train_probe(A, W, T, S, Y, tr, va, dev, epochs=40):
    n_tok = 0 if A is None else A.shape[1]
    d_in = 128 if A is None else A.shape[2]
    ym, ys = Y[tr].mean(0), Y[tr].std(0) + 1e-6
    sm, ss = S[tr].mean(0), S[tr].std(0) + 1e-6
    to = lambda x: torch.tensor(x, device=dev, dtype=torch.float32)
    At = None if A is None else to(A); Wt = None if W is None else to(W)
    Tt, St, Yt = to(T), to((S - sm) / ss), to((Y - ym) / ys)
    if At is not None:
        mu = At[torch.tensor(np.where(tr)[0], device=dev)].mean((0, 1), keepdim=True)
        sd = At[torch.tensor(np.where(tr)[0], device=dev)].std((0, 1), keepdim=True) + 1e-6
        At, Wt = (At - mu) / sd, (Wt - mu) / sd
    net = Probe(n_tok, d=d_in).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-3)
    tri = torch.tensor(np.where(tr)[0], device=dev); vai = torch.tensor(np.where(va)[0], device=dev)
    fwd = lambda i: net(None if At is None else At[i], None if Wt is None else Wt[i], Tt[i], St[i])
    best, state = 1e9, None
    for ep in range(epochs):
        net.train()
        perm = tri[torch.randperm(len(tri), device=dev)]
        for k in range(0, len(perm), 512):
            i = perm[k:k + 512]
            loss = nn.functional.smooth_l1_loss(fwd(i), Yt[i])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = sum(nn.functional.smooth_l1_loss(fwd(vai[k:k + 2048]), Yt[vai[k:k + 2048]], reduction="sum").item()
                     for k in range(0, len(vai), 2048)) / len(vai)
        if vl < best:
            best, state = vl, {k_: v.clone() for k_, v in net.state_dict().items()}
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        P = torch.cat([fwd(vai[k:k + 2048]) for k in range(0, len(vai), 2048)]).cpu().numpy()
    return P * ys + ym


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["cache/bakeoff/teacher.npz", "cache/bakeoff/student.npz"])
    ap.add_argument("--variants", nargs="+", default=["blind", "clip_pool@128", "clip_patch@224", "siglip@224", "dinov2@224", "dinov2@128"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--full-dim", action="store_true", help="keep full token dimension (no random projection)")
    ap.add_argument("--student-frac", type=float, nargs="+", default=[1.0],
                    help="fractions of VLA-driven training episodes (learning curve)")
    ap.add_argument("--probe-seeds", type=int, default=1)
    args = ap.parse_args()
    dev = args.device
    metas = [np.load(p) for p in args.data]
    his = [(np.load(p + ".agent_hi.npy", mmap_mode="r"), np.load(p + ".wrist_hi.npy", mmap_mode="r")) for p in args.data]
    cat = lambda k: np.concatenate([m[k] for m in metas])
    task = cat("task").astype(int)
    T = metas[0]["text"][task].astype(np.float32)
    S = cat("state").astype(np.float32); Y = cat("priv")[:, :3].astype(np.float32)
    ph = cat("phase"); src = np.concatenate([np.full(len(m["label"]), i) for i, m in enumerate(metas)])
    ep = np.concatenate([m["episode"] + 10_000_000 * i for i, m in enumerate(metas)])
    va = (ep % 5) == 0; tr = ~va
    dist = np.linalg.norm(Y, axis=1)
    near = (dist < 0.06) & np.isin(ph, ["approach", "descend", "close"])
    print(f"{len(Y)} states from {len(np.unique(ep))} episodes; held-out {va.sum()}; near-grasp held-out: "
          f"teacher-driven {int((near & va & (src == 0)).sum())}, VLA-driven {int((near & va & (src == 1)).sum())}")
    g = torch.Generator().manual_seed(0)
    proj = None if args.full_dim else (torch.randn(1152, 128, generator=g) / np.sqrt(128)).to(dev)
    results = []
    for v in args.variants:
        if v == "blind":
            A = W = None
        else:
            name, res = v.split("@"); res = int(res)
            kind = {"clip_pool": "clip", "clip_patch": "clip", "siglip": "siglip", "dinov2": "dino"}[name]
            mode = "pool" if name == "clip_pool" else "patch"
            A = np.concatenate([encode(h[0], kind, mode, res, dev, proj) for h in his])
            W = np.concatenate([encode(h[1], kind, mode, res, dev, proj) for h in his])
        student_eps = np.unique(ep[tr & (src == 1)])
        rng = np.random.default_rng(0)
        order = rng.permutation(student_eps)
        for frac in args.student_frac:
            keep = set(order[: max(1, int(round(frac * len(order))))].tolist())
            tr_f = tr & ((src == 0) | np.isin(ep, list(keep)))
            for seed in range(args.probe_seeds):
                torch.manual_seed(seed)
                pred = train_probe(A, W, T, S, Y, tr_f, va, dev)
                err = np.linalg.norm(pred - Y[va], axis=1) * 1000
                nv, sv = near[va], src[va]
                stat = lambda m: (np.median(err[m]), np.percentile(err[m], 90)) if m.sum() > 20 else (np.nan, np.nan)
                r = dict(variant=v, frac=frac, seed=seed, all=stat(np.ones(len(err), bool)),
                         t_near=stat(nv & (sv == 0)), s_near=stat(nv & (sv == 1)))
                results.append(r)
                print(f"  {v:16s} VLA-driven train episodes {int(round(frac * len(order))):3d} ({frac:.2f}) seed {seed}  "
                      f"near grasp, teacher-driven {r['t_near'][0]:5.1f}/{r['t_near'][1]:5.1f}"
                      f"   near grasp, VLA-driven {r['s_near'][0]:5.1f}/{r['s_near'][1]:5.1f}   (mm, median/p90)", flush=True)
    print("\nreference: the teacher closes within ~6 mm of the grasp pose; the bowl wall is 3 mm thick")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
