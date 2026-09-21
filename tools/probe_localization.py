#!/usr/bin/env python
"""How precisely can the VLA's own inputs locate the grasp?

The scripted teacher grasps a 3 mm bowl wall and closes only within ~6 mm of the
grasp pose. The VLA sees each camera as one pooled frozen-CLIP vector. If those
inputs cannot place the grasp to that precision, more labels will not help --
the student cannot reproduce a servo on a quantity it cannot perceive.

This trains probes from exactly the student's inputs to the privileged grasp
error recorded by tools/distill.py (p_grasp - p_tool, base frame; analysis only)
and reports held-out error in millimetres -- overall and near the grasp, where
the tolerance bites -- for four input sets:

  all     both cameras + language + tool pose   (what the VLA gets)
  blind   language + tool pose                  (no vision: task identity and arm only)
  agent   third-person camera + pose
  wrist   wrist camera + pose
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch import nn

NEAR_GRASP = 0.06          # m, tool-to-grasp distance: "tool <60 mm"
VERY_NEAR_GRASP = 0.025    # m, "tool <25 mm"
MIN_SLICE = 20             # fewer held-out states than this and a slice reports n/a


def load(paths):
    parts = [np.load(p) for p in paths]
    cat = lambda k: np.concatenate([p[k] for p in parts])
    src = np.concatenate([np.full(len(p["label"]), i) for i, p in enumerate(parts)])
    ep = np.concatenate([p["episode"] + 10_000_000 * i for i, p in enumerate(parts)])
    text = parts[0]["text"]
    task = cat("task").astype(int)
    return dict(agent=cat("agent").astype(np.float32), wrist=cat("wrist").astype(np.float32),
                text=text[task].astype(np.float32), state=cat("state").astype(np.float32),
                priv=cat("priv").astype(np.float32), phase=cat("phase"), episode=ep, src=src)


def fit(X, Y, tr, va, epochs=60, dev="cuda"):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    ym, ys = Y[tr].mean(0), Y[tr].std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd, device=dev); Yt = torch.tensor((Y - ym) / ys, device=dev)
    net = nn.Sequential(nn.Linear(X.shape[1], 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 512), nn.GELU(),
                        nn.Linear(512, Y.shape[1])).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-3)
    tri = torch.tensor(np.where(tr)[0], device=dev); vai = torch.tensor(np.where(va)[0], device=dev)
    best, best_state = 1e9, None
    for _ in range(epochs):
        net.train()
        for _ in range(0, len(tri), 1024):
            i = tri[torch.randperm(len(tri), device=dev)[:1024]]
            loss = nn.functional.smooth_l1_loss(net(Xt[i]), Yt[i])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = nn.functional.smooth_l1_loss(net(Xt[vai]), Yt[vai]).item()
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        return net(Xt[vai]).cpu().numpy() * ys + ym


def _median_p90(err, m) -> str:
    """'median / p90' of the errors selected by mask m, or n/a when the slice is too thin."""
    if m.sum() > MIN_SLICE:
        return f"{np.median(err[m]):5.1f} / {np.percentile(err[m], 90):5.1f}"
    return "   n/a       "


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+")
    args = ap.parse_args()
    D = load(args.data)
    Y = D["priv"][:, :3]                                   # grasp position error, m
    va = (D["episode"] % 5) == 0; tr = ~va
    dist = np.linalg.norm(Y, axis=1)
    phases_close = np.isin(D["phase"], ["descend", "close", "approach"])
    print(f"{len(Y)} states, {len(np.unique(D['episode']))} episodes; validation {va.sum()} states")
    sets = {
        "all":   np.concatenate([D["agent"], D["wrist"], D["text"], D["state"]], 1),
        "blind": np.concatenate([D["text"], D["state"]], 1),
        "agent": np.concatenate([D["agent"], D["text"], D["state"]], 1),
        "wrist": np.concatenate([D["wrist"], D["text"], D["state"]], 1),
    }
    Yv, dv, sv, pv = Y[va], dist[va], D["src"][va], phases_close[va]
    print(f"\n{'inputs':6s} | grasp-position error on held-out episodes, mm (median / p90)")
    print(f"{'':6s} | {'all states':>14s} {'tool <60 mm':>14s} {'tool <25 mm':>14s} | {'teacher-driven <60':>18s} {'VLA-driven <60':>15s}")
    for name, X in sets.items():
        P = fit(X, Y, tr, va)
        err = np.linalg.norm(P - Yv, axis=1) * 1000
        near, vnear = (dv < NEAR_GRASP) & pv, (dv < VERY_NEAR_GRASP) & pv
        print(f"{name:6s} | {_median_p90(err, np.ones(len(err), bool)):>14s} {_median_p90(err, near):>14s} "
              f"{_median_p90(err, vnear):>14s} | "
              f"{_median_p90(err, near & (sv == 0)):>18s} {_median_p90(err, near & (sv == 1)):>15s}")
    print("\nreference: the teacher closes only within ~6 mm of the grasp pose; the wall is 3 mm thick")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
