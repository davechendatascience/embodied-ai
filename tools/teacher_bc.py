#!/usr/bin/env python
"""Clone the privileged teacher from demonstrations.

Standardises observations and every action channel from training statistics --
the rotation channels peak at 0.03-0.04 of their range against 0.26 for
translation, so an unscaled loss trains translation and ignores rotation
(FM-joint-normalisation), and an unscaled Gaussian policy would later explore
rotation at several times its demonstrated magnitude.

Reports the object-pose ablation beside the result. A teacher cloned from demos
with 12 mm of placement variance has little reason to read the object pose it
is given; if zeroing it costs nothing, the clone has the student's shortcut and
only the randomised fine-tune can remove it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
OBJECT_SLICE = slice(17, 35)       # bowl pose, plate position, both relative vectors
LAB = ["wx", "wy", "wz", "vx", "vy", "vz", "grip"]


class TeacherMLP(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, act_dim))

    def forward(self, x):
        return self.net(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/teacher_bc")
    ap.add_argument("--out", default="checkpoints/teacher_bc.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-demos", type=int, default=5)
    ap.add_argument("--zero-objects", action="store_true",
                    help="train and evaluate with the object-pose features zeroed")
    ap.add_argument("--drop-joints", action="store_true",
                    help="mask the 7 raw joint angles. The twist servo reproduces tool pose, "
                         "not the null-space posture (measured drift 0.26-0.36 rad), so a "
                         "teacher reading raw joints sees configurations it never trained on.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    O, A, V = [], [], []
    for f in sorted(Path(args.cache).glob("task*.npz")):
        d = np.load(f)
        val = set(sorted(np.unique(d["demo"]).tolist())[: args.val_demos])
        O.append(d["obs"]); A.append(d["act"]); V.append(np.isin(d["demo"], list(val)))
    O, A, V = np.concatenate(O), np.concatenate(A), np.concatenate(V)
    if args.zero_objects:
        O = O.copy(); O[:, OBJECT_SLICE] = 0.0
    mask = np.ones(O.shape[1], np.float32)
    if args.drop_joints:
        mask[0:7] = 0.0
    if args.zero_objects:
        mask[OBJECT_SLICE] = 0.0
    tr = ~V
    obs_mu, obs_sd = O[tr].mean(0), O[tr].std(0).clip(1e-6)
    act_sd = A[tr].std(0).clip(1e-6)
    print(f"{len(O)} frames: train {int(tr.sum())} val {int(V.sum())}")
    print("action std:", dict(zip(LAB, act_sd.round(4))))

    dev = args.device
    X = torch.tensor((O - obs_mu) / obs_sd * mask, dtype=torch.float32, device=dev)
    Y = torch.tensor(A / act_sd, dtype=torch.float32, device=dev)
    tri, vai = torch.nonzero(torch.tensor(tr)).flatten(), torch.nonzero(torch.tensor(V)).flatten()
    model = TeacherMLP(O.shape[1], A.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best, best_state = float("inf"), None
    for ep in range(args.epochs):
        model.train()
        perm = tri[torch.randperm(len(tri), device="cpu")].to(dev)
        for k in range(0, len(perm) - args.batch + 1, args.batch):
            i = perm[k:k + args.batch]
            loss = nn.functional.smooth_l1_loss(model(X[i]), Y[i])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = nn.functional.smooth_l1_loss(model(X[vai.to(dev)]), Y[vai.to(dev)]).item()
        if vl < best:
            best, best_state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d}  train {loss.item():.4f}  val {vl:.4f}{'  *' if vl == best else ''}")
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        P = model(X[vai.to(dev)]).cpu().numpy() * act_sd
    Yv = A[V]
    corr = [float(np.corrcoef(P[:, j], Yv[:, j])[0, 1]) for j in range(A.shape[1])]
    print(f"best val {best:.4f}   corr " + "  ".join(f"{l}:{c:.2f}" for l, c in zip(LAB, corr)))

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "obs_mu": obs_mu, "obs_sd": obs_sd, "act_sd": act_sd,
                "obs_dim": int(O.shape[1]), "act_dim": int(A.shape[1]), "val": best, "obs_mask": mask,
                "corr": corr, "zero_objects": args.zero_objects, "args": vars(args)}, out)
    Path(str(out) + ".json").write_text(json.dumps({"val": best, "corr": corr,
                                                    "zero_objects": args.zero_objects}, indent=2))
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
