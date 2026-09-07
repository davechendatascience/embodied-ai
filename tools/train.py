#!/usr/bin/env python
"""Train an action head on cached frozen features.

Both policies share this loop, this data, and this optimiser. Only the target
and the conditioning differ, so a difference in transfer is attributable to the
action representation.

Splitting is BY DEMONSTRATION, not by frame. Consecutive frames of one
demonstration are near-duplicates; a frame-wise split puts near-copies of the
validation set into training and reports a validation loss that means nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from screwhead.interface import ActionSpec          # noqa: E402
from screwhead.libero import panda_chain            # noqa: E402
from screwhead.policy import MAX_DOF, BaselineHead, ScrewHead  # noqa: E402
from screwhead.spec import encode                   # noqa: E402


def load(cache: Path, chunk: int, policy: str):
    ag, wr, tx, st, tgt, demo, task = [], [], [], [], [], [], []
    spec = ActionSpec()
    for f in sorted(cache.glob("task*.npz")):
        d = np.load(f, allow_pickle=True)
        n = len(d["agent"])
        dem = d["demo"]
        # Chunk targets, never crossing a demonstration boundary.
        if policy == "baseline":
            act = np.concatenate([d["dq"], d["gripper"][:, None]], -1)      # (n, 8)
            act = np.pad(act, ((0, 0), (0, MAX_DOF + 1 - act.shape[1])))
        else:
            twist = d["twist"] / np.array(
                [spec.rot_scale * spec.control_hz] * 3 + [spec.pos_scale * spec.control_hz] * 3,
                np.float32)                                                  # embodiment-free scale
            act = np.concatenate([twist, d["gripper"][:, None]], -1)         # (n, 7)
        idx = np.arange(n)
        keep = np.array([i + chunk <= n and dem[i] == dem[min(i + chunk - 1, n - 1)]
                         for i in idx])
        idx = idx[keep]
        ag.append(d["agent"][idx]); wr.append(d["wrist"][idx])
        tx.append(np.repeat(d["text"][None], len(idx), 0))
        q = d["qpos"][idx]
        q = np.pad(q, ((0, 0), (0, MAX_DOF - q.shape[1])))
        st.append(np.concatenate([q, np.zeros((len(idx), 1), np.float32)], -1))
        tgt.append(np.stack([act[i:i + chunk] for i in idx]))
        demo.append(dem[idx].astype(np.int32) + 1000 * int(d["task_index"]))
        task.append(np.full(len(idx), int(d["task_index"]), np.int32))
    pack = lambda xs, dt=torch.float32: torch.tensor(np.concatenate(xs), dtype=dt)
    return (pack(ag), pack(wr), pack(tx), pack(st), pack(tgt),
            pack(demo, torch.long), pack(task, torch.long))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["baseline", "screwhead"], required=True)
    ap.add_argument("--cache", default="cache/libero_spatial")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-demos", type=int, default=5, help="held-out demos per task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    ag, wr, tx, st, tgt, demo, task = load(Path(args.cache), args.chunk, args.policy)
    print(f"{len(ag)} samples, {len(demo.unique())} demonstrations, {len(task.unique())} tasks")

    # split by demonstration id, so validation frames share no trajectory with training
    val_ids = set()
    for t in task.unique().tolist():
        ids = sorted(demo[task == t].unique().tolist())[: args.val_demos]
        val_ids.update(ids)
    is_val = torch.tensor([d.item() in val_ids for d in demo])
    print(f"  train {int((~is_val).sum())}  val {int(is_val.sum())} "
          f"({len(val_ids)} held-out demonstrations)")

    dev = args.device
    model = (BaselineHead(chunk=args.chunk) if args.policy == "baseline"
             else ScrewHead(chunk=args.chunk)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * max(1, int((~is_val).sum()) // args.batch))

    chain = panda_chain()
    sp = encode(chain)
    tok_all, mask_all = sp.padded(MAX_DOF)
    tok_all = tok_all.float().to(dev); mask_all = mask_all.to(dev)

    def batch_forward(i):
        a, w, t, s = ag[i].to(dev), wr[i].to(dev), tx[i].to(dev), st[i].to(dev)
        if args.policy == "baseline":
            return model(a, w, t, s, torch.zeros(len(i), dtype=torch.long, device=dev))
        return model(a, w, t, s, tok_all.expand(len(i), -1, -1), mask_all.expand(len(i), -1))

    tr_idx = torch.nonzero(~is_val).flatten()
    va_idx = torch.nonzero(is_val).flatten()
    best = float("inf")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ckpt = out / f"{args.policy}_libero_spatial.pt"
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = tr_idx[torch.randperm(len(tr_idx))]
        tot = n = 0
        for k in range(0, len(perm) - args.batch + 1, args.batch):
            i = perm[k:k + args.batch]
            loss = nn.functional.smooth_l1_loss(batch_forward(i), tgt[i].to(dev))
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
            tot += loss.item() * len(i); n += len(i)
        model.eval(); vtot = vn = 0
        with torch.no_grad():
            for k in range(0, len(va_idx), args.batch):
                i = va_idx[k:k + args.batch]
                vtot += nn.functional.smooth_l1_loss(
                    batch_forward(i), tgt[i].to(dev), reduction="sum").item()
                vn += i.numel() * tgt.shape[1] * tgt.shape[2]
        val = vtot / max(vn, 1)
        if val < best:
            best = val
            torch.save({"state_dict": model.state_dict(), "policy": args.policy,
                        "chunk": args.chunk, "val": val, "epoch": ep,
                        "args": vars(args)}, ckpt)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d}  train {tot/max(n,1):.5f}  val {val:.6f}"
                  f"{'  *' if val == best else ''}", flush=True)
    print(f"best val {best:.6f} -> {ckpt}  ({time.time()-t0:.0f}s)")
    (out / f"{args.policy}_libero_spatial.json").write_text(
        json.dumps({"best_val": best, "args": vars(args)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
