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

JOINT_ACTION_SCALE = 0.05      # robosuite joint_position.json output_max
from screwhead.libero import panda_chain            # noqa: E402
from screwhead.policy import MAX_DOF, BaselineHead, ScrewHead  # noqa: E402
from screwhead.state import STATE_DIM, tool_state  # noqa: E402
from screwhead.spec import encode                   # noqa: E402


def load(cache: Path, chunk: int, policy: str):
    ag, wr, tx, st, tgt, demo, task = [], [], [], [], [], [], []
    spec = ActionSpec()
    chain = panda_chain()          # the arm the demonstrations were collected on
    for f in sorted(cache.glob("task*.npz")):
        d = np.load(f, allow_pickle=True)
        n = len(d["agent"])
        dem = d["demo"]
        # Chunk targets, never crossing a demonstration boundary.
        if policy == "baseline":
            # Normalise by the controller's own action scale, exactly as the
            # twist path divides by pos_scale/rot_scale. Without it the gripper
            # (+/-1) and the joint deltas (std 0.015) share one loss at a 68x
            # scale ratio, so the gripper carries 99.8% of the squared error and
            # the seven joints that matter get almost no gradient. That is
            # FM-joint-normalisation, and it produces a policy whose action
            # magnitude is right and whose direction is uncorrelated.
            act = np.concatenate([d["dq"] / JOINT_ACTION_SCALE,
                                  d["gripper"][:, None]], -1)               # (n, 8)
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
        # Embodiment-free proprioception: tool pose from this arm's own forward
        # kinematics, not its joint angles. Raw q is arm-specific -- the Panda's
        # j5 spans [+1.02,+3.44] while a UR5e sits at -1.991 there -- so a policy
        # fed q fails an arm swap on the STATE before its action representation
        # is ever tested.
        gs = d["gripper_state"][idx] if "gripper_state" in d.files else None
        st.append(tool_state(chain,
                             torch.tensor(d["qpos"][idx], dtype=torch.float64),
                             None if gs is None else torch.tensor(gs, dtype=torch.float64),
                             ).float().numpy())
        tgt.append(np.stack([act[i:i + chunk] for i in idx]))
        demo.append(dem[idx].astype(np.int32) + 1000 * int(d["task_index"]))
        task.append(np.full(len(idx), int(d["task_index"]), np.int32))
    pack = lambda xs, dt=torch.float32: torch.tensor(np.concatenate(xs), dtype=dt)
    tgt = pack(tgt)

    # Standardise every action channel to unit variance before the loss sees it.
    # Without this the gripper (+/-1) drowns the rest: measured on this data it
    # carries 99.8% of the squared error for the joint head, and 96.5% for the
    # twist head where the ANGULAR channels come to 0.05%. A shared loss over
    # channels spanning 68x in scale trains the largest and ignores the others --
    # FM-joint-normalisation, whose signature is exactly what we saw: action
    # magnitude correct, direction uncorrelated.
    #
    # Statistics are pooled over the whole dataset and never fitted per robot,
    # so they stay embodiment-independent (FM-embodiment-statistics).
    std = tgt.reshape(-1, tgt.shape[-1]).std(0).clamp(min=1e-6)
    tgt = tgt / std
    return (pack(ag), pack(wr), pack(tx), pack(st), tgt,
            pack(demo, torch.long), pack(task, torch.long), std)


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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    ag, wr, tx, st, tgt, demo, task, act_std = load(Path(args.cache), args.chunk, args.policy)
    print("action channel std before normalisation:",
          [round(float(v), 4) for v in act_std])
    print(f"{len(ag)} samples, {len(demo.unique())} demonstrations, {len(task.unique())} tasks")

    # split by demonstration id, so validation frames share no trajectory with training
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
        ids = sorted(demo[task == t].unique().tolist())[: args.val_demos]
        val_ids.update(ids)
    is_val = torch.tensor([d.item() in val_ids for d in demo])
    if args.center:
        for f in (ag, wr):
            m = f[~is_val].mean(0, keepdim=True)
            sd = f[~is_val].std(0, keepdim=True).clamp(min=1e-6)
            f.sub_(m).div_(sd)
        print("features standardised per dimension on the training split")
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
                        "act_std": act_std, "args": vars(args)}, ckpt)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d}  train {tot/max(n,1):.5f}  val {val:.6f}"
                  f"{'  *' if val == best else ''}", flush=True)
    print(f"best val {best:.6f} -> {ckpt}  ({time.time()-t0:.0f}s)")
    (out / f"{args.policy}_libero_spatial.json").write_text(
        json.dumps({"best_val": best, "args": vars(args)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
