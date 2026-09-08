#!/usr/bin/env python
"""Fine-tune Qwen2-VL with LoRA and an action head, on raw frames.

Everything below the backbone is unchanged from the CLIP runs on purpose: the
same tool-pose state, the same per-channel action standardisation, the same
twist/joint-delta split, the same split BY DEMONSTRATION. The backbone is the
only variable, so a difference in the result is attributable to it.

Only the adapters and the head are saved. The 2.2B base stays frozen and is
reloaded from the hub, so a checkpoint is megabytes.
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
from screwhead.interface import ActionSpec                    # noqa: E402
from screwhead.libero import panda_chain                      # noqa: E402
from screwhead.policy import MAX_DOF                          # noqa: E402
from screwhead.qwen_backbone import HIDDEN, QwenBackbone      # noqa: E402
from screwhead.spec import TOKEN_DIM, encode                  # noqa: E402
from screwhead.state import STATE_DIM, tool_state             # noqa: E402

JOINT_ACTION_SCALE = 0.05


class QwenHead(nn.Module):
    """Action head over a jointly-grounded VLM feature.

    Deliberately larger than the 4.26M CLIP head: with a 2B backbone the
    bottleneck is no longer the features, and the head has a 1536-d jointly
    grounded vector to work with rather than two concatenated unimodal ones.
    """

    def __init__(self, out_dim: int, chunk: int = 8, width: int = 1024,
                 spec_tokens: bool = True, heads: int = 8):
        super().__init__()
        self.chunk, self.out_dim = chunk, out_dim
        self.feat = nn.Linear(HIDDEN, width)
        self.state = nn.Linear(STATE_DIM, width)
        self.norm1 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(),
                                 nn.Linear(width * 2, width))
        self.norm2 = nn.LayerNorm(width)
        self.use_spec = spec_tokens
        if spec_tokens:
            self.spec_in = nn.Linear(TOKEN_DIM, width)
            self.spec_attn = nn.MultiheadAttention(width, heads, batch_first=True)
            self.norm3 = nn.LayerNorm(width)
        else:
            self.embodiment = nn.Embedding(32, width)
            self.emb_attn = nn.MultiheadAttention(width, heads, batch_first=True)
            self.norm3 = nn.LayerNorm(width)
        self.out = nn.Linear(width, chunk * out_dim)

    def forward(self, feat, state, tok=None, mask=None, eid=None):
        h = self.norm1(self.feat(feat) + self.state(state))
        h = self.norm2(h + self.mlp(h))
        if self.use_spec:
            t = self.spec_in(tok)
            att, _ = self.spec_attn(h[:, None], t, t, key_padding_mask=~mask,
                                    need_weights=False)
        else:
            t = self.embodiment(eid)[:, None]
            att, _ = self.emb_attn(h[:, None], t, t, need_weights=False)
        h = self.norm3(h + att[:, 0])
        return self.out(h).view(-1, self.chunk, self.out_dim)


def load_meta(cache: Path, chunk: int, policy: str):
    """Window index plus per-task arrays. Frames stay on disk as memmaps."""
    spec, chain = ActionSpec(), panda_chain()
    sc = np.array([spec.rot_scale * spec.control_hz] * 3 +
                  [spec.pos_scale * spec.control_hz] * 3, np.float32)
    items, tasks = [], {}
    for meta in sorted(cache.glob("task*_meta.npz")):
        d = np.load(meta, allow_pickle=True)
        ti, dem = int(d["task_index"]), d["demo"]
        act = (np.concatenate([d["twist"] / sc, d["gripper"][:, None]], -1)
               if policy == "screwhead" else
               np.concatenate([d["dq"] / JOINT_ACTION_SCALE, d["gripper"][:, None]], -1))
        st = tool_state(chain, torch.tensor(d["qpos"], dtype=torch.float64),
                        torch.tensor(d["gripper_state"], dtype=torch.float64)).float().numpy()
        tasks[ti] = dict(act=act.astype(np.float32), state=st, demo=dem,
                         instruction=str(d["instruction"]),
                         agent=cache / f"task{ti:02d}_agent.npy",
                         wrist=cache / f"task{ti:02d}_wrist.npy")
        # A window must lie inside ONE demonstration: spanning a boundary would
        # ask the policy to predict the start of the next episode from the end
        # of the previous one.
        for i in range(len(dem) - chunk):
            if dem[i] == dem[i + chunk - 1]:
                items.append((ti, i, int(dem[i])))
    return items, tasks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["baseline", "screwhead"], required=True)
    ap.add_argument("--cache", default="cache/libero_spatial_img")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--val-demos", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    cache = Path(args.cache)
    items, tasks = load_meta(cache, args.chunk, args.policy)
    mm = {ti: (np.load(t["agent"], mmap_mode="r"), np.load(t["wrist"], mmap_mode="r"))
          for ti, t in tasks.items()}
    val_ids = {ti: set(sorted(np.unique(t["demo"]).tolist())[: args.val_demos])
               for ti, t in tasks.items()}
    tr = [x for x in items if x[2] not in val_ids[x[0]]]
    va = [x for x in items if x[2] in val_ids[x[0]]]
    print(f"{len(items)} windows: train {len(tr)} val {len(va)} over {len(tasks)} tasks")

    out_dim = 7 if args.policy == "screwhead" else MAX_DOF + 1
    all_act = np.concatenate([tasks[t]["act"] for t in sorted(tasks)])
    act_std = torch.tensor(all_act.std(0), dtype=torch.float32).clamp(min=1e-6)
    print("action channel std:", [round(float(v), 4) for v in act_std])

    backbone = QwenBackbone(lora_r=args.lora_r, device="cuda")
    head = QwenHead(out_dim, chunk=args.chunk,
                    spec_tokens=(args.policy == "screwhead")).cuda()
    tok, mask = encode(panda_chain()).padded(MAX_DOF)
    tok, mask = tok.float().cuda(), mask.cuda()
    print(f"trainable: LoRA {backbone.trainable_parameters()/1e6:.2f}M + "
          f"head {sum(p.numel() for p in head.parameters())/1e6:.2f}M")

    params = [p for p in backbone.model.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    def batch_of(sel):
        ag = np.stack([mm[t][0][i] for t, i, _ in sel])
        wr = np.stack([mm[t][1][i] for t, i, _ in sel])
        instr = [tasks[t]["instruction"] for t, _, _ in sel]
        st = torch.tensor(np.stack([tasks[t]["state"][i] for t, i, _ in sel])).cuda()
        y = torch.tensor(np.stack([tasks[t]["act"][i:i + args.chunk] for t, i, _ in sel])).cuda()
        return ag, wr, instr, st, y / act_std.cuda()

    def forward(sel):
        ag, wr, instr, st, y = batch_of(sel)
        f = backbone.encode(ag, wr, instr).float()
        n = len(sel)
        p = (head(f, st, tok.expand(n, -1, -1), mask.expand(n, -1))
             if args.policy == "screwhead"
             else head(f, st, eid=torch.zeros(n, dtype=torch.long, device="cuda")))
        return nn.functional.smooth_l1_loss(p, y)

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    ck = outdir / f"qwen_{args.policy}_libero_spatial"
    best, t0 = float("inf"), time.time()
    rng = np.random.default_rng(args.seed)
    for ep in range(args.epochs):
        backbone.model.train(); head.train()
        perm = rng.permutation(len(tr)); tot = n = 0
        opt.zero_grad(set_to_none=True)
        for k in range(0, len(perm) - args.batch + 1, args.batch):
            sel = [tr[j] for j in perm[k:k + args.batch]]
            loss = forward(sel) / args.accum
            loss.backward()
            if (k // args.batch + 1) % args.accum == 0:
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
            tot += loss.item() * args.accum * len(sel); n += len(sel)
            if (k // args.batch) % 50 == 0:
                print(f"  ep{ep} step {k//args.batch} loss {tot/max(n,1):.5f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        backbone.model.eval(); head.eval()
        vt = vn = 0
        with torch.no_grad():
            for k in range(0, min(len(va), 400), args.batch):
                sel = va[k:k + args.batch]
                if len(sel) < 2: break
                vt += forward(sel).item() * len(sel); vn += len(sel)
        val = vt / max(vn, 1)
        print(f"epoch {ep}  train {tot/max(n,1):.5f}  val {val:.5f}", flush=True)
        if val < best:
            best = val
            ck.mkdir(parents=True, exist_ok=True)
            backbone.save_adapter(ck)                       # LoRA only
            torch.save({"head": head.state_dict(), "policy": args.policy,
                        "chunk": args.chunk, "act_std": act_std, "val": val,
                        "epoch": ep, "args": vars(args)}, ck / "head.pt")
    print(f"best val {best:.5f} -> {ck}  ({time.time()-t0:.0f}s)")
    (ck / "summary.json").write_text(json.dumps({"best_val": best, "args": vars(args)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
