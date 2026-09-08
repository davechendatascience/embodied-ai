#!/usr/bin/env python
"""Train an action head on SigLIP PATCH tokens, backbone run live.

The variable under test is spatial detail. Everything else is held at what the
CLIP runs established: tool-pose state, per-channel action standardisation, the
twist/joint-delta split, split by demonstration. The decisive metric is not the
loss but the per-channel correlation on wx/wy, where pooled CLIP tops out at
0.58 for a linear probe and 0.54 for a trained head.

The head cross-attends a learned query over 128 visual tokens (64 per camera)
plus the instruction, so it can look WHERE the task refers rather than reading
one averaged vector.
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
from screwhead.interface import ActionSpec                     # noqa: E402
from screwhead.libero import panda_chain                       # noqa: E402
from screwhead.policy import MAX_DOF                           # noqa: E402
from screwhead.siglip_backbone import DIM, SiglipBackbone      # noqa: E402
from screwhead.spec import TOKEN_DIM, encode                   # noqa: E402
from screwhead.state import STATE_DIM, tool_state              # noqa: E402

JOINT_ACTION_SCALE = 0.05


class PatchHead(nn.Module):
    """Cross-attention over visual patches, conditioned on state and spec."""

    def __init__(self, out_dim: int, chunk: int = 8, width: int = 512,
                 heads: int = 8, spec_tokens: bool = True, queries: int = 4):
        super().__init__()
        self.chunk, self.out_dim = chunk, out_dim
        self.proj = nn.Linear(DIM, width)
        self.cam = nn.Parameter(torch.zeros(2, 1, width))      # which camera a token came from
        self.state = nn.Linear(STATE_DIM, width)
        # The instruction enters as an extra key/value the queries can attend to,
        # rather than being added to the query. That lets the model select image
        # regions BY the instruction, which is the point of keeping patches.
        self.text = nn.Linear(DIM, width)
        self.q = nn.Parameter(torch.randn(queries, width) * 0.02)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(),
                                 nn.Linear(width * 2, width))
        self.norm2 = nn.LayerNorm(width)
        self.use_spec = spec_tokens
        if spec_tokens:
            self.spec_in = nn.Linear(TOKEN_DIM, width)
            self.spec_attn = nn.MultiheadAttention(width, heads, batch_first=True)
        else:
            self.embodiment = nn.Embedding(32, width)
            self.spec_attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm3 = nn.LayerNorm(width)
        self.out = nn.Linear(width * queries, chunk * out_dim)

    def forward(self, agent, wrist, state, text, tok=None, mask=None, eid=None):
        b, n, _ = agent.shape
        v = torch.cat([self.proj(agent) + self.cam[0],
                       self.proj(wrist) + self.cam[1],
                       self.text(text)[:, None]], 1)
        s = self.state(state)[:, None]
        q = self.q[None].expand(b, -1, -1) + s
        h, _ = self.attn(q, v, v, need_weights=False)
        h = self.norm1(h + q)
        h = self.norm2(h + self.mlp(h))
        ctx = (self.spec_in(tok) if self.use_spec else self.embodiment(eid)[:, None])
        a, _ = self.spec_attn(h, ctx, ctx,
                              key_padding_mask=(~mask if self.use_spec else None),
                              need_weights=False)
        h = self.norm3(h + a)
        return self.out(h.flatten(1)).view(b, self.chunk, self.out_dim)


def load_meta(cache: Path, chunk: int, policy: str):
    spec, chain = ActionSpec(), panda_chain()
    sc = np.array([spec.rot_scale * spec.control_hz] * 3 +
                  [spec.pos_scale * spec.control_hz] * 3, np.float32)
    items, tasks = [], {}
    for meta in sorted(cache.glob("task*_meta.npz")):
        d = np.load(meta, allow_pickle=True)
        ti, dem = int(d["task_index"]), np.asarray(d["demo"])
        act = (np.concatenate([np.asarray(d["twist"]) / sc, np.asarray(d["gripper"])[:, None]], -1)
               if policy == "screwhead" else
               np.concatenate([np.asarray(d["dq"]) / JOINT_ACTION_SCALE,
                               np.asarray(d["gripper"])[:, None]], -1))
        st = tool_state(chain, torch.tensor(np.asarray(d["qpos"]), dtype=torch.float64),
                        torch.tensor(np.asarray(d["gripper_state"]),
                                     dtype=torch.float64)).float().numpy()
        tasks[ti] = dict(act=act.astype(np.float32), state=st, demo=dem,
                         text=np.asarray(d["text"], dtype=np.float32).reshape(-1),
                         agent=cache / f"task{ti:02d}_agent.npy",
                         wrist=cache / f"task{ti:02d}_wrist.npy")
        for i in range(len(dem) - chunk):
            if dem[i] == dem[i + chunk - 1]:
                items.append((ti, i, int(dem[i])))
    return items, tasks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["baseline", "screwhead"], default="screwhead")
    ap.add_argument("--cache", default="cache/libero_spatial_siglip",
                    help="either cached SigLIP patches (N,tokens,1152) or raw frames "
                         "(N,H,W,3 uint8). Detected from the array itself rather than a "
                         "flag, so the two cannot be mismatched.")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=0, help="0 freezes the tower")
    ap.add_argument("--val-demos", type=int, default=5)
    ap.add_argument("--val-cap", type=int, default=600)
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
    print(f"{len(items)} windows: train {len(tr)} val {len(va)}")

    out_dim = 7 if args.policy == "screwhead" else MAX_DOF + 1
    act_std = torch.tensor(np.concatenate([tasks[t]["act"] for t in sorted(tasks)]).std(0),
                           dtype=torch.float32).clamp(min=1e-6)
    print("action channel std:", [round(float(v), 4) for v in act_std])

    probe = np.load(tasks[min(tasks)]["agent"], mmap_mode="r")
    cached_patches = (probe.ndim == 3 and probe.shape[-1] == DIM)
    del probe
    if cached_patches and args.lora_r:
        raise SystemExit("--lora-r needs raw frames: cached patches are already encoded "
                         "and a frozen encoding cannot be fine-tuned. Point --cache at "
                         "cache/libero_spatial_img.")
    back = None if cached_patches else SiglipBackbone(grid=args.grid, lora_r=args.lora_r,
                                                      device="cuda")
    print("features:", "cached SigLIP patches" if cached_patches
          else f"live SigLIP tower ({args.grid}x{args.grid})")
    head = PatchHead(out_dim, chunk=args.chunk,
                     spec_tokens=(args.policy == "screwhead")).cuda()
    tok, mask = encode(panda_chain()).padded(MAX_DOF)
    tok, mask = tok.float().cuda(), mask.cuda()
    tower_tr = 0.0 if back is None else back.trainable_parameters() / 1e6
    print(f"trainable: tower {tower_tr:.2f}M + "
          f"head {sum(p.numel() for p in head.parameters())/1e6:.2f}M")

    params = list(head.parameters())
    if back is not None:
        params += [p for p in back.vis.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    stdc = act_std.cuda()

    def step(sel, train: bool):
        ag = np.stack([mm[t][0][i] for t, i, _ in sel])
        wr = np.stack([mm[t][1][i] for t, i, _ in sel])
        st = torch.tensor(np.stack([tasks[t]["state"][i] for t, i, _ in sel])).cuda()
        tx = torch.tensor(np.stack([tasks[t]["text"] for t, _, _ in sel])).cuda()
        y = torch.tensor(np.stack([tasks[t]["act"][i:i + args.chunk]
                                   for t, i, _ in sel])).cuda() / stdc
        if back is None:
            fa = torch.tensor(np.asarray(ag, dtype=np.float32)).cuda()
            fw = torch.tensor(np.asarray(wr, dtype=np.float32)).cuda()
        else:
            ctx = torch.enable_grad() if (train and args.lora_r) else torch.no_grad()
            with ctx:
                fa, fw = back.encode(ag).float(), back.encode(wr).float()
        n = len(sel)
        p = (head(fa, fw, st, tx, tok.expand(n, -1, -1), mask.expand(n, -1))
             if args.policy == "screwhead" else
             head(fa, fw, st, tx, eid=torch.zeros(n, dtype=torch.long, device="cuda")))
        return nn.functional.smooth_l1_loss(p, y), p.detach(), y

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    ck = outdir / f"siglip_{args.policy}_libero_spatial.pt"
    rng = np.random.default_rng(args.seed)
    best, t0 = float("inf"), time.time()
    for ep in range(args.epochs):
        head.train(); perm = rng.permutation(len(tr)); tot = n = 0
        for k in range(0, len(perm) - args.batch + 1, args.batch):
            sel = [tr[j] for j in perm[k:k + args.batch]]
            loss, _, _ = step(sel, True)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            tot += loss.item() * len(sel); n += len(sel)
            if (k // args.batch) % 100 == 0:
                print(f"  ep{ep} step {k//args.batch} loss {tot/max(n,1):.5f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        head.eval(); P=[]; Y=[]; vt=vn=0
        with torch.no_grad():
            for k in range(0, min(len(va), args.val_cap), args.batch):
                sel = va[k:k + args.batch]
                if len(sel) < 2: break
                l, p, y = step(sel, False)
                vt += l.item()*len(sel); vn += len(sel)
                P.append((p*stdc).cpu().numpy()); Y.append((y*stdc).cpu().numpy())
        val = vt/max(vn,1)
        P=np.concatenate(P); Y=np.concatenate(Y)
        lab = (['wx','wy','wz','vx','vy','vz','grip'] if args.policy=="screwhead"
               else [f'j{i}' for i in range(7)]+['grip'])
        c=[float(np.corrcoef(P[:,0,j],Y[:,0,j])[0,1]) for j in range(out_dim)]
        print(f"epoch {ep}  train {tot/max(n,1):.5f}  val {val:.5f}")
        print("   corr " + "  ".join(f"{l}:{v:.2f}" for l,v in zip(lab,c)), flush=True)
        if val < best:
            best = val
            torch.save({"head": head.state_dict(), "policy": args.policy,
                        "chunk": args.chunk, "act_std": act_std, "val": val,
                        "epoch": ep, "grid": args.grid, "corr": c,
                        "args": vars(args)}, ck)
    print(f"best val {best:.5f} -> {ck}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
