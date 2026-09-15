#!/usr/bin/env python
"""Token datasets and training for the DINOv2 patch-token VLA.

  encode  raw 128 px frames -> <dir>/{agent,wrist}_tok.npy (N, 64, 768 float16, memory-mapped)
          + <dir>/meta.npz (state, label, task, episode, step, text, flags)
          Sources: tools/collect_scripted.py shard dirs (teacher demonstrations, successes
          only) or tools/distill.py rounds collected with --save-frames (student-driven).
  train   TokenHead on one or more token datasets. Teacher-only rounds are already
          success-filtered at collection; student-driven rounds keep every frame -- the
          teacher's labels on the student's mistakes are what DAgger is for.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def encode(args):
    import torch
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
    from screwhead.clip_features import clip_encoder
    from screwhead.dino_features import DIM, GRID, DinoFeatures
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    feats = DinoFeatures(args.device)
    _, enc_txt = clip_encoder(args.device)
    src = Path(args.source)
    if src.is_dir():                                     # collect_scripted shards
        parts = [np.load(f, allow_pickle=True) for f in sorted(src.glob("task*.npz"))]
        parts = [p for p in parts if len(p["label"])]
        get = lambda p, k: p[k]
        frames = lambda p: (p["agent"], p["wrist"])
        langs = {int(p["task"][0]): str(p["language"]) for p in parts}
        teacher_round = True
    else:                                                # distill.py round with saved frames
        meta = np.load(src, allow_pickle=True)
        parts = [meta]
        get = lambda p, k: p[k]
        frames = lambda p: (np.load(str(src) + ".agent_img.npy", mmap_mode="r"), np.load(str(src) + ".wrist_img.npy", mmap_mode="r"))
        langs = json.loads(str(meta["languages"]))
        langs = {int(k): v for k, v in langs.items()}
        teacher_round = float(meta["beta"]) >= 1.0
    n = sum(len(get(p, "label")) for p in parts)
    A = np.lib.format.open_memmap(out / "agent_tok.npy", mode="w+", dtype=np.float16, shape=(n, GRID * GRID, DIM))
    W = np.lib.format.open_memmap(out / "wrist_tok.npy", mode="w+", dtype=np.float16, shape=(n, GRID * GRID, DIM))
    keys = ("state", "label", "task", "episode", "step")
    meta_out = {k: [] for k in keys}
    success = []
    k0 = 0
    for p in parts:
        fa, fw = frames(p)
        m = len(get(p, "label"))
        for k in range(0, m, args.batch):
            A[k0 + k:k0 + min(k + args.batch, m)] = feats(np.asarray(fa[k:k + args.batch])).cpu().numpy()
            W[k0 + k:k0 + min(k + args.batch, m)] = feats(np.asarray(fw[k:k + args.batch])).cpu().numpy()
        for kk in keys:
            meta_out[kk].append(np.asarray(get(p, kk)))
        success.append(np.asarray(p["episode_success"]) if "episode_success" in p.files else np.ones(m, bool))
        k0 += m
        print(f"  encoded {m} frames ({k0}/{n})", flush=True)
    A.flush(); W.flush()
    text = np.zeros((10, 512), np.float32)
    for t, s in langs.items():
        text[t] = enc_txt(s).float().cpu().numpy()
    extra = {}
    if not src.is_dir() and "gripper_target" in parts[0].files and bool(parts[0]["gripper_target"]):
        extra["label_gt"] = np.concatenate(meta_out["label"])       # collected in target mode already
    if not src.is_dir() and "phase" in parts[0].files:
        extra["phase"] = np.asarray(parts[0]["phase"])
    np.savez(out / "meta.npz", **{k: np.concatenate(v) for k, v in meta_out.items()},
             episode_success=np.concatenate(success), text=text, teacher_round=teacher_round, **extra)
    print(f"-> {out}  ({n} frames)")
    return 0


APERTURE = 9          # index of the gripper aperture in tool_state
CONTROL_HZ = 20


def aperture_rate(meta) -> np.ndarray:
    """Finite-difference gripper aperture rate (m/s) per frame, 0 at an episode's first step.

    Frames of a DAgger round are interleaved across workers, so the previous frame is
    found by (episode, step), not by position. The drawer program's gripper law reads
    aperture + rate * tau; without the rate its labels contradict each other at equal
    aperture once student-driven states break the aperture-rate correlation of a demo."""
    ep, st, ap = meta["episode"], meta["step"], meta["state"][:, APERTURE]
    where = {(int(e), int(s)): i for i, (e, s) in enumerate(zip(ep, st))}
    prev = np.array([where.get((int(e), int(s) - 1), -1) for e, s in zip(ep, st)])
    rate = np.zeros(len(ap), np.float32)
    ok = prev >= 0
    rate[ok] = (ap[ok] - ap[prev[ok]]) * CONTROL_HZ
    return rate


def train(args):
    import torch
    from torch import nn
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
    from distill import spec_tokens
    from screwhead.token_head import TokenHead
    torch.manual_seed(args.seed)
    dev = args.device
    sets = []
    for k, d in enumerate(args.data):
        # dict(), not the NpzFile: an NpzFile re-reads a whole array on EVERY key access,
        # and rows are indexed per frame -- O(N^2) reads that grew to 100 GB RSS at 100k
        # frames and starved the (unified-memory) GPU
        meta = dict(np.load(Path(d) / "meta.npz"))
        A = np.load(Path(d) / "agent_tok.npy", mmap_mode="r"); W = np.load(Path(d) / "wrist_tok.npy", mmap_mode="r")
        keep = np.ones(len(meta["label"]), bool)
        if bool(meta["teacher_round"]):
            keep = meta["episode_success"].astype(bool)
        state = meta["state"].astype(np.float32)
        if args.aperture_rate:
            state = np.concatenate([state, aperture_rate(meta)[:, None]], 1)
        sets.append(dict(A=A, W=W, meta=meta, state=state, idx=np.where(keep)[0], tag=k))
    text = sets[0]["meta"]["text"]
    rows = [(k, i) for k, s in enumerate(sets) for i in s["idx"]]
    ep = np.array([sets[k]["meta"]["episode"][i] + 10_000_000 * k for k, i in rows])
    val = (ep % 10) == 0
    key = "label_gt" if args.gripper_target else "label"
    for d, s_ in zip(args.data, sets):
        if key not in s_["meta"]:
            raise SystemExit(f"{d}: no {key}; run tools/relabel_gripper.py first")
    lab = np.stack([sets[k]["meta"][key][i] for k, i in rows]).astype(np.float32)
    act_std = lab[~val].std(0).clip(1e-6)
    st_all = np.stack([sets[k]["state"][i] for k, i in rows])
    if args.standardize_state:
        # The drawer's gripper law switches inside a 3 mm band on a 26 mm aperture. Raw, the
        # aperture enters as ~0.03 +- 0.035 next to rotation columns of +-0.75; standardized,
        # every state column is O(1) and a millimetre is visible to the first linear layer.
        state_mean, state_std = st_all[~val].mean(0), st_all[~val].std(0).clip(1e-6)
    else:
        state_mean, state_std = np.zeros(st_all.shape[1], np.float32), np.ones(st_all.shape[1], np.float32)
    for s_ in sets:
        s_["state"] = ((s_["state"] - state_mean) / state_std).astype(np.float32)
    print(f"{len(rows)} frames from {len(sets)} dataset(s): train {int((~val).sum())} val {int(val.sum())}"
          f"{'  ABLATION: images zeroed' if args.zero == 'image' else ''}", flush=True)
    tok, tmask = spec_tokens(dev)
    state_dim = sets[0]["state"].shape[1]
    model = TokenHead(state_dim=state_dim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    tri, vai = np.where(~val)[0], np.where(val)[0]
    steps = args.epochs * (len(tri) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(steps, 1))

    def batch(ids):
        by = {}
        for j in ids:
            by.setdefault(rows[j][0], []).append(j)
        order, a_l, w_l = [], [], []
        for k, js in by.items():
            ii = np.array([rows[j][1] for j in js]); srt = np.argsort(ii)
            a_l.append(np.asarray(sets[k]["A"][ii[srt]])); w_l.append(np.asarray(sets[k]["W"][ii[srt]]))
            order += [js[s] for s in srt]
        a = torch.tensor(np.concatenate(a_l), device=dev); w = torch.tensor(np.concatenate(w_l), device=dev)
        if args.zero == "image":
            a.zero_(); w.zero_()
        st = torch.tensor(np.stack([sets[rows[j][0]]["state"][rows[j][1]] for j in order]), device=dev)
        tk = torch.tensor(np.stack([text[int(sets[rows[j][0]]["meta"]["task"][rows[j][1]])] for j in order]), device=dev)
        y = torch.tensor(lab[order] / act_std, device=dev)[:, None]
        return a, w, tk, st, y, order

    fwd = lambda a, w, tk, st: model(a, w, tk, st, tok.expand(len(a), -1, -1), tmask.expand(len(a), -1))
    best, best_state = float("inf"), None
    rng = np.random.default_rng(args.seed)
    for ep_i in range(args.epochs):
        model.train()
        perm = rng.permutation(tri)
        for k in range(0, len(perm) - args.batch + 1, args.batch):
            a, w, tk, st, y, _ = batch(perm[k:k + args.batch])
            loss = nn.functional.smooth_l1_loss(fwd(a, w, tk, st), y)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if sched.last_epoch < sched.total_steps - 1: sched.step()
        model.eval(); vt = 0.0
        with torch.no_grad():
            for k in range(0, len(vai), 1024):
                a, w, tk, st, y, _ = batch(vai[k:k + 1024])
                vt += nn.functional.smooth_l1_loss(fwd(a, w, tk, st), y, reduction="sum").item()
        vl = vt / (len(vai) * 7)
        if vl < best:
            best, best_state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"  epoch {ep_i:3d}  train {loss.item():.4f}  val {vl:.4f}{'  *' if vl == best else ''}", flush=True)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in best_state.items()}, "act_std": act_std, "chunk": 1,
                "kind": "token", "zero": args.zero, "state_dim": state_dim, "aperture_rate": bool(args.aperture_rate),
                "state_mean": state_mean, "state_std": state_std, "gripper_target": bool(args.gripper_target), "spec_mask_fixed": True, "val": best, "data": args.data, "args": vars(args)}, out)
    print(f"best val {best:.4f} -> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode")
    e.add_argument("--source", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--batch", type=int, default=256)
    e.add_argument("--device", default="cuda")
    t = sub.add_parser("train")
    t.add_argument("--data", nargs="+", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--zero", default="none", choices=["none", "image"])
    t.add_argument("--aperture-rate", action="store_true", help="append the gripper aperture rate to the state")
    t.add_argument("--standardize-state", action="store_true", help="z-score the state with training statistics")
    t.add_argument("--gripper-target", action="store_true", help="train on target-aperture gripper labels (label_gt)")
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cuda")
    args = ap.parse_args()
    return encode(args) if args.cmd == "encode" else train(args)


if __name__ == "__main__":
    raise SystemExit(main())
