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


def _read_source(src):
    """-> (parts, frames of a part, task -> instruction, teacher_round) for a source of frames."""
    if src.is_dir():                                     # collect_scripted shards
        parts = [np.load(f, allow_pickle=True) for f in sorted(src.glob("task*.npz"))]
        parts = [p for p in parts if len(p["label"])]
        langs = {int(p["task"][0]): str(p["language"]) for p in parts}
        return parts, lambda p: (p["agent"], p["wrist"]), langs, True
    meta = np.load(src, allow_pickle=True)               # distill.py round with saved frames
    frames = lambda _p: (np.load(str(src) + ".agent_img.npy", mmap_mode="r"),
                         np.load(str(src) + ".wrist_img.npy", mmap_mode="r"))
    langs = json.loads(str(meta["languages"]))
    langs = {int(k): v for k, v in langs.items()}
    return [meta], frames, langs, float(meta["beta"]) >= 1.0


def encode(args):
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
    from screwhead.student.clip_features import clip_encoder
    from screwhead.student.dino_features import DIM, GRID, DinoFeatures
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    feats = DinoFeatures(args.device)
    _, enc_txt = clip_encoder(args.device)
    src = Path(args.source)
    parts, frames, langs, teacher_round = _read_source(src)
    n = sum(len(p["label"]) for p in parts)
    A = np.lib.format.open_memmap(out / "agent_tok.npy", mode="w+", dtype=np.float16, shape=(n, GRID * GRID, DIM))
    W = np.lib.format.open_memmap(out / "wrist_tok.npy", mode="w+", dtype=np.float16, shape=(n, GRID * GRID, DIM))
    keys = ("state", "label", "task", "episode", "step")
    meta_out = {k: [] for k in keys}
    success = []
    k0 = 0
    for p in parts:
        fa, fw = frames(p)
        m = len(p["label"])
        for k in range(0, m, args.batch):
            A[k0 + k:k0 + min(k + args.batch, m)] = feats(np.asarray(fa[k:k + args.batch])).cpu().numpy()
            W[k0 + k:k0 + min(k + args.batch, m)] = feats(np.asarray(fw[k:k + args.batch])).cpu().numpy()
        for kk in keys:
            meta_out[kk].append(np.asarray(p[kk]))
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
    where = {(int(e), int(s)): i for i, (e, s) in enumerate(zip(ep, st, strict=True))}
    prev = np.array([where.get((int(e), int(s) - 1), -1) for e, s in zip(ep, st, strict=True)])
    rate = np.zeros(len(ap), np.float32)
    ok = prev >= 0
    rate[ok] = (ap[ok] - ap[prev[ok]]) * CONTROL_HZ
    return rate


def _cap_per_episode(meta, idx, cap, name):
    """At most `cap` evenly spaced frames of each episode among rows `idx`."""
    ep_k = meta["episode"][idx]
    kept = []
    for e in np.unique(ep_k):
        ii = idx[ep_k == e]
        ii = ii[np.argsort(meta["step"][ii])]
        kept.append(ii if len(ii) <= cap else ii[np.linspace(0, len(ii) - 1, cap).round().astype(int)])
    print(f"  {name}: {len(idx)} -> {sum(len(x) for x in kept)} frames (<= {cap} per episode)", flush=True)
    return np.sort(np.concatenate(kept))


def _load_set(args, k, d):
    """One token dataset: memory-mapped tokens, metadata, state, and the rows kept for training."""
    # dict(), not the NpzFile: an NpzFile re-reads a whole array on EVERY key access,
    # and rows are indexed per frame -- O(N^2) reads that grew to 100 GB RSS at 100k
    # frames and starved the (unified-memory) GPU
    meta = dict(np.load(Path(d) / "meta.npz"))
    A = np.load(Path(d) / "agent_tok.npy", mmap_mode="r"); W = np.load(Path(d) / "wrist_tok.npy", mmap_mode="r")
    keep = np.ones(len(meta["label"]), bool)
    if bool(meta["teacher_round"]):
        keep = meta["episode_success"].astype(bool)
    if args.exclude_tasks:
        # held-out tasks: never trained on, so evaluating them measures what carries
        # across tasks rather than what was memorised per task
        keep &= ~np.isin(meta["task"], args.exclude_tasks)
    state = meta["state"].astype(np.float32)
    if args.aperture_rate:
        state = np.concatenate([state, aperture_rate(meta)[:, None]], 1)
    idx = np.where(keep)[0]
    if args.max_frames_per_episode:
        # A DAgger episode that stalls contributes hundreds of near-identical frames: 48% of
        # round 3 came from episodes at the 400-step limit, and training on it dropped the
        # student from 151/200 to 73/200. Keep at most B evenly spaced frames per episode,
        # so the stuck states stay covered without outweighing everything else.
        idx = _cap_per_episode(meta, idx, args.max_frames_per_episode, d)
    return dict(A=A, W=W, meta=meta, state=state, idx=idx, tag=k)


def _rows(args, sets):
    """-> (rows as (dataset, frame), validation mask, labels). Every tenth episode validates."""
    rows = [(k, i) for k, s in enumerate(sets) for i in s["idx"]]
    ep = np.array([sets[k]["meta"]["episode"][i] + 10_000_000 * k for k, i in rows])
    val = (ep % 10) == 0
    key = "label_gt" if args.gripper_target else "label"
    for d, s_ in zip(args.data, sets, strict=True):
        if key not in s_["meta"]:
            raise SystemExit(f"{d}: no {key}; collect it through the same servo and decode the "
                             "policy will run under (tools/distill.py collect --gripper-target)")
    return rows, val, np.stack([sets[k]["meta"][key][i] for k, i in rows]).astype(np.float32)


def _clone_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _state_stats(args, st_all, val):
    """Mean and std the state is standardised with (zeros and ones without --standardize-state)."""
    if args.standardize_state:
        # The drawer's gripper law switches inside a 3 mm band on a 26 mm aperture. Raw, the
        # aperture enters as ~0.03 +- 0.035 next to rotation columns of +-0.75; standardized,
        # every state column is O(1) and a millimetre is visible to the first linear layer.
        return st_all[~val].mean(0), st_all[~val].std(0).clip(1e-6)
    return np.zeros(st_all.shape[1], np.float32), np.ones(st_all.shape[1], np.float32)


def _gripper_classes(args, lab, sets, rows):
    """-> (classes, per-row class, per-row change point); all None unless --gripper-classes."""
    if not args.gripper_classes:
        return None, None, None
    # the gripper as a classification over the program apertures: a regression averages
    # 80 mm and 0 at the grasp, and any snap threshold between them becomes a trap
    # (measured: snapped regression never closed in 22 of 62 failures)
    if not (args.gripper_target and args.gripper_levels):
        raise SystemExit("--gripper-classes needs --gripper-target and --gripper-levels")
    from screwhead.sim.gripper_servo import channel_to_target
    classes = np.asarray(sorted(args.gripper_levels), np.float32)
    cls = np.abs(channel_to_target(lab[:, 6])[:, None] - classes[None]).argmin(1)
    print("gripper classes " + ", ".join(f"{c*1000:.0f} mm: {int((cls == i).sum())}" for i, c in enumerate(classes)), flush=True)
    # change points: the class differs from the same episode's previous step (~1.5% of
    # frames). "Keep the current aperture" is right on ~91% of all frames and 0-9% of these,
    # so overall accuracy says little. These do NOT isolate vision either: on teacher-driven
    # frames the round-1 blind model got 60% of change points vs 40% sighted -- once the tool
    # is at the grasp, proprioception says "close". Vision shows in the twist near the grasp.
    key = {(k, int(sets[k]["meta"]["episode"][i]), int(sets[k]["meta"]["step"][i])): n for n, (k, i) in enumerate(rows)}
    change = np.zeros(len(rows), bool)
    for n, (k, i) in enumerate(rows):
        prev = key.get((k, int(sets[k]["meta"]["episode"][i]), int(sets[k]["meta"]["step"][i]) - 1))
        change[n] = prev is not None and cls[prev] != cls[n]
    return classes, cls, change


class _Batches:
    """Minibatches over rows (dataset k, frame i): tokens are gathered per dataset in index
    order, so the memory-mapped reads are sequential."""

    def __init__(self, sets, rows, lab, act_std, cls, zero_images, dev):
        self.sets, self.rows, self.lab, self.act_std = sets, rows, lab, act_std
        self.cls, self.zero_images, self.dev = cls, zero_images, dev
        self.text = sets[0]["meta"]["text"]

    def __call__(self, ids):
        import torch
        sets, rows, dev = self.sets, self.rows, self.dev
        by = {}
        for j in ids:
            by.setdefault(rows[j][0], []).append(j)
        order, a_l, w_l = [], [], []
        for k, js in by.items():
            ii = np.array([rows[j][1] for j in js]); srt = np.argsort(ii)
            a_l.append(np.asarray(sets[k]["A"][ii[srt]])); w_l.append(np.asarray(sets[k]["W"][ii[srt]]))
            order += [js[s] for s in srt]
        a = torch.tensor(np.concatenate(a_l), device=dev); w = torch.tensor(np.concatenate(w_l), device=dev)
        if self.zero_images:
            a.zero_(); w.zero_()
        st = torch.tensor(np.stack([sets[rows[j][0]]["state"][rows[j][1]] for j in order]), device=dev)
        tk = torch.tensor(np.stack([self.text[int(sets[rows[j][0]]["meta"]["task"][rows[j][1]])] for j in order]), device=dev)
        y = torch.tensor(self.lab[order] / self.act_std, device=dev)[:, None]
        c = torch.tensor(self.cls[order], device=dev) if self.cls is not None else None
        return a, w, tk, st, (y, c), order


def _loss(out, yc, classes, gripper_weight, reduction="mean"):
    from torch import nn
    y, c = yc
    if classes is None:
        return nn.functional.smooth_l1_loss(out, y, reduction=reduction)
    tw = nn.functional.smooth_l1_loss(out[:, 0, :6], y[:, 0, :6], reduction=reduction)
    ce = nn.functional.cross_entropy(out[:, 0, 6:], c, reduction=reduction)
    # per frame: mean twist loss + gripper cross-entropy (a sum over 6 twist elements
    # carries the cross-entropy 6 times, so val = sum / (N * 6) keeps the same weighting)
    g = gripper_weight
    return tw + g * ce if reduction == "mean" else tw + 6.0 * g * ce


def _train_epoch(model, opt, sched, perm, batch_size, step_loss):
    """One pass over the shuffled training rows; returns the last batch's loss."""
    from torch import nn
    model.train()
    loss = None
    for k in range(0, len(perm) - batch_size + 1, batch_size):
        loss = step_loss(perm[k:k + batch_size])
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if sched.last_epoch < sched.total_steps - 1: sched.step()
    return loss


def _validate(fwd, batch, vai, loss_of, classes, change):
    """Summed validation losses over `vai`: total, and with classes the twist, the gripper
    cross-entropy and the class hits (overall and at change points), which are printed."""
    import torch
    from torch import nn
    v = dict(total=0.0, hit=0, twist=0.0, ce=0.0, change_hit=0)
    with torch.no_grad():
        for k in range(0, len(vai), 1024):
            a, w, tk, st, y, order = batch(vai[k:k + 1024])
            o = fwd(a, w, tk, st)
            v["total"] += loss_of(o, y, reduction="sum").item()
            if classes is not None:
                ok = (o[:, 0, 6:].argmax(-1) == y[1]).cpu().numpy()
                v["hit"] += int(ok.sum()); v["change_hit"] += int(ok[change[order]].sum())
                v["twist"] += nn.functional.smooth_l1_loss(o[:, 0, :6], y[0][:, 0, :6], reduction="sum").item()
                v["ce"] += nn.functional.cross_entropy(o[:, 0, 6:], y[1], reduction="sum").item()
    if classes is not None:
        n_ch = int(change[vai].sum())
        print(f"  val twist {v['twist'] / (len(vai) * 6):.4f}  gripper CE {v['ce'] / len(vai):.4f}  "
              f"gripper accuracy {v['hit'] / len(vai):.3f} (change points {v['change_hit']}/{n_ch} = "
              f"{v['change_hit'] / max(n_ch, 1):.3f})", flush=True)
    return v


def train(args):
    import torch
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
    from distill import spec_tokens
    from screwhead.student.token_head import TokenHead
    torch.manual_seed(args.seed)
    dev = args.device
    sets = [_load_set(args, k, d) for k, d in enumerate(args.data)]
    rows, val, lab = _rows(args, sets)
    act_std = lab[~val].std(0).clip(1e-6)
    state_mean, state_std = _state_stats(args, np.stack([sets[k]["state"][i] for k, i in rows]), val)
    for s_ in sets:
        s_["state"] = ((s_["state"] - state_mean) / state_std).astype(np.float32)
    print(f"{len(rows)} frames from {len(sets)} dataset(s): train {int((~val).sum())} val {int(val.sum())}"
          f"{'  ABLATION: images zeroed' if args.zero == 'image' else ''}", flush=True)
    tok, tmask = spec_tokens(dev)
    state_dim = sets[0]["state"].shape[1]
    classes, cls, change = _gripper_classes(args, lab, sets, rows)
    out_dim = 6 + len(classes) if classes is not None else 7
    model = TokenHead(state_dim=state_dim, out_dim=out_dim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    tri, vai = np.where(~val)[0], np.where(val)[0]
    steps = args.epochs * (len(tri) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(steps, 1))
    batch = _Batches(sets, rows, lab, act_std, cls, args.zero == "image", dev)
    loss_of = lambda out, yc, reduction="mean": _loss(out, yc, classes, args.gripper_weight, reduction)
    fwd = lambda a, w, tk, st: model(a, w, tk, st, tok.expand(len(a), -1, -1), tmask.expand(len(a), -1))
    twist_select = classes is not None and args.select == "twist"
    best, best_state = float("inf"), None
    best_total, best_total_state = float("inf"), None      # fallback when --select twist never qualifies
    rng = np.random.default_rng(args.seed)

    def step_loss(ids):
        a, w, tk, st, y, _ = batch(ids)
        return loss_of(fwd(a, w, tk, st), y)

    for ep_i in range(args.epochs):
        loss = _train_epoch(model, opt, sched, rng.permutation(tri), args.batch, step_loss)
        model.eval()
        vs = _validate(fwd, batch, vai, loss_of, classes, change)
        vl = vs["total"] / (len(vai) * (6 if classes is not None else 7))
        score = vl
        if twist_select:
            # choose on the motion, as long as the gripper class holds up: the cross-entropy
            # grows overconfident late while the twist keeps improving
            score = vs["twist"] / (len(vai) * 6) if vs["hit"] / len(vai) >= args.min_gripper_acc else float("inf")
        if vl < best_total:
            best_total, best_total_state = vl, _clone_state(model)
        saved = score < best
        if saved:
            best, best_state = score, _clone_state(model)
        print(f"  epoch {ep_i:3d}  train {loss.item():.4f}  val {vl:.4f}{'  *' if saved else ''}", flush=True)
    if best_state is None:
        # no epoch cleared --min-gripper-acc (the blind ablation does not: it cannot see when to
        # close). Fall back to the total-loss checkpoint rather than saving nothing.
        print(f"  no epoch reached gripper accuracy {args.min_gripper_acc}; selecting by total val loss", flush=True)
        best, best_state = best_total, best_total_state
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in best_state.items()}, "act_std": act_std, "chunk": 1,
                "kind": "token", "zero": args.zero, "state_dim": state_dim, "aperture_rate": bool(args.aperture_rate),
                "state_mean": state_mean, "state_std": state_std, "gripper_target": bool(args.gripper_target), "spec_mask_fixed": True,
                "gripper_levels": list(args.gripper_levels) if args.gripper_levels else None,
                "gripper_classes": classes.tolist() if classes is not None else None, "out_dim": out_dim, "val": best, "data": args.data, "args": vars(args)}, out)
    print(f"best val {best:.4f}{' (twist)' if twist_select else ''} -> {out}")
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
    t.add_argument("--exclude-tasks", type=int, nargs="*", default=None, help="drop these task ids from training")
    t.add_argument("--max-frames-per-episode", type=int, default=0, help="subsample longer episodes to this many frames")
    t.add_argument("--gripper-weight", type=float, default=1.0, help="weight of the gripper cross-entropy against the twist loss")
    t.add_argument("--select", default="total", choices=["total", "twist"],
                   help="checkpoint by total val loss, or by val twist loss with --min-gripper-acc")
    t.add_argument("--min-gripper-acc", type=float, default=0.95)
    t.add_argument("--gripper-classes", action="store_true", help="classify the gripper over --gripper-levels instead of regressing it")
    t.add_argument("--gripper-levels", type=float, nargs="*", default=None,
                   help="stored in the checkpoint: every rollout of it (DAgger and eval) snaps the target to these (m)")
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cuda")
    args = ap.parse_args()
    return encode(args) if args.cmd == "encode" else train(args)


if __name__ == "__main__":
    raise SystemExit(main())
