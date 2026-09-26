#!/usr/bin/env python
"""Train the Panda VLA on LIBERO's human demonstrations as they are stored (BRN-vla-learns-the-recorded-motion).

  train_vla.py --suites libero_spatial --out checkpoints/vla_spatial_s0.pt [--blind] [--seed 0]
               [--demos 50] [--val-demos 2] [--steps 30000] [--batch 16]

Every pair is a stored observation and the recorded motion from the state it shows; the model and its
preprocessing are screwhead/student/qwen_vla.py. The normalization constants are fitted on the training
labels and saved with the weights (BRN-vla-sees-and-acts-as-trained). The last --val-demos demonstrations
of each task are held out for the offline error, reported every --eval-every steps. --blind trains the twin
whose camera images are zeros.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Pairs:
    """Every pair of the chosen demonstrations, in memory (the stored images are 128 px)."""

    def __init__(self, suites: list[str], demos: slice, tasks: list[int] | None):
        import h5py
        from libero.libero import benchmark

        from screwhead.student import libero_data as D
        chain = D.panda_chain()
        self.demos, self.language, self.files = [], [], []
        for suite in suites:
            bm = benchmark.get_benchmark_dict()[suite]()
            for t in (tasks if tasks is not None else range(bm.n_tasks)):
                task = bm.get_task(t)
                self.files.append(str(D.task_file(suite, task.name)))
                with h5py.File(D.task_file(suite, task.name)) as f:
                    keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))[demos]
                    for k in keys:
                        self.demos.append(D.load_demo(chain, f["data"][k]))
                        self.language.append(task.language)
        self.index = [(i, j) for i, d in enumerate(self.demos) for j in range(len(d.twist))]

    def stats(self) -> dict:
        tw = np.concatenate([d.twist for d in self.demos]); pr = np.concatenate([d.proprio for d in self.demos])
        return dict(twist_mean=tw.mean(0).tolist(), twist_std=(tw.std(0) + 1e-6).tolist(),
                    proprio_mean=pr.mean(0).tolist(), proprio_std=(pr.std(0) + 1e-6).tolist())


class Dataset:
    """(pair index) -> one model sample: tokens and pixels, normalized proprio, the chunk of normalized labels."""

    def __init__(self, pairs: Pairs, proc, cfg):
        self.p, self.proc, self.cfg = pairs, proc, cfg
        self.tm, self.ts = np.array(cfg.twist_mean, np.float32), np.array(cfg.twist_std, np.float32)
        self.pm, self.ps = np.array(cfg.proprio_mean, np.float32), np.array(cfg.proprio_std, np.float32)

    def __len__(self):
        return len(self.p.index)

    def __getitem__(self, n):
        import torch

        from screwhead.student.qwen_vla import prompt_ids
        i, j = self.p.index[n]
        d, H = self.p.demos[i], self.cfg.chunk
        s = prompt_ids(self.proc, self.p.language[i], [d.agentview[j], d.wrist[j]], self.cfg)
        k = np.arange(j, j + H)
        idx = np.minimum(k, len(d.twist) - 1)
        s.update(proprio=torch.from_numpy((d.proprio[j] - self.pm) / self.ps),
                 twist=torch.from_numpy((d.twist[idx] - self.tm) / self.ts), gripper=torch.from_numpy(d.gripper[idx]),
                 valid=torch.from_numpy(k < len(d.twist)))
        return s


def loss_fn(model, batch):
    import torch.nn.functional as F
    tw, gl = model(batch)
    v = batch["valid"].float()
    l1 = ((tw - batch["twist"]).abs().mean(-1) * v).sum() / v.sum()
    bce = (F.binary_cross_entropy_with_logits(gl, (batch["gripper"] + 1) / 2, reduction="none") * v).sum() / v.sum()
    grip_ok = (((gl > 0).float() * 2 - 1) == batch["gripper"]).float()
    return l1, bce, (grip_ok * v).sum() / v.sum()


def _parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", nargs="+", default=["libero_spatial"])
    ap.add_argument("--tasks", type=int, nargs="*", default=None)
    ap.add_argument("--demos", type=int, default=50, help="demonstrations per task, held-out ones included")
    ap.add_argument("--val-demos", type=int, default=2)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--execute", type=int, default=4)
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def _validate(model, vloader) -> dict:
    import torch
    model.eval()
    vs = []
    with torch.no_grad():
        for vb in vloader:
            vs.append([x.item() for x in loss_fn(model, {k: v.cuda() for k, v in vb.items()})])
    model.train()
    vs = np.array(vs)
    return dict(val_l1=round(float(vs[:, 0].mean()), 4), val_bce=round(float(vs[:, 1].mean()), 4),
                val_grip_acc=round(float(vs[:, 2].mean()), 4))


def _train(model, loader, vloader, args) -> int:
    """The optimisation loop: AdamW with warm-up and cosine decay; a log line every 50 steps."""
    import torch
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup)
                                              * 0.5 * (1 + np.cos(np.pi * min(1.0, s / args.steps))))
    log = Path(args.out).with_suffix(".log.jsonl")
    step, t0, hist = 0, time.time(), []
    model.train()
    while step < args.steps:
        for batch in loader:
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            l1, bce, acc = loss_fn(model, batch)
            opt.zero_grad(set_to_none=True)
            (l1 + bce).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            hist.append((l1.item(), bce.item(), acc.item()))
            if step % 50 == 0:
                h = np.array(hist[-50:])
                rec = dict(step=step, l1=round(float(h[:, 0].mean()), 4), bce=round(float(h[:, 1].mean()), 4),
                           grip_acc=round(float(h[:, 2].mean()), 4), lr=sched.get_last_lr()[0],
                           s_per_step=round((time.time() - t0) / step, 3))
                if vloader is not None and step % args.eval_every == 0:
                    rec.update(_validate(model, vloader))
                print(json.dumps(rec), flush=True)
                with open(log, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
            if step % args.save_every == 0 or step >= args.steps:
                model.save(args.out, read=args.read)
            if step >= args.steps:
                break
    return step


def main() -> int:
    args = _parse()
    import torch
    from torch.utils.data import DataLoader

    from screwhead.student.qwen_vla import VLAConfig, build, collate
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    n_train = args.demos - args.val_demos
    t0 = time.time()
    train = Pairs(args.suites, slice(0, n_train), args.tasks)
    val = Pairs(args.suites, slice(n_train, args.demos), args.tasks) if args.val_demos else None
    print(f"pairs: train {len(train.index)} from {len(train.demos)} demonstrations, "
          f"val {len(val.index) if val else 0} ({time.time() - t0:.0f} s)", flush=True)
    cfg = VLAConfig(chunk=args.chunk, execute=args.execute, blind=args.blind, **train.stats())
    model, proc = build(cfg)
    pad = proc.tokenizer.pad_token_id
    loader = DataLoader(Dataset(train, proc, cfg), batch_size=args.batch, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(args.seed), num_workers=args.workers,
                        persistent_workers=True, collate_fn=lambda s: collate(s, pad))
    vloader = DataLoader(Dataset(val, proc, cfg), batch_size=args.batch, shuffle=False, num_workers=2,
                         collate_fn=lambda s: collate(s, pad)) if val else None
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    args.read = sorted(set(train.files + (val.files if val else [])))
    step = _train(model, loader, vloader, args)
    model.save(args.out, read=args.read)
    # BRN-vla-reported-beside-a-blind-twin: the configuration as values and the digest of the model training ended
    # with, beside the checkpoint (which cannot hold its own digest); an evaluation checks the checkpoint against it
    from screwhead.student.qwen_vla import environment_record, file_digest
    config = {k: v for k, v in vars(args).items() if k not in ("read",)}
    Path(args.out + ".record.json").write_text(json.dumps(
        {"checkpoint_digest": file_digest(args.out), "steps": step, "configuration": config,
         "precision": "bfloat16 backbone, float32 heads", "trained_in": environment_record(read=args.read)},
        indent=1, sort_keys=True, default=str))
    print(f"saved {args.out} after {step} steps ({(time.time() - t0) / 3600:.2f} h)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
