#!/usr/bin/env python
"""Distil the privileged teacher into the vision-language student with DAgger.

  collect  -- run episodes at randomised placement; at every visited state the
              teacher's action is recorded as the label. beta is the probability
              the TEACHER's action is executed, so beta=1 is teacher-driven data
              and beta=0 is the student driving (and, without --out, an eval).
  train    -- fit the student (CLIP ScrewHead) on every round collected so far.

Why the pieces are what they are:

  PLACEMENT. Episodes draw the bowl from a 60 mm disc, where a perfect open-loop
  replay of the demonstrations succeeds 0.40. Imitating the teacher here cannot
  be done from proprioception, which is the point.

  LABELS AT VISITED STATES. Teacher-driven data alone teaches the student what to
  do on the teacher's states; its own mistakes take it elsewhere. Later rounds
  let the student drive and label where it actually goes.

  SINGLE-STEP. The teacher acts every step and visual servoing needs feedback;
  an 8-step chunk executed open-loop would throw away exactly what vision adds.

  SAME ACTION UNITS. Teacher and student both emit normalised body twist plus
  gripper, executed through TwistServo. No conversion sits between them.

Features are the frozen CLIP ViT-B/32 pooled outputs the student already uses,
encoded on the GPU at collection time; images are not kept unless --save-images.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------ workers
def _worker(remote, task, seed, cpu, teacher_ckpt, horizon, radius, env_kw):
    os.sched_setaffinity(0, {cpu})
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
    from screwhead.teacher_env import PrivilegedEnv
    from teacher_rl import build
    teacher, _, st, _ = build(teacher_ckpt, "cpu")
    mu, sd, mask, asd = (st["obs_mu"].numpy(), st["obs_sd"].numpy(),
                         st["obs_mask"].numpy(), st["act_sd"].numpy())
    env = PrivilegedEnv(task, radius_m=radius, horizon=horizon, seed=seed, render=True, **env_kw)

    def label(o):
        with torch.no_grad():
            m = teacher(torch.from_numpy(((o - mu) / sd * mask).astype(np.float32))).numpy()
        return np.clip(m * asd, -1.0, 1.0).astype(np.float32)

    def payload(o, done=False, info=None):
        a, w = env.images()
        return (a, w, env.student_state(), label(o), done, info)

    remote.send(env.language)
    while True:
        cmd, arg = remote.recv()
        if cmd == "reset":
            o = env.reset()
            remote.send(payload(o))
        elif cmd == "step":
            o, _, done, info = env.step(arg)
            if done:
                end = dict(success=bool(info["success"]), length=env.t,
                           placement_mm=float(np.linalg.norm(env.placement[:2]) * 1000))
                o = env.reset()
                remote.send(payload(o, True, end))
            else:
                remote.send(payload(o))
        elif cmd == "close":
            env.close(); remote.close(); return


# ------------------------------------------------------------------------ student
def load_student(ckpt, device):
    import torch
    from screwhead.policy import ScrewHead
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = ScrewHead(chunk=ck["chunk"]); m.load_state_dict(ck["state_dict"]); m.to(device).eval()
    return m, torch.as_tensor(np.asarray(ck["act_std"]), dtype=torch.float32, device=device), ck


def spec_tokens(device):
    import torch
    from screwhead.libero import panda_chain
    from screwhead.policy import MAX_DOF
    from screwhead.spec import encode
    tok, mask = encode(panda_chain()).padded(MAX_DOF)
    return tok.float().to(device)[None], mask.to(device)[None]


# ------------------------------------------------------------------------ collect
def collect(args):
    import torch
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
    from rollout import clip_encoder
    from teacher_rl import start_kw
    dev = args.device
    rng = np.random.default_rng(args.seed)
    enc_img, enc_txt = clip_encoder(dev)
    student = act_std = None
    if args.beta < 1.0:
        student, act_std, _ = load_student(args.student, dev)
    tok, tmask = spec_tokens(dev)

    tasks = list(range(10))
    cpus = [int(c) for c in args.cpus.split(",")]
    ctx = mp.get_context("spawn")
    remotes, procs = [], []
    for i, t in enumerate(tasks):
        a, b = ctx.Pipe()
        p = ctx.Process(target=_worker, args=(b, t, args.seed * 100 + t, cpus[i % len(cpus)],
                                              args.teacher, args.horizon, args.radius, start_kw(args)), daemon=True)
        p.start(); b.close(); remotes.append(a); procs.append(p)
    langs = [r.recv() for r in remotes]
    text = torch.stack([enc_txt(s) for s in langs])                     # (10, 512)
    for r in remotes: r.send(("reset", None))
    cur = [r.recv() for r in remotes]

    buf = {k: [] for k in ("agent", "wrist", "state", "label", "task", "episode", "step", "exec_teacher")}
    ep_id = [t * 10000 for t in tasks]; ep_step = [0] * len(tasks)
    done_eps = {t: [] for t in tasks}
    active = [True] * len(tasks)
    t0, frames = time.time(), 0
    while any(active):
        idx = [i for i in range(len(tasks)) if active[i]]
        imgs = [x for i in idx for x in (cur[i][0], cur[i][1])]
        with torch.no_grad():
            f = enc_img(*imgs).float()
            agent, wrist = f[0::2], f[1::2]
            state = torch.as_tensor(np.stack([cur[i][2] for i in idx]), device=dev)
            if student is not None:
                sa, sw = (torch.zeros_like(agent), torch.zeros_like(wrist)) if args.zero in ("image",) \
                    else (agent, wrist)
                st_text = torch.zeros_like(text[idx]) if args.zero == "text" else text[idx]
                pred = student(sa, sw, st_text, state, tok.expand(len(idx), -1, -1),
                               tmask.expand(len(idx), -1))[:, 0] * act_std
                s_act = pred.clamp(-1, 1).cpu().numpy()
        for j, i in enumerate(idx):
            teacher_exec = student is None or rng.random() < args.beta
            act = cur[i][3] if teacher_exec else s_act[j]
            if args.out:
                buf["agent"].append(agent[j].cpu().numpy().astype(np.float16))
                buf["wrist"].append(wrist[j].cpu().numpy().astype(np.float16))
                buf["state"].append(cur[i][2]); buf["label"].append(cur[i][3])
                buf["task"].append(tasks[i]); buf["episode"].append(ep_id[i])
                buf["step"].append(ep_step[i]); buf["exec_teacher"].append(teacher_exec)
            remotes[i].send(("step", act))
        frames += len(idx)
        for i in idx:
            cur[i] = remotes[i].recv()
            ep_step[i] += 1
            if cur[i][4]:
                done_eps[tasks[i]].append(dict(cur[i][5], episode=ep_id[i]))
                ep_id[i] += 1; ep_step[i] = 0
                if len(done_eps[tasks[i]]) >= args.episodes:
                    active[i] = False
    for r in remotes: r.send(("close", None))
    for p in procs: p.join(timeout=10)

    allw = []
    for t in tasks:
        w = [e["success"] for e in done_eps[t]]; allw += w
        print(f"  task {t}: {sum(w)}/{len(w)}   placement median "
              f"{np.median([e['placement_mm'] for e in done_eps[t]]):5.1f} mm", flush=True)
    driver = "teacher" if student is None else f"beta={args.beta}" + (f" zero={args.zero}" if args.zero != "none" else "")
    print(f"{driver} @ {args.radius*1000:.0f} mm: {sum(allw)}/{len(allw)} = {np.mean(allw):.2f}   "
          f"({frames} frames, {frames/(time.time()-t0):.0f} frames/s)")
    if args.out:
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        succ = {e["episode"]: e["success"] for t in tasks for e in done_eps[t]}
        ep = np.array(buf["episode"])
        np.savez(out, agent=np.stack(buf["agent"]), wrist=np.stack(buf["wrist"]),
                 state=np.stack(buf["state"]).astype(np.float32), label=np.stack(buf["label"]),
                 task=np.array(buf["task"], np.int8), episode=ep, step=np.array(buf["step"], np.int32),
                 exec_teacher=np.array(buf["exec_teacher"]),
                 episode_success=np.array([succ.get(e, False) for e in ep]),
                 text=text.cpu().numpy(), radius=args.radius, beta=args.beta)
        Path(str(out) + ".json").write_text(json.dumps({
            "driver": driver, "radius_m": args.radius, "episodes_per_task": args.episodes,
            "success": float(np.mean(allw)), "frames": frames,
            "per_task": {t: [e["success"] for e in done_eps[t]] for t in tasks}}, indent=2))
        print("->", out)
    return 0


# -------------------------------------------------------------------------- train
def train(args):
    import torch
    from torch import nn
    sys.path.insert(0, str(ROOT))
    from screwhead.policy import ScrewHead
    torch.manual_seed(args.seed)
    dev = args.device
    parts = [np.load(r) for r in args.rounds]
    # A teacher-driven episode that FAILED is the teacher in states it does not
    # understand -- measured, a failing teacher can climb to 0.89 m and stay
    # there -- so its labels teach flailing. Keep those rounds' successes only.
    # Student-driven rounds keep everything: the teacher's labels on the
    # student's mistakes are what DAgger is for.
    keeps = []
    for k, p in enumerate(parts):
        teacher_round = float(p["beta"]) >= 1.0
        keep = p["episode_success"] if (teacher_round and not args.keep_failures) else np.ones(len(p["label"]), bool)
        keeps.append(keep)
        print(f"  round {args.rounds[k]}: beta {float(p['beta'])}, {len(keep)} frames, keeping {int(keep.sum())}")
    cat = lambda k: np.concatenate([p[k][m] for p, m in zip(parts, keeps)])
    agent, wrist = cat("agent").astype(np.float32), cat("wrist").astype(np.float32)
    state, label, task = cat("state"), cat("label"), cat("task").astype(np.int64)
    episode = np.concatenate([p["episode"][m] + 10_000_000 * k for k, (p, m) in enumerate(zip(parts, keeps))])
    text = parts[0]["text"]
    if args.zero == "image":
        agent[:] = 0; wrist[:] = 0
    val = (episode % 10) == 0
    if val.all() or not val.any():
        raise SystemExit(f"degenerate split: {int(val.sum())}/{len(val)} validation frames -- collect more episodes")
    act_std = label[~val].std(0).clip(1e-6)
    print(f"{len(label)} frames from {len(args.rounds)} round(s): train {int((~val).sum())} val {int(val.sum())}"
          f"{'  ABLATION: images zeroed' if args.zero == 'image' else ''}")
    T = lambda x: torch.as_tensor(x, device=dev)
    A, W, S, Y = T(agent), T(wrist), T(state), T(label / act_std)[:, None]
    TX = T(text)[T(task)]
    if args.zero == "text":
        TX = torch.zeros_like(TX)
    tok, tmask = spec_tokens(dev)
    model = ScrewHead(chunk=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    tri = torch.nonzero(T(~val)).flatten(); vai = torch.nonzero(T(val)).flatten()
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.epochs * max(1, len(tri) // args.batch))
    fwd = lambda i: model(A[i], W[i], TX[i], S[i], tok.expand(len(i), -1, -1), tmask.expand(len(i), -1))
    best, best_state = float("inf"), None
    for ep in range(args.epochs):
        model.train()
        perm = tri[torch.randperm(len(tri), device=dev)]
        for k in range(0, len(perm) - args.batch + 1, args.batch):
            i = perm[k:k + args.batch]
            loss = nn.functional.smooth_l1_loss(fwd(i), Y[i])
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if sched.last_epoch < sched.total_steps - 1: sched.step()
        model.eval()
        with torch.no_grad():
            vl = sum(nn.functional.smooth_l1_loss(fwd(vai[k:k + 4096]), Y[vai[k:k + 4096]], reduction="sum").item()
                     for k in range(0, len(vai), 4096)) / (len(vai) * 7)
        if vl < best:
            best, best_state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d}  train {loss.item():.4f}  val {vl:.4f}{'  *' if vl == best else ''}", flush=True)
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        P = torch.cat([fwd(vai[k:k + 4096])[:, 0] for k in range(0, len(vai), 4096)]).cpu().numpy() * act_std
    Yv = label[val]
    lab = ["wx", "wy", "wz", "vx", "vy", "vz", "grip"]
    corr = [float(np.corrcoef(P[:, j], Yv[:, j])[0, 1]) for j in range(7)]
    print(f"best val {best:.4f}   corr " + "  ".join(f"{l}:{c:.2f}" for l, c in zip(lab, corr)))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in best_state.items()}, "act_std": act_std, "chunk": 1,
                "policy": "screwhead", "val": best, "corr": corr, "zero": args.zero,
                "rounds": args.rounds, "args": vars(args)}, out)
    print("->", out)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--teacher", default="checkpoints/teacher_bc_nojoints.pt")
    c.add_argument("--student", default="")
    c.add_argument("--beta", type=float, default=1.0)
    c.add_argument("--zero", default="none", choices=["none", "image", "text"])
    c.add_argument("--radius", type=float, default=0.06)
    c.add_argument("--episodes", type=int, default=30, help="per task")
    c.add_argument("--horizon", type=int, default=300)
    c.add_argument("--cpus", default="5,6,7,8,9,15,16,17,18,19", help="performance cores")
    c.add_argument("--out", default="", help="omit to evaluate without saving")
    c.add_argument("--seed", type=int, default=0)
    sys.path.insert(0, str(ROOT / "tools"))
    from teacher_rl import add_start_args
    add_start_args(c)
    c.add_argument("--device", default="cuda")
    t = sub.add_parser("train")
    t.add_argument("--rounds", nargs="+", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--zero", default="none", choices=["none", "image", "text"])
    t.add_argument("--keep-failures", action="store_true")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--batch", type=int, default=512)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cuda")
    args = ap.parse_args()
    return collect(args) if args.cmd == "collect" else train(args)


if __name__ == "__main__":
    raise SystemExit(main())
