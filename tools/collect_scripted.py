#!/usr/bin/env python
"""Teacher-driven demonstrations from the per-task programs, collected without lockstep.

tools/distill.py steps every environment together because a student-driven round
needs a GPU encode per step. A teacher-driven round does not: the programs run on
the CPU. Stepping in lockstep made every worker wait for the slowest reset -- with
object layouts a reset samples, settles, and checks reachability, and takes seconds
-- so ten workers ran at the pace of one.

Here each worker runs whole episodes on its own and keeps only the successful ones,
writing raw frames, student state and the teacher's clean labels to a shard as it
goes. Encoding with the frozen CLIP tower happens afterwards, in GPU batches
(`encode`). The resulting file has the same fields tools/distill.py train reads.

DART noise is applied to the executed action in free-space phases only (approach,
rise, carry); labels are always the program's clean action.
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
NOISY_PHASES = {"approach", "rise", "carry"}


def worker(a):
    task, cpu, n_success, seed, env_kw, noise, out_dir, max_episodes, wid = a
    os.sched_setaffinity(0, {cpu}); os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT))
    from screwhead.teacher_env import PrivilegedEnv
    from screwhead.scripted_teacher import ScriptedTeacher
    env = PrivilegedEnv(task, seed=seed, render=True, **env_kw)
    prog = ScriptedTeacher(env)
    if env.layout_radius > 0:
        env.layout_check = prog.layout_feasible
    rng = np.random.default_rng(seed)
    scale = np.array([0.12] * 3 + [0.25] * 3 + [0.0], np.float32)
    kept = {k: [] for k in ("agent", "wrist", "state", "label", "episode", "step")}
    n_ok = n_ep = frames = 0
    t0 = time.time()
    tag = f"task{task:02d}" + (f"_w{wid}" if wid is not None else "")
    log = open(Path(out_dir) / f"{tag}.log", "a")
    while n_ok < n_success and n_ep < max_episodes:
        env.reset()
        ep = {k: [] for k in kept}
        done, info = False, {}
        while not done:
            a_img, w_img = env.images()
            label = prog.act().astype(np.float32)
            ep["agent"].append(a_img.copy()); ep["wrist"].append(w_img.copy())
            ep["state"].append(env.student_state()); ep["label"].append(label)
            ep["step"].append(env.t)
            act = label
            if noise > 0 and prog.phase in NOISY_PHASES:
                act = np.clip(label + noise * scale * rng.standard_normal(7).astype(np.float32), -1, 1)
            _, _, done, info = env.step(act)
        n_ep += 1
        if info.get("success"):
            eid = task * 100000 + (0 if wid is None else wid) * 1000 + n_ok
            ep["episode"] = [eid] * len(ep["label"])
            for k in kept:
                kept[k] += ep[k]
            n_ok += 1; frames += len(ep["label"])
        log.write(json.dumps(dict(task=task, episode=n_ep, success=bool(info.get("success")), steps=env.t,
                                  kept=n_ok, frames=frames, elapsed=round(time.time() - t0, 1))) + "\n"); log.flush()
    env.close()
    path = Path(out_dir) / f"{tag}.npz"
    np.savez(path, agent=np.stack(kept["agent"]), wrist=np.stack(kept["wrist"]),
             state=np.stack(kept["state"]).astype(np.float32), label=np.stack(kept["label"]),
             episode=np.array(kept["episode"], np.int64), step=np.array(kept["step"], np.int32),
             task=np.full(len(kept["label"]), task, np.int8), language=env.language,
             attempts=n_ep, successes=n_ok)
    return task, n_ok, n_ep, frames, time.time() - t0


def collect(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    env_kw = dict(horizon=args.horizon, layout_radius=args.layout_radius, start_xy_m=args.start_xy,
                  start_z_m=args.start_z, start_yaw_deg=args.start_yaw, start_tilt_deg=args.start_tilt,
                  start_null_rad=args.start_null)
    cpus = [int(c) for c in args.cpus.split(",")]
    wpt = args.workers_per_task
    jobs = []
    for t in args.tasks:
        for w in range(wpt):
            quota = args.successes // wpt + (1 if w < args.successes % wpt else 0)
            jobs.append((t, None, quota, args.seed * 1000 + t * 10 + w, env_kw, args.exec_noise, str(out),
                         args.max_episodes, w if wpt > 1 else None))
    jobs = [(j[0], cpus[i % len(cpus)], *j[2:]) for i, j in enumerate(jobs)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(len(cpus)) as pool:
        for task, ok, n, frames, dt in pool.imap_unordered(worker, jobs):
            print(f"  task {task}: {ok}/{n} successful, {frames} frames, {dt/60:.1f} min", flush=True)
    print(f"collected in {(time.time()-t0)/60:.1f} min -> {out}")
    return 0


def encode(args):
    """Raw frames -> frozen CLIP features, in GPU batches, in the format distill.py trains on."""
    import torch
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
    from rollout import clip_encoder
    enc_img, enc_txt = clip_encoder(args.device)
    shards = sorted(Path(args.shards).glob("task*.npz"))
    A, W, S, L, E, ST, T = [], [], [], [], [], [], []
    text = torch.zeros(10, 512, device=args.device)
    for f in shards:
        d = np.load(f, allow_pickle=True)
        if len(d["label"]) == 0:
            continue
        t = int(d["task"][0]); text[t] = enc_txt(str(d["language"]))
        for k in range(0, len(d["label"]), args.batch):
            frames = list(d["agent"][k:k + args.batch]) + list(d["wrist"][k:k + args.batch])
            with torch.no_grad():
                feats = enc_img(*frames).float().cpu().numpy().astype(np.float16)
            n = len(feats) // 2
            A.append(feats[:n]); W.append(feats[n:])
        S.append(d["state"]); L.append(d["label"]); E.append(d["episode"]); ST.append(d["step"]); T.append(d["task"])
        print(f"  encoded {f.name}: {len(d['label'])} frames ({int(d['successes'])}/{int(d['attempts'])} episodes kept)", flush=True)
    ep = np.concatenate(E)
    np.savez(args.out, agent=np.concatenate(A), wrist=np.concatenate(W), state=np.concatenate(S),
             label=np.concatenate(L), task=np.concatenate(T), episode=ep, step=np.concatenate(ST),
             exec_teacher=np.ones(len(ep), bool), episode_success=np.ones(len(ep), bool),
             text=text.cpu().numpy(), radius=0.0, beta=1.0)
    print(f"-> {args.out}  ({len(ep)} frames, {len(np.unique(ep))} episodes)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--out", required=True)
    c.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    c.add_argument("--successes", type=int, default=20, help="successful episodes kept per task")
    c.add_argument("--max-episodes", type=int, default=200)
    c.add_argument("--workers-per-task", type=int, default=1)
    c.add_argument("--horizon", type=int, default=400)
    c.add_argument("--layout-radius", type=float, default=0.08)
    c.add_argument("--start-xy", type=float, default=0.10)
    c.add_argument("--start-z", type=float, default=0.05)
    c.add_argument("--start-yaw", type=float, default=30.0)
    c.add_argument("--start-tilt", type=float, default=10.0)
    c.add_argument("--start-null", type=float, default=0.3)
    c.add_argument("--exec-noise", type=float, default=0.3)
    c.add_argument("--cpus", default="5,6,7,8,9,15,16,17,18,19")
    c.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("encode")
    e.add_argument("--shards", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--batch", type=int, default=256)
    e.add_argument("--device", default="cuda")
    args = ap.parse_args()
    return collect(args) if args.cmd == "collect" else encode(args)


if __name__ == "__main__":
    raise SystemExit(main())
