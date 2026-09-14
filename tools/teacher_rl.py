#!/usr/bin/env python
"""PPO fine-tune of the privileged teacher under randomised placement.

  train  -- starts from the BC clone, fine-tunes on task success
  eval   -- deterministic success per task at a fixed placement radius

Why each piece is here:

  BC INITIALISATION. Grasp-lift-place is a long, sparse problem; discovering it
  from scratch is not what we are paying for. The clone supplies the motion, RL
  supplies the reason to read the object pose.

  CRITIC WARM-UP. A fresh value function on a good policy produces large, wrong
  advantages, and the first actor updates destroy the clone. The critic trains
  alone first.

  BC REGULARISER, ANNEALED. Pulls the actor mean toward the clone early, decays
  to zero, so the fine-tune can leave the demonstrations once it has signal.

  PLACEMENT CURRICULUM. Radius grows only when success at the current radius
  clears a target, so sparse reward keeps arriving.

  STANDARDISED ACTIONS. The Gaussian lives in per-channel standardised units
  (the clone's act_sd), so exploration noise on rotation is proportional to
  demonstrated rotation, not to translation.

  TRUNCATION BOOTSTRAPPED. A timeout is not failure: the value of the last state
  is bootstrapped, only success is terminal.

Workers are pinned to one torch thread; each hosts one task's environment.
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


# ----------------------------------------------------------------------- workers
def _worker(remote, task, seed, horizon):
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT))
    from screwhead.teacher_env import PrivilegedEnv
    env = PrivilegedEnv(task, radius_m=0.0, horizon=horizon, seed=seed)
    ep_ret = 0.0
    while True:
        cmd, arg = remote.recv()
        if cmd == "reset":
            env.radius = float(arg)
            remote.send(env.reset())
        elif cmd == "step":
            o, r, done, info = env.step(arg)
            final = o
            if done:
                info = dict(info, placement_mm=float(np.linalg.norm(env.placement[:2]) * 1000),
                            length=env.t)
                o = env.reset()
            remote.send((o, r, done, info, final))
        elif cmd == "radius":
            env.radius = float(arg); remote.send(True)
        elif cmd == "close":
            env.close(); remote.close(); return


class VecEnv:
    def __init__(self, tasks, seed, horizon):
        ctx = mp.get_context("spawn")
        self.remotes, self.procs = [], []
        for i, t in enumerate(tasks):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, t, seed * 1000 + i, horizon), daemon=True)
            p.start(); b.close()
            self.remotes.append(a); self.procs.append(p)
        self.tasks = list(tasks)

    def reset(self, radius):
        for r in self.remotes: r.send(("reset", radius))
        return np.stack([r.recv() for r in self.remotes])

    def set_radius(self, radius):
        for r in self.remotes: r.send(("radius", radius))
        for r in self.remotes: r.recv()

    def step(self, actions):
        for r, a in zip(self.remotes, actions): r.send(("step", a))
        out = [r.recv() for r in self.remotes]
        o, rew, done, info, final = zip(*out)
        return np.stack(o), np.array(rew, np.float32), np.array(done), list(info), np.stack(final)

    def close(self):
        for r in self.remotes:
            try: r.send(("close", None))
            except Exception: pass
        for p in self.procs: p.join(timeout=10)


# ------------------------------------------------------------------------ policy
def build(ckpt, device):
    import torch
    from torch import nn
    sys.path.insert(0, str(ROOT / "tools"))
    from teacher_bc import TeacherMLP
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    actor = TeacherMLP(ck["obs_dim"], ck["act_dim"]); actor.load_state_dict(ck["state_dict"])
    critic = nn.Sequential(nn.Linear(ck["obs_dim"], 512), nn.LayerNorm(512), nn.GELU(),
                           nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 1))
    stats = {k: torch.tensor(np.asarray(ck[k]), dtype=torch.float32, device=device)
             for k in ("obs_mu", "obs_sd", "act_sd")}
    stats["obs_mask"] = torch.tensor(np.asarray(ck.get("obs_mask", np.ones(ck["obs_dim"]))),
                                     dtype=torch.float32, device=device)
    return actor.to(device), critic.to(device), stats, ck


def to_env_action(mean_std_units, act_sd):
    return np.clip(mean_std_units * act_sd, -1.0, 1.0)


# ------------------------------------------------------------------------- train
def train(args):
    import torch
    from torch import nn
    dev = args.device
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    actor, critic, st, ck = build(args.init, dev)
    bc_actor, _, _, _ = build(args.init, dev)
    for p in bc_actor.parameters(): p.requires_grad_(False)
    log_std = nn.Parameter(torch.full((ck["act_dim"],), args.init_log_std, device=dev))
    opt_a = torch.optim.Adam(list(actor.parameters()) + [log_std], lr=args.lr_actor)
    opt_c = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)
    act_sd_np = st["act_sd"].cpu().numpy()

    tasks = [t for t in range(10) for _ in range(args.envs_per_task)]
    venv = VecEnv(tasks, args.seed, args.horizon)
    radius = args.radius_start
    obs = venv.reset(radius)
    norm = lambda o: (torch.as_tensor(o, device=dev) - st["obs_mu"]) / st["obs_sd"] * st["obs_mask"]

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "a")
    window = []           # (success, placement_mm) of recent episodes at the current radius
    steps, t0 = 0, time.time()
    N, T = len(tasks), args.rollout
    for it in range(args.iters):
        O = torch.zeros(T, N, ck["obs_dim"], device=dev); A = torch.zeros(T, N, ck["act_dim"], device=dev)
        LP = torch.zeros(T, N, device=dev); R = torch.zeros(T, N, device=dev)
        TERM = torch.zeros(T, N, device=dev); VAL = torch.zeros(T + 1, N, device=dev)
        ep_info = []
        for t in range(T):
            with torch.no_grad():
                x = norm(obs)
                mu = actor(x); std = log_std.exp()
                a = mu + std * torch.randn_like(mu)
                lp = torch.distributions.Normal(mu, std).log_prob(a).sum(-1)
                v = critic(x).squeeze(-1)
            obs2, rew, done, info, final = venv.step(to_env_action(a.cpu().numpy(), act_sd_np))
            rew_t = torch.as_tensor(rew, device=dev)
            for i, (d, inf) in enumerate(zip(done, info)):
                if d and inf["truncated"] and not inf["success"]:
                    with torch.no_grad():   # bootstrap a timeout from the state it stopped in
                        rew_t[i] += args.gamma * critic(norm(final[i:i + 1])).squeeze()
                if d:
                    ep_info.append(inf); window.append(inf["success"])
            O[t], A[t], LP[t], VAL[t] = x, a, lp, v
            R[t] = rew_t
            TERM[t] = torch.as_tensor(np.array([bool(d) for d in done]), device=dev).float()
            obs = obs2
        steps += T * N
        with torch.no_grad():
            VAL[T] = critic(norm(obs)).squeeze(-1)
            adv = torch.zeros(T, N, device=dev); last = torch.zeros(N, device=dev)
            for t in reversed(range(T)):
                nonterm = 1.0 - TERM[t]
                delta = R[t] + args.gamma * VAL[t + 1] * nonterm - VAL[t]
                last = delta + args.gamma * args.lam * nonterm * last
                adv[t] = last
            ret = adv + VAL[:T]

        bO, bA, bLP = O.reshape(T * N, -1), A.reshape(T * N, -1), LP.reshape(-1)
        bAdv, bRet = adv.reshape(-1), ret.reshape(-1)
        bAdv = (bAdv - bAdv.mean()) / (bAdv.std() + 1e-8)
        warm = it < args.critic_warmup
        bc_coef = 0.0 if warm else args.bc_coef * max(0.0, 1 - (it - args.critic_warmup) / args.bc_anneal)
        kl_sum = clipfrac = vloss_sum = 0.0; nb = 0
        for _ in range(args.epochs):
            perm = torch.randperm(T * N, device=dev)
            for k in range(0, T * N, args.minibatch):
                i = perm[k:k + args.minibatch]
                v = critic(bO[i]).squeeze(-1)
                vloss = 0.5 * (v - bRet[i]).pow(2).mean()
                opt_c.zero_grad(set_to_none=True); vloss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0); opt_c.step()
                vloss_sum += vloss.item(); nb += 1
                if warm:
                    continue
                mu = actor(bO[i]); dist = torch.distributions.Normal(mu, log_std.exp())
                lp = dist.log_prob(bA[i]).sum(-1)
                ratio = (lp - bLP[i]).exp()
                pg = -torch.min(ratio * bAdv[i],
                                ratio.clamp(1 - args.clip, 1 + args.clip) * bAdv[i]).mean()
                with torch.no_grad():
                    bc_target = bc_actor(bO[i])
                bc = (mu - bc_target).pow(2).mean()
                loss = pg + bc_coef * bc - args.ent_coef * dist.entropy().sum(-1).mean()
                opt_a.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + [log_std], 1.0); opt_a.step()
                with torch.no_grad():
                    kl_sum += (bLP[i] - lp).mean().item()
                    clipfrac += ((ratio - 1).abs() > args.clip).float().mean().item()
                with torch.no_grad():
                    log_std.clamp_(args.min_log_std, args.max_log_std)

        succ = float(np.mean([e["success"] for e in ep_info])) if ep_info else float("nan")
        # curriculum: judged over a fixed-size window at the CURRENT radius only
        advanced = False
        if len(window) >= args.curriculum_window:
            rate = float(np.mean(window[-args.curriculum_window:]))
            if rate >= args.curriculum_target and radius < args.radius_max:
                radius = min(args.radius_max, radius + args.radius_step)
                venv.set_radius(radius); window = []; advanced = True
        rec = dict(it=it, steps=steps, sps=round(steps / (time.time() - t0)), radius_mm=radius * 1000,
                   episodes=len(ep_info), success=succ, warmup=warm, bc_coef=round(bc_coef, 4),
                   value_loss=vloss_sum / max(nb, 1),
                   approx_kl=kl_sum / max(nb, 1) if not warm else None,
                   clipfrac=clipfrac / max(nb, 1) if not warm else None,
                   log_std=log_std.detach().cpu().numpy().round(3).tolist(), advanced=advanced)
        log.write(json.dumps(rec) + "\n"); log.flush()
        if it % args.print_every == 0 or advanced:
            print(f"it {it:4d} steps {steps:8d} ({rec['sps']}/s)  radius {radius*1000:4.0f}mm  "
                  f"eps {len(ep_info):3d} success {succ:.2f}  vloss {rec['value_loss']:.4f}  "
                  f"{'WARMUP' if warm else 'kl %.4f clip %.2f' % (rec['approx_kl'], rec['clipfrac'])}"
                  f"{'  -> radius up' if advanced else ''}", flush=True)
        if (it + 1) % args.save_every == 0 or it == args.iters - 1:
            torch.save({**{k: ck[k] for k in ("obs_mu", "obs_sd", "act_sd", "obs_dim", "act_dim")},
                        "obs_mask": st["obs_mask"].cpu().numpy(),
                        "state_dict": {k: v.cpu() for k, v in actor.state_dict().items()},
                        "critic": {k: v.cpu() for k, v in critic.state_dict().items()},
                        "log_std": log_std.detach().cpu(), "radius_m": radius, "it": it,
                        "steps": steps, "args": vars(args)}, out / "teacher_rl.pt")
    venv.close(); log.close()
    return 0


# -------------------------------------------------------------------------- eval
def _eval_task(a):
    ckpt, task, radius, episodes, seed, horizon = a
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT))
    from screwhead.teacher_env import PrivilegedEnv
    actor, _, st, _ = build(ckpt, "cpu")
    env = PrivilegedEnv(task, radius_m=radius, horizon=horizon, seed=seed)
    wins, places, lifts, closest = [], [], [], []
    for ep in range(episodes):
        o = env.reset(init_index=ep % len(env.init_states))
        places.append(float(np.linalg.norm(env.placement[:2]) * 1000))
        z0, lift, near = float(o[19]), 0.0, 1e9
        done, info = False, {}
        while not done:
            with torch.no_grad():
                mu = actor((torch.as_tensor(o) - st["obs_mu"]) / st["obs_sd"] * st["obs_mask"]).numpy()
            o, _, done, info = env.step(to_env_action(mu, st["act_sd"].numpy()))
            lift = max(lift, float(o[19]) - z0)                 # bowl height rise
            near = min(near, float(np.linalg.norm(o[29:32])))   # closest tool approach to bowl
        wins.append(bool(info["success"])); lifts.append(lift * 1000); closest.append(near * 1000)
    env.close()
    return task, wins, places, lifts, closest


def evaluate(args):
    ctx = mp.get_context("spawn")
    jobs = [(args.ckpt, t, args.radius, args.episodes, args.seed + t, args.horizon) for t in range(10)]
    with ctx.Pool(args.procs) as pool:
        res = pool.map(_eval_task, jobs)
    allw, alll, allc = [], [], []
    for t, w, p, l, c in sorted(res):
        allw += w; alll += l; allc += c
        print(f"  task {t}: {sum(w)}/{len(w)}   placement median {np.median(p):5.1f} mm   "
              f"lifted>30mm {np.mean(np.array(l) > 30):.1f}   closest approach median {np.median(c):5.1f} mm")
    L, W = np.array(alll), np.array(allw)
    print(f"radius {args.radius*1000:.0f} mm: {W.sum()}/{len(W)} = {W.mean():.2f}   "
          f"| bowl lifted >30 mm in {np.mean(L > 30):.2f} of episodes; "
          f"success given lifted {W[L > 30].mean() if (L > 30).any() else float('nan'):.2f}; "
          f"closest approach median {np.median(allc):.1f} mm")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"ckpt": args.ckpt, "radius_m": args.radius,
                                              "per_task": {t: w for t, w, *_ in res}}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--init", default="checkpoints/teacher_bc.pt")
    tr.add_argument("--out", default="checkpoints/teacher_rl")
    tr.add_argument("--envs-per-task", type=int, default=2)
    tr.add_argument("--iters", type=int, default=2000)
    tr.add_argument("--rollout", type=int, default=128)
    tr.add_argument("--horizon", type=int, default=300)
    tr.add_argument("--epochs", type=int, default=4)
    tr.add_argument("--minibatch", type=int, default=1024)
    tr.add_argument("--gamma", type=float, default=0.99)
    tr.add_argument("--lam", type=float, default=0.95)
    tr.add_argument("--clip", type=float, default=0.2)
    tr.add_argument("--lr-actor", type=float, default=3e-5)
    tr.add_argument("--lr-critic", type=float, default=3e-4)
    tr.add_argument("--init-log-std", type=float, default=-1.0)
    tr.add_argument("--min-log-std", type=float, default=-3.0)
    tr.add_argument("--max-log-std", type=float, default=0.0)
    tr.add_argument("--ent-coef", type=float, default=0.0)
    tr.add_argument("--critic-warmup", type=int, default=20)
    tr.add_argument("--bc-coef", type=float, default=1.0)
    tr.add_argument("--bc-anneal", type=int, default=300)
    tr.add_argument("--radius-start", type=float, default=0.0)
    tr.add_argument("--radius-step", type=float, default=0.01)
    tr.add_argument("--radius-max", type=float, default=0.06,
                    help="60 mm: open-loop replay of the demos succeeds 0.40 under a uniform disc here, measured; beyond it the curve is extrapolated and ejections reach 20%%")
    tr.add_argument("--curriculum-target", type=float, default=0.6)
    tr.add_argument("--curriculum-window", type=int, default=100)
    tr.add_argument("--print-every", type=int, default=5)
    tr.add_argument("--save-every", type=int, default=25)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--device", default="cuda")
    ev = sub.add_parser("eval")
    ev.add_argument("--ckpt", required=True)
    ev.add_argument("--radius", type=float, default=0.0)
    ev.add_argument("--episodes", type=int, default=10)
    ev.add_argument("--horizon", type=int, default=300)
    ev.add_argument("--procs", type=int, default=10)
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--out", default="")
    args = ap.parse_args()
    return train(args) if args.cmd == "train" else evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
