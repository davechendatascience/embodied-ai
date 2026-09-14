#!/usr/bin/env python
"""PPO from scratch on the waypoint-progress reward. No demonstrations anywhere.

  train  -- random-initialised actor, decentralised collection, shaped reward
  eval   -- deterministic policy on held-out start poses

Reward is PrivilegedEnv's potential-based progress shaping (screwhead/progress.py)
plus a task bonus only for a bowl that was actually lifted. Observations are the
privileged state with joint angles, plus progress features (phi, stage, the
tool's error to the grasp waypoint); normalised by running statistics, since
there is no dataset to take them from.

Logged per iteration, because progress is visible long before success is:
  success_lifted rate, mean peak phi, the distribution of the highest stage
  reached, action noise, KL.
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


def mlp(i, o, h=256):
    from torch import nn
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))


class RunningNorm:
    def __init__(self, dim):
        self.mean, self.var, self.n = np.zeros(dim), np.ones(dim), 1e-4

    def update(self, x):
        bm, bv, bn = x.mean(0), x.var(0), len(x)
        tot = self.n + bn
        d = bm - self.mean
        self.mean = self.mean + d * bn / tot
        self.var = (self.var * self.n + bv * bn + d ** 2 * self.n * bn / tot) / tot
        self.n = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10, 10)


def env_kwargs(args):
    return dict(horizon=args.horizon, shaping=True, rich_obs=True, gamma=args.gamma,
                shaping_gamma=getattr(args, "shaping_gamma", 1.0),
                success_bonus=args.success_bonus, start_xy_m=args.start_xy, start_z_m=args.start_z,
                start_yaw_deg=args.start_yaw, start_tilt_deg=args.start_tilt, start_null_rad=args.start_null)


# --------------------------------------------------------------------- collectors
def _collector(remote, task, seed, cpu, kw, obs_dim, act_dim):
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT))
    from screwhead.teacher_env import PrivilegedEnv
    actor = mlp(obs_dim, act_dim)
    env = PrivilegedEnv(task, seed=seed, **kw)
    rng = np.random.default_rng(seed)
    obs = env.reset()
    ep = dict(peak_phi=0.0, peak_stage=0)
    remote.send(obs.shape[0])
    while True:
        cmd, arg = remote.recv()
        if cmd == "close":
            env.close(); remote.close(); return
        weights, log_std, mean, var, T, deterministic = arg
        actor.load_state_dict(weights)
        std = np.exp(np.asarray(log_std, np.float32))
        norm = lambda o: np.clip((o - mean) / np.sqrt(var + 1e-8), -10, 10).astype(np.float32)
        RAW = np.zeros((T, obs_dim), np.float32); O = np.zeros((T, obs_dim), np.float32)
        A = np.zeros((T, act_dim), np.float32); LP = np.zeros(T, np.float32); R = np.zeros(T, np.float32)
        TERM = np.zeros(T, np.float32); TOUT = np.zeros(T, np.float32); FINAL = np.zeros((T, obs_dim), np.float32)
        eps = []
        for t in range(T):
            x = norm(obs)
            with torch.no_grad():
                m = actor(torch.from_numpy(x)).numpy()
            a = m if deterministic else m + std * rng.standard_normal(act_dim).astype(np.float32)
            LP[t] = float(-0.5 * (((a - m) / std) ** 2).sum() - np.log(std).sum() - 0.5 * act_dim * np.log(2 * np.pi))
            RAW[t], O[t], A[t] = obs, x, a
            o2, r, done, info = env.step(np.clip(a, -1.0, 1.0))
            R[t] = r
            ep["peak_phi"] = max(ep["peak_phi"], info.get("phi", 0.0))
            ep["peak_stage"] = max(ep["peak_stage"], info.get("stage", 0))
            if done:
                if info["success"]:
                    TERM[t] = 1.0
                elif info["truncated"]:
                    TOUT[t] = 1.0; FINAL[t] = norm(o2)
                eps.append(dict(ep, success=info["success"], success_lifted=info.get("success_lifted", False),
                                length=env.t))
                ep = dict(peak_phi=0.0, peak_stage=0)
                o2 = env.reset()
            obs = o2
        remote.send((RAW, O, A, LP, R, TERM, TOUT, FINAL, norm(obs), eps))


class Collector:
    def __init__(self, tasks, seed, cpus, kw, obs_dim, act_dim):
        ctx = mp.get_context("spawn")
        self.remotes, self.procs = [], []
        for i, t in enumerate(tasks):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_collector, args=(b, t, seed * 1000 + i, cpus[i % len(cpus)] if cpus else None,
                                                     kw, obs_dim, act_dim), daemon=True)
            p.start(); b.close(); self.remotes.append(a); self.procs.append(p)
        dims = [r.recv() for r in self.remotes]
        assert all(d == obs_dim for d in dims), f"observation dim mismatch: {dims} vs {obs_dim}"

    def collect(self, *arg):
        for r in self.remotes: r.send(("collect", arg))
        return [r.recv() for r in self.remotes]

    def close(self):
        for r in self.remotes:
            try: r.send(("close", None))
            except Exception: pass
        for p in self.procs: p.join(timeout=10)


def parse_cpus(spec):
    out = []
    for part in filter(None, spec.split(",")):
        lo, _, hi = part.partition("-"); out += list(range(int(lo), int(hi or lo) + 1))
    return out


# -------------------------------------------------------------------------- train
def train(args):
    import torch
    from torch import nn
    sys.path.insert(0, str(ROOT))
    from screwhead.teacher_env import OBS_DIM, PrivilegedEnv
    torch.manual_seed(args.seed)
    dev = args.device
    obs_dim, act_dim = OBS_DIM + PrivilegedEnv.RICH_DIM, 7
    actor, critic = mlp(obs_dim, act_dim).to(dev), mlp(obs_dim, 1).to(dev)
    with torch.no_grad():
        actor[-1].weight.mul_(0.01); actor[-1].bias.zero_()
    # initial exploration per channel: rotation 0.1, translation 0.3, gripper 0.5 (normalised action units)
    log_std = nn.Parameter(torch.tensor(np.log([0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.5]), dtype=torch.float32, device=dev))
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()) + [log_std], lr=args.lr, eps=1e-5)
    norm = RunningNorm(obs_dim)
    if args.learner_cpus:
        os.sched_setaffinity(0, set(parse_cpus(args.learner_cpus)))
    tasks = [t for t in args.tasks for _ in range(args.envs_per_task)]
    col = Collector(tasks, args.seed, parse_cpus(args.worker_cpus), env_kwargs(args), obs_dim, act_dim)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    log = open(out / "log.jsonl", "a")
    steps, t0, T = 0, time.time(), args.rollout
    for it in range(args.iters):
        tc = time.perf_counter()
        w = {k: v.detach().cpu() for k, v in actor.state_dict().items()}
        segs = col.collect(w, log_std.detach().cpu().numpy(), norm.mean.copy(), norm.var.copy(), T, False)
        t_col = time.perf_counter() - tc
        RAW = np.concatenate([s[0] for s in segs]); norm.update(RAW)
        st = lambda j: torch.as_tensor(np.stack([s[j] for s in segs], 1), device=dev)
        O, A, LP, R, TERM, TOUT, FINAL = (st(j) for j in range(1, 8))
        LAST = torch.as_tensor(np.stack([s[8] for s in segs]), device=dev)
        eps = [e for s in segs for e in s[9]]
        N = len(segs); steps += T * N
        R = R * args.reward_scale
        with torch.no_grad():
            V = critic(O.reshape(T * N, -1)).reshape(T, N)
            Vl = critic(LAST).squeeze(-1)
            R = R + args.gamma * critic(FINAL.reshape(T * N, -1)).reshape(T, N) * TOUT
            Vn = torch.cat([V[1:], Vl[None]], 0)
            ended = ((TERM + TOUT) > 0).float()
            adv = torch.zeros(T, N, device=dev); last = torch.zeros(N, device=dev)
            for t in reversed(range(T)):
                delta = R[t] + args.gamma * Vn[t] * (1 - ended[t]) - V[t]
                last = delta + args.gamma * args.lam * (1 - ended[t]) * last
                adv[t] = last
            ret = adv + V
        bO, bA, bLP, bAdv, bRet = O.reshape(T * N, -1), A.reshape(T * N, -1), LP.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        bAdv = (bAdv - bAdv.mean()) / (bAdv.std() + 1e-8)
        kls, clips, vls = [], [], []
        for _ in range(args.epochs):
            perm = torch.randperm(T * N, device=dev)
            for k in range(0, T * N, args.minibatch):
                i = perm[k:k + args.minibatch]
                mu = actor(bO[i]); dist = torch.distributions.Normal(mu, log_std.exp())
                lp = dist.log_prob(bA[i]).sum(-1); ratio = (lp - bLP[i]).exp()
                pg = -torch.min(ratio * bAdv[i], ratio.clamp(1 - args.clip, 1 + args.clip) * bAdv[i]).mean()
                vl = 0.5 * (critic(bO[i]).squeeze(-1) - bRet[i]).pow(2).mean()
                loss = pg + args.vf_coef * vl - args.ent_coef * dist.entropy().sum(-1).mean()
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()) + [log_std], 0.5)
                opt.step()
                with torch.no_grad():
                    log_std.clamp_(np.log(0.02), np.log(1.0))
                    kls.append((bLP[i] - lp).mean().item()); clips.append(((ratio - 1).abs() > args.clip).float().mean().item())
                vls.append(vl.item())
        stages = np.bincount([e["peak_stage"] for e in eps], minlength=7) if eps else np.zeros(7, int)
        rec = dict(it=it, steps=steps, sps=round(steps / (time.time() - t0)), episodes=len(eps),
                   success=float(np.mean([e["success"] for e in eps])) if eps else None,
                   success_lifted=float(np.mean([e["success_lifted"] for e in eps])) if eps else None,
                   peak_phi=float(np.mean([e["peak_phi"] for e in eps])) if eps else None,
                   peak_stage_hist=stages.tolist(), kl=float(np.mean(kls)), clipfrac=float(np.mean(clips)),
                   value_loss=float(np.mean(vls)), std=log_std.exp().detach().cpu().numpy().round(3).tolist(),
                   collect_s=round(t_col, 2))
        log.write(json.dumps(rec) + "\n"); log.flush()
        if it % args.print_every == 0:
            reached = (np.cumsum(stages[::-1])[::-1] / max(len(eps), 1)).round(2) if eps else []
            print(f"it {it:4d} steps {steps:8d} ({rec['sps']}/s) eps {len(eps):3d}  "
                  f"success_lifted {rec['success_lifted'] if rec['success_lifted'] is None else round(rec['success_lifted'], 2)}  "
                  f"peak_phi {rec['peak_phi'] if rec['peak_phi'] is None else round(rec['peak_phi'], 2)}  "
                  f"reached>=stage {list(reached)}  kl {rec['kl']:.4f}  std {rec['std']}", flush=True)
        if (it + 1) % args.save_every == 0 or it == args.iters - 1:
            ck = dict(actor={k: v.cpu() for k, v in actor.state_dict().items()},
                      critic={k: v.cpu() for k, v in critic.state_dict().items()},
                      log_std=log_std.detach().cpu(), norm_mean=norm.mean, norm_var=norm.var,
                      obs_dim=obs_dim, act_dim=act_dim, it=it, steps=steps, args=vars(args))
            torch.save(ck, out / "policy.pt")
            if args.keep_every and (it + 1) % args.keep_every == 0:
                torch.save(ck, out / f"policy_it{it+1:05d}.pt")
    col.close(); log.close()
    return 0


# --------------------------------------------------------------------------- eval
def evaluate(args):
    import torch
    sys.path.insert(0, str(ROOT))
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = argparse.Namespace(**{**ck["args"], **{k: v for k, v in vars(args).items() if v is not None}})
    col = Collector([t for t in args.tasks for _ in range(args.envs_per_task)], args.seed,
                    parse_cpus(args.worker_cpus), env_kwargs(a), ck["obs_dim"], ck["act_dim"])
    eps = []
    while len(eps) < args.episodes:
        segs = col.collect(ck["actor"], ck["log_std"].numpy(), ck["norm_mean"], ck["norm_var"], 300, True)
        eps += [e for s in segs for e in s[9]]
    col.close()
    eps = eps[: args.episodes]
    st = np.bincount([e["peak_stage"] for e in eps], minlength=7)
    reached = (np.cumsum(st[::-1])[::-1] / len(eps)).round(2)
    print(f"{args.ckpt}: {len(eps)} held-out episodes  success_lifted {np.mean([e['success_lifted'] for e in eps]):.2f}  "
          f"success(any) {np.mean([e['success'] for e in eps]):.2f}  peak_phi {np.mean([e['peak_phi'] for e in eps]):.2f}  "
          f"reached>=stage {list(reached)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for p in (tr := sub.add_parser("train"), ev := sub.add_parser("eval")):
        p.add_argument("--tasks", type=int, nargs="+", default=[2])
        p.add_argument("--envs-per-task", type=int, default=10)
        p.add_argument("--worker-cpus", default="5-9,15-19")
        p.add_argument("--seed", type=int, default=0)
    tr.add_argument("--out", required=True)
    tr.add_argument("--iters", type=int, default=1200)
    tr.add_argument("--rollout", type=int, default=256)
    tr.add_argument("--horizon", type=int, default=300)
    tr.add_argument("--gamma", type=float, default=0.99)
    tr.add_argument("--lam", type=float, default=0.95)
    tr.add_argument("--clip", type=float, default=0.2)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--epochs", type=int, default=5)
    tr.add_argument("--minibatch", type=int, default=2048)
    tr.add_argument("--vf-coef", type=float, default=0.5)
    tr.add_argument("--ent-coef", type=float, default=0.0)
    tr.add_argument("--reward-scale", type=float, default=1.0)
    tr.add_argument("--shaping-gamma", type=float, default=1.0,
                    help="discount inside the potential difference; <1 adds a drag proportional to progress")
    tr.add_argument("--success-bonus", type=float, default=10.0)
    tr.add_argument("--start-xy", type=float, default=0.10)
    tr.add_argument("--start-z", type=float, default=0.05)
    tr.add_argument("--start-yaw", type=float, default=30.0)
    tr.add_argument("--start-tilt", type=float, default=10.0)
    tr.add_argument("--start-null", type=float, default=0.3)
    tr.add_argument("--learner-cpus", default="0-4,10-14")
    tr.add_argument("--print-every", type=int, default=10)
    tr.add_argument("--save-every", type=int, default=25)
    tr.add_argument("--keep-every", type=int, default=100)
    tr.add_argument("--device", default="cuda")
    ev.add_argument("--ckpt", required=True)
    ev.add_argument("--episodes", type=int, default=100)
    args = ap.parse_args()
    return train(args) if args.cmd == "train" else evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
