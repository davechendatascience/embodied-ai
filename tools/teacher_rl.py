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
def _worker(remote, task, seed, horizon, cpu):
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
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
    def __init__(self, tasks, seed, horizon, cpus=None):
        ctx = mp.get_context("spawn")
        self.remotes, self.procs = [], []
        for i, t in enumerate(tasks):
            a, b = ctx.Pipe()
            cpu = None if not cpus else cpus[i % len(cpus)]
            p = ctx.Process(target=_worker, args=(b, t, seed * 1000 + i, horizon, cpu), daemon=True)
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


def parse_cpus(spec: str) -> list[int]:
    out = []
    for part in filter(None, spec.split(",")):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


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



# ------------------------------------------------------- decentralised collection
def _collector(remote, task, seed, horizon, cpu, init_ckpt):
    """Collects a whole rollout segment locally with its own CPU copy of the actor.

    The per-step synchronous vector env paid every worker's tail latency on every
    step (21 ms per step against a 10 ms isolated worker). Here the barrier is
    once per iteration, so efficiency cores contribute instead of stalling.
    """
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT))
    from screwhead.teacher_env import PrivilegedEnv
    actor, _, st, ck = build(init_ckpt, "cpu")
    mu_o, sd_o, mask, act_sd = (st["obs_mu"].numpy(), st["obs_sd"].numpy(),
                                st["obs_mask"].numpy(), st["act_sd"].numpy())
    env = PrivilegedEnv(task, radius_m=0.0, horizon=horizon, seed=seed)
    rng = np.random.default_rng(seed)
    obs = None
    while True:
        cmd, arg = remote.recv()
        if cmd == "close":
            env.close(); remote.close(); return
        weights, log_std, radius, T = arg
        actor.load_state_dict(weights)
        std = np.exp(np.asarray(log_std, np.float32))
        if radius != env.radius or obs is None:
            env.radius = radius
            obs = env.reset()
        n_o, n_a = len(obs), len(act_sd)
        O = np.zeros((T, n_o), np.float32); A = np.zeros((T, n_a), np.float32)
        LP = np.zeros(T, np.float32); R = np.zeros(T, np.float32)
        TERM = np.zeros(T, np.float32); TIMEOUT = np.zeros(T, np.float32)
        FINAL = np.zeros((T, n_o), np.float32); infos = []
        for t in range(T):
            x = (obs - mu_o) / sd_o * mask
            with torch.no_grad():
                m = actor(torch.from_numpy(x.astype(np.float32))).numpy()
            a = m + std * rng.standard_normal(n_a).astype(np.float32)
            LP[t] = float(-0.5 * (((a - m) / std) ** 2).sum() - np.log(std).sum()
                          - 0.5 * n_a * np.log(2 * np.pi))
            O[t], A[t] = x, a
            o2, r, done, info = env.step(np.clip(a * act_sd, -1.0, 1.0))
            R[t] = r
            if done:
                TERM[t] = 1.0 if info["success"] else 0.0
                if info["truncated"] and not info["success"]:
                    TIMEOUT[t] = 1.0
                    FINAL[t] = (o2 - mu_o) / sd_o * mask
                infos.append(dict(success=info["success"],
                                  placement_mm=float(np.linalg.norm(env.placement[:2]) * 1000),
                                  length=env.t))
                o2 = env.reset()
            obs = o2
        last = ((obs - mu_o) / sd_o * mask).astype(np.float32)
        remote.send((O, A, LP, R, TERM, TIMEOUT, FINAL, last, infos))


class Collector:
    def __init__(self, tasks, seed, horizon, cpus, init_ckpt):
        ctx = mp.get_context("spawn")
        self.remotes, self.procs = [], []
        for i, t in enumerate(tasks):
            a, b = ctx.Pipe()
            cpu = None if not cpus else cpus[i % len(cpus)]
            p = ctx.Process(target=_collector, args=(b, t, seed * 1000 + i, horizon, cpu, init_ckpt),
                            daemon=True)
            p.start(); b.close()
            self.remotes.append(a); self.procs.append(p)

    def collect(self, weights, log_std, radius, T):
        for r in self.remotes:
            r.send(("collect", (weights, log_std, radius, T)))
        return [r.recv() for r in self.remotes]

    def close(self):
        for r in self.remotes:
            try: r.send(("close", None))
            except Exception: pass
        for p in self.procs: p.join(timeout=10)

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
    if args.decentralised:
        return train_decentralised(args, actor, critic, bc_actor, log_std, opt_a, opt_c, st, ck, tasks)
    # The GB10 has 10 Cortex-X925 cores (3.9 GHz) and 10 Cortex-A725 (2.8 GHz).
    # A synchronous vector env steps at the pace of its SLOWEST worker, so an
    # unpinned worker landing on an efficiency core sets everyone's rate.
    # Workers get the performance cores, the learner the efficiency cores.
    cpus = parse_cpus(args.worker_cpus)
    if args.learner_cpus:
        os.sched_setaffinity(0, set(parse_cpus(args.learner_cpus)))
    venv = VecEnv(tasks, args.seed, args.horizon, cpus)
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
        t_pol = t_env = t_book = 0.0
        for t in range(T):
            _a = time.perf_counter()
            with torch.no_grad():
                x = norm(obs)
                mu = actor(x); std = log_std.exp()
                a = mu + std * torch.randn_like(mu)
                lp = torch.distributions.Normal(mu, std).log_prob(a).sum(-1)
                v = critic(x).squeeze(-1)
            act_np = to_env_action(a.cpu().numpy(), act_sd_np)
            _b = time.perf_counter()
            obs2, rew, done, info, final = venv.step(act_np)
            _c = time.perf_counter()
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
            t_pol += _b - _a; t_env += _c - _b; t_book += time.perf_counter() - _c
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
                   log_std=log_std.detach().cpu().numpy().round(3).tolist(), advanced=advanced,
                   t_policy_ms=round(t_pol / T * 1000, 2), t_env_ms=round(t_env / T * 1000, 2),
                   t_book_ms=round(t_book / T * 1000, 2))
        log.write(json.dumps(rec) + "\n"); log.flush()
        if it % args.print_every == 0 or advanced:
            print(f"it {it:4d} steps {steps:8d} ({rec['sps']}/s)  radius {radius*1000:4.0f}mm  "
                  f"eps {len(ep_info):3d} success {succ:.2f}  vloss {rec['value_loss']:.4f}  "
                  f"{'WARMUP' if warm else 'kl %.4f clip %.2f' % (rec['approx_kl'], rec['clipfrac'])}"
                  f"{'  -> radius up' if advanced else ''}  [per step: policy {rec['t_policy_ms']}ms env {rec['t_env_ms']}ms book {rec['t_book_ms']}ms]", flush=True)
        if (it + 1) % args.save_every == 0 or it == args.iters - 1:
            torch.save({**{k: ck[k] for k in ("obs_mu", "obs_sd", "act_sd", "obs_dim", "act_dim")},
                        "obs_mask": st["obs_mask"].cpu().numpy(),
                        "state_dict": {k: v.cpu() for k, v in actor.state_dict().items()},
                        "critic": {k: v.cpu() for k, v in critic.state_dict().items()},
                        "log_std": log_std.detach().cpu(), "radius_m": radius, "it": it,
                        "steps": steps, "args": vars(args)}, out / "teacher_rl.pt")
    venv.close(); log.close()
    return 0




def ppo_update(args, it, actor, critic, bc_actor, log_std, opt_a, opt_c, bO, bA, bLP, bAdv, bRet):
    import torch
    from torch import nn
    bAdv = (bAdv - bAdv.mean()) / (bAdv.std() + 1e-8)
    warm = it < args.critic_warmup
    bc_coef = 0.0 if warm else args.bc_coef * max(0.0, 1 - (it - args.critic_warmup) / args.bc_anneal)
    n = len(bO); kl_sum = clipfrac = vloss_sum = 0.0; nb = 0
    for _ in range(args.epochs):
        perm = torch.randperm(n, device=bO.device)
        for k in range(0, n, args.minibatch):
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
            pg = -torch.min(ratio * bAdv[i], ratio.clamp(1 - args.clip, 1 + args.clip) * bAdv[i]).mean()
            with torch.no_grad():
                bc_target = bc_actor(bO[i])
            loss = pg + bc_coef * (mu - bc_target).pow(2).mean() - args.ent_coef * dist.entropy().sum(-1).mean()
            opt_a.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(list(actor.parameters()) + [log_std], 1.0); opt_a.step()
            with torch.no_grad():
                kl_sum += (bLP[i] - lp).mean().item()
                clipfrac += ((ratio - 1).abs() > args.clip).float().mean().item()
                log_std.clamp_(args.min_log_std, args.max_log_std)
    return dict(warmup=warm, bc_coef=round(bc_coef, 4), value_loss=vloss_sum / max(nb, 1),
                approx_kl=None if warm else kl_sum / max(nb, 1),
                clipfrac=None if warm else clipfrac / max(nb, 1))


def train_decentralised(args, actor, critic, bc_actor, log_std, opt_a, opt_c, st, ck, tasks):
    import torch
    dev = args.device
    cpus = parse_cpus(args.worker_cpus)
    col = Collector(tasks, args.seed, args.horizon, cpus, args.init)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "a")
    radius, window, steps, t0 = args.radius_start, [], 0, time.time()
    T = args.rollout
    for it in range(args.iters):
        tc = time.perf_counter()
        weights = {k: v.detach().cpu() for k, v in actor.state_dict().items()}
        segs = col.collect(weights, log_std.detach().cpu().numpy(), radius, T)
        t_collect = time.perf_counter() - tc
        N = len(segs)
        O, A, LP, R, TERM, TOUT, FINAL = [torch.as_tensor(np.stack([sg[j] for sg in segs], 1), device=dev)
                                          for j in range(7)]                 # (T, N, ...)
        LAST = torch.as_tensor(np.stack([sg[7] for sg in segs], 0), device=dev)   # (N, obs)
        ep_info = [inf for sg in segs for inf in sg[8]]
        for inf in ep_info:
            window.append(inf["success"])
        steps += T * N
        with torch.no_grad():
            VAL = critic(O.reshape(T * N, -1)).reshape(T, N)
            V_last = critic(LAST).squeeze(-1)
            R = R + args.gamma * critic(FINAL.reshape(T * N, -1)).reshape(T, N) * TOUT   # bootstrap timeouts
            V_next = torch.cat([VAL[1:], V_last[None]], 0)
            ended = ((TERM + TOUT) > 0).float()           # episode boundary inside the segment
            adv = torch.zeros(T, N, device=dev); last = torch.zeros(N, device=dev)
            for t in reversed(range(T)):
                delta = R[t] + args.gamma * V_next[t] * (1 - ended[t]) - VAL[t]
                last = delta + args.gamma * args.lam * (1 - ended[t]) * last
                adv[t] = last
            ret = adv + VAL
        upd = ppo_update(args, it, actor, critic, bc_actor, log_std, opt_a, opt_c,
                         O.reshape(T * N, -1), A.reshape(T * N, -1), LP.reshape(-1),
                         adv.reshape(-1), ret.reshape(-1))
        succ = float(np.mean([e["success"] for e in ep_info])) if ep_info else float("nan")
        advanced = False
        if len(window) >= args.curriculum_window:
            rate = float(np.mean(window[-args.curriculum_window:]))
            if rate >= args.curriculum_target and radius < args.radius_max:
                radius = min(args.radius_max, radius + args.radius_step); window = []; advanced = True
        rec = dict(it=it, steps=steps, sps=round(steps / (time.time() - t0)), radius_mm=radius * 1000,
                   episodes=len(ep_info), success=succ, t_collect_s=round(t_collect, 2),
                   log_std=log_std.detach().cpu().numpy().round(3).tolist(), advanced=advanced, **upd)
        log.write(json.dumps(rec) + "\n"); log.flush()
        if it % args.print_every == 0 or advanced:
            print(f"it {it:4d} steps {steps:8d} ({rec['sps']}/s)  radius {radius*1000:4.0f}mm  "
                  f"eps {len(ep_info):3d} success {succ:.2f}  vloss {rec['value_loss']:.4f}  "
                  f"{'WARMUP' if upd['warmup'] else 'kl %.4f clip %.2f' % (upd['approx_kl'], upd['clipfrac'])}"
                  f"  collect {t_collect:.1f}s{'  -> radius up' if advanced else ''}", flush=True)
        if (it + 1) % args.save_every == 0 or it == args.iters - 1:
            torch.save({**{k: ck[k] for k in ("obs_mu", "obs_sd", "act_sd", "obs_dim", "act_dim")},
                        "obs_mask": st["obs_mask"].cpu().numpy(),
                        "state_dict": {k: v.cpu() for k, v in actor.state_dict().items()},
                        "critic": {k: v.cpu() for k, v in critic.state_dict().items()},
                        "log_std": log_std.detach().cpu(), "radius_m": radius, "it": it,
                        "steps": steps, "args": vars(args)}, out / "teacher_rl.pt")
            if args.keep_every and (it + 1) % args.keep_every == 0:
                import shutil
                shutil.copy(out / "teacher_rl.pt", out / f"teacher_rl_it{it+1:05d}.pt")
    col.close(); log.close()
    return 0


# -------------------------------------------------------------------------- eval
def _eval_task(a):
    ckpt, task, radius, episodes, seed, horizon, env_kw = a
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT))
    from screwhead.teacher_env import PrivilegedEnv
    actor, _, st, _ = build(ckpt, "cpu")
    env = PrivilegedEnv(task, radius_m=radius, horizon=horizon, seed=seed, **env_kw)
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
    env_kw = dict(hard_reset=args.hard_reset, servo_iters=args.servo_iters, settle_steps=args.settle_steps)
    jobs = [(args.ckpt, t, args.radius, args.episodes, args.seed + t, args.horizon, env_kw) for t in range(10)]
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
    tr.add_argument("--envs-per-task", type=int, default=1)
    tr.add_argument("--worker-cpus", default="5-9,15-19", help="performance cores on the GB10")
    tr.add_argument("--learner-cpus", default="0-4,10-14", help="efficiency cores")
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
    tr.add_argument("--decentralised", action="store_true",
                    help="workers collect whole segments with a local actor copy")
    tr.add_argument("--keep-every", type=int, default=0, help="also keep numbered checkpoints")
    ev = sub.add_parser("eval")
    ev.add_argument("--ckpt", required=True)
    ev.add_argument("--radius", type=float, default=0.0)
    ev.add_argument("--episodes", type=int, default=10)
    ev.add_argument("--horizon", type=int, default=300)
    ev.add_argument("--procs", type=int, default=10)
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--out", default="")
    ev.add_argument("--hard-reset", action="store_true")
    ev.add_argument("--servo-iters", type=int, default=1)
    ev.add_argument("--settle-steps", type=int, default=5)
    args = ap.parse_args()
    return train(args) if args.cmd == "train" else evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
