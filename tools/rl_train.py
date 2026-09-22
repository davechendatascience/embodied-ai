#!/usr/bin/env python
"""Train the teacher on one LIBERO task with PPO (screwhead/teacher/rl_env.py; the objective is
BRN-rl-teacher-equilibrium-settle, work in progress -- this is its pilot).

  OMP_NUM_THREADS=1 taskset -c 5-9,15-19 rl_train.py --suite libero_goal --task 8 --workers 10 \
      --segment 256 --iters 2000 --out runs/rl/goal8

(taskset keeps the learner and every thread on the performance cores -- half the machine.)

Workers, pinned to the performance cores, each run one environment and a CPU copy of the policy
and collect a whole segment; the learner (GPU) computes values, advantages at gamma = 1 (GAE
lambda) with truncation at the horizon bootstrapped and termination not, updates the policy and
sends it back. The policy is a Gaussian over the normalised twist and a categorical over the
student's gripper snap levels; observations are normalised by running statistics the learner keeps.
Logged per iteration: episodes finished, success, violations by kind, return, steps per second.
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
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
PERF_CORES = (5, 6, 7, 8, 9, 15, 16, 17, 18, 19)
N_LEVELS = 3
THREAD_POOL_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")


class Policy(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.mu = nn.Linear(hidden, 6)
        self.logits = nn.Linear(hidden, N_LEVELS)
        self.log_std = nn.Parameter(torch.full((6,), -0.5))
        self.value = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                   nn.Linear(hidden, 1))

    def dist(self, obs):
        h = self.body(obs)
        return (torch.distributions.Normal(self.mu(h), self.log_std.exp()),
                torch.distributions.Categorical(logits=self.logits(h)))

    def act(self, obs, mode: bool = False):
        twist, grip = self.dist(obs)
        a = twist.mean if mode else twist.sample()
        g = grip.probs.argmax(-1) if mode else grip.sample()
        return a, g, twist.log_prob(a).sum(-1) + grip.log_prob(g)


class RunningNorm:
    def __init__(self, dim: int):
        self.n, self.mean, self.m2 = 1e-4, np.zeros(dim), np.ones(dim)

    def update(self, x: np.ndarray) -> None:
        for row in x:
            self.n += 1
            delta = row - self.mean
            self.mean += delta / self.n
            self.m2 += delta * (row - self.mean)

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.m2 / self.n + 1e-8), -10, 10)


def _worker(remote, suite: str, task: int, seed: int, cpu: int, segment: int, phi_scale: float) -> None:
    os.sched_setaffinity(0, {cpu})
    sys.path.insert(0, str(ROOT))
    torch.set_num_threads(1)
    from screwhead.teacher.rl_env import RLTaskEnv
    env = RLTaskEnv(suite, task, seed=seed, phi_scale=phi_scale)
    rng = np.random.default_rng(seed)
    obs = env.reset(int(rng.integers(len(env.env.init_states))))
    remote.send(("dim", obs.shape[0]))
    policy = None
    ret = 0.0                                          # the running episode's return, across segments
    while True:
        msg = remote.recv()
        if msg[0] == "stop":
            break
        state, norm_mean, norm_scale = msg[1], msg[2], msg[3]
        if policy is None:
            policy = Policy(obs.shape[0])
        policy.load_state_dict(state)
        buf = {k: [] for k in ("obs", "twist", "grip", "logp", "reward", "done", "trunc")}
        reached = {}                                   # row -> the state a truncated row reached
        episodes = []
        for _ in range(segment):
            o = np.clip((obs - norm_mean) / norm_scale, -10, 10).astype(np.float32)
            with torch.no_grad():
                a, g, lp = policy.act(torch.from_numpy(o)[None])
            nxt, r, done, info = env.step(a[0].numpy(), int(g[0]))
            ret += r
            for k, v in (("obs", obs), ("twist", a[0].numpy()), ("grip", int(g[0])), ("logp", float(lp[0])),
                         ("reward", r), ("done", done), ("trunc", info.truncated and not done)):
                buf[k].append(v)
            if info.truncated and not done:
                reached[len(buf["obs"]) - 1] = nxt
            if done or info.truncated:
                episodes.append(dict(ret=ret, steps=env.t, success=info.success, violation=info.violation.split(":")[0]))
                ret = 0.0
                nxt = env.reset(int(rng.integers(len(env.env.init_states))))
            obs = nxt
        remote.send(("seg", {k: np.asarray(v) for k, v in buf.items()}, obs, episodes, reached))
    remote.close()


def _advantages(v: np.ndarray, v_last: float, r: np.ndarray, done: np.ndarray, trunc: np.ndarray,
                v_next_at_trunc: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """GAE at gamma = 1. A terminal step has no successor value; a step truncated at the horizon
    bootstraps from the value of the state it reached (which the next row does not hold, since
    the environment was reset)."""
    n = len(r)
    adv = np.zeros(n)
    nxt, running = v_last, 0.0
    for t in reversed(range(n)):
        if done[t]:
            nxt, running = 0.0, 0.0
        elif trunc[t]:
            nxt, running = v_next_at_trunc[t], 0.0
        delta = r[t] + nxt - v[t]
        running = delta + lam * running
        adv[t] = running
        nxt = v[t]
    return adv, adv + v


def _batch(segs, policy: Policy, norm: RunningNorm, dev, lam: float) -> tuple[dict, list]:
    """Advantages and returns for the workers' segments, as learner tensors; then the running
    normaliser takes in the new observations (after, so each segment is judged by the
    statistics its actions were chosen under)."""
    def value(x):
        with torch.no_grad():
            return policy.value(torch.as_tensor(norm(x), dtype=torch.float32, device=dev)).squeeze(-1).cpu().numpy()
    batch = {k: [] for k in ("obs", "twist", "grip", "logp", "adv", "ret")}
    episodes = []
    for _, seg, last_obs, eps, reached in segs:
        episodes += eps
        v = value(seg["obs"])
        v_reached = np.zeros(len(v))
        if reached:
            rows = sorted(reached)
            v_reached[rows] = value(np.stack([reached[r] for r in rows]))
        adv, ret = _advantages(v, float(value(last_obs[None])[0]), seg["reward"], seg["done"], seg["trunc"],
                               v_reached, lam)
        for k, val in (("obs", norm(seg["obs"])), ("twist", seg["twist"]), ("grip", seg["grip"]),
                       ("logp", seg["logp"]), ("adv", adv), ("ret", ret)):
            batch[k].append(val)
    for _, seg, *_ in segs:
        norm.update(seg["obs"])
    b = {k: torch.as_tensor(np.concatenate(v), dtype=torch.float32, device=dev) for k, v in batch.items()}
    b["grip"] = b["grip"].long()
    b["adv"] = (b["adv"] - b["adv"].mean()) / (b["adv"].std() + 1e-8)
    return b, episodes


def _update(policy: Policy, opt, b: dict, args) -> None:
    """PPO's clipped surrogate, a value loss scaled by the batch's return variance, entropy bonus."""
    n = len(b["obs"])
    ret_var = b["ret"].var() + 1.0
    for _ in range(args.epochs):
        for idx in torch.randperm(n, device=b["obs"].device).split(args.minibatch):
            twist, grip = policy.dist(b["obs"][idx])
            lp = twist.log_prob(b["twist"][idx]).sum(-1) + grip.log_prob(b["grip"][idx])
            ratio = (lp - b["logp"][idx]).exp()
            a = b["adv"][idx]
            pg = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
            vf = (policy.value(b["obs"][idx]).squeeze(-1) - b["ret"][idx]).pow(2).mean()
            ent = twist.entropy().sum(-1).mean() + grip.entropy().mean()
            loss = pg + 0.5 * vf / ret_var - args.entropy * ent
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()


def _summary(it: int, n: int, dt: float, episodes: list, policy: Policy) -> dict:
    kinds: dict[str, int] = {}
    for e in episodes:
        key = "success" if e["success"] else (e["violation"] or "timeout")
        kinds[key] = kinds.get(key, 0) + 1
    return dict(iter=it, steps=n, sps=round(n / dt), episodes=len(episodes), outcomes=kinds,
                ret=round(float(np.mean([e["ret"] for e in episodes])), 2) if episodes else None,
                steps_per_episode=round(float(np.mean([e["steps"] for e in episodes])), 1) if episodes else None,
                success=round(sum(e["success"] for e in episodes) / max(len(episodes), 1), 3),
                std=[round(float(s), 3) for s in policy.log_std.exp().detach().cpu()])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--segment", type=int, default=256)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--entropy", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phi-scale", type=float, default=100.0,
                    help="steps of time cost per unit of potential; cancels out of every return")
    ap.add_argument("--out", default="runs/rl/pilot")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # one thread per worker: numpy, torch and numba start their pools at import, before the
    # worker pins itself, and idle pool threads spinning on its core starve the simulation
    for var in THREAD_POOL_VARS:
        os.environ[var] = "1"
    ctx = mp.get_context("spawn")
    remotes = []
    for w in range(args.workers):
        a, b = ctx.Pipe()
        ctx.Process(target=_worker, args=(b, args.suite, args.task, args.seed * 1000 + w,
                                          PERF_CORES[w % len(PERF_CORES)], args.segment, args.phi_scale),
                    daemon=True).start()
        b.close()
        remotes.append(a)
    dim = [r.recv()[1] for r in remotes][0]
    policy = Policy(dim).to(dev)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    norm = RunningNorm(dim)
    with open(out / "log.jsonl", "a") as log:
        for it in range(args.iters):
            t0 = time.perf_counter()
            state = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
            scale = np.sqrt(norm.m2 / norm.n + 1e-8)
            for r in remotes:
                r.send(("go", state, norm.mean.copy(), scale))
            b, episodes = _batch([r.recv() for r in remotes], policy, norm, dev, args.lam)
            _update(policy, opt, b, args)
            row = _summary(it, len(b["obs"]), time.perf_counter() - t0, episodes, policy)
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
            if it % 20 == 19 or it == args.iters - 1:
                torch.save(dict(policy=policy.state_dict(), norm_mean=norm.mean, norm_m2=norm.m2, norm_n=norm.n,
                                obs_dim=dim, suite=args.suite, task=args.task, iter=it, args=vars(args)),
                           out / "policy.pt")
    for r in remotes:
        r.send(("stop",))
    return 0


if __name__ == "__main__":
    sys.exit(main())
