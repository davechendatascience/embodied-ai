#!/usr/bin/env python
"""Evaluate the Panda VLA by LIBERO's protocol: each task's initial states once, in order, up to the suite's
step limit, success by LIBERO's own predicate -- through the VLA's decode (BRN-vla-decodes-twists-exactly) and
its one interface (BRN-vla-sees-and-acts-as-trained).

  eval_vla.py --ckpt checkpoints/vla_spatial_s0.pt --suite libero_spatial [--tasks 0 1] [--episodes 50]
              --trials $OUT [--cpus 5,6,7,8] [--seed 555]

Each worker holds its own copy of the model on the GPU and runs its share of the episodes on one core. A
trial records success, the steps taken, the model's queries and their latency, the servo's counted decode
events and the peak GPU memory of its worker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# LIBERO's step limits per suite, as its published evaluations use them (the demonstrations' longest plus margin)
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}



def _pin(cpus) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.sched_setaffinity(0, {cpus.get()})


def act(model, proc, cfg, chain, env, stats) -> tuple[np.ndarray, float]:
    """One query: the chunk of actions in the environment's units, and the query's latency in seconds."""
    import torch

    from screwhead.student.libero_data import proprio, upright
    from screwhead.student.qwen_vla import collate, prompt_ids
    raw = env.raw
    p = (proprio(chain, raw["robot0_joint_pos"], raw["robot0_gripper_qpos"]) - stats["pm"]) / stats["ps"]
    s = prompt_ids(proc, env.language, [upright(raw["agentview_image"]), upright(raw["robot0_eye_in_hand_image"])], cfg)
    s["proprio"] = torch.from_numpy(p.astype(np.float32))
    batch = {k: v.cuda() for k, v in collate([s], proc.tokenizer.pad_token_id).items()}
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad():
        tw, gl = model(batch)
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    twist = tw[0].float().cpu().numpy() * stats["ts"] + stats["tm"]          # back to the label's units
    grip = np.where(gl[0].float().cpu().numpy() > 0, 1.0, -1.0)
    return np.concatenate([np.clip(twist, -1.0, 1.0), grip[:, None]], 1), dt


def _task(job: tuple) -> list[dict]:
    ckpt, suite, task, episodes, seed, rev = job
    import torch
    torch.set_num_threads(1)
    from screwhead.sim.sim_arm import Execution
    from screwhead.sim.task_env import TaskEnv
    from screwhead.student.libero_data import panda_chain
    from screwhead.student.qwen_vla import VLA_EXECUTION, load
    model, proc = load(ckpt)
    cfg = model.cfg
    stats = dict(tm=np.array(cfg.twist_mean), ts=np.array(cfg.twist_std),
                 pm=np.array(cfg.proprio_mean, np.float32), ps=np.array(cfg.proprio_std, np.float32))
    chain = panda_chain()
    env = TaskEnv(suite, task, horizon=MAX_STEPS[suite], seed=seed * 100 + task, render=cfg.image_px,
                  execution=Execution(**VLA_EXECUTION))
    rows = []
    for e in episodes:
        env.reset(init_index=e)
        sv = env.servo
        ev0 = (sv.acc_limited, sv.scaled, sv.iter_cap, sv.reanchors)
        success, lat, queries = False, [], 0
        while env.t < MAX_STEPS[suite] and not success:
            chunk, dt = act(model, proc, cfg, chain, env, stats)
            lat.append(dt); queries += 1
            for a in chunk[:cfg.execute]:
                _r, _x, _d, info = env.step(a)
                if info["success"]:
                    success = True
                    break
                if env.t >= MAX_STEPS[suite]:
                    break
        ev = [n - n0 for n, n0 in zip((sv.acc_limited, sv.scaled, sv.iter_cap, sv.reanchors), ev0, strict=True)]
        rows.append({"metrics": {"success": success, "steps": env.t, "queries": queries,
                                 "latency_ms_median": round(1000 * float(np.median(lat)), 1),
                                 "latency_ms_max": round(1000 * float(np.max(lat)), 1),
                                 "acc_limited_steps": ev[0], "scaled_steps": ev[1], "iter_cap_steps": ev[2],
                                 "reanchored_steps": ev[3],
                                 "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)},
                     "conditions": {"task_suite": suite, "task": task, "init_index": e, "blind": cfg.blind},
                     "repro": {"task_suite": suite, "task": task, "init_index": e, "seed": seed * 100 + task,
                               "horizon": MAX_STEPS[suite], "policy_revision": rev,
                               "execution": json.dumps(VLA_EXECUTION, default=list), "blind": cfg.blind}})
    env.close()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", type=int, nargs="*", default=None)
    ap.add_argument("--episodes", type=int, default=50, help="initial states 0..N-1 of each task, in order")
    ap.add_argument("--seed", type=int, default=555)
    ap.add_argument("--cpus", default="5,6,7,8")
    ap.add_argument("--trials", required=True)
    args = ap.parse_args()
    tasks = args.tasks if args.tasks is not None else list(range(90 if args.suite == "libero_90" else 10))
    rev = "vla:" + hashlib.sha1(Path(args.ckpt).read_bytes()).hexdigest()[:10]
    cpus = [int(c) for c in args.cpus.split(",")]
    free = multiprocessing.Manager().Queue()
    for c in cpus:
        free.put(c)
    # a task's episodes split across workers so each worker's share is about the same length
    jobs, per = [], max(1, (len(tasks) * args.episodes) // len(cpus) // max(1, len(tasks)) or 1)
    for t in tasks:
        eps = list(range(args.episodes))
        for i in range(0, len(eps), max(per, 5)):
            jobs.append((args.ckpt, args.suite, t, eps[i:i + max(per, 5)], args.seed, rev))
    t0 = time.time()
    with ProcessPoolExecutor(len(cpus), initializer=_pin, initargs=(free,),
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        trials = [r for rows in pool.map(_task, jobs) for r in rows]
    Path(args.trials).parent.mkdir(parents=True, exist_ok=True)
    Path(args.trials).write_text(json.dumps({"trials": trials}, indent=1))
    ok = sum(r["metrics"]["success"] for r in trials)
    print(f"{args.suite} {rev}: {ok}/{len(trials)} ({100 * ok / len(trials):.1f}%) in {(time.time() - t0) / 60:.1f} min")
    for t in tasks:
        rs = [r for r in trials if r["conditions"]["task"] == t]
        print(f"  task {t}: {sum(r['metrics']['success'] for r in rs)}/{len(rs)}")
    lat = [r["metrics"]["latency_ms_median"] for r in trials]
    print(f"  query latency median {np.median(lat):.1f} ms; GPU peak per worker {max(r['metrics']['gpu_peak_gib'] for r in trials):.2f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
