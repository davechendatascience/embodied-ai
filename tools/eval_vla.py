#!/usr/bin/env python
"""Evaluate the Panda VLA by LIBERO's protocol: each task's initial states once, in order, up to the suite's
step limit, success by LIBERO's own predicate -- through the VLA's decode (BRN-vla-decodes-twists-exactly) and
its one interface (BRN-vla-sees-and-acts-as-trained).

  eval_vla.py --ckpt checkpoints/vla_spatial_s0.pt --suite libero_spatial [--tasks 0 1] [--episodes 50]
              --trials $OUT [--cpus 5,6,7,8] [--seed 555]

Each worker holds its own copy of the model on the GPU and runs its share of the episodes on one core. Every
episode is executed twice, in two rounds of fresh worker processes, and its trial records whether the two agree
in every observation, every action and the outcome (BRN-vla-reported-beside-a-blind-twin): a rate is evidence
only if all did. From a randomized set (--starts), a trial records whether its episode was placed: its task admitted
and its model matching the set's (BRN-random-starts-test-set); an episode not placed is reported, not scored. It also
records the set file's digest and the digest of the integration state at placement, which begins the episode's
digest; two models are compared on a start only where both agree. A
trial also records success, the
steps taken, the model's queries and their latency, clipped steps, the servo's counted decode events and the
peak GPU memory of its worker.
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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# LIBERO's step limits per suite, as its published evaluations use them (the demonstrations' longest plus margin)
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}



def _pin(cpus) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.sched_setaffinity(0, {cpus.get()})


@dataclass(frozen=True)
class Policy:
    """The loaded model and everything its inputs and outputs are made with."""
    model: object
    proc: object
    cfg: object
    chain: object
    stats: dict


def act(pol: Policy, env) -> tuple[np.ndarray, float, np.ndarray]:
    """One query: the chunk of actions in the environment's units, the query's latency in seconds, and which
    actions had a twist component outside the action range (clipped; BRN-vla-sees-and-acts-as-trained)."""
    import torch

    from screwhead.student.libero_data import proprio, upright
    from screwhead.student.qwen_vla import collate, prompt_ids
    raw, stats = env.raw, pol.stats
    p = (proprio(pol.chain, raw["robot0_joint_pos"], raw["robot0_gripper_qpos"]) - stats["pm"]) / stats["ps"]
    s = prompt_ids(pol.proc, env.language, [upright(raw["agentview_image"]), upright(raw["robot0_eye_in_hand_image"])],
                   pol.cfg)
    s["proprio"] = torch.from_numpy(p.astype(np.float32))
    batch = {k: v.cuda() for k, v in collate([s], pol.proc.tokenizer.pad_token_id).items()}
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad():
        tw, gl = pol.model(batch)
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    twist = tw[0].float().cpu().numpy() * stats["ts"] + stats["tm"]          # back to the label's units
    grip = np.where(gl[0].float().cpu().numpy() > 0, 1.0, -1.0)
    return np.concatenate([np.clip(twist, -1.0, 1.0), grip[:, None]], 1), dt, (np.abs(twist) > 1.0).any(1)


def _episode(pol: Policy, env, suite: str, e: int, start=None) -> dict:
    """One episode from LIBERO's initial state e, or from a stored start (state, fixtures, fingerprint); the digest of
    every observation the model received and every action executed identifies what it saw and did."""
    if start is None:
        env.reset(init_index=e)
    elif not env.place_stored(*start):
        return dict(placed=False, steps=0, queries=0, digest="model differs")      # reported, not scored
    sv = env.servo
    ev0 = (sv.acc_limited, sv.scaled, sv.iter_cap, sv.reanchors)
    success, lat, queries, clipped = False, [], 0, 0
    digest = hashlib.sha1((env.placed_digest if start is not None else "").encode())
    while env.t < MAX_STEPS[suite] and not success:
        raw = env.raw                   # the observation the model receives, in the digest with the actions it chose
        for key in ("agentview_image", "robot0_eye_in_hand_image", "robot0_joint_pos", "robot0_gripper_qpos"):
            digest.update(np.ascontiguousarray(raw[key]).tobytes())
        chunk, dt, n_clip = act(pol, env)
        lat.append(dt)
        queries += 1
        for i, a in enumerate(chunk[:pol.cfg.execute]):
            digest.update(np.ascontiguousarray(a, np.float64).tobytes())
            clipped += int(n_clip[i])
            _r, _x, _d, info = env.step(a)
            if info["success"]:
                success = True
                break
            if env.t >= MAX_STEPS[suite]:
                break
    ev = [n - n0 for n, n0 in zip((sv.acc_limited, sv.scaled, sv.iter_cap, sv.reanchors), ev0, strict=True)]
    return dict(**({} if start is None else {"placed": True, "placed_digest": env.placed_digest}), success=success,
                steps=env.t, queries=queries,
                digest=digest.hexdigest(),
                latency_ms_median=round(1000 * float(np.median(lat)), 1), latency_ms_max=round(1000 * float(np.max(lat)), 1),
                clipped_steps=clipped, acc_limited_steps=ev[0], scaled_steps=ev[1], iter_cap_steps=ev[2],
                reanchored_steps=ev[3])


def _task(job: tuple) -> list[dict]:
    ckpt, suite, task, episodes, seed, rev, starts_file, arguments = job
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    from screwhead.sim.sim_arm import Execution
    from screwhead.sim.task_env import TaskEnv
    from screwhead.student.libero_data import panda_chain
    from screwhead.student.qwen_vla import VLA_EXECUTION, environment_record, file_digest, load
    model, proc = load(ckpt)
    trained_in = torch.load(ckpt, map_location="cpu", weights_only=False).get("trained_in", {})
    record = Path(ckpt + ".record.json")
    trained = json.loads(record.read_text()) if record.exists() else {}
    ended_with = trained.get("checkpoint_digest")
    # the training run this checkpoint ended: its per-run inputs, from that run's own record
    training_run = json.dumps({k: trained.get("configuration", {}).get(k) for k in ("seed", "blind")}, sort_keys=True)
    cfg = model.cfg
    stats = dict(tm=np.array(cfg.twist_mean), ts=np.array(cfg.twist_std),
                 pm=np.array(cfg.proprio_mean, np.float32), ps=np.array(cfg.proprio_std, np.float32))
    pol = Policy(model, proc, cfg, panda_chain(), stats)
    env = TaskEnv(suite, task, horizon=MAX_STEPS[suite], seed=seed * 100 + task, render=cfg.image_px,
                  execution=Execution(**VLA_EXECUTION))
    rows = []
    stored = None
    if starts_file:
        # BRN-random-starts-test-set: admitted only if the task file, the simulator and its stepping match the set's
        import mujoco

        from libero.libero import benchmark, get_libero_path
        from screwhead.sim.task_env_place import load_starts, stepping_digests
        stored = load_starts(starts_file).get(task)
        spec = benchmark.get_benchmark_dict()[suite]().get_task(task)
        ref = (stored or {}).get("references") or {}
        admitted = bool(ref) and ref["task_file_digest"] == file_digest(os.path.join(
            get_libero_path("bddl_files"), spec.problem_folder, spec.bddl_file)) and ref["simulator"] == mujoco.__version__ \
            and ref["stepping"] == stepping_digests(env, stored["states"][0], stored["fixtures"][0])
        if not admitted:
            env.close()
            return [{"metrics": {"admitted": False, "placed": False},
                     "conditions": {"task_suite": suite, "task": task, "init_index": e, "blind": cfg.blind},
                     "repro": {"task_suite": suite, "task": task, "init_index": e, "starts": starts_file}} for e in episodes]
        episodes = [e for e in episodes if e < len(stored["states"])]
    for e in episodes:
        start = None if stored is None else (stored["states"][e], stored["fixtures"][e], stored["references"]["model_fingerprint"])
        m = _episode(pol, env, suite, e, start)
        m.update(gpu_peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        placed = {} if start is None else {"placed_digest": m.pop("placed_digest", None),
                                          "starts_digest": file_digest(starts_file)}
        rows.append({"metrics": m,
                     "conditions": {"task_suite": suite, "task": task, "init_index": e, "blind": cfg.blind},
                     "repro": {"task_suite": suite, "task": task, "init_index": e, "seed": seed * 100 + task,
                               "horizon": MAX_STEPS[suite], "policy_revision": rev, "starts": starts_file,
                               "execution": json.dumps(VLA_EXECUTION, default=list), "blind": cfg.blind,
                               "trained_in": json.dumps(trained_in, sort_keys=True),
                               "configuration": json.dumps({"chunk": cfg.chunk, "execute": cfg.execute,
                                                            "image_px": cfg.image_px, "precision": "bfloat16",
                                                            "gripper": "positive score closes"}, sort_keys=True),
                               "checkpoint_digest": file_digest(ckpt),
                               "checkpoint_is_training_end": ended_with == file_digest(ckpt),
                               "training_run": training_run, "arguments": arguments, **placed}})
    from libero.libero import benchmark, get_libero_path
    spec = benchmark.get_benchmark_dict()[suite]().get_task(task)
    read = [os.path.join(get_libero_path("bddl_files"), spec.problem_folder, spec.bddl_file),
            os.path.join(get_libero_path("init_states"), spec.problem_folder, spec.init_states_file),
            *([starts_file] if starts_file else [])]
    # at the end: every source the episodes imported is in it
    evaluated_in = environment_record(read=read, models={f"{suite}:{task}": env.model_fingerprint()})
    for r in rows:
        r["repro"]["evaluated_in"] = json.dumps(evaluated_in, sort_keys=True)
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
    ap.add_argument("--starts", default=None, help="a randomized set's file (tools/random_starts.py) instead of LIBERO's starts")
    args = ap.parse_args()
    tasks = args.tasks if args.tasks is not None else list(range(90 if args.suite == "libero_90" else 10))
    rev = "vla:" + hashlib.sha1(Path(args.ckpt).read_bytes()).hexdigest()[:10]
    cpus = [int(c) for c in args.cpus.split(",")]
    # every launch argument as a value, but the checkpoint loaded, the output's location and the cores used
    arguments = json.dumps({k: v for k, v in vars(args).items() if k not in ("ckpt", "trials", "cpus")}, sort_keys=True)
    free = multiprocessing.Manager().Queue()
    for c in cpus:
        free.put(c)
    # a task's episodes split across workers so each worker's share is about the same length
    jobs, per = [], max(1, (len(tasks) * args.episodes) // len(cpus) // max(1, len(tasks)) or 1)
    for t in tasks:
        eps = list(range(args.episodes))
        for i in range(0, len(eps), max(per, 5)):
            jobs.append((args.ckpt, args.suite, t, eps[i:i + max(per, 5)], args.seed, rev, args.starts, arguments))
    t0 = time.time()
    # BRN-vla-reported-beside-a-blind-twin: every episode executed twice, in two rounds of fresh worker processes
    # started at different times; the rounds are compared episode by episode in the digest of every observation the
    # model received and every action it chose, and in the outcome
    rounds = []
    for _ in range(2):
        with ProcessPoolExecutor(len(cpus), initializer=_pin, initargs=(free,),
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            rounds.append([r for rows in pool.map(_task, jobs) for r in rows])
        for c in cpus:
            free.put(c)
    trials = reproduce(rounds[0], rounds[1])
    Path(args.trials).parent.mkdir(parents=True, exist_ok=True)
    Path(args.trials).write_text(json.dumps({"trials": trials}, indent=1))
    _report(args.suite, rev, tasks, trials, time.time() - t0)
    return 0


def reproduce(first: list[dict], second: list[dict]) -> list[dict]:
    """The first round's trials, each executed episode marked reproduced if the second round's execution of it agrees
    in the digest of what the model received and chose and in the outcome. An episode of a task that was not admitted
    was never executed and has nothing to compare."""
    other = {(r["repro"]["task"], r["repro"]["init_index"]): r["metrics"] for r in second}
    for r in first:
        m, o = r["metrics"], other.get((r["repro"]["task"], r["repro"]["init_index"]), {})
        if "digest" in m:
            m["reproduced"] = m.pop("digest") == o.get("digest") and m.get("success") == o.get("success")
    return first


def _report(suite: str, rev: str, tasks: list[int], trials: list[dict], seconds: float) -> None:
    """The rate over scored episodes; episodes not scored (task not admitted, model differs) and episodes that did not
    reproduce are reported, the latter as a failed measurement."""
    scored = [r for r in trials if "success" in r["metrics"]]
    ok = sum(r["metrics"]["success"] for r in scored)
    unrepro = sum(not r["metrics"].get("reproduced", False) for r in trials)
    print(f"{suite} {rev}: {ok}/{len(scored)} ({100 * ok / max(1, len(scored)):.1f}%) in {seconds / 60:.1f} min"
          + (f"; {len(trials) - len(scored)} episodes not scored (task not admitted or model differs)"
             if len(scored) < len(trials) else "")
          + (f" -- MEASUREMENT FAILED: {unrepro} episodes not executed or not reproduced, the rate is not evidence"
             if unrepro else "; every episode reproduced"))
    for t in tasks:
        rs = [r for r in scored if r["conditions"]["task"] == t]
        print(f"  task {t}: {sum(r['metrics']['success'] for r in rs)}/{len(rs)}")
    if scored:
        print(f"  query latency median {np.median([r['metrics']['latency_ms_median'] for r in scored]):.1f} ms; "
              f"GPU peak per worker {max(r['metrics']['gpu_peak_gib'] for r in scored):.2f} GiB")


if __name__ == "__main__":
    sys.exit(main())
