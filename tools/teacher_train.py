#!/usr/bin/env python
"""The optimization teacher's alternation (BRN-optimization-teacher): search, keep, fit, repeat.

  collect  run episodes whose every control step is solved by the search, recording the state it
           saw and the plan it chose, and keep an episode only if it ends settled with no
           violation, free-motion acceleration p95 <= 6 m/s^2 and no servo re-anchor (DEF-smooth-motion)
  fit      regress pi_theta onto the kept plans -- squared error on the twist, cross-entropy on
           the level -- and write the weights with the feature layout and search settings
  round    collect, fit, and collect again starting from the fitted pi_theta

  PYTHONPATH=third_party/LIBERO:. tools/teacher_train.py collect libero_goal 7 --episodes 20 \\
      --out runs/teacher/goal7_r0 --trials runs/evidence/search_goal7.json
  PYTHONPATH=third_party/LIBERO:. tools/teacher_train.py fit runs/teacher/goal7_r0 \\
      --out checkpoints/teacher/libero_goal_7.pt

An episode costs about 3 minutes at the default settings, so a round of 200 runs in about an hour
and a half across six processes; parallelize by running several collects with different --seed
into the same --out directory, each pinned with taskset to the performance cores.
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

from screwhead.sim.contacts import robot_in_contact  # noqa: E402
from screwhead.sim.episode_record import Episode  # noqa: E402
from screwhead.sim.sim_arm import Execution  # noqa: E402
from screwhead.sim.task_env import StartNoise, TaskEnv  # noqa: E402
from screwhead.teacher import policy as pi  # noqa: E402
from screwhead.teacher.search import LEVELS, Search, Settings  # noqa: E402
from screwhead.teacher.task_loss import TaskLoss  # noqa: E402
from screwhead.teacher.verdicts import Verdicts  # noqa: E402

def _revision() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


SMOOTH_ACC = 6.0        # DEF-smooth-motion (b): the ramped envelope, free motion only
STUDENT_NOISE = StartNoise(xy_m=0.10, z_m=0.05, yaw_deg=30.0, tilt_deg=10.0, null_rad=0.3)


def _executed_track(env, watch) -> tuple[list, list, callable]:
    """Grip-site positions after every physics substep of the EXECUTED trajectory.

    Execution hands the caller a substep hook, and the carried watch is what the episode passes
    to it; the recorder wraps that one. The search's rollouts drive forks of the watch through
    their own hooks and restore afterwards, so nothing they simulate reaches this track -- which
    is the point, since DEF-smooth-motion is a property of what was executed.
    """
    m, d = env.scene.m, env.scene.d
    site, track, contact = m.site_name2id("gripper0_grip_site"), [], []

    def recorded(i=None) -> None:
        watch.substep(i)
        track.append(d.site_xpos[site].copy())
        contact.append(bool(robot_in_contact(m, d)))      # (b) is a bar on free motion

    return track, contact, recorded


def episode(env, settings: Settings, policy, features, init: int | None) -> dict:
    """One episode driven entirely by the search, with the labels it produced."""
    env.reset(init)
    loss = TaskLoss(env)
    v = Verdicts(env, loss)
    search = Search(env, v, loss, settings, policy=policy)
    watch, start_ref = v.watch(), v.reference()
    track, contact, substep = _executed_track(env, watch)
    previous_ref, reanchors = dict(start_ref), 0
    states, twists, levels, executed = [], [], [], []
    outcome, t0 = "timeout", time.perf_counter()

    for _ in range(env.horizon):
        states.append(features.vector(watch, start_ref))
        action, report = search.act(watch, start_ref, previous_ref)
        twists.append(report.best.twist.astype(np.float32))
        levels.append(report.best.level.astype(np.int64))
        executed.append(np.asarray(action, float).copy())
        before = env.servo.reanchors                       # the rollouts moved it too; only this step counts
        env.execute(action, substep=substep)
        reanchors += env.servo.reanchors - before
        success = env.success()
        watch.period_end(success)
        violation = watch.pending() or v.disturbed(start_ref, previous_ref) or v.lost()
        if violation:
            outcome = f"violation: {violation}"
            break
        if success and v.settled(watch):
            outcome = "settled"
            break
        previous_ref = v.reference()

    final = watch.finish()
    dt = float(env.scene.m.opt.timestep)
    acc_p95 = acc_p95_free = float("nan")
    if len(track) > 3:
        acc = np.linalg.norm(np.diff(np.diff(np.array(track), axis=0) / dt, axis=0) / dt, axis=1)
        free = ~np.array(contact[2:], dtype=bool)
        acc_p95 = float(np.percentile(acc, 95))
        acc_p95_free = float(np.percentile(acc[free], 95)) if free.any() else float("nan")
    kept = (outcome == "settled" and not final                 # finish() judges the still-open losses
            and acc_p95_free <= SMOOTH_ACC and reanchors == 0)
    return dict(init=env.init_index, outcome=outcome, steps=len(states), kept=bool(kept),
                acc_p95=round(acc_p95, 3), acc_p95_free=round(acc_p95_free, 3),
                reanchors=int(reanchors), final_watch=final,
                seconds=round(time.perf_counter() - t0, 1),
                state=np.asarray(states), twist=np.asarray(twists), level=np.asarray(levels),
                executed=np.asarray(executed))


def collect(args) -> int:
    settings = Settings(samples=args.samples, horizon=args.horizon, segments=args.segments,
                        bound_segments=args.bound)
    env = TaskEnv(args.suite, args.task, seed=args.seed, render=False,
                  start=STUDENT_NOISE if args.start_noise else StartNoise(),
                  execution=Execution(lean=True, anchor=True, scale_lead=True))
    loss = TaskLoss(env)
    features = pi.Features(env, Verdicts(env, loss))
    policy = None
    if args.policy:
        policy, blob = pi.load(args.policy, env, Verdicts(env, loss))
        settings = Settings(**blob["settings"])
        print(json.dumps({"policy": args.policy, "rounds": blob["rounds"], "kept": blob["kept_episodes"]}))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for e in range(args.episodes):
        r = episode(env, settings, policy, features, None if args.init < 0 else args.init)
        name = f"{args.suite}_{args.task}_s{args.seed}_e{e}"
        np.savez_compressed(out / f"{name}.npz", state=r["state"], twist=r["twist"], level=r["level"],
                            kept=r["kept"], outcome=r["outcome"], init=r["init"],
                            suite=args.suite, task=args.task, seed=args.seed,
                            settings=json.dumps(vars(settings)), state_dim=features.dim)
        summary = {k: r[k] for k in ("init", "outcome", "steps", "kept", "acc_p95",
                                     "acc_p95_free", "reanchors", "seconds")}
        # the episode as the few numbers that play it again: no states, no frames
        Episode.of(env, r["executed"], suite=args.suite, task=args.task, init=r["init"],
                   seed=args.seed, noise=STUDENT_NOISE if args.start_noise else StartNoise(),
                   outcome={k: summary[k] for k in ("outcome", "steps", "kept", "acc_p95_free",
                                                    "reanchors")} | {"settled": r["outcome"] == "settled"},
                   provenance={"tool": "tools/teacher_train.py collect", "revision": _revision(),
                               "settings": vars(settings), "policy": args.policy or "search only"},
                   ).save(out / f"{name}.episode.json")
        rows.append(summary)
        print(json.dumps({"episode": e, **summary}), flush=True)

    if args.trials:
        p = Path(args.trials)
        prior = json.loads(p.read_text()) if p.exists() else []
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(prior + [{
            "metrics": {"settled": r["outcome"] == "settled", "steps": r["steps"]},
            "conditions": {"task_suite": args.suite, "task": args.task, "init": r["init"],
                           "start_noise": bool(args.start_noise)},
        } for r in rows], indent=1))
    kept = sum(r["kept"] for r in rows)
    print(json.dumps({"episodes": len(rows), "settled": sum(r["outcome"] == "settled" for r in rows),
                      "kept": kept, "out": str(out)}))
    return 0


def _dataset(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], dict]:
    states, twists, levels, episode_id, meta = [], [], [], [], {}
    for i, p in enumerate(sorted(paths)):
        z = np.load(p, allow_pickle=False)
        if not bool(z["kept"]):
            continue
        states.append(z["state"]); twists.append(z["twist"]); levels.append(z["level"])
        episode_id += [i] * len(z["state"])
        meta = {"settings": str(z["settings"]), "state_dim": int(z["state_dim"]),
                "suite": str(z["suite"]), "task": int(z["task"])}
    if not states:
        raise SystemExit("no kept episode in the data: nothing passed the keep-filter")
    return (np.concatenate(states), np.concatenate(twists), np.concatenate(levels),
            episode_id, meta)


def fit(args) -> int:
    import torch
    from torch import nn

    paths = sorted(Path(args.data).glob("*.npz"))
    x, y_twist, y_level, episode_id, meta = _dataset(paths)
    settings = Settings(**json.loads(meta["settings"].replace("'", '"')))
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    ep = np.asarray(episode_id)
    val = (ep % 10) == 0                                   # every tenth episode validates
    xt = torch.tensor(x, dtype=torch.float32, device=dev)
    tw = torch.tensor(y_twist, dtype=torch.float32, device=dev)
    lv = torch.tensor(y_level, dtype=torch.long, device=dev)
    mu, sd = xt[~val].mean(0), xt[~val].std(0).clamp_min(1e-6)
    xt = (xt - mu) / sd

    net = pi.TeacherNet(x.shape[1], settings.segments, len(LEVELS)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    best, best_state = float("inf"), None
    idx = np.flatnonzero(~val)
    for epoch in range(args.epochs):
        net.train()
        order = np.random.default_rng(epoch).permutation(idx)
        for k in range(0, len(order), args.batch):
            b = torch.tensor(order[k:k + args.batch], device=dev)
            twist, logits = net(xt[b])
            loss = (nn.functional.mse_loss(twist, tw[b])
                    + nn.functional.cross_entropy(logits.reshape(-1, len(LEVELS)), lv[b].reshape(-1)))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vb = torch.tensor(np.flatnonzero(val), device=dev)
            twist, logits = net(xt[vb])
            v_loss = float(nn.functional.mse_loss(twist, tw[vb])
                           + nn.functional.cross_entropy(logits.reshape(-1, len(LEVELS)), lv[vb].reshape(-1)))
            acc = float((logits.argmax(-1) == lv[vb]).float().mean())
        if v_loss < best:
            best, best_state = v_loss, {k: v.detach().clone() for k, v in net.state_dict().items()}
        print(json.dumps({"epoch": epoch, "val_loss": round(v_loss, 5), "level_acc": round(acc, 4)}), flush=True)

    net.load_state_dict(best_state)
    # the normalization belongs with the weights: the features are standardized the same way at
    # labelling time, so it is folded into the first layer rather than carried separately
    with torch.no_grad():
        first = net.trunk[0]
        first.bias.copy_(first.bias - first.weight @ (mu / sd))
        first.weight.copy_(first.weight / sd)

    env = TaskEnv(meta["suite"], meta["task"], seed=0, render=False,
                  execution=Execution(lean=True, anchor=True, scale_lead=True))
    features = pi.Features(env, Verdicts(env, TaskLoss(env)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pi.save(args.out, net.cpu(), features, settings, meta["suite"], meta["task"],
            rounds=args.round, kept=len(set(episode_id)))
    print(json.dumps({"checkpoint": args.out, "val_loss": round(best, 5),
                      "frames": int(len(x)), "episodes": len(set(episode_id))}))
    return 0


def rounds(args) -> int:
    """The alternation itself: collect, fit, and collect again starting from pi_theta.

    Each round's data lives in its own directory and each checkpoint records the round it came
    from, so a later round is never fitted on labels a different pi_theta produced without saying
    so. The search settings are the first round's throughout: they are part of the label.
    """
    base, ckpt = Path(args.out), ""
    for r in range(args.rounds):
        data = base / f"round{r}"
        collect_args = argparse.Namespace(
            suite=args.suite, task=args.task, episodes=args.episodes, init=-1, seed=args.seed + r,
            samples=args.samples, horizon=args.horizon, segments=args.segments,
            start_noise=args.start_noise, policy=ckpt, out=str(data), trials=args.trials)
        collect(collect_args)
        ckpt = str(base / f"pi_theta_r{r}.pt")
        fit(argparse.Namespace(data=str(data), out=ckpt, epochs=args.epochs, batch=256, lr=3e-4,
                               seed=args.seed, round=r, cpu=args.cpu))
        print(json.dumps({"round": r, "data": str(data), "checkpoint": ckpt}), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect")
    c.add_argument("suite"); c.add_argument("task", type=int)
    c.add_argument("--episodes", type=int, default=20)
    c.add_argument("--init", type=int, default=-1, help="-1 draws a LIBERO init state per episode")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--samples", type=int, default=Settings.samples)
    c.add_argument("--horizon", type=int, default=Settings.horizon)
    c.add_argument("--segments", type=int, default=Settings.segments)
    c.add_argument("--start-noise", action="store_true", default=True)
    c.add_argument("--bound", action="store_true", default=True,
                   help="hold each segment's twist within what its periods can absorb")
    c.add_argument("--no-bound", dest="bound", action="store_false")
    c.add_argument("--no-start-noise", dest="start_noise", action="store_false")
    c.add_argument("--policy", default="", help="checkpoint the search starts from")
    c.add_argument("--out", required=True)
    c.add_argument("--trials", default="", help="append CTR-search-settles trial rows here")
    c.set_defaults(func=collect)

    f = sub.add_parser("fit")
    f.add_argument("data"); f.add_argument("--out", required=True)
    f.add_argument("--epochs", type=int, default=60)
    f.add_argument("--batch", type=int, default=256)
    f.add_argument("--lr", type=float, default=3e-4)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--round", type=int, default=0)
    f.add_argument("--cpu", action="store_true")
    f.set_defaults(func=fit)

    r = sub.add_parser("round")
    r.add_argument("suite"); r.add_argument("task", type=int)
    r.add_argument("--rounds", type=int, default=2)
    r.add_argument("--episodes", type=int, default=20)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--samples", type=int, default=Settings.samples)
    r.add_argument("--horizon", type=int, default=Settings.horizon)
    r.add_argument("--segments", type=int, default=Settings.segments)
    r.add_argument("--start-noise", action="store_true", default=True)
    r.add_argument("--epochs", type=int, default=60)
    r.add_argument("--cpu", action="store_true")
    r.add_argument("--out", required=True)
    r.add_argument("--trials", default="")
    r.set_defaults(func=rounds)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
