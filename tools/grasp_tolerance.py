#!/usr/bin/env python
"""How far can the goal object be from where a trajectory expects it?

Replays recorded demonstrations open-loop -- the purest blind policy there is --
after displacing the goal object from its recorded start. Success against
displacement is the basin a mean-reaching trajectory tolerates, which is the
number that says how much placement randomisation makes perception necessary.
Object width does not answer that; this does.

The bowl is moved before the settling steps, so it may slide, drop off a
support, or be pushed by a neighbour. Both the commanded offset and the
displacement that actually survived settling are recorded, and the summary is
binned on the latter.

State-only environments (no renderer), one process per task.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TARGET = "akita_black_bowl_1"      # the goal object in every libero_spatial task (BDDL)


def run_task(ti: int, args) -> list[dict]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
    import h5py
    from screwhead.libero import task_files
    from screwhead.libero_env import register_ur5e, remap_init_state
    register_ur5e()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from rollout import JOINT_ACTION_SCALE, set_joint_gains

    bm = benchmark.get_benchmark_dict()[args.suite]()
    task = bm.get_task(ti)
    files = {f.stem.replace("_demo", ""): f for f in task_files(args.suite)}
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, robots=["Panda"], gripper_types="PandaGripper",
                             controller="JOINT_POSITION", use_camera_obs=False,
                             has_offscreen_renderer=False)
    env.reset()          # robosuite builds (and may rebuild) the sim on reset

    offsets = [(0.0, 0.0)]
    for r in args.radii_mm:
        for k in range(args.directions):
            th = 2 * np.pi * k / args.directions
            offsets.append((r / 1000 * np.cos(th), r / 1000 * np.sin(th)))

    out = []
    with h5py.File(files[task.name], "r") as h:
        for k in list(h["data"].keys())[: args.demos]:
            g = h["data"][k]
            q = np.asarray(g["obs"]["joint_states"][:], np.float64)
            grip = np.asarray(g["actions"][:, 6], np.float64)
            for dx, dy in offsets:
                env.reset()
                sim = env.sim                  # re-fetch: a reset can replace it
                jid = sim.model.joint_name2id(f"{TARGET}_joint0")
                qadr, vadr = int(sim.model.jnt_qposadr[jid]), int(sim.model.jnt_dofadr[jid])
                env.set_init_state(remap_init_state(np.asarray(g["states"][0]), sim))
                p0 = sim.data.qpos[qadr:qadr + 3].copy()
                sim.data.qpos[qadr:qadr + 2] = p0[:2] + (dx, dy)
                sim.data.qvel[vadr:vadr + 6] = 0.0
                sim.forward()
                set_joint_gains(env, args.kp)
                for _ in range(3):
                    env.step(np.zeros(env.env.action_dim))
                p1 = sim.data.qpos[qadr:qadr + 3].copy()
                success = False
                for t in range(len(q) - 1):
                    set_joint_gains(env, args.kp)
                    cur = env.env._get_observations()["robot0_joint_pos"]
                    a = np.zeros(env.env.action_dim)
                    a[:7] = np.clip((q[t + 1][:7] - cur[:7]) / JOINT_ACTION_SCALE, -1, 1)
                    a[-1] = np.clip(grip[t], -1, 1)
                    _, _, done, _ = env.step(a)
                    if done:
                        success = True
                        break
                out.append(dict(task=ti, demo=k, cmd_mm=float(np.hypot(dx, dy) * 1000),
                                dx_mm=dx * 1000, dy_mm=dy * 1000,
                                actual_mm=float(np.linalg.norm(p1[:2] - p0[:2]) * 1000),
                                dz_mm=float((p1[2] - p0[2]) * 1000), success=success))
    env.env.close()
    ok = sum(r["success"] for r in out)
    print(f"  [{ti}] {task.name[:48]:48s} {ok}/{len(out)}", flush=True)
    return out


def _star(a):
    return run_task(*a)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--demos", type=int, default=8)
    ap.add_argument("--radii-mm", type=float, nargs="+", default=[10, 20, 30, 45, 60])
    ap.add_argument("--directions", type=int, default=8)
    ap.add_argument("--kp", type=float, default=4000.0)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--out", default="runs/grasp_tolerance.json")
    args = ap.parse_args()

    mp.set_start_method("spawn")
    with mp.Pool(args.procs) as pool:
        rows = [r for chunk in pool.map(_star, [(ti, args) for ti in range(10)]) for r in chunk]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows))

    # A bowl placed intersecting a neighbour is ejected on settling; that row
    # measures a collision, not a grasp basin, so it is excluded and counted.
    dz = np.array([abs(r["dz_mm"]) for r in rows])
    keep = dz <= 10.0
    print(f"\nexcluded {int((~keep).sum())}/{len(rows)} settles with |dz| > 10 mm")
    by_cmd = {}
    for r, k in zip(rows, keep):
        c = round(r["cmd_mm"]); t = by_cmd.setdefault(c, [0, 0]); t[1] += 1; t[0] += int(not k)
    print("  ejected fraction by commanded offset: " +
          "  ".join(f"{c}mm:{e/n:.2f}" for c, (e, n) in sorted(by_cmd.items())))
    rows = [r for r, k in zip(rows, keep) if k]
    s = np.array([r["success"] for r in rows], float)
    act = np.array([r["actual_mm"] for r in rows])
    cmd = np.array([r["cmd_mm"] for r in rows])
    print("\nby commanded offset:")
    for c in sorted(set(cmd.round(1))):
        m = np.isclose(cmd, c)
        print(f"  {c:5.0f} mm   {s[m].mean():.2f}  (n={int(m.sum())}, "
              f"median actual {np.median(act[m]):5.1f} mm)")
    print("\nby displacement that survived settling:")
    edges = [0, 5, 15, 25, 35, 50, 70, 1e9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (act >= lo) & (act < hi)
        if m.sum():
            p = s[m].mean()
            se = np.sqrt(p * (1 - p) / m.sum())
            print(f"  [{lo:3.0f},{hi if hi < 1e8 else 'inf':>4}) mm   {p:.2f} +/- {se:.2f}  (n={int(m.sum())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
