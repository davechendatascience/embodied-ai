#!/usr/bin/env python
"""Validate the waypoint progress function on recorded demonstrations.

The demonstrations are used here only as a test of the reward, never for
training. If progress is a faithful measure of how far a state is from task
success, then along a successful demonstration it should climb through the
stages in order, reach 6 at the end, and rarely fall. Where it does fall, the
stage it falls in names the predicate or waypoint that disagrees with how the
task is physically done.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--demos", type=int, default=5)
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    import h5py
    import libero.libero.benchmark as B
    from screwhead.libero import task_files
    from screwhead.libero_env import remap_init_state
    from screwhead.teacher_env import PrivilegedEnv

    files = {f.stem.replace("_demo", ""): f for f in task_files("libero_spatial")}
    bm = B.get_benchmark_dict()["libero_spatial"]()
    allrows = []
    for ti in args.tasks:
        env = PrivilegedEnv(ti)
        with h5py.File(files[bm.get_task(ti).name], "r") as h:
            for k in list(h["data"].keys())[: args.demos]:
                states = np.asarray(h["data"][k]["states"])
                phis, stages = [], []
                for t, st in enumerate(states):
                    sim = env.env.sim
                    sim.set_state_from_flattened(remap_init_state(st, sim)); sim.forward()
                    env.raw = env.env.env._get_observations(force_update=True)
                    snap = env.snapshot()
                    if t == 0:
                        env.set_reference(snap)
                    phi, stage, _ = env.progress(snap)
                    phis.append(phi); stages.append(stage)
                    if stage == 6:        # the environment ends the episode at first success
                        break
                phis, stages = np.array(phis), np.array(stages)
                seq = [int(stages[0])] + [int(b) for a, b in zip(stages[:-1], stages[1:]) if b != a]
                dphi = np.diff(phis)
                drops = dphi < -0.05
                worst = int(np.argmin(dphi)) if len(dphi) else 0
                row = dict(task=ti, demo=k, T=len(phis), final=float(phis[-1]), max=float(phis.max()),
                           seq=seq, n_drops=int(drops.sum()), worst=float(dphi.min()) if len(dphi) else 0.0,
                           worst_stage=f"{stages[worst]}->{stages[worst+1]}" if len(dphi) else "",
                           reached={s: int(np.argmax(stages >= s)) if (stages >= s).any() else -1 for s in range(7)})
                allrows.append(row)
                print(f"  task {ti} {k:8s} T={row['T']:3d} final {row['final']:.2f} max {row['max']:.2f} "
                      f"drops>{0.05}: {row['n_drops']:2d} worst {row['worst']:+.2f} ({row['worst_stage']})  "
                      f"stages {''.join(map(str, seq))}", flush=True)
        env.close()
    fin = np.array([r["final"] for r in allrows])
    print(f"\n{len(allrows)} demos: final phi = 6 in {np.mean(fin >= 5.999):.2f}; median final {np.median(fin):.2f}; "
          f"demos with any drop > 0.05: {np.mean([r['n_drops'] > 0 for r in allrows]):.2f}")
    import collections
    c = collections.Counter(r["worst_stage"] for r in allrows if r["n_drops"] > 0)
    print("where the worst drop happens:", dict(c))
    worst = np.array([r["worst"] for r in allrows])
    print(f"worst single-step drop per demo: median {np.median(worst):+.2f}; demos with a drop below -0.5: "
          f"{np.mean(worst < -0.5):.2f}; below -0.25: {np.mean(worst < -0.25):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
