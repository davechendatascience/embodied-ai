#!/usr/bin/env python
"""Privileged behaviour-cloning data for the teacher.

Pairs each recorded demonstration frame with the teacher's observation at that
simulator state and the action the demonstrator took, in the student's action
units: body twist divided by the action-space scale, plus the gripper command.

Frames near a kinematic singularity are dropped with the same `usable` filter
the student's caches use, since a twist there is not decodable into joints.

This initialises the teacher; it does not make it look. The demos carry only
12 mm of placement variance, so a teacher cloned from them inherits the same
shortcut the student has. Fine-tuning under randomised placement is what makes
the object pose matter.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def run_task(ti: int, out: str, demos: int) -> str:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(ROOT))
    import h5py
    import torch
    from screwhead.interface import ActionSpec
    from screwhead.libero import panda_chain, task_files
    from screwhead.retarget import to_twists, usable
    from screwhead.teacher_env import PrivilegedEnv

    env = PrivilegedEnv(ti, radius_m=0.0)
    spec = ActionSpec()
    sc = np.array([spec.rot_scale * spec.control_hz] * 3 + [spec.pos_scale * spec.control_hz] * 3)
    chain = panda_chain()
    files = {f.stem.replace("_demo", ""): f for f in task_files("libero_spatial")}
    import libero.libero.benchmark as B
    name = B.get_benchmark_dict()["libero_spatial"]().get_task(ti).name
    O, A, D, S = [], [], [], []
    with h5py.File(files[name], "r") as h:
        for di, k in enumerate(list(h["data"].keys())[:demos]):
            g = h["data"][k]
            q = torch.tensor(g["obs"]["joint_states"][:], dtype=torch.float64)
            if len(q) < 3:
                continue
            V = to_twists(chain, q, spec).numpy()
            keep = torch.nonzero(usable(chain, q[:-1])).flatten().numpy()
            states = np.asarray(g["states"][:])
            grip = np.asarray(g["actions"][:, 6], np.float64)
            for t in keep:
                O.append(env.obs_at(states[t]))
                A.append(np.concatenate([V[t] / sc, [grip[t]]]).astype(np.float32))
                D.append(di); S.append(t)
    env.close()
    path = Path(out) / f"task{ti:02d}.npz"
    np.savez(path, obs=np.stack(O), act=np.stack(A), demo=np.array(D, np.int16),
             step=np.array(S, np.int32))
    print(f"  [{ti}] {len(O)} frames", flush=True)
    return str(path)


def _star(a):
    return run_task(*a)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cache/teacher_bc")
    ap.add_argument("--demos", type=int, default=50)
    ap.add_argument("--procs", type=int, default=10)
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    mp.set_start_method("spawn")
    with mp.Pool(args.procs) as pool:
        pool.map(_star, [(ti, args.out, args.demos) for ti in range(10)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
