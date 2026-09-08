#!/usr/bin/env python
"""Can the frozen visual features locate the objects at all?

If a policy ignores its cameras, there are two possible reasons and they lead
opposite ways: the encoder never represented the scene (fix the encoder), or it
did and the objective never asked for it (fix the objective). A ridge probe from
the frozen features to the simulator's own object poses separates them, with no
policy in the loop.

Object coordinates are found without loading a MuJoCo model: in a robosuite
state vector the qpos block holds each free-floating body as 3 positions
followed by a unit quaternion, so a 7-wide window whose last four entries keep
unit norm across every frame is a free joint, and its first three are a position
in world coordinates.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from screwhead.libero import task_files            # noqa: E402


def free_joint_offsets(S: np.ndarray, tol: float = 1e-3) -> list[int]:
    """Offsets o where S[:, o+3:o+7] is a unit quaternion in every frame."""
    out, o = [], 0
    while o + 7 <= S.shape[1]:
        n = np.linalg.norm(S[:, o + 3:o + 7], axis=1)
        if np.abs(n - 1).max() < tol and S[:, o:o + 3].std(0).max() > 1e-6:
            out.append(o); o += 7
        else:
            o += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--cache", default="cache/libero_spatial")
    ap.add_argument("--demos-per-task", type=int, default=50)
    ap.add_argument("--val-demos", type=int, default=5)
    ap.add_argument("--alpha", type=float, nargs="+", default=[1.0, 10.0, 100.0, 1000.0])
    args = ap.parse_args()

    import h5py
    import torch
    from screwhead.interface import ActionSpec
    from screwhead.libero import panda_chain
    from screwhead.retarget import to_twists, usable

    chain, spec = panda_chain(), ActionSpec()
    rows = []
    for ti, f in enumerate(task_files(args.suite)):
        st, dm = [], []
        with h5py.File(f, "r") as h:
            for di, k in enumerate(list(h["data"].keys())[: args.demos_per_task]):
                g = h["data"][k]
                q = torch.tensor(g["obs"]["joint_states"][:], dtype=torch.float64)
                if len(q) - 1 < 2:
                    continue
                to_twists(chain, q, spec)              # same call, same filtering
                idx = torch.nonzero(usable(chain, q[:-1])).flatten().numpy()
                if len(idx) == 0:
                    continue
                st.append(np.asarray(g["states"][:-1], np.float64)[idx])
                dm.append(np.full(len(idx), di, np.int16))
        S = np.concatenate(st); D = np.concatenate(dm)

        d = np.load(Path(args.cache) / f"task{ti:02d}.npz", allow_pickle=True)
        F = np.concatenate([np.asarray(d["agent"], np.float64),
                            np.asarray(d["wrist"], np.float64)], 1)
        if len(F) != len(S):
            print(f"[{ti}] MISALIGNED: {len(F)} features vs {len(S)} states"); continue

        offs = free_joint_offsets(S)
        if not offs:
            print(f"[{ti}] no free joints found"); continue
        # An object the robot carries moves WITH the gripper, so predicting it
        # can be done from the wrist camera's view of the gripper and says
        # nothing about scene understanding. The layout question is about the
        # objects that never move: their position is fixed inside a demo and
        # randomised between demos, so predicting one means reading the scene.
        static = []
        for o in offs:
            within = max(S[D == di, o:o + 3].std(0).max() for di in np.unique(D))
            across = np.stack([S[D == di, o:o + 3][0] for di in np.unique(D)]).std(0).max()
            (static if within < 1e-3 and across > 5e-3 else []).append(o)

        moved = [o for o in offs if o not in static]
        if not static:
            print(f"[{ti}] no static randomised object"); continue
        Y = np.concatenate([S[:, o:o + 3] for o in static + moved], 1)
        nst = 3 * len(static)

        val = set(sorted(np.unique(D).tolist())[: args.val_demos])
        m = np.array([x in val for x in D])
        Xtr, Xva = F[~m], F[m]
        Ytr, Yva = Y[~m], Y[m]
        mu, sd = Xtr.mean(0), Xtr.std(0).clip(1e-6)
        Xtr = np.c_[(Xtr - mu) / sd, np.ones(len(Xtr))]
        Xva = np.c_[(Xva - mu) / sd, np.ones(len(Xva))]
        ym = Ytr.mean(0)

        best = (-np.inf, None)
        G = Xtr.T @ Xtr
        for a in args.alpha:
            W = np.linalg.solve(G + a * np.eye(G.shape[0]), Xtr.T @ (Ytr - ym))
            P = Xva @ W + ym
            ss = 1 - ((P - Yva) ** 2).sum(0) / ((Yva - Yva.mean(0)) ** 2).sum(0).clip(1e-12)
            if ss.mean() > best[0]:
                best = (float(ss.mean()), (a, ss, np.abs(P - Yva).mean(0)))
        a, ss, err = best[1]
        r_st = ss[:nst].reshape(-1, 3).mean(1)
        r_mv = ss[nst:].reshape(-1, 3).mean(1) if len(ss) > nst else np.array([np.nan])
        print(f"[{ti}] alpha={a:<6g}  static objects={len(static)} "
              f"R2 median {np.median(r_st):+.3f} best {r_st.max():+.3f} "
              f"(err {err[:nst].mean()*1000:5.1f} mm)   "
              f"carried R2 median {np.nanmedian(r_mv):+.3f}", flush=True)
        rows.append(np.median(r_st))
    if rows:
        print(f"\nacross {len(rows)} tasks: median R2 on static randomised "
              f"objects = {np.median(rows):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
