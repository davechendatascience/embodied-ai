#!/usr/bin/env python
"""Does the teacher steer toward where the bowl IS? A counterfactual in observation space.

The image-space displacement probe, without a simulator: take approach-phase
states from held-out demonstrations, shift the bowl-dependent features by a
displacement D, and measure how the commanded tool velocity responds. A policy
that servos to the bowl turns its velocity toward D; one that uses the bowl pose
only as a phase flag does not.

Features moved together, so the counterfactual stays self-consistent:
    p_bowl += D,  (bowl - tool) += D,  (plate - bowl) -= D

The head emits a body twist; its linear part is rotated into the base frame by
the tool rotation carried in the observation before comparing with D -- the
frame error that once inverted the image probe's sign.

A checkpoint with the object features masked must score exactly zero.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from screwhead.interface import ActionSpec    # noqa: E402
from teacher_bc import TeacherMLP             # noqa: E402

P_TOOL, R6_TOOL, GRIP = slice(7, 10), slice(10, 16), 16
P_BOWL, BOWL_TOOL, PLATE_BOWL = slice(17, 20), slice(29, 32), slice(32, 35)


def load(ckpt):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = TeacherMLP(ck["obs_dim"], ck["act_dim"]); m.load_state_dict(ck["state_dict"]); m.eval()
    mask = np.asarray(ck.get("obs_mask", np.ones(ck["obs_dim"])), np.float32)
    return m, ck["obs_mu"], ck["obs_sd"], ck["act_sd"], mask


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--cache", default="cache/teacher_bc")
    ap.add_argument("--val-demos", type=int, default=5)
    ap.add_argument("--radius", type=float, default=0.06)
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--min-dist", type=float, default=0.08, help="approach phase: tool this far from bowl")
    ap.add_argument("--max-dist", type=float, default=1e9)
    ap.add_argument("--states", type=int, default=400)
    args = ap.parse_args()

    O = []
    for f in sorted(Path(args.cache).glob("task*.npz")):
        d = np.load(f)
        val = np.isin(d["demo"], sorted(np.unique(d["demo"]).tolist())[: args.val_demos])
        O.append(d["obs"][val])
    O = np.concatenate(O)
    dist = np.linalg.norm(O[:, BOWL_TOOL], axis=1)
    approach = (dist > args.min_dist) & (dist <= args.max_dist) & (O[:, GRIP] > 0.03)   # gripper still open
    O = O[approach]
    O = O[np.random.default_rng(0).choice(len(O), min(args.states, len(O)), replace=False)]
    spec = ActionSpec()
    lin_scale = spec.pos_scale * spec.control_hz
    offs = np.linspace(-args.radius, args.radius, args.grid)
    D = np.array([[dx, dy, 0.0] for dx in offs for dy in offs])          # (G, 3)
    print(f"{len(O)} states, tool {args.min_dist*1000:.0f}-{min(args.max_dist, 9.99)*1000:.0f} mm from bowl, gripper open, "
          f"{len(D)} displacements within +/-{args.radius*1000:.0f} mm")

    for ck in args.ckpts:
        m, mu, sd, asd, mask = load(ck)
        X = np.repeat(O[:, None], len(D), 1)                                 # (S, G, obs)
        X[..., P_BOWL] += D; X[..., BOWL_TOOL] += D; X[..., PLATE_BOWL] -= D
        with torch.no_grad():
            A = m(torch.tensor((X.reshape(-1, X.shape[-1]) - mu) / sd * mask, dtype=torch.float32)).numpy()
        A = A.reshape(len(O), len(D), -1) * asd
        r6 = O[:, R6_TOOL]; c1, c2 = r6[:, 0:3], r6[:, 3:6]
        R = np.stack([c1, c2, np.cross(c1, c2)], axis=-1)                   # (S, 3, 3) columns
        v = np.einsum("sij,sgj->sgi", R, A[..., 3:6] * lin_scale)           # base-frame velocity
        Dc = D - D.mean(0)
        vc = v - v.mean(1, keepdims=True)
        gains, coss = [], []
        for s in range(len(O)):
            G = np.linalg.lstsq(Dc[:, :2], vc[s, :, :2], rcond=None)[0]
            gains.append(np.linalg.norm(G, 2))
            den = np.linalg.norm(Dc[:, :2]) * np.linalg.norm(vc[s, :, :2])
            coss.append(float((Dc[:, :2] * vc[s, :, :2]).sum() / den) if den > 1e-12 else 0.0)
        gains, coss = np.array(gains), np.array(coss)
        print(f"  {Path(ck).parent.name + '/' + Path(ck).name:48s} gain median {np.median(gains):6.3f} "
              f"(m/s per m)  alignment median {np.median(coss):+.3f}  frac>0.5 {np.mean(coss > 0.5):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
