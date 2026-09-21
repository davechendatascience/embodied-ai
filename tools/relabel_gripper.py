#!/usr/bin/env python
"""Add target-aperture gripper labels (label_gt) to existing token caches.

The saved labels are the program's -1/0/+1 commands. The target it was regulating
toward is a function of its phase (screwhead/gripper_servo.py:program_target):

  DAgger rounds    the distill.py npz stored the phase per frame -- exact.
  teacher shards   no phase was stored. Only the drawer task pre-shapes; every other
                   task's command already IS its target (-1 open, +1 closed). For the
                   drawer, a successful demonstration closes once for good at the grasp
                   and opens once at release, so: frames before the final closing run
                   regulate the pre-shape, the run is closed, frames after are open.
                   Checked against phase-labelled drawer episodes with --check.

Usage:
  relabel_gripper.py --tokens cache/tokens/round1 --phases cache/distill_scripted/tok_round1.npz
  relabel_gripper.py --tokens cache/tokens/round0f          # teacher shards, drawer heuristic
  relabel_gripper.py --check cache/distill_scripted/tok_round1.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from screwhead.gripper_servo import A_OPEN, program_target, target_to_channel  # noqa: E402
from screwhead.scripted_teacher import PROGRAMS  # noqa: E402


def from_phases(task, phase, command):
    return np.array([target_to_channel(program_target(str(p), float(c), PROGRAMS[int(t)].preshape_aperture))
                     for t, p, c in zip(task, phase, command, strict=True)], np.float32)


def from_sequence(task, episode, step, command):
    """No phases: per episode, in step order."""
    out = np.where(command < 0, target_to_channel(A_OPEN), target_to_channel(0.0)).astype(np.float32)
    for e in np.unique(episode[np.array([PROGRAMS[int(t)].preshape_aperture is not None for t in task])]):
        idx = np.where(episode == e)[0]
        idx = idx[np.argsort(step[idx])]
        c = np.sign(np.round(command[idx]))
        pre = PROGRAMS[int(task[idx[0]])].preshape_aperture
        closed = np.where(c > 0)[0]
        if len(closed) == 0:
            out[idx] = target_to_channel(pre); continue
        # the final closing run: walk back from the last close while it stays closed
        end = closed[-1]; start = end
        while start > 0 and c[start - 1] > 0:
            start -= 1
        out[idx[:start]] = target_to_channel(pre)
        out[idx[start:end + 1]] = target_to_channel(0.0)
        out[idx[end + 1:]] = target_to_channel(A_OPEN)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens")
    ap.add_argument("--phases", default="")
    ap.add_argument("--check", default="")
    args = ap.parse_args()
    if args.check:
        z = np.load(args.check)
        exact = from_phases(z["task"], z["phase"], z["label"][:, 6])
        guess = from_sequence(z["task"], z["episode"], z["step"], z["label"][:, 6])
        for name, sel in (("drawer, successful episodes", (z["task"] == 4) & z["episode_success"]),
                          ("drawer, all episodes", z["task"] == 4), ("other tasks", z["task"] != 4)):
            print(f"{name}: sequence heuristic agrees with phases on {np.mean(np.isclose(exact[sel], guess[sel])):.3f} of {sel.sum()} frames")
        return 0
    meta_path = Path(args.tokens) / "meta.npz"
    meta = dict(np.load(meta_path))
    if args.phases:
        z = np.load(args.phases)
        assert np.array_equal(z["label"], meta["label"]), "phase source does not match the token cache frame order"
        g = from_phases(meta["task"], z["phase"], meta["label"][:, 6])
        meta["phase"] = z["phase"]
    else:
        g = from_sequence(meta["task"], meta["episode"], meta["step"], meta["label"][:, 6])
    lab = meta["label"].copy(); lab[:, 6] = g
    meta["label_gt"] = lab
    np.savez(meta_path, **meta)
    vals, counts = np.unique(np.round(g, 3), return_counts=True)
    values = dict(zip(vals.tolist(), counts.tolist(), strict=True))
    print(f"{meta_path}: label_gt written; gripper channel values {values}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
