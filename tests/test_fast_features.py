"""Does the GPU feature assembly agree with PointWorld's own `gather_features`?

`fast_features.py` exists only for speed -- it recomputes four channel groups
on the GPU instead of running the upstream pipeline once per candidate. That is
a shortcut around a reference implementation, so it is worth exactly as much as
its agreement with that reference and nothing more.

A silent divergence here would not crash. It would feed the world model
slightly wrong velocities or distances and show up as degraded predictions,
which is indistinguishable from a worse model -- the most expensive failure
mode this project has hit repeatedly. Hence a control, not a smoke test:
identical candidates, both paths, tensors compared elementwise.

    CUMM_CUDA_VERSION=13.0 CUMM_CUDA_ARCH_LIST=12.0 PYTHONPATH=src \
        .venv-pw/bin/python tests/test_fast_features.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pointworld_bridge.episode import build_data_dict, load_episode, rigid_trajectory  # noqa: E402
from pointworld_bridge.fast_features import FastFeatures  # noqa: E402
from pointworld_bridge.model import inference_args, data_info_from_checkpoint, CKPT  # noqa: E402

EPISODE = ROOT / "data" / "pw_episodes" / "libero_goal_0_ep0.npz"
K = 8


def main():
    dev = torch.device("cuda")
    ep = load_episode(EPISODE)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    args = inference_args(ck, dev)
    info = data_info_from_checkpoint(ck)
    n_grip = 2 if info["robot_features_dim"] == 17 else 1
    print(f"checkpoint: robot {info['robot_features_dim']}ch scene "
          f"{info['scene_features_dim']}ch -> {n_grip} gripper slot(s)")

    ref0, meta = build_data_dict(ep, args, dev)
    fast = FastFeatures(ref0, meta, args, dev, n_grip)

    # Candidates the planner would actually produce: rigid pushes in a spread
    # of directions, in the WORLD frame.
    robot0 = ep["robot_flows"][0, 0].astype(np.float32)
    dirs = [(0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0),
            (0, 0, 1), (0, 0, -1), (0.7, 0.7, 0), (0, 0, 0)]
    cands = np.stack([rigid_trajectory(robot0, d, 0.013, meta["T"]) for d in dirs])

    # ---- reference path, one candidate at a time -------------------------
    t0 = time.perf_counter()
    refs = [build_data_dict(ep, args, dev, robot_flows=c, centre=meta["centre"])[0]
            for c in cands]
    t_ref = time.perf_counter() - t0

    # ---- fast path, all at once ------------------------------------------
    # Warm up first. The first call pays CUDA kernel selection, and quoting it
    # as the steady-state cost is the same mistake that made `observe` look 7x
    # more expensive than it is (`NOTES.md` section 4).
    for _ in range(3):
        fast.batch(cands)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        got = fast.batch(cands)
    torch.cuda.synchronize()
    t_fast = (time.perf_counter() - t0) / 10

    print(f"\nreference : {t_ref*1e3:7.1f} ms for K={K} "
          f"({t_ref*1e3/K:.1f} ms/candidate, CPU)")
    print(f"fast      : {t_fast*1e3:7.1f} ms for K={K} "
          f"({t_fast*1e3/K:.2f} ms/candidate, GPU)  -> {t_ref/t_fast:.0f}x\n")

    ok = True
    for key in ("scene_flows", "scene_features", "robot_flows", "robot_features"):
        want = torch.cat([r[key] for r in refs], dim=0)
        have = got[key]
        if want.shape != have.shape:
            print(f"{key:16s} SHAPE MISMATCH {tuple(want.shape)} vs {tuple(have.shape)}")
            ok = False
            continue
        err = (want - have).abs().max().item()
        # float32 assembly, so exact equality is not the right bar; anything
        # above single-precision noise is a real difference.
        good = err < 1e-5
        ok &= good
        print(f"{key:16s} max |ref - fast| = {err:.3e}   {'ok' if good else 'DIVERGES'}")

    # Name the worst channel group when something is off, so the next person
    # does not have to bisect a 42-wide vector by hand.
    if not ok:
        want = torch.cat([r["scene_features"] for r in refs], dim=0)[:, 0]
        have = got["scene_features"][:, 0]
        per_ch = (want - have).abs().amax(dim=(0, 1))
        bad = torch.nonzero(per_ch > 1e-5).flatten().tolist()
        print(f"\nscene channels that differ: {bad}")

    print(f"\nverdict : {'GPU assembly MATCHES the reference' if ok else 'DIVERGES — do not use it'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
