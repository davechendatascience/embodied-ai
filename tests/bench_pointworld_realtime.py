"""What would a real-time control loop cost, on real data, on the GB10?

`bench_pointworld.py` timed `DynamicsPredictor` on synthetic points. That is
the right thing to time for MPPI throughput, but it is not the loop. The loop
splits in two, and the split is the whole point:

  PERCEPTION   DINOv3 + projection to points. Runs ONCE per camera
               observation, and `BaseModel.forward` accepts
               `encoded_scene_feat0` precisely so it can be reused.
  ROLLOUT      the PTv3 trunk and heads. Runs once per candidate trajectory,
               every planning tick.

A controller that re-runs perception per candidate is paying the wrong cost.
This measures both, on the recorded episode rather than on random points, so
the voxel occupancy is real.

    CUMM_CUDA_VERSION=13.0 CUMM_CUDA_ARCH_LIST=12.0 PYTHONPATH=src \
        .venv-pw/bin/python tests/bench_pointworld_realtime.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pointworld_bridge.episode import build_data_dict, load_episode  # noqa: E402
from pointworld_bridge.model import load_base_model  # noqa: E402

EPISODE = ROOT / "data" / "pw_episodes" / "libero_goal_0_ep0.npz"


def timeit(fn, n=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2], ts[0], ts[-1]


def main():
    dev = torch.device("cuda")
    ep = load_episode(EPISODE)
    model, args, _ = load_base_model(device=dev, verbose=False)
    dd, meta = build_data_dict(ep, args, dev)
    print(f"episode : {meta['Ns']} scene pts, {meta['Nr']} robot pts, "
          f"{meta['cameras']} cameras, horizon {meta['T'] - 1}")

    with torch.no_grad():
        # Perception. `_current_domain_indices` is set by forward(), so prime it.
        model(dd, training=False)
        enc = lambda: model.encode_scene_features(dd)          # noqa: E731
        p_med, p_min, p_max = timeit(enc, n=10)
        feat = model.encode_scene_features(dd)

        # Rollout, reusing the encoded features the way a controller would.
        roll = lambda: model(dd, training=False, encoded_scene_feat0=feat)   # noqa: E731
        r_med, r_min, r_max = timeit(roll, n=10)

        # And the naive version, re-running DINOv3 for every candidate.
        full = lambda: model(dd, training=False)               # noqa: E731
        f_med, _, _ = timeit(full, n=10)

    print(f"\nperception (DINOv3, once per observation): "
          f"{p_med*1000:6.1f} ms  [{p_min*1000:.1f}-{p_max*1000:.1f}]")
    print(f"rollout    (PTv3 + heads, per candidate) : "
          f"{r_med*1000:6.1f} ms  [{r_min*1000:.1f}-{r_max*1000:.1f}]")
    print(f"naive      (perception re-run per call)  : {f_med*1000:6.1f} ms")
    print(f"paper's stated inference for one 10-step chunk: 100 ms")

    print(f"\nA planning tick that scores K candidate trajectories costs "
          f"{p_med*1000:.0f} + K x {r_med*1000:.0f} ms:")
    for k in (1, 4, 8, 16, 32):
        total = p_med + k * r_med
        print(f"  K={k:3d}  {total*1000:7.1f} ms  ->  {1/total:5.1f} Hz")
    print("\nCandidates batch on one GPU, so K x is an upper bound; a batched "
          "rollout amortises far better than this table suggests.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
