"""How fast is PointWorld's dynamics predictor on the GB10?

The paper reports "real-time (0.1 s) inference" for one 10-step chunk, on
whatever GPU they used. That number is what makes the MPPI control loop viable,
so it is the first thing worth knowing about this port -- and it is knowable
WITHOUT DINOv3, because compute cost does not depend on whether the scene
features are real or zeros. Only prediction accuracy does.

What this measures is `DynamicsPredictor` alone: the PTv3 backbone (spconv +
flash-attn + torch-scatter) and the dynamics head. It excludes the DINOv3
featurizer, which runs once per observation rather than once per MPPI rollout,
and excludes MPPI sampling itself.

Shapes come from the released checkpoint's own training args, not from guesses:
    max_scene_points 12000, max_robot_points 500, pred_horizon 10,
    grid_size 0.015, ptv3_size small, predictor_dim 128

    CUMM_CUDA_VERSION=13.0 CUMM_CUDA_ARCH_LIST=12.0 \
        .venv-pw/bin/python tests/bench_pointworld.py
"""

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PW = ROOT / "third_party" / "PointWorld"
sys.path.insert(0, str(PW))

CKPT = PW / "pretrained_checkpoints" / "small-droid" / "model-best.pt"


def main():
    if not CKPT.exists():
        print(f"checkpoint not found: {CKPT}")
        return 1

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    args = ck["args"]
    dev = torch.device("cuda")

    from pointworld.base import DynamicsPredictor

    C = args.predictor_dim
    # T is the SEQUENCE length, and only T-1 steps are predicted because the
    # first is the input. The checkpoint's dynamics head is 3*(T-1)=30 wide, so
    # T = pred_horizon + 1 = 11 rather than pred_horizon.
    T = args.pred_horizon + 1
    Ns, Nr = args.max_scene_points, args.max_robot_points
    print(f"config : ptv3={args.ptv3_size} dim={C} horizon={T} "
          f"scene_pts={Ns} robot_pts={Nr} grid={args.grid_size}")

    model = DynamicsPredictor(args, C, T).to(dev).eval()

    # load just the dynamics predictor's weights out of the full checkpoint
    sd = ck["model"]
    prefix = "dynamics_predictor."
    sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    missing, unexpected = model.load_state_dict(sub, strict=False)
    print(f"weights: loaded {len(sub)} tensors "
          f"({len(missing)} missing, {len(unexpected)} unexpected)")

    # Zeros would be unrepresentative: the spconv voxelisation cost depends on
    # how many distinct voxels the points occupy, so use a realistic spread.
    g = torch.Generator(device="cpu").manual_seed(0)
    scene_coord = torch.rand(1, Ns, 3, generator=g) * 0.8 - 0.4
    scene_feat = torch.zeros(1, Ns, C)          # DINOv3 slot — cost is the same
    scene_exists = torch.ones(1, Ns, dtype=torch.bool)
    robot_coord = torch.rand(1, T, Nr, 3, generator=g) * 0.3
    robot_feat = torch.zeros(1, T, Nr, C)
    robot_exists = torch.ones(1, T, Nr, dtype=torch.bool)

    tensors = [t.to(dev) for t in (scene_coord, scene_feat, scene_exists,
                                   robot_coord, robot_feat, robot_exists)]

    with torch.no_grad():
        for i in range(3):                       # warm up: JIT + autotune
            model(*tensors, training=False)
        torch.cuda.synchronize()

        times = []
        for _ in range(10):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(*tensors, training=False)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    times.sort()
    med = times[len(times) // 2]
    print(f"\nforward: median {med * 1000:.1f} ms  "
          f"(min {times[0] * 1000:.1f}, max {times[-1] * 1000:.1f})")
    print(f"paper   : 0.1 s per 10-step chunk")
    print(f"verdict : {'FASTER than' if med < 0.1 else 'SLOWER than'} the paper's "
          f"0.1 s — {0.1 / med:.2f}x")
    print(f"\nA 30-step MPPI rollout is 3 chunks: ~{med * 3 * 1000:.0f} ms per "
          f"candidate trajectory.")
    print("MPPI samples K trajectories per step, so multiply by K (batched).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
