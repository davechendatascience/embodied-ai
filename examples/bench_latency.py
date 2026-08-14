"""How much time is there for a second vision layer? Measure, don't budget.

The grounding question has a deployment half that the condition sweep does not
touch: a detector bolted in front of a VLA has to fit inside the VLA's own
inference cadence, or it is not a second vision layer, it is a stall.

WHAT THE CADENCE ACTUALLY IS, and why it is not the frame rate.
pi-0.5 emits an ACTION CHUNK -- `action_horizon=10` for pi05_libero -- and the
host replays some prefix of it open loop before querying again. openpi's own
LIBERO example replans every 5 steps. So the budget per policy call is

    replan_steps x control_period

and NOT one control step. A detector that costs 30 ms looks fatal against a
20 ms control loop and is comfortable against a 5-step replan at 20 Hz. Quoting
detector latency against the wrong denominator is the usual way this decision
gets made badly.

WHAT THIS MEASURES. Round-trip `infer()` from the simulator's side: serialise,
websocket, model, deserialise. That is the number that competes with a
detector, because the detector would sit in the same loop. It is NOT the model's
raw forward time, which is smaller and not what anyone waits on.

FIRST CALL IS COMPILATION. JAX traces and compiles on the first shape it sees,
which here costs seconds. It is reported separately and excluded from the
statistics -- averaging it in makes a fast policy look slow and hides whatever
the steady state actually is.

THIS BOX IS NOT THE DEPLOY TARGET, AND THE FLAG MATTERS. GB10 needs
`--xla_gpu_enable_triton_gemm=false` (see scripts/serve_pi05.sh) to run at all,
which routes matmuls through cuBLAS instead of fused Triton kernels. Every
number here is therefore a pessimistic bound for this hardware, and says
nothing directly about Orin or Thor. Re-measure there.

Run (needs `bash scripts/serve_pi05.sh`):
  PYTHONPATH=. ./.venv/bin/python examples/bench_latency.py --n 30
"""
import argparse
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--replan-steps", type=int, default=5,
                    help="openpi's LIBERO default; sets the per-call budget")
    ap.add_argument("--control-hz", type=float, default=20.0)
    a = ap.parse_args()

    from openpi_client import websocket_client_policy as wcp

    client = wcp.WebsocketClientPolicy(a.host, a.port)
    obs = {
        "observation/image": np.zeros((224, 224, 3), np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), np.uint8),
        "observation/state": np.zeros(8, np.float32),
        "prompt": "put both the alphabet soup and the tomato sauce in the basket",
    }

    t0 = time.perf_counter()
    first = client.infer(obs)
    compile_s = time.perf_counter() - t0
    horizon = np.asarray(first["actions"]).shape[0]

    ts = []
    for _ in range(a.n):
        t = time.perf_counter()
        client.infer(obs)
        ts.append((time.perf_counter() - t) * 1000.0)
    ts = np.array(ts)

    budget_ms = a.replan_steps / a.control_hz * 1000.0
    print(f"pi-0.5 (pi05_libero) round-trip infer, n={a.n}")
    print(f"  first call (trace+compile)  {compile_s:8.2f} s   [excluded below]")
    print(f"  median                      {np.median(ts):8.1f} ms")
    print(f"  mean +/- sd                 {ts.mean():8.1f} +/- {ts.std():.1f} ms")
    print(f"  min / max                   {ts.min():8.1f} / {ts.max():.1f} ms")
    print(f"  action horizon              {horizon:8d} steps")
    print()
    print(f"budget at {a.control_hz:g} Hz, replan every {a.replan_steps}: "
          f"{budget_ms:.0f} ms per call")
    head = budget_ms - np.median(ts)
    print(f"  headroom for a second vision layer: {head:.0f} ms "
          f"({'fits' if head > 0 else 'DOES NOT FIT -- policy alone is over budget'})")
    print()
    print("Detector cost must be measured on the same box before this headroom")
    print("means anything. A ViT-L/14 patch encoder is not a YOLO-n, and neither")
    print("was measured here.")


if __name__ == "__main__":
    main()
