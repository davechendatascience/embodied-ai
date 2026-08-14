#!/usr/bin/env bash
# Serve pi-0.5 fine-tuned for LIBERO, on port 8000.
#
# WHY THIS IS A SEPARATE PROCESS AND A SEPARATE VENV.
# openpi pins jax==0.5.3 with the CUDA 12 plugin, torch 2.7.1, numpy<2 and
# transformers 4.53.2. The LIBERO venv pins robosuite 1.4.1 against mujoco
# 3.1.6. These two stacks cannot share an interpreter, and every attempt to make
# them do so ends with one of them silently resolving to a version that imports
# but behaves differently. So openpi lives in third_party/openpi/.venv, managed
# by uv, and speaks to the simulator over a websocket. The only thing installed
# on BOTH sides is `openpi-client`, which is pure python -- msgpack, websockets,
# pillow, numpy -- and carries no framework with it.
#
# WHAT GETS DOWNLOADED. `--env LIBERO` resolves to openpi's own default:
#   config  pi05_libero
#   dir     gs://openpi-assets/checkpoints/pi05_libero
# fetched on first run into ~/.cache/openpi and reused after. Several GB.
#
# THE SERVER IS STATELESS. One `infer` call in, one full action chunk out
# (action_horizon=10 for this config). There is no episode state, no ensembler
# and no step counter to get wrong -- which is the single reason a static probe
# can query it repeatedly and trust every answer independently.
#
# Run:
#   bash scripts/serve_pi05.sh            # foreground, Ctrl-C to stop
#   bash scripts/serve_pi05.sh --port 8001
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI="$ROOT/third_party/openpi"

[ -d "$OPENPI/.venv" ] || {
    echo "openpi venv missing. Build it with:" >&2
    echo "  cd $OPENPI && GIT_LFS_SKIP_SMUDGE=1 uv sync && uv pip install -e ." >&2
    exit 1
}

# GB10 (aarch64 Blackwell) WORKAROUND -- REQUIRED, NOT A TUNING KNOB.
# Without this the server loads the checkpoint, accepts a connection, and then
# aborts on the first inference with:
#     Unsupported conversion from bf16 to f16
#     LLVM ERROR: Unsupported rounding mode for conversion.
# It is an XLA codegen failure in the fused Triton GEMM path for this compute
# capability under jax 0.5.3 / CUDA 12.9, not a configuration error: the model
# is bf16 (Pi0Config.dtype) and the fusion emits an f16 convert this backend
# cannot lower. Disabling the Triton GEMM path routes those matmuls through
# cuBLAS instead and inference succeeds. Expect it to cost some throughput.
#
# Re-test after any jax upgrade; if a later jaxlib lowers it correctly, delete
# this line rather than carrying it forever.
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"

cd "$OPENPI"
exec uv run scripts/serve_policy.py --env LIBERO "$@"
