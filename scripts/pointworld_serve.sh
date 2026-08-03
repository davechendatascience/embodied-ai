#!/usr/bin/env bash
# Start the PointWorld service in .venv-pw. Nothing else may do this.
#
# The two stacks must not know about each other: .venv is mujoco 3.1.6 +
# numpy 1.26, .venv-pw is torch 2.11 + numpy 2.5, and the socket is the whole
# boundary between them. So the client does NOT launch the server -- if it
# did, the LIBERO venv would be choosing the PointWorld venv's interpreter and
# setting its CUDA variables, which is the coupling this project exists to
# avoid. Lifecycle lives here, in shell, which belongs to neither.
#
# The CUMM variables are not optional: spconv builds kernels at import time and
# must be told which CUDA and which arch, because the GB10 is newer than the
# toolchain PointWorld pins (NOTES.md section 4).
#
#     scripts/pointworld_serve.sh                    # foreground
#     scripts/pointworld_serve.sh --socket /tmp/x.sock --max-batch 8
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x .venv-pw/bin/python ]]; then
    echo "no .venv-pw -- rebuild it with scripts/setup_pointworld.sh" >&2
    exit 1
fi

export CUMM_CUDA_VERSION=13.0
export CUMM_CUDA_ARCH_LIST=12.0
export PYTHONPATH="$ROOT/src"

exec .venv-pw/bin/python -m pointworld_bridge.server "$@"
