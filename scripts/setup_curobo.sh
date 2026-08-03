#!/usr/bin/env bash
# cuRobo on aarch64 / CUDA 13 / Blackwell (GB10, sm_121).
#
# OWN VENV (.venv-curobo), for the reason NOTES.md keeps repeating: `.venv` is
# pinned for ReKep/LIBERO and currently works, and cuRobo compiles CUDA
# extensions against whatever torch it finds. A failed build that half-upgrades
# torch would take the drawer result with it.
#
# The interface to the rest of the stack is therefore the same shape as the
# PointWorld bridge, and just as small:
#
#     (start joint config, goal SE(3) pose, world) -> joint trajectory
#
# JSON on disk for offline work, a socket for the control loop. Neither venv
# imports the other; orchestration lives in shell.
#
# sm_120 not sm_121: they are binary compatible, and 12.0 is what this
# toolchain emits. Same reasoning as setup_pointworld.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv-curobo"
TP="$ROOT/third_party"
INDEX="https://pypi.jetson-ai-lab.io/sbsa/cu130/+simple"

export TORCH_CUDA_ARCH_LIST="12.0"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

echo "==> venv"
[ -d "$VENV" ] || python3.12 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip wheel setuptools

echo "==> torch (prebuilt aarch64/cu130, same index as .venv-pw)"
"$VENV/bin/pip" install --index-url "$INDEX" --extra-index-url https://pypi.org/simple \
    torch torchvision

echo "==> cuRobo build deps"
# yourdfpy parses the URDF; trimesh does mesh IO for the world; numpy-quaternion
# and networkx are cuRobo's own runtime deps. usd-core is OPTIONAL (Isaac
# visualisation only) and has no reliable aarch64 wheel -- left out on purpose.
"$VENV/bin/pip" install -q numpy scipy networkx trimesh yourdfpy pyyaml \
    numpy-quaternion tqdm wheel ninja

echo "==> cuRobo from source"
[ -d "$TP/curobo" ] || git clone https://github.com/NVlabs/curobo.git "$TP/curobo"

# --no-build-isolation so the build sees the torch installed above rather than
# pulling a fresh x86/CPU one into an isolated env.
MAX_JOBS=8 "$VENV/bin/pip" install --no-build-isolation -e "$TP/curobo"

echo "==> verify"
"$VENV/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
from curobo.types.base import TensorDeviceType
from curobo.util_file import get_robot_configs_path
import os
print("curobo import OK")
cfgs = sorted(f for f in os.listdir(get_robot_configs_path()) if f.endswith(".yml"))
print("stock robot configs:", len(cfgs))
print("  ur5e     ->", [c for c in cfgs if "ur5" in c])
print("  franka   ->", [c for c in cfgs if "franka" in c or "panda" in c])
PY

echo
echo "Done. cuRobo lives in $VENV and may not be imported from .venv."
