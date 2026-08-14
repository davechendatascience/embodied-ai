#!/usr/bin/env bash
# Build .venv: the LIBERO / robosuite stack, and nothing else.
#
# SCOPE. The minimum to reset a LIBERO scene and render it -- a simulator and a
# benchmark, no policy. VLA-JEPA, cuRobo and PointWorld each need their own venv
# and their own script; they must not share this one. A venv here is a pinned
# stack for exactly one dependency set, and the moment two of them reference
# each other both become unreproducible.
#
# DO NOT `pip install -r third_party/LIBERO/requirements.txt`.
# That file pins `numpy==1.22.4`, which has no cp312 wheel and cannot build on
# Python 3.12 -- its build pins setuptools<60, whose pkg_resources touches
# `pkgutil.ImpImporter`, removed in 3.12. It also pins robosuite==1.4.0,
# transformers==4.21.1, opencv==4.6.0.66 and a training stack (wandb, hydra,
# robomimic, thop) that nothing here imports. Installed instead is the curated
# set below: every third-party module actually imported under
# libero/libero/{envs,benchmark,utils}, and nothing else. LIBERO itself is not
# pip-installed at all -- see the note further down, it cannot be.
#
# THE PINS, AND WHY. Each was a real failure, not a precaution.
#
#   robosuite==1.4.1   LIBERO imports
#                      `robosuite.environments.manipulation.single_arm_env`,
#                      which 1.5.x removed. 1.5 is what RoboCasa needs -- that
#                      is a different venv, deliberately.
#   mujoco==3.1.6      robosuite 1.4.1 calls `sim.data.qM`.
#   numpy==1.26.4      robosuite 1.4.1 and LIBERO predate the numpy 2 ABI, and
#                      1.26.4 is the last 1.x with a cp312 wheel. "numpy<2"
#                      alone resolves to something older that tries to build.
#
# TORCH IS CPU-ONLY, ON PURPOSE. This venv imports torch only because
# `libero.libero.benchmark` does, and to `torch.load` the init-state files. The
# policy runs in its own process behind a websocket, in its own venv, with its
# own CUDA build. Installing a CUDA torch here would be ~2.5 GB to support two
# call sites. If that stops being true, change it here and say so.
#
# ~/.libero/config.yaml IS WRITTEN UP FRONT. LIBERO's package __init__ calls
# input() on first import when that file is missing, which hangs any
# non-interactive build. Writing it first is not a convenience, it is what makes
# this script runnable unattended.
#
# NOT INSTALLED: datasets. The probe needs only BDDL task files, init states and
# object assets, all of which ship inside the LIBERO repo.
#
# Run from the repo root:
#   bash scripts/setup_libero.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-python3.12}"
VENV="$ROOT/.venv"
LIBERO_DIR="$ROOT/third_party/LIBERO"
LIBERO_URL="https://github.com/Lifelong-Robot-Learning/LIBERO.git"

command -v "$PY" >/dev/null || { echo "no $PY on PATH; set PY=..." >&2; exit 1; }

echo "== venv =="
[ -d "$VENV" ] || "$PY" -m venv "$VENV"
PIP="$VENV/bin/pip"
"$PIP" install --upgrade pip wheel setuptools

echo "== LIBERO checkout =="
mkdir -p "$ROOT/third_party"
if [ ! -d "$LIBERO_DIR/.git" ]; then
    git clone "$LIBERO_URL" "$LIBERO_DIR"
fi
# Record what we actually got. LIBERO's main branch moves, and a probe whose
# object names shift underneath it is not a probe. Pin this hash once a run has
# been trusted:  git -C third_party/LIBERO checkout <hash>
git -C "$LIBERO_DIR" rev-parse HEAD > "$ROOT/third_party/LIBERO.commit"
echo "LIBERO @ $(cat "$ROOT/third_party/LIBERO.commit")"

echo "== pinned core =="
"$PIP" install "numpy==1.26.4" "robosuite==1.4.1" "mujoco==3.1.6"

echo "== torch (CPU) =="
"$PIP" install torch --index-url https://download.pytorch.org/whl/cpu

echo "== what libero/libero actually imports =="
"$PIP" install bddl cloudpickle easydict gym h5py imageio imageio-ffmpeg \
               matplotlib termcolor tqdm pyyaml

# LIBERO IS NOT PIP-INSTALLED, AND MUST NOT BE.
# Its `libero/` directory has no `__init__.py`, so setup.py's `find_packages()`
# returns an empty list. `pip install -e` then SUCCEEDS, writes a dist-info, and
# maps nothing: `__editable___libero_0_1_0_finder.py` ships `MAPPING = {}`, and
# `import libero` still raises ModuleNotFoundError. A install that reports
# success and installs nothing is worse than one that fails, so it is not run.
# It works as a namespace package instead, via sys.path -- which is what every
# script here already does:
#     sys.path.insert(0, R + "/third_party/LIBERO")
"$PIP" uninstall -y libero >/dev/null 2>&1 || true

echo "== ~/.libero/config.yaml (pre-written; the import prompts otherwise) =="
LIBERO_PKG="$LIBERO_DIR/libero/libero"
mkdir -p "$HOME/.libero"
cat > "$HOME/.libero/config.yaml" <<YAML
benchmark_root: $LIBERO_PKG
bddl_files: $LIBERO_PKG/bddl_files
init_states: $LIBERO_PKG/init_files
datasets: $LIBERO_DIR/libero/datasets
assets: $LIBERO_PKG/assets
YAML
cat "$HOME/.libero/config.yaml"

echo "== assert =="
MUJOCO_GL=egl PYTHONPATH="$LIBERO_DIR" "$VENV/bin/python" - <<'PY'
import numpy, mujoco, robosuite
assert robosuite.__version__ == "1.4.1", robosuite.__version__
assert mujoco.__version__ == "3.1.6", mujoco.__version__
assert numpy.__version__.startswith("1."), numpy.__version__
# The import that decides whether this stack is usable at all -- 1.5.x drops it.
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
# The two LIBERO entry points the probe uses. Importing them here means a
# missing dependency fails the BUILD, not the first experiment.
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
n = len(benchmark.get_benchmark_dict())
print(f"robosuite {robosuite.__version__} | mujoco {mujoco.__version__} "
      f"| numpy {numpy.__version__} | {n} benchmark suites | OK")
PY

echo
echo "next:"
echo "  MUJOCO_GL=egl PYTHONPATH=. ./.venv/bin/python examples/probe_grounding.py \\"
echo "      --suite libero_10 --task-id 0 --overlay pairs/diag/boxes.png"
echo
echo "NOTE: MUJOCO_GL=egl needs libEGL present; on a headless box install libegl1."
