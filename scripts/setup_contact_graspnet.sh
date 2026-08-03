#!/usr/bin/env bash
# Wire Contact-GraspNet into this venv, reusing the robot project's install.
#
# WHY THIS IS NOT A pip install
# -----------------------------
# Contact-GraspNet needs two CUDA extensions, `pointnet2` and `knn_pytorch`,
# which are compiled against a specific torch build. They are already built for
# cp312 / aarch64 / torch 2.13.0+cu130 in
#
#     wardmate_ws/.venv/lib/python3.12/site-packages
#
# and this venv is on exactly that python and that torch, so the .so files load
# as-is. Rebuilding them would produce byte-identical artifacts and take far
# longer than it is worth, so this links them instead. If either venv's torch
# moves, DELETE the symlinks and rebuild -- a torch ABI mismatch surfaces as an
# undefined-symbol ImportError at first grasp, not at install time.
#
# The weights are shared for a different reason: the robot and this project
# should propose the same grasp for the same object. A copied checkpoint is a
# checkpoint that can drift.
#
# NAMING, because it has already caused one wrong conclusion:
# `wardmate_ws/src/llm_robot_control/third_party/graspgen` is a ROS 2 package
# that WRAPS Contact-GraspNet. It is unrelated to NVIDIA GraspGen, which cannot
# be installed here at all -- it pins torch==2.1.0, which has no cp312 wheel for
# aarch64.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CGN_ROOT="${CGN_ROOT:-$HOME/contact_graspnet_pytorch}"
DONOR="${DONOR_VENV:-$HOME/Documents/GitHub/wardmate_ws/.venv}"

SITE="$ROOT/.venv/lib/python3.12/site-packages"
DONOR_SITE="$DONOR/lib/python3.12/site-packages"

echo "==> checking the model and weights"
if [ ! -d "$CGN_ROOT/checkpoints/contact_graspnet" ]; then
  echo "ERROR: no Contact-GraspNet checkpoint at $CGN_ROOT/checkpoints/contact_graspnet"
  echo "       Clone https://github.com/elchun/contact_graspnet_pytorch and fetch its"
  echo "       weights, or set CGN_ROOT to an existing install."
  exit 1
fi

echo "==> checking the two venvs agree on torch"
mine="$("$ROOT/.venv/bin/python" -c 'import torch; print(torch.__version__)')"
theirs="$("$DONOR/bin/python" -c 'import torch; print(torch.__version__)' 2>/dev/null || echo missing)"
echo "    this venv : $mine"
echo "    donor     : $theirs"
if [ "$mine" != "$theirs" ]; then
  echo "ERROR: torch differs, so the compiled extensions will not load."
  echo "       Build pointnet2 and knn_pytorch against $mine instead of linking."
  exit 1
fi

echo "==> pure-python deps"
# pyrender is imported by contact_graspnet_pytorch/data.py for training-time
# scene rendering only, but the import is unconditional, so it must be present.
#
# Its metadata pins PyOpenGL==3.1.0, and that pin is WRONG for us: 3.1.0 lacks
# EGL.EGLDeviceEXT, which mujoco.egl needs, so installing it unmodified breaks
# every headless render in this project. pyrender imports fine on 3.1.10, so
# reinstall the newer one afterwards and accept the pip warning.
"$ROOT/.venv/bin/pip" install trimesh pyrender "numpy==1.26.4" >/dev/null
"$ROOT/.venv/bin/pip" install "PyOpenGL==3.1.10" 2>&1 | grep -v "^ERROR: pip's dependency resolver" || true

echo "==> linking the compiled CUDA extensions"
for pkg in pointnet2 knn_pytorch; do
  if [ ! -d "$DONOR_SITE/$pkg" ]; then
    echo "ERROR: $DONOR_SITE/$pkg not found"
    exit 1
  fi
  ln -sfn "$DONOR_SITE/$pkg" "$SITE/$pkg"
  echo "    $pkg -> $DONOR_SITE/$pkg"
done
[ -f "$DONOR_SITE/pointnet2_flat.pth" ] && ln -sfn "$DONOR_SITE/pointnet2_flat.pth" "$SITE/pointnet2_flat.pth"

echo "==> verifying"
MUJOCO_GL=egl "$ROOT/.venv/bin/python" - <<'PY'
import sys
sys.path.insert(0, "src")
import pointnet2, knn_pytorch, torch          # noqa: F401
import mujoco.egl                             # the PyOpenGL regression guard
from rekep_libero import grasp_cgn
assert grasp_cgn.available(), "checkpoint missing"
grasp_cgn.estimator()
print("    OK: extensions load, EGL intact, weights load")
PY

cat <<'EOF'

Done. Select the proposer per run:
    MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python -m rekep_libero.runner \
        --backend robosuite --grasp contact_graspnet
or persistently via `grasp.proposer` in configs/rekep_libero.yaml.

Compare against the PCA baseline:
    MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/compare_grasp_proposers.py 0
EOF
