#!/usr/bin/env bash
# Reproduce third_party/ from upstream sources + our patches.
#
# third_party/ is NOT our code. ReKep is a pinned upstream checkout with one
# patch applied; LIBERO is a vendored copy. Everything we wrote lives in src/,
# tests/, configs/ and scripts/. Re-run this to rebuild third_party/ from
# scratch, or to verify our patch still applies to upstream.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

REKEP_URL="https://github.com/huangwl18/ReKep.git"
REKEP_COMMIT="63c43fdba60354980258beaeb8a7d48e088e1e3e"
PATCH="patches/0001-rekep-libero-integration.patch"

mkdir -p third_party

if [ ! -d third_party/ReKep/.git ]; then
  echo "==> cloning ReKep"
  git clone "$REKEP_URL" third_party/ReKep
fi

echo "==> pinning ReKep to $REKEP_COMMIT"
git -C third_party/ReKep checkout --quiet "$REKEP_COMMIT"
git -C third_party/ReKep checkout --quiet -- .

echo "==> applying $PATCH"
# Two changes, both load-bearing:
#   utils.py                 drop the open3d import (no aarch64/cp312 wheel);
#                            farthest_point_sampling reimplemented in numpy
#   constraint_generation.py route the VLM through rekep_libero.vlm_backends
#                            instead of a hard-coded OpenAI client
git -C third_party/ReKep apply "$ROOT/$PATCH"

echo "==> verifying"
git -C third_party/ReKep --no-pager diff --stat

if [ ! -d third_party/LIBERO ]; then
  echo
  echo "NOTE: third_party/LIBERO is missing. It is a vendored copy of the LIBERO"
  echo "      benchmark; clone it from https://github.com/Lifelong-Robot-Learning/LIBERO"
  echo "      into third_party/LIBERO."
fi

echo
echo "Done. Install the pinned stack with:"
echo "  .venv/bin/pip install -r requirements-libero.txt"
echo "Run headless with MUJOCO_GL=egl."
