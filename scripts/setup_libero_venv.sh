#!/usr/bin/env bash
# Rebuild .venv-libero. Separate from .venv on purpose: LIBERO pins an old
# stack, and mixing it with the geometry venv would drag those pins into both.
# Orchestration between the two lives in the shell, not in either environment.
set -euo pipefail
cd "$(dirname "$0")/.."

uv venv .venv-libero --python 3.11
uv pip install --python .venv-libero/bin/python \
    "numpy<2" \
    "robosuite==1.4.0" \
    "mujoco==3.8.1" \
    bddl h5py easydict "gym==0.25.2" cloudpickle termcolor matplotlib
uv pip install --python .venv-libero/bin/python torch "torchvision==0.29.0" --index-url https://download.pytorch.org/whl/cu130
# the Panda VLA's backbone (Qwen3-VL-2B): transformers' Qwen image processor needs torchvision
uv pip install --python .venv-libero/bin/python "transformers==5.16.1" "peft==0.20.0"
uv pip install --python .venv-libero/bin/python -e third_party/LIBERO --no-deps

# mujoco 3.8.1, NOT latest. On 3.12, robosuite 1.4.0's get_joint_qpos_addr
# asserts the joint is hinge or slide and that fires for every robot joint
# before a frame renders -- a bare AssertionError carrying no message.
#
# Rendering is EGL; osmesa is not installed here. The EGLError raised from
# EGLGLContext.__del__ at teardown is cosmetic and does not affect frames.
echo
echo "Run rollouts with:"
echo "  PYTHONPATH=third_party/LIBERO:. MUJOCO_GL=egl .venv-libero/bin/python ..."
