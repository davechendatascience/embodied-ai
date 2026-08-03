#!/usr/bin/env bash
# Geometric Action Model (GAM) on aarch64 / CUDA 13 / Blackwell (GB10, sm_121).
#
# GAM's README pins torch 2.5.1 + cu124 on x86_64. CUDA 12.4 predates Blackwell,
# so that toolchain cannot generate code for this GPU. Its requirements.txt is
# honest about it though -- the real constraint is `torch>=2.5` (the predictor
# needs torch.nn.attention.flex_attention / BlockMask), and the cu124 line is an
# example, not a pin. So this installs the aarch64/cu130 wheels we already know
# work on this machine and leaves everything else at GAM's versions.
#
# Runs in its OWN venv (.venv-gam). It may not reference .venv or .venv-pw, and
# they may not reference it. GAM's stack is INCOMPATIBLE with .venv on purpose:
# mujoco 3.6.0 against .venv's 3.1.6, plus its own LIBERO checkout. Orchestration
# between them belongs in shell, never in either interpreter.
#
# What GAM actually is, which the README leaves implicit: the backbone is
# Depth-Anything-3, split at an intermediate layer. `setup_sources.sh` clones it
# alongside LIBERO and LIBERO-Plus and puts all three on PYTHONPATH rather than
# installing them -- so nothing here is pip-visible and every shell that runs
# GAM needs the exports printed at the end.
#
# Idempotent: re-run it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv-gam"
GAM="$ROOT/third_party/GAM"
INDEX="https://pypi.jetson-ai-lab.io/sbsa/cu130/+simple"

[ -d "$GAM" ] || {
    echo "==> cloning GAM"
    git clone --depth 1 https://github.com/cvlab-kaist/Geometric-Action-Model.git "$GAM"
}

echo "==> venv"
[ -d "$VENV" ] || python3.12 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip wheel setuptools

# Same index as .venv-pw: prebuilt aarch64 CUDA-13 wheels. Install torch FIRST so
# that requirements.txt's `torch>=2.5` is already satisfied and pip does not go
# to PyPI for a CPU-only or x86 build.
echo "==> torch + torchvision (prebuilt aarch64/cu130)"
"$VENV/bin/pip" install --index-url "$INDEX" --extra-index-url https://pypi.org/simple \
    torch torchvision

echo "==> GAM requirements, minus the torch lines (already installed)"
# deepspeed is TRAINING-only and is a long aarch64 build; drop it for inference.
# Everything else is left at GAM's pinned version.
grep -vE '^(torch|torchvision|deepspeed)\b' "$GAM/requirements.txt" > /tmp/gam-req.txt
"$VENV/bin/pip" install -r /tmp/gam-req.txt

echo "==> source-only deps (Depth-Anything-3, LIBERO, LIBERO-Plus)"
# GAM's own script, run with OUR interpreter on PATH so its `pip` calls land in
# .venv-gam rather than wherever the invoking shell points.
PATH="$VENV/bin:$PATH" bash "$GAM/scripts/setup_sources.sh"

# The HF repo is 56.6 GB: FIVE checkpoints of 14.16 GB each -- one per LIBERO
# suite (goal / object / spatial / long) plus `pretrained`. Pulling all of them
# takes ~66 min on this link and only one is needed to run a given suite, so
# default to the suite we are actually evaluating. Override to fetch more:
#   GAM_SUITES="goal object" scripts/setup_gam.sh
#   GAM_SUITES=all          scripts/setup_gam.sh
GAM_SUITES="${GAM_SUITES:-goal}"
CKPT_DIR="$GAM/checkpoints_hf/3da-libero-gam"

# DOWNLOAD THROUGHPUT, measured on this link rather than assumed:
#
#   5 files in flight   12.5 MB/s aggregate  (~2.5 MB/s each)
#   1 file  in flight    2.3 MB/s
#
# The repo is xet-backed (hf-xet ships with huggingface_hub 1.10) and xet
# parallelises ACROSS files, not within one. So `--include goal/*` does NOT
# make `goal` arrive sooner -- it only avoids fetching the 42 GB of suites we
# are not evaluating. Budget ~95 min for one 14.16 GB checkpoint.
#
# HF_XET_HIGH_PERFORMANCE=1 with HF_XET_NUM_CONCURRENT_RANGE_GETS=32 was tried
# and made it WORSE: the transfer stalled outright, 0.0 MB written in 30 s
# against 2.3 MB/s without it (verified through /proc/<pid>/io, not just file
# size). Do not re-add them without measuring; they are left unset on purpose.

echo "==> checkpoints: $GAM_SUITES (14.16 GB each; resumes, skips what is present)"
if [ "$GAM_SUITES" = "all" ]; then
    "$VENV/bin/hf" download SeonghuJeon/3da-libero-gam --local-dir "$CKPT_DIR"
else
    for suite in $GAM_SUITES; do
        if [ -f "$CKPT_DIR/$suite/gam.pt" ]; then
            echo "    $suite already present, skipping."
            continue
        fi
        "$VENV/bin/hf" download SeonghuJeon/3da-libero-gam \
            --include "$suite/*" --local-dir "$CKPT_DIR"
    done
fi

cat <<EOF

==> done. Every GAM shell needs:

  export DA3_ROOT="$GAM"
  export DA3_LIBERO_SOURCE_DIR="$GAM/LIBERO"
  export DA3_LIBERO_PLUS_DIR="$GAM/LIBERO-plus"
  export PYTHONPATH="$GAM/src:$GAM/LIBERO-plus:$GAM/LIBERO"
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

  ...and \$VENV/bin/python, which is $VENV/bin/python.
EOF
