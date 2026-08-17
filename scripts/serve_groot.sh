#!/usr/bin/env bash
# Serve GR00T-N1.6-3B for RoboCasa, on port 5555.
#
# WHY THIS MODEL AND NOT pi-0.5.
# pi05_libero is fine-tuned on LIBERO and is not competent on RoboCasa. That is
# measured here, not assumed: on three RoboCasa tasks, eight IDENTICAL queries
# disagreed with each other by up to 110 degrees (floor cos_dir down to -0.34),
# so no prompt manipulation could be distinguished from the sampler and the
# grounding question could not be asked at all. GR00T-N1.6-3B is evaluated
# zero-shot on RoboCasa at 66.22% average across 24 tasks. Competence on the
# benchmark is a PRECONDITION for the measurement, not a nice-to-have.
#
# EMBODIMENT TAG IS LOAD-BEARING. ROBOCASA_PANDA_OMRON selects the state and
# action heads for RoboCasa's mobile-base Panda. The wrong tag produces a model
# that runs, returns arrays of a plausible shape, and means nothing.
#
# TWO SERVERS, NEVER AT ONCE. openpi's JAX server preallocates most of the GPU,
# and with it running torch here fails a 512x512 matmul with
# "CUDA error: out of memory" on a 121 GB machine. Stop one before starting the
# other:  pkill -f "[s]erve_policy.py"
# The bracket is not decoration -- `pkill -f serve_policy.py` matches its own
# command line and kills the calling shell.
#
# THE GB10 TRITON WORKAROUND DOES NOT APPLY HERE. That flag is XLA-specific;
# this is torch 2.9 + cu128, which handles bf16 on sm_121 natively (verified).
#
# Run:
#   bash scripts/serve_groot.sh                 # foreground, Ctrl-C to stop
#   bash scripts/serve_groot.sh --port 5556
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GROOT="$ROOT/third_party/Isaac-GR00T"

[ -d "$GROOT/.venv" ] || {
    echo "Isaac-GR00T venv missing. Build it with:" >&2
    echo "  cd $GROOT && uv sync" >&2
    exit 1
}

if pgrep -f "[s]erve_policy.py" >/dev/null; then
    echo "openpi's pi-0.5 server is running and will hold the GPU." >&2
    echo "Stop it first:  pkill -f \"[s]erve_policy.py\"" >&2
    exit 1
fi

# CALL THE VENV'S PYTHON DIRECTLY, NOT `uv run`.
# `uv run` re-resolves the project's dependencies before executing, and on this
# box that resolution fails: n1.6.1-release pins flash-attn 2.7.4.post1, which
# publishes no aarch64 wheel, so uv tries to compile it and the build aborts
# because the system toolkit is CUDA 13.0 while torch is cu128
# ("The detected CUDA version (13.0) mismatches ... PyTorch (12.8)").
#
# The venv itself is fine -- it carries torch 2.9 + flash-attn 2.8.3 from the
# `main` resolution, both with real aarch64 wheels, and bf16 on sm_121 is
# verified. gr00t is installed editable, so it already resolves to the
# n1.6.1-release source tree. Running the interpreter directly uses that
# working environment and skips the resolution that cannot succeed here.
#
# CONSEQUENCE, STATED SO IT IS NOT DISCOVERED LATER: the N1.6 model code is
# running against transformers 4.57.3 and torch 2.9, not the 4.51.3 / 2.7.1 it
# pins. `Gr00tN1d6Config` imports cleanly under them, but if inference produces
# something structurally odd, this mismatch is the first place to look.
cd "$GROOT"
# MODEL is overridable so a fine-tuned checkpoint can be served with the same
# script the baseline was measured through -- same server, same client, same
# probe. Evaluating a fine-tune against a differently-served baseline would
# confound the comparison with the serving path.
#   MODEL=checkpoints/groot_overlay/checkpoint-5000 bash scripts/serve_groot.sh
exec ./.venv/bin/python gr00t/eval/run_gr00t_server.py \
    --model-path "${MODEL:-nvidia/GR00T-N1.6-3B}" \
    --embodiment-tag ROBOCASA_PANDA_OMRON \
    "$@"
