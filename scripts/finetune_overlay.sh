#!/usr/bin/env bash
# Fine-tune GR00T N1.6 on box-overlaid RoboCasa. Point-VLA's recipe, our data.
#
# WHAT THIS IS TESTING. Neither prompt-serialised boxes nor pixel overlays steer
# an UNTRAINED policy (README). Every published overlay success fine-tuned on
# overlaid frames first, so the open question is whether training closes the gap
# -- and the probe measures the thing those papers do not report: whether MOVING
# the mark moves the robot.
#
# THE DATASET IS HALF OVERLAID, 1:1. Point-VLA co-trains overlaid against plain
# so the policy does not collapse into "always go to the mark" and lose language
# conditioning. `emit_lerobot.py` emits both variants from one replay, so they
# are pixel-identical apart from the drawn box.
#
# WAITS FOR EMISSION, THEN GENERATES STATS, THEN TRAINS. The three steps are
# chained here rather than run by hand because the middle one is easy to forget
# and its absence fails late: `LeRobotEpisodeLoader` ASSERTS meta/stats.json
# exists, so a missing stats file surfaces as a crash after model load.
#
# relative_stats.json IS NOT GENERATED and that is expected: GR00T's
# MODALITY_CONFIGS has entries for libero_panda, oxe_* and unitree_g1 but NOT
# robocasa_panda_omron, so the relative-stats step raises KeyError for this
# embodiment. The loader guards on the file's existence, so training proceeds.
#
# BATCH SIZE IS DELIBERATELY SMALL. Defaults are tuned for datacentre GPUs; this
# is a GB10 with unified memory shared with the desktop. Raise it only after
# watching a few hundred steps without an OOM.
#
# CHECKPOINTS EVERY 1000 STEPS, so an unattended run that dies at step 7000 has
# not lost the weekend.
#
# Run:
#   bash scripts/finetune_overlay.sh                  # waits for any running emit
#   MAX_STEPS=2000 bash scripts/finetune_overlay.sh   # short smoke run
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GROOT="$ROOT/third_party/Isaac-GR00T"
DATASET="${DATASET:-$ROOT/data/robocasa_overlay}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/checkpoints/groot_overlay}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.6-3B}"
# The enum NAME, not its value: launch_finetune.py takes
# ROBOCASA_PANDA_OMRON and rejects robocasa_panda_omron, while
# gr00t.data.stats takes the same uppercase form. serve_policy.py
# accepts either, which is why the lowercase form looked right.
TAG="${TAG:-ROBOCASA_PANDA_OMRON}"
export MAX_STEPS="${MAX_STEPS:-10000}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
export USE_WANDB="${USE_WANDB:-0}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"

echo "== waiting for any running emission =="
while pgrep -f "[e]mit_lerobot.py" >/dev/null; do sleep 30; done
echo "emission not running"

[ -d "$DATASET/data/chunk-000" ] || { echo "no dataset at $DATASET" >&2; exit 1; }
echo "episodes: $(ls "$DATASET"/data/chunk-000/*.parquet | wc -l)"

# The GPU cannot hold an inference server and a 3B fine-tune at once. The GR00T
# server is only needed to PROBE, and the probe runs after training.
if pgrep -f "[r]un_gr00t_server.py" >/dev/null; then
    echo "stopping the GR00T inference server to free the GPU"
    pkill -f "[r]un_gr00t_server.py"; sleep 5
fi

echo "== stats (required; loader asserts on it) =="
"$GROOT/.venv/bin/python" -m gr00t.data.stats --dataset-path "$DATASET" \
    --embodiment-tag ROBOCASA_PANDA_OMRON 2>&1 | tail -3 || true
[ -f "$DATASET/meta/stats.json" ] || { echo "stats.json missing" >&2; exit 1; }
echo "stats.json ok"

echo "== finetune =="
cd "$GROOT"
# PATH FIRST, THEN activate_spark.sh. THE ORDER IS LOAD-BEARING.
# activate_spark.sh resolves torch's shared libraries with the AMBIENT python3:
#   TORCH_LIB_DIR="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
# and in this repo the ambient python3 resolves to the LIBERO venv, which holds
# torch 2.7.1+cpu. Sourcing it first pointed LD_LIBRARY_PATH at that CPU
# libtorch, and GR00T's torch 2.9.0+cu128 then loaded the wrong shared objects
# and died on import with
#   AttributeError: module 'torch._C' has no attribute 'AcceleratorError'
# which reads like a broken install and is really one venv reaching into
# another. Putting the GR00T venv on PATH first makes activate_spark.sh resolve
# its own torch.
#
# `uv run` is avoided for a separate reason: it re-resolves and tries to build
# flash-attn 2.7.4.post1, which has no aarch64 wheel -- see scripts/serve_groot.sh.
export PATH="$GROOT/.venv/bin:$PATH"
unset LD_LIBRARY_PATH
# shellcheck disable=SC1091
source scripts/activate_spark.sh 2>/dev/null || true

# Assert the interpreter that will train is the one we think it is. Cross-venv
# contamination is silent until it is fatal, and it already cost one launch.
"$GROOT/.venv/bin/python" - <<'ASSERT'
import torch, sys
assert torch.__version__.startswith("2.9"), f"wrong torch: {torch.__version__}"
assert torch.cuda.is_available(), "cuda unavailable in the training venv"
print(f"training venv OK: torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
ASSERT
# --modality-config-path is REQUIRED for this embodiment, not optional.
# n1.6.1-release ships no robocasa_panda_omron entry in FinetuneConfig's
# modality_configs, so validate() raises KeyError before training starts. The
# file registers it, read out of the checkpoint's own processor_config.json so
# the fine-tune uses the contract the model was pretrained with.
exec bash examples/finetune.sh \
    --base-model-path "$BASE_MODEL" \
    --dataset-path "$DATASET" \
    --embodiment-tag "$TAG" \
    --output-dir "$OUTPUT_DIR" \
    --modality-config-path "$ROOT/configs/robocasa_modality_config.py"
