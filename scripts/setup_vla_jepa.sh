#!/usr/bin/env bash
# VLA-JEPA (ECCV 2026, arXiv 2602.10098) on aarch64 / CUDA 13 / Blackwell.
#
# VLA-JEPA ALREADY SPLITS POLICY FROM SIMULATOR, over a websocket:
#
#   deployment/model_server/server_policy.py   the policy   -> .venv-jepa
#   examples/LIBERO/eval_libero.py             the LIBERO sim -> $sim_python
#
# Their own eval script points `sim_python` at a separate interpreter. That is
# the same two-venv shape this repo already uses, so `.venv-jepa` needs the
# POLICY deps only -- no mujoco, no robosuite, no LIBERO. The sim side is our
# existing `.venv`, which already runs LIBERO on mujoco 3.1.6 and is where the
# E1 trace gets recorded.
#
# That split also disposes of the scary half of requirements.txt: decord,
# eva-decord, pipablepytorch3d, deepspeed and av are training-time video and
# data-loader deps, none of which the policy server imports. They are the ones
# with no aarch64/py3.12 wheels, and skipping them is not a compromise.
#
# THE CHECKPOINT IS NOT SELF-CONTAINED. `checkpoints_hf/LIBERO/config.yaml`
# carries ABSOLUTE paths from the authors' machine:
#     base_vlm:     /home/dataset-local/models/Qwen3-VL-2B-Instruct
#     base_encoder: /home/dataset-local/models/vjepa2-vitl-fpc64-256
# Both must be fetched separately and the paths rewritten, or the server dies
# loading a directory that does not exist. Note the README says 2B while
# QwenOFT.py defaults to 4B -- the CONFIG is authoritative, and it says 2B.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv-jepa"
REPO="$ROOT/third_party/VLA_JEPA"
MODELS="$REPO/base_models"
INDEX="https://pypi.jetson-ai-lab.io/sbsa/cu130/+simple"

[ -d "$REPO" ] || git clone --depth 1 https://github.com/ginwind/VLA-JEPA.git "$REPO"

echo "==> venv"
[ -d "$VENV" ] || python3.12 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip wheel setuptools

# TORCH IS PINNED TO 2.11.0, and the pin is load bearing.
#
# `starVLA/model/modules/vlm/QWen3.py:60` hardcodes
# `attn_implementation="flash_attention_2"` -- not read from config, so it
# cannot be turned off without patching third_party. flash-attn therefore has
# to import, and the only aarch64/cu130 wheel that exists (2.8.4, same one
# .venv-pw uses) is built against torch 2.11. On torch 2.13 it dies with:
#
#   ImportError: flash_attn_2_cuda...so: undefined symbol:
#   _ZN3c104impl3cow23materialize_cow_storageERNS_11StorageImplE
#
# Pinning torch keeps third_party pristine AND preserves the authors' attention
# numerics, which matters because we intend to quote their LIBERO number. The
# repo's own torchvision==0.21.0 pin implies torch 2.6 on x86 and cannot be
# honoured on Blackwell at all; it is ignored.
echo "==> torch 2.11.0 + flash-attn (prebuilt aarch64/cu130 -- see the note above)"
"$VENV/bin/pip" install --index-url "$INDEX" --extra-index-url https://pypi.org/simple \
    "torch==2.11.0" torchvision flash-attn

echo "==> policy-server deps"
# transformers 4.57 is the floor for Qwen3-VL. numpy is pinned to 1.26.4 to
# match the repo; torch 2.13 is fine with it (so is .venv).
"$VENV/bin/pip" install \
    "transformers==4.57.0" accelerate safetensors tokenizers \
    qwen-vl-utils timm einops omegaconf "pydantic==2.10.6" numpydantic \
    "numpy==1.26.4" pillow scipy rich tiktoken transformers_stream_generator \
    websockets websocket-client msgpack "huggingface_hub" diffusers
# diffusers backs the DiT-B action head. Without it every framework submodule
# fails to import -- and you will NOT see that, because
# starVLA/model/framework/__init__.py catches the ImportError and then dies in
# its own handler (`logger.log` does not exist on PureOverwatch), so the real
# `ModuleNotFoundError: diffusers` never reaches you. Another swallowed
# exception that reports as something unrelated; NOTES.md §1's rule again.

echo "==> base models (NOT in the checkpoint; ~5.6 GB)"
mkdir -p "$MODELS"
for spec in "Qwen/Qwen3-VL-2B-Instruct:Qwen3-VL-2B-Instruct" \
            "facebook/vjepa2-vitl-fpc64-256:vjepa2-vitl-fpc64-256"; do
    repo="${spec%%:*}"; dir="${spec##*:}"
    if [ -d "$MODELS/$dir" ] && [ -n "$(ls -A "$MODELS/$dir" 2>/dev/null)" ]; then
        echo "    $dir present, skipping."
    else
        "$VENV/bin/hf" download "$repo" --local-dir "$MODELS/$dir"
    fi
done

echo "==> rewriting the authors' absolute paths in the checkpoint config"
"$VENV/bin/python" - "$REPO" <<'PY'
import json, pathlib, sys, re
repo = pathlib.Path(sys.argv[1])
models = repo / "base_models"
sub = {
    "/home/dataset-local/models/Qwen3-VL-2B-Instruct": str(models / "Qwen3-VL-2B-Instruct"),
    "/home/dataset-local/models/vjepa2-vitl-fpc64-256": str(models / "vjepa2-vitl-fpc64-256"),
}
for name in ("config.yaml", "config.json"):
    p = repo / "checkpoints_hf" / "LIBERO" / name
    if not p.exists():
        print(f"    {name}: missing, skipped")
        continue
    text = p.read_text()
    orig = text
    for old, new in sub.items():
        text = text.replace(old, new)
    if text != orig:
        bak = p.with_suffix(p.suffix + ".orig")
        if not bak.exists():
            bak.write_text(orig)          # keep the authors' file recoverable
        p.write_text(text)
        print(f"    {name}: rewrote {sum(orig.count(o) for o in sub)} path(s)")
    else:
        print(f"    {name}: already local")
PY

cat <<EOF

==> done. The policy server runs in .venv-jepa:

  $VENV/bin/python deployment/model_server/server_policy.py \\
      --ckpt_path $REPO/checkpoints_hf/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt \\
      --port 15084

...and the LIBERO sim side runs in .venv, which already has mujoco/robosuite.
Neither venv imports the other; the websocket is the whole interface.
EOF
