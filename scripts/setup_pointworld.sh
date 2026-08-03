#!/usr/bin/env bash
# Port PointWorld to aarch64 / CUDA 13 / Blackwell (GB10, sm_121).
#
# PointWorld ships pinned to torch 2.5.1 + cu124 on x86_64. CUDA 12.4 predates
# Blackwell, so that toolchain cannot generate code for this GPU at all. This
# rebuilds the stack against CUDA 13 and sm_120.
#
# Runs in its OWN venv (.venv-pw). Do not merge it with .venv -- that one is
# pinned for ReKep/LIBERO (numpy 1.26.4, mujoco 3.1.6) and currently works.
#
# WHY sm_120 AND NOT sm_121:
#   sm_120 and sm_121 are binary compatible, so a build targeting 12.0 runs on
#   the GB10. This is also why stock torch works here despite its arch list
#   stopping at sm_120.
#
# THE FOUR THINGS THAT BITE, in the order they bite:
#   1. --no-build-isolation skips build deps, so pccm must be installed FIRST
#      or cumm's setup.py dies with ModuleNotFoundError: pccm.
#   2. spconv 2.3.8 pins cumm>=0.7.11,<0.8.0 while cumm master is 0.8.2. Build
#      cumm at tag v0.7.13.
#   3. CUMM_CUDA_ARCH_LIST is read at IMPORT time, not install time, because the
#      editable install compiles kernels via JIT on first use. It must be set in
#      whatever process first imports spconv, not just during pip install.
#   4. cumm 0.7.13's arch table predates Blackwell and rejects "12.0", and pccm
#      defaults to c++14 while CUDA 13's libcu++ hard-requires c++17. Both are
#      patched below.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv-pw"
TP="$ROOT/third_party"
INDEX="https://pypi.jetson-ai-lab.io/sbsa/cu130/+simple"

export CUMM_CUDA_VERSION="13.0"
export CUMM_CUDA_ARCH_LIST="12.0"

echo "==> venv (python3.12; urdfpy is NOT needed for inference, so 3.10 is not required)"
[ -d "$VENV" ] || python3.12 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip wheel setuptools

# The sbsa/cu130 index carries prebuilt aarch64 CUDA-13 wheels, including
# flash-attn -- which would otherwise be a multi-hour compile, or an attention
# rewrite, since ptv3.py imports flash_attn_varlen_qkvpacked_func at module
# scope with no fallback.
echo "==> torch + torchvision + flash-attn (prebuilt aarch64/cu130)"
"$VENV/bin/pip" install --index-url "$INDEX" --extra-index-url https://pypi.org/simple \
    torch==2.11.0 torchvision flash-attn

echo "==> build deps (see bite #1)"
"$VENV/bin/pip" install -q pccm ccimport pybind11 fire

echo "==> torch-scatter from source"
TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=8 FORCE_CUDA=1 \
    "$VENV/bin/pip" install --no-build-isolation torch-scatter

echo "==> cumm at v0.7.13 (see bite #2)"
[ -d "$TP/cumm" ] || git clone https://github.com/FindDefinition/cumm "$TP/cumm"
git -C "$TP/cumm" fetch --tags -q
git -C "$TP/cumm" checkout -q v0.7.13

# bite #4a: teach cumm about Blackwell
python3 - "$TP/cumm/cumm/common.py" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
if "'Blackwell'" not in s:
    s = s.replace("        ('Lovelace', '8.9+PTX'),\n    ])",
                  "        ('Lovelace', '8.9+PTX'),\n"
                  "        ('Blackwell', '10.0;12.0+PTX'),\n    ])")
    s = s.replace("'8.6', '8.9', '9.0'\n    ]",
                  "'8.6', '8.9', '9.0',\n        '10.0', '11.0', '12.0', '12.1',\n    ]")
    open(p, "w").write(s)
    print("    patched cumm arch table for Blackwell")
else:
    print("    cumm arch table already patched")
PY
(cd "$TP/cumm" && "$VENV/bin/pip" install -e . --no-build-isolation)

echo "==> spconv 2.3.8 from source"
[ -d "$TP/spconv" ] || git clone --depth 1 https://github.com/traveller59/spconv "$TP/spconv"

# bite #4b: CUDA 13's libcu++ requires c++17; pccm defaults to c++14
python3 - "$TP/spconv/spconv/build.py" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
if 'std="c++17"' not in s:
    s = s.replace("                              namespace_root=PACKAGE_ROOT,\n"
                  "                              load_library=False,",
                  "                              namespace_root=PACKAGE_ROOT,\n"
                  '                              std="c++17",\n'
                  "                              load_library=False,")
    open(p, "w").write(s)
    print("    patched spconv to build with c++17")
else:
    print("    spconv already patched")
PY
# bite #5: spconv reads the GPU capability as (12,1), finds no prebuilt kernels
# for it, and silently falls back to NVRTC -- which then fails on CUDA 13
# because libcu++ moved to targets/<arch>/include/cccl and NVRTC is not given
# that path ("cannot open source file cuda/std/cassert"). sm_120 kernels run
# natively on sm_121, so map the capability and the AOT kernels are used.
python3 - "$TP/spconv/spconv/pytorch/cppcore.py" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
if "arch == (12, 1)" not in s:
    s = s.replace(
        "    arch = torch.cuda.get_device_capability()\n",
        "    arch = torch.cuda.get_device_capability()\n"
        "    if arch == (12, 1) and (12, 0) in COMPILED_CUDA_ARCHS:\n"
        "        arch = (12, 0)   # sm_120/sm_121 are binary compatible\n", 1)
    open(p, "w").write(s)
    print("    patched spconv sm_121 -> sm_120 arch mapping")
else:
    print("    spconv arch mapping already patched")
PY
(cd "$TP/spconv" && "$VENV/bin/pip" install -e . --no-build-isolation)

echo "==> checkpoint"
"$VENV/bin/pip" install -q huggingface_hub==0.26.2
"$VENV/bin/huggingface-cli" download nvidia/PointWorld_models \
    --local-dir "$TP/PointWorld/pretrained_checkpoints" \
    --include "small-droid/model-best.pt"

echo "==> verifying (first spconv use JIT-compiles ~783 kernels; this is slow ONCE)"
cd "$ROOT" && "$VENV/bin/python" - <<'PY'
import torch, flash_attn, torch_scatter
import spconv.pytorch as spconv
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func as fa
print("torch", torch.__version__, "| flash_attn", flash_attn.__version__,
      "| torch_scatter", torch_scatter.__version__)
qkv = torch.randn(64, 3, 4, 32, device="cuda", dtype=torch.float16)
cu = torch.tensor([0, 32, 64], device="cuda", dtype=torch.int32)
assert fa(qkv, cu, 32).isfinite().all(), "flash-attn"
N, C = 500, 128
idx = torch.cat([torch.zeros(N, 1, dtype=torch.int32),
                 torch.randint(0, 16, (N, 3), dtype=torch.int32)], 1).cuda()
x = spconv.SparseConvTensor(torch.randn(N, C).cuda(), idx, [16, 16, 16], 1)
y = spconv.SubMConv3d(C, C, kernel_size=3, bias=True, indice_key="t").cuda()(x)
assert y.features.isfinite().all(), "spconv"
print("ALL OK — flash-attn, torch-scatter and spconv all run on this GPU")
PY

cat <<'EOF'

Done. NOTE: export CUMM_CUDA_VERSION=13.0 and CUMM_CUDA_ARCH_LIST=12.0 in any
shell that imports spconv, until the JIT cache is warm (see bite #3).

STILL BLOCKED ON: DINOv3 weights, which are access-gated by Meta and supply
`scene_features`. Everything except prediction ACCURACY can be measured without
them -- stack validation, speed benchmarking, and the MuJoCo data pipeline.
EOF
