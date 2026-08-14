#!/usr/bin/env bash
# Build .venv-robocasa: RoboCasa on robosuite master, python 3.11.
#
# WHY A THIRD VENV AND NOT A FLAG.
# RoboCasa requires robosuite MASTER (1.5.x). LIBERO requires robosuite 1.4.1,
# because it imports `robosuite.environments.manipulation.single_arm_env`, which
# 1.5 deleted. There is no version that satisfies both, and there is no import
# order that hides it. So: LIBERO owns .venv, RoboCasa owns .venv-robocasa,
# openpi owns third_party/openpi/.venv, and the three talk to each other only
# through a websocket carrying arrays. Orchestration lives in shell, never in
# a python path that reaches across them.
#
# PYTHON IS 3.11 HERE, 3.12 IN .venv. RoboCasa's README specifies 3.11 and its
# numba/asset pipeline is what pins it. uv fetches the interpreter, so this does
# not require a system python3.11.
#
# ASSETS ARE ~10 GB and are NOT in git (third_party/ is ignored). They are
# fetched by robocasa's own script, which is also the only supported way to get
# the kitchen object meshes and textures.
#
# COMMITS ARE RECORDED, NOT PINNED. Both upstreams track moving branches; a
# probe whose object names or scene layouts shift underneath it is not a probe.
# The hashes land in third_party/*.commit. Pin them once a run is trusted.
#
# Run from the repo root:
#   bash scripts/setup_robocasa.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv-robocasa"
RS_DIR="$ROOT/third_party/robosuite_master"
RC_DIR="$ROOT/third_party/robocasa"

command -v uv >/dev/null || { echo "uv not on PATH" >&2; exit 1; }

echo "== venv (python 3.11) =="
[ -d "$VENV" ] || uv venv --python 3.11 "$VENV"
PIP() { VIRTUAL_ENV="$VENV" uv pip install --python "$VENV/bin/python" "$@"; }

echo "== checkouts =="
mkdir -p "$ROOT/third_party"
[ -d "$RS_DIR/.git" ] || git clone https://github.com/ARISE-Initiative/robosuite "$RS_DIR"
[ -d "$RC_DIR/.git" ] || git clone https://github.com/robocasa/robocasa "$RC_DIR"
git -C "$RS_DIR" rev-parse HEAD > "$ROOT/third_party/robosuite_master.commit"
git -C "$RC_DIR" rev-parse HEAD > "$ROOT/third_party/robocasa.commit"
echo "robosuite @ $(cat "$ROOT/third_party/robosuite_master.commit")"
echo "robocasa  @ $(cat "$ROOT/third_party/robocasa.commit")"

echo "== robosuite (master) =="
PIP -e "$RS_DIR"

echo "== robocasa =="
PIP -e "$RC_DIR"

# The probe talks to pi-0.5 over a websocket. Pure python, no framework, so it
# is safe in every venv -- this is the ONE thing all three share.
#
# --no-deps IS REQUIRED HERE, AND IT IS A REAL CONFLICT, NOT A SHORTCUT.
# robocasa/__init__.py asserts `numpy.__version__ == "2.2.5"` and refuses to
# import otherwise. openpi-client's metadata declares `numpy>=1.22.4,<2.0.0`,
# inherited from openpi's jax stack. Installing it normally downgrades numpy and
# RoboCasa stops importing; pinning numpy back breaks the client's declared
# constraint. The constraint is the thing that is wrong: the client's own code
# is msgpack, websockets and pillow, and touches numpy only to pack and unpack
# arrays. So its transitive deps are installed explicitly and numpy is left at
# the version RoboCasa demands -- then asserted, because an unasserted
# workaround is a bug waiting for a reinstall.
echo "== openpi-client (--no-deps; see comment) =="
PIP --no-deps "$ROOT/third_party/openpi/packages/openpi-client"
PIP msgpack websockets pillow typing_extensions

echo "== macros =="
"$VENV/bin/python" -m robocasa.scripts.setup_macros || true

echo "== kitchen assets (~10 GB) =="
if [ "${SKIP_ASSETS:-0}" = "1" ]; then
    echo "SKIP_ASSETS=1, skipping"
else
    # `yes |` feeds the downloader's confirmation prompts. When it finishes and
    # stops reading, `yes` takes SIGPIPE and exits 141 -- and under `pipefail`
    # that becomes the pipeline's status, aborting a run whose 23 GB of assets
    # downloaded and extracted perfectly. Disable pipefail for this line only:
    # the download's real success is asserted below by importing the env.
    set +o pipefail
    yes | "$VENV/bin/python" -m robocasa.scripts.download_kitchen_assets
    set -o pipefail
fi

echo "== assert =="
MUJOCO_GL=egl "$VENV/bin/python" - <<'PY'
import numpy
# RoboCasa asserts this itself on import; asserting it here too means a bad
# install fails the BUILD rather than the first experiment, and names the cause.
assert numpy.__version__ == "2.2.5", (
    f"numpy is {numpy.__version__}, RoboCasa requires exactly 2.2.5 -- "
    "something re-resolved it, most likely openpi-client without --no-deps")
import robosuite, robocasa
print("robosuite", robosuite.__version__, "| numpy", numpy.__version__)
# The import the LIBERO venv CANNOT do, and the reason these are separate.
import robocasa.environments.kitchen.kitchen as _k
# The client must survive numpy 2 despite declaring <2. Round-trip an array to
# prove the packer works, rather than trusting that it does.
from openpi_client import msgpack_numpy
a = numpy.arange(6, dtype=numpy.float32).reshape(2, 3)
b = msgpack_numpy.unpackb(msgpack_numpy.Packer().pack({"x": a}))["x"]
assert numpy.array_equal(a, b) and a.dtype == b.dtype, "msgpack round trip failed"
print("robocasa OK; openpi-client msgpack round-trip OK under numpy 2")
PY

echo
echo "next:"
echo "  MUJOCO_GL=egl PYTHONPATH=. ./.venv-robocasa/bin/python \\"
echo "      examples/probe_robocasa.py --task PnPCounterToCab"
