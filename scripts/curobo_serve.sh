#!/usr/bin/env bash
# Collision-aware IK service, in .venv-curobo. Neither venv imports the other;
# the UNIX socket is the whole interface. Lifecycle lives here, in shell.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
YML="${1:-$ROOT/configs/curobo/ur5e_pandagripper.yml}"
exec "$ROOT/.venv-curobo/bin/python" "$ROOT/src/curobo_bridge/server.py" "$YML"
