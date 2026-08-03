#!/usr/bin/env bash
# Closed-loop drawer opening, planned by PointWorld.
#
# Service in .venv-pw, planner in .venv, meeting at a socket and nowhere else.
# Neither python process may reference the other's interpreter -- orchestration
# lives here, in shell, which belongs to neither venv.
#
#     scripts/run_drawer_planner.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SOCK="${SOCK:-/tmp/pointworld_planner.sock}"
LOG="${LOG:-/tmp/pointworld_planner_service.log}"

cleanup() {
    if [[ -n "${SRV_PID:-}" ]] && kill -0 "$SRV_PID" 2>/dev/null; then
        kill "$SRV_PID" 2>/dev/null
        wait "$SRV_PID" 2>/dev/null
    fi
    rm -f "$SOCK"
}
trap cleanup EXIT

rm -f "$SOCK"
scripts/pointworld_serve.sh --socket "$SOCK" --max-batch 8 >"$LOG" 2>&1 &
SRV_PID=$!

echo "waiting for the service (log: $LOG)"
for _ in $(seq 1 240); do
    [[ -S "$SOCK" ]] && break
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
        echo "service died before binding the socket:" >&2
        tail -20 "$LOG" >&2
        exit 1
    fi
    sleep 0.5
done
[[ -S "$SOCK" ]] || { echo "service did not bind $SOCK" >&2; tail -20 "$LOG" >&2; exit 1; }

MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python -u tests/test_pointworld_grasp.py \
    --socket "$SOCK" "$@"
