#!/usr/bin/env bash
# Bring the service up in .venv-pw, run the round-trip test in .venv, tear down.
#
# The orchestration is here rather than inside either test because neither venv
# may reach into the other. Each python process below sees only its own
# interpreter; they meet at a socket path and nowhere else.
#
#     scripts/run_bridge_test.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SOCK="${SOCK:-/tmp/pointworld_bridge_test.sock}"
LOG="${LOG:-/tmp/pointworld_bridge_test.log}"

cleanup() {
    if [[ -n "${SRV_PID:-}" ]] && kill -0 "$SRV_PID" 2>/dev/null; then
        kill "$SRV_PID" 2>/dev/null
        wait "$SRV_PID" 2>/dev/null
    fi
    rm -f "$SOCK"
}
trap cleanup EXIT

rm -f "$SOCK"
scripts/pointworld_serve.sh --socket "$SOCK" >"$LOG" 2>&1 &
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
if [[ ! -S "$SOCK" ]]; then
    echo "service did not bind $SOCK in 120s" >&2
    tail -20 "$LOG" >&2
    exit 1
fi

PYTHONPATH=src .venv/bin/python tests/test_bridge_roundtrip.py --socket "$SOCK"
STATUS=$?
echo
echo "--- service log ---"
tail -5 "$LOG"
exit $STATUS
