#!/usr/bin/env bash
# Hold the DGX Spark under a temperature ceiling by pausing the trainer.
#
# WHY PAUSING AND NOT A SMALLER BATCH.
# The obvious knobs change the experiment. Batch size changes the effective
# gradient and therefore the result; fewer steps changes what is being measured.
# SIGSTOP/SIGCONT changes only WALL-CLOCK: the same batches, in the same order,
# with the same optimiser state, just spread over more time. A run governed this
# way is bit-for-bit the run that would have happened uncooled.
#
# WHY NOT CLOCK OR POWER CAPS. They would be better -- lower clocks are more
# efficient than duty cycling -- but GB10 exposes no power limit through
# nvidia-smi (every field reads N/A) and `nvidia-smi -lgc` requires root:
#   "The current user does not have permission to change clocks"
# If you have sudo, prefer:  sudo nvidia-smi -lgc 300,2000
# and skip this script; it is the fallback for an unprivileged user.
#
# WHAT IT WATCHES. The GPU die is not the hot part. Measured under load: GPU 80C
# while the SoC/board ACPI zones sat at 88C, with SW power capping already
# engaged and SM clocks at 2483 of 3003 MHz. So the ceiling is applied to the
# HOTTEST thermal zone on the machine, GPU included, not to the GPU alone.
#
# Run:
#   nohup bash scripts/thermal_governor.sh > /tmp/thermal.log 2>&1 &
#   CEIL=82 FLOOR=76 bash scripts/thermal_governor.sh
set -uo pipefail

CEIL="${CEIL:-84}"      # pause above this
FLOOR="${FLOOR:-78}"    # resume below this (hysteresis prevents flapping)
PATTERN="${PATTERN:-[l]aunch_finetune.py}"
INTERVAL="${INTERVAL:-15}"

hottest() {
    local m=0 t
    for z in /sys/class/thermal/thermal_zone*/temp; do
        t=$(cat "$z" 2>/dev/null) || continue
        t=$((t / 1000))
        [ "$t" -gt "$m" ] && m=$t
    done
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | tr -dc '0-9')
    [ -n "$t" ] && [ "$t" -gt "$m" ] && m=$t
    echo "$m"
}

paused=0
echo "governor: ceiling ${CEIL}C, resume below ${FLOOR}C, watching '$PATTERN'"
while true; do
    pids=$(pgrep -f "$PATTERN" || true)
    if [ -z "$pids" ]; then
        # Never leave a stopped process behind if the trainer vanished.
        [ "$paused" = 1 ] && echo "trainer gone; nothing to resume"
        echo "no trainer matching '$PATTERN'; governor exiting"
        exit 0
    fi
    t=$(hottest)
    if [ "$paused" = 0 ] && [ "$t" -ge "$CEIL" ]; then
        # shellcheck disable=SC2086
        kill -STOP $pids 2>/dev/null && paused=1
        echo "$(date +%H:%M:%S) ${t}C >= ${CEIL}C -> PAUSED"
    elif [ "$paused" = 1 ] && [ "$t" -le "$FLOOR" ]; then
        # shellcheck disable=SC2086
        kill -CONT $pids 2>/dev/null && paused=0
        echo "$(date +%H:%M:%S) ${t}C <= ${FLOOR}C -> RESUMED"
    fi
    sleep "$INTERVAL"
done
