#!/usr/bin/env bash
# The DINOv2 patch-token VLA, end to end, with ONE execution path throughout: the arm through
# TwistServo, the gripper as a target aperture through GripperServo, and the student's gripper
# decode (snap levels) stored in its checkpoint -- so teacher demonstrations, DAgger rollouts
# and evaluation all execute actions identically.
#
#   scripts/token_vla.sh round0                    teacher demos through the servo, 20 per task
#   scripts/token_vla.sh dagger N DRIVER.pt        round N driven by DRIVER at beta 0.5
#   scripts/token_vla.sh train OUT.pt "R0 R1 .."   sighted OUT.pt and OUT_blind.pt (images zeroed)
#   scripts/token_vla.sh eval CKPT.pt EPS [FLAGS]  seed 555, VLA alone, trials in runs/evidence/
#   scripts/token_vla.sh all                       round0 -> r0 -> dagger 1 -> r1 -> dagger 2 -> r2 -> eval
#
# Every stage skips itself when its output exists, so `all` resumes after an interruption.
# Data: cache/tokens/gt_roundN. Checkpoints: checkpoints/vla_gt_rN.pt (+ _blind).
#
# Simulation workers run on the performance cores only. On this GB10 the efficiency cores
# measured ~4x slower per worker (70 vs 280 frames/min for the same 9 workers), and a task
# whose workers land there becomes the straggler every stage waits for.
# Train one model at a time: the GPU shares host memory.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 PYTHONPATH=third_party/LIBERO:. MUJOCO_GL=egl
PY=.venv-libero/bin/python
RAND="--horizon 400 --layout-radius 0.08 --start-xy 0.10 --start-z 0.05 --start-yaw 30 --start-tilt 10 --start-null 0.3"
LEVELS="0 0.026 0.08"          # the apertures the programs use: closed, drawer pre-shape, open
PERF=5,6,7,8,9,15,16,17,18,19

collected() { [[ -f cache/distill_scripted/gt_round$1.npz ]]; }
encoded()   { [[ -f cache/tokens/gt_round$1/meta.npz ]]; }

encode() {
  encoded "$1" || $PY tools/token_data.py encode --source cache/distill_scripted/gt_round$1.npz --out cache/tokens/gt_round$1
}
round0() {
  collected 0 || $PY tools/distill.py collect --teacher scripted --beta 1 --gripper-target --exec-noise 0.3 \
    --episodes 20 --cpus $PERF $RAND --seed 400 --save-frames --out cache/distill_scripted/gt_round0.npz
  encode 0
}
dagger() {
  local n=$1 driver=$2
  collected "$n" || $PY tools/distill.py collect --teacher scripted --student "$driver" --gripper-target --beta 0.5 \
    --episodes 20 --cpus $PERF $RAND --seed $((400 + n)) --save-frames --out cache/distill_scripted/gt_round$n.npz
  encode "$n"
}
train() {
  local out=$1 rounds=$2 data=""
  for r in $rounds; do data="$data cache/tokens/gt_round$r"; done
  [[ -f "$out" ]] || $PY tools/token_data.py train --data $data --gripper-target --gripper-levels $LEVELS --zero none --out "$out"
  [[ -f "${out%.pt}_blind.pt" ]] || \
    $PY tools/token_data.py train --data $data --gripper-target --gripper-levels $LEVELS --zero image --out "${out%.pt}_blind.pt"
}
evaluate() {
  local ckpt=$1 eps=$2; shift 2
  local zero=none; [[ "$ckpt" == *_blind.pt ]] && zero=image
  $PY tools/distill.py collect --teacher scripted --student "$ckpt" --gripper-target --zero $zero --beta 0 --episodes "$eps" $RAND \
    --cpus $PERF --seed 555 --trials "runs/evidence/$(basename "${ckpt%.pt}")_s555.trials.json" "$@"
}

case "${1:-}" in
  round0) round0 ;;
  dagger) dagger "$2" "$3" ;;
  train)  train "$2" "$3" ;;
  eval)   evaluate "${@:2}" ;;
  all)
    echo "[stage] round 0";   round0
    echo "[stage] train r0";  train checkpoints/vla_gt_r0.pt "0"
    echo "[stage] dagger 1";  dagger 1 checkpoints/vla_gt_r0.pt
    echo "[stage] train r1";  train checkpoints/vla_gt_r1.pt "0 1"
    echo "[stage] dagger 2";  dagger 2 checkpoints/vla_gt_r1.pt
    echo "[stage] train r2";  train checkpoints/vla_gt_r2.pt "0 1 2"
    echo "[stage] eval r2";       evaluate checkpoints/vla_gt_r2.pt 20
    echo "[stage] eval r2 blind"; evaluate checkpoints/vla_gt_r2_blind.pt 20
    echo "[stage] PIPELINE DONE" ;;
  *) sed -n '2,19p' "$0"; exit 2 ;;
esac
