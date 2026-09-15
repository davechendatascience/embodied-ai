#!/usr/bin/env bash
# The DINOv2 patch-token VLA, end to end: teacher demonstrations -> tokens -> train ->
# DAgger rounds -> randomized evaluation with a blind control.
#
#   scripts/token_vla.sh round0                      200 successful teacher episodes (20/task) -> cache/tokens/round0f
#   scripts/token_vla.sh dagger N DRIVER.pt [FLAGS]  round N driven by DRIVER at beta 0.5    -> cache/tokens/roundN
#   scripts/token_vla.sh train OUT.pt ROUNDS [FLAGS] sighted model + OUT_blind.pt (images zeroed)
#   scripts/token_vla.sh eval CKPT.pt EPISODES [FLAGS]  seed 555, VLA drives alone, ledger trials in runs/evidence/
#
# FLAGS pass through, e.g. --gripper-target (target-aperture gripper, screwhead/gripper_servo.py).
# Results so far (seed 555): r0 80/100, r1 68/100, r2 165/200 (blind r2 9/200); the drawer task
# fails without the gripper servo. Train one model at a time: the GPU shares host memory.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 PYTHONPATH=third_party/LIBERO:. MUJOCO_GL=egl
PY=.venv-libero/bin/python
RAND="--horizon 400 --layout-radius 0.08 --start-xy 0.10 --start-z 0.05 --start-yaw 30 --start-tilt 10 --start-null 0.3"

case "${1:-}" in
  round0)
    $PY tools/collect_scripted.py collect --out cache/distill_scripted/round0f_shards --successes 20 --workers-per-task 2
    $PY tools/token_data.py encode --source cache/distill_scripted/round0f_shards --out cache/tokens/round0f
    $PY tools/relabel_gripper.py --tokens cache/tokens/round0f ;;
  dagger)
    n=$2; driver=$3; shift 3
    $PY tools/distill.py collect --teacher scripted --student "$driver" --beta 0.5 --episodes 20 $RAND \
      --seed $((299 + n)) --save-frames --out cache/distill_scripted/tok_round$n.npz "$@"
    $PY tools/token_data.py encode --source cache/distill_scripted/tok_round$n.npz --out cache/tokens/round$n
    $PY tools/relabel_gripper.py --tokens cache/tokens/round$n --phases cache/distill_scripted/tok_round$n.npz ;;
  train)
    out=$2; rounds=$3; shift 3
    data=$(for r in $rounds; do echo -n "cache/tokens/$r "; done)
    $PY tools/token_data.py train --data $data --zero none --out "$out" "$@"
    $PY tools/token_data.py train --data $data --zero image --out "${out%.pt}_blind.pt" "$@" ;;
  eval)
    ckpt=$2; eps=$3; shift 3
    zero=none; [[ "$ckpt" == *_blind.pt || "$ckpt" == *_image.pt ]] && zero=image
    $PY tools/distill.py collect --teacher scripted --student "$ckpt" --zero $zero --beta 0 --episodes "$eps" $RAND \
      --seed 555 --trials "runs/evidence/$(basename "${ckpt%.pt}")_s555.trials.json" "$@" ;;
  *) sed -n '2,13p' "$0"; exit 2 ;;
esac
