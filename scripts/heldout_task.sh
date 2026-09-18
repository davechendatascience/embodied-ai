#!/usr/bin/env bash
# Train on the other tasks, evaluate on the held-out ones (belief.yaml: CTR-held-out-task).
#   scripts/heldout_task.sh "2 5 9"      -> checkpoints/vla_ho_<ids>.pt, 20 episodes per held-out task
# The data, teacher, gripper and decode are the gate-2 ones; the only change is which tasks
# the student ever saw. Bar 0.3: blind-level 0.05 is refuted, a true 0.5 supported.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 PYTHONPATH=third_party/LIBERO:. MUJOCO_GL=egl
PY=.venv-libero/bin/python
HO=${1:?held-out task ids, e.g. "2 5 9"}
IDS=$(echo "$HO" | tr -d ' ')
CKPT=checkpoints/vla_ho_$IDS.pt
DATA="cache/tokens/gt_v05_round0 cache/tokens/gc_v05_round1 cache/tokens/gc_v05_round2 cache/tokens/gc_v05_round3"
F="--gripper-target --gripper-classes --gripper-levels 0 0.026 0.08 --gripper-weight 0.1 --select twist"
RAND="--horizon 400 --layout-radius 0.08 --start-xy 0.10 --start-z 0.05 --start-yaw 30 --start-tilt 10 --start-null 0.3"
[[ -f $CKPT ]] || $PY tools/token_data.py train --data $DATA $F --exclude-tasks $HO --out $CKPT
$PY tools/distill.py collect --teacher scripted --student $CKPT --gripper-target --teacher-v-min 0.05 --beta 0 \
  --tasks $HO --episodes 20 $RAND --cpus 5,6,7,8,9,15,16,17,18,19 --seed 555 --heldout-tasks "$IDS" \
  --trials runs/evidence/$(basename ${CKPT%.pt})_s555.trials.json
