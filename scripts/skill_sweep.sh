#!/usr/bin/env bash
# The geometry-driven teacher over LIBERO, one suite at a time.
#
#   scripts/skill_sweep.sh                      # the three 10-task suites, 5 episodes each
#   scripts/skill_sweep.sh 5 libero_10          # one suite
#   EPISODES=10 scripts/skill_sweep.sh 10 libero_90
#
# Writes runs/skill_<suite>.log and runs/evidence/skill_<suite>.trials.json (the ledger's
# CTR-skill-teacher-solves-unseen), then prints every task that is not perfect -- the
# teacher labels the data, so anything short of 100% is a hole in the demonstrations.
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HUB_OFFLINE=1 PYTHONPATH=third_party/LIBERO:. MUJOCO_GL=egl
PY=.venv-libero/bin/python
PERF=5,6,7,8,9,15,16,17,18,19          # the GB10's performance cores; 0-4,10-14 are ~4x slower
EPISODES=${1:-${EPISODES:-5}}
HORIZON=${HORIZON:-500}
SUITES=${*:2}
SUITES=${SUITES:-"libero_object libero_spatial libero_goal"}

mkdir -p runs/evidence
for S in $SUITES; do
  echo "=== $S ($EPISODES episodes/task)"
  $PY tools/skill_eval.py --suite "$S" --episodes "$EPISODES" --horizon "$HORIZON" \
      --cpus "$PERF" --trials "runs/evidence/skill_${S}.trials.json" \
      > "runs/skill_${S}.log" 2>&1 || true
  grep -E "^  task [0-9]+: " "runs/skill_${S}.log" | grep -v "$EPISODES/$EPISODES" || echo "  (all tasks perfect)"
  grep -E "^${S}: " "runs/skill_${S}.log"
done
