#!/usr/bin/env bash
# Roll out both policies across all four cells of the arm x gripper factorial.
#
# Sequential on purpose. Each rollout holds an EGL context and a MuJoCo model,
# and two live LIBERO environments corrupt each other's context -- the symptom
# is a depth-buffer assertion many calls later, nowhere near the cause.
#
# ~13 min per cell measured, so ~1h45 for the eight.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs videos

EPISODES=${EPISODES:-10}
STEPS=${STEPS:-400}

for POLICY in screwhead baseline; do
  for CELL in source arm_only gripper_only both; do
    OUT="runs/${POLICY}_${CELL}.json"
    if [ -s "$OUT" ]; then
      echo "== skip $POLICY/$CELL (already have $OUT)"
      continue
    fi
    echo "== $POLICY / $CELL  ->  $OUT"
    HF_HUB_OFFLINE=1 PYTHONPATH=third_party/LIBERO:. MUJOCO_GL=egl \
      .venv-libero/bin/python tools/rollout.py \
        --out "$OUT" --policy "$POLICY" --cell "$CELL" \
        --episodes-per-task "$EPISODES" --max-steps "$STEPS" \
        --video videos --video-every 5 \
      2>&1 | grep -viE "warning|^\[info\]|gym has been|upgrade to gymnasium|migration guide"
    # grep eats the exit status, so check the artifact instead of $?
    [ -s "$OUT" ] || echo "!! $POLICY/$CELL produced no output"
  done
done

echo
echo "== summary =="
.venv/bin/python - <<'PY'
import json, glob, os
rows=[]
for f in sorted(glob.glob("runs/*.json")):
    t=json.load(open(f))["trials"]
    if not t: continue
    ok=sum(x["metrics"]["success"] for x in t)
    c=t[0]["conditions"]
    rows.append((os.path.basename(f)[:-5], c["policy_revision"], c["robot"], c["gripper"],
                 c["dof"], ok, len(t)))
print(f'{"run":28s} {"arm":7s} {"gripper":16s} {"dof":>3s} {"success":>9s}  rate')
for n,p,r,g,d,ok,tot in rows:
    print(f'{n:28s} {r:7s} {g:16s} {d:3d} {ok:5d}/{tot:3d}  {ok/tot:.2f}')
PY
