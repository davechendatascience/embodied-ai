#!/usr/bin/env bash
# Pairs for ONE target gripper across four suites.
#   usage: collect_gripper_pairs.sh <GripperName> <outdir>
#
# The existing 450 PandaGripper pairs carry no `geom` column, which featurise()
# reads as "Panda geometry" -- a genuine zero, not a gap. Collecting a second
# and third gripper gives the conditioning column real spread
# (Panda 0 mm, Rethink 12 mm, Robotiq85 48 mm from the training geometry)
# instead of a single step it can only memorise as a bias.
set -u
GRIPPER=${1:?usage: collect_gripper_pairs.sh <GripperName> <outdir>}
OUT=${2:?usage: collect_gripper_pairs.sh <GripperName> <outdir>}
cd "$(dirname "$0")/.."
mkdir -p "$OUT"
export MUJOCO_GL=egl PYTHONPATH=.
P=.venv/bin/python

run () {  # suite init tag [traj]
    local suite=$1 init=$2 tag=$3 traj=${4:-}
    local args=(--suite "$suite" --task-id 0 --init-state "$init" --n 40
                --gripper "$GRIPPER" --out "$OUT/p_${tag}.npz")
    [ -n "$traj" ] && args+=(--traj "$traj")
    echo "### $suite init$init -> $OUT/p_${tag}.npz"
    $P examples/collect_pairs.py "${args[@]}" 2>&1 | grep -E "^pairs|^  "
}

for i in 0 1 2; do
    run libero_spatial $i "s$i" "pairs/traj/traj_libero_spatial_Panda_raw_init${i}.npy"
done
for i in 5 6 7; do
    run libero_goal $i "g$i" "pairs/traj/traj_libero_goal_Panda_raw_init${i}.npy"
done
for i in 0 1 2; do
    run libero_10 $i "t$i" "pairs/traj/traj_libero_10_Panda_raw_init${i}.npy"
done
for i in 0 1 2; do
    # No Panda traj was saved for libero_object, so fall back to the recorded
    # jsonl trace -- and match the EPISODE to the init state, since the default
    # trace is ep0 and would otherwise label all three inits with ep0's poses.
    TRACE=/tmp/vla_jepa_obj/libero_object_task0_ep${i}.jsonl
    echo "### libero_object init$i -> $OUT/p_o${i}.npz"
    $P examples/collect_pairs.py --suite libero_object --task-id 0 \
        --init-state $i --n 40 --gripper "$GRIPPER" \
        --trace "$TRACE" --out "$OUT/p_o${i}.npz" 2>&1 |
        grep -E "^pairs|^  "
done
echo "### done"
