#!/usr/bin/env bash
# Robotiq85 pairs across four suites. The existing 450 PandaGripper pairs carry
# no `geom` column, which featurise() reads as "Panda geometry" -- a genuine
# zero, not a gap. So these are the contrast half of the same dataset, and the
# corrector sees the depth difference vary instead of as an unlearnable constant.
set -u
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl PYTHONPATH=.
P=.venv/bin/python

run () {  # suite init tag [traj]
    local suite=$1 init=$2 tag=$3 traj=${4:-}
    local args=(--suite "$suite" --task-id 0 --init-state "$init" --n 40
                --gripper Robotiq85Gripper --out "pairs/robotiq85/r_${tag}.npz")
    [ -n "$traj" ] && args+=(--traj "$traj")
    echo "### $suite init$init -> pairs/robotiq85/r_${tag}.npz"
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
    echo "### libero_object init$i -> pairs/robotiq85/r_o${i}.npz"
    $P examples/collect_pairs.py --suite libero_object --task-id 0 \
        --init-state $i --n 40 --gripper Robotiq85Gripper \
        --trace "$TRACE" --out "pairs/robotiq85/r_o${i}.npz" 2>&1 |
        grep -E "^pairs|^  "
done
echo "### done"
