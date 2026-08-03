"""Record every pose the scripted drawer test ASKS for, and what it got.

On the UR5e that test fails without ever touching the drawer:

    pre     : requested [-0.029, -0.071, 1.033]  reached [-0.024, +0.012, 1.025]
    at grasp: requested [ 0.002, -0.143, 1.016]  reached [ 0.020, -0.110, 1.026]
    closed  : 0 gripper-drawer contacts

`_move_to_waypoint` descends locally with OSC, so a miss means only that the
LOCAL descent failed. It does not distinguish "the UR5e cannot reach this pose"
from "OSC could not get there from where it started". Those two have completely
different consequences: the first is a workspace limit and kills the task on
this embodiment, the second is exactly what a global planner exists to fix.

This dumps the requested poses so cuRobo's IK can answer that. No policy, no
perception -- a scripted task, identical scene, identical TCP -- so whatever
comes back is kinematics and nothing else.

    MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/dump_drawer_waypoints.py \
        --robot UR5e --out /tmp/drawer_waypoints_ur5e.json
"""

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tests"))

from rekep_libero import add_rekep_to_path  # noqa: E402

add_rekep_to_path()

from rekep_libero.environment_libero import ReKepLiberoEnv  # noqa: E402
from rekep_libero.frames import flange_from_obs, homo, inv  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--robot", default="UR5e")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    records = []
    original = ReKepLiberoEnv._move_to_waypoint

    def logging_move(self, target_pose, *a, **kw):
        # callers pass thresholds positionally as well as by keyword
        before = self.get_ee_pose()
        out = original(self, target_pose, *a, **kw)
        after = self.get_ee_pose()
        tp = np.asarray(target_pose, dtype=np.float64)
        records.append({
            "requested_pos": tp[:3].tolist(),
            "requested_quat_xyzw": tp[3:].tolist(),
            "start_pos": before[:3].tolist(),
            "start_quat_xyzw": before[3:].tolist(),
            "reached_pos": after[:3].tolist(),
            "error_mm": float(np.linalg.norm(after[:3] - tp[:3]) * 1000.0),
            "arm_qpos": self.get_arm_joint_postions().tolist(),
        })
        return out

    ReKepLiberoEnv._move_to_waypoint = logging_move
    try:
        import test_drawer_open

        sys.argv = ["test_drawer_open.py", "--robot", args.robot]
        test_drawer_open.main()
    finally:
        ReKepLiberoEnv._move_to_waypoint = original

    # T_world_base is needed to express the goals in the frame cuRobo plans in.
    # Rebuilt rather than captured mid-run: it is a constant (verified in
    # tests/test_frame_math.py, 0.00000 mm over 40 configurations).
    from rekep_libero.config import load_config

    cfg = load_config()
    ec = dict(cfg["env"])
    ec["bounds_min"] = cfg["workspace"]["bounds_min"]
    ec["bounds_max"] = cfg["workspace"]["bounds_max"]
    ec["interpolate_pos_step_size"] = cfg["main"]["interpolate_pos_step_size"]
    ec["interpolate_rot_step_size"] = cfg["main"]["interpolate_rot_step_size"]
    env = ReKepLiberoEnv(ec, task_suite="libero_goal", task_id=0,
                         robot=args.robot,
                         gripper="default" if args.robot == "Panda" else "PandaGripper",
                         resolution=cfg["libero"]["resolution"])
    m, d = env.sim.model, env.sim.data
    b = m.body_name2id("robot0_base")
    T_wb = homo(d.body_xpos[b], d.body_xmat[b].reshape(3, 3))
    lo, hi = m.jnt_range[env.env.robots[0]._ref_joint_indexes].T

    for r in records:
        T_wf = flange_from_obs(r["requested_pos"], r["requested_quat_xyzw"])
        r["goal_base_flange"] = (inv(T_wb) @ T_wf).tolist()

    out = {
        "robot": args.robot,
        "base_link": "robot0_base",
        "tool_frame": "robot0_right_hand",
        "T_world_base": T_wb.tolist(),
        "joint_names": [m.joint_id2name(j)
                        for j in env.env.robots[0]._ref_joint_indexes],
        "joint_limits": {"lower": lo.tolist(), "upper": hi.tolist()},
        "waypoints": records,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n{len(records)} waypoints -> {args.out}")
    for i, r in enumerate(records):
        print(f"  [{i}] requested {np.round(r['requested_pos'], 4)} "
              f"missed by {r['error_mm']:7.2f} mm")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
