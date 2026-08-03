"""Can the UR5e actually reach the poses OSC missed? cuRobo IK decides.

The scripted drawer test fails on the UR5e without touching the drawer: it
misses its pre-grasp by 83 mm and its grasp by 35 mm, closes on air, and the
drawer stays at exactly 0.0000. `_move_to_waypoint` descends locally with OSC,
so that only proves the LOCAL descent failed. Two very different worlds:

  * the poses are UNREACHABLE for a UR5e -> a workspace limit, and this task
    is off the table for this embodiment no matter what plans it
  * the poses are REACHABLE and OSC could not descend to them -> exactly the
    failure a global planner exists to remove, and the strongest evidence yet
    for the keypose + cuRobo thesis

Collision is deliberately OFF here (`self_collision_check=False`,
`load_collision_spheres=False`): the sphere set does not exist yet, and this
question is about kinematic reachability alone. A pose that IK cannot reach in
free space will not become reachable once obstacles are added.

    PYTHONPATH=src .venv-curobo/bin/python tests/test_curobo_ik.py \
        --waypoints /tmp/drawer_wp_ur5e.json \
        --urdf $PWD/configs/curobo/ur5e_pandagripper.urdf
"""

import argparse
import json
import sys
import time

import numpy as np


def mat_to_quat_wxyz(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
             (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s,
             (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s,
             (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
             (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--waypoints", required=True)
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--num-seeds", type=int, default=64)
    args = ap.parse_args()

    import torch
    from curobo.inverse_kinematics import (
        InverseKinematics,
        InverseKinematicsCfg,
    )
    from curobo.types import GoalToolPose

    data = json.load(open(args.waypoints))
    robot = {
        "robot_cfg": {
            "kinematics": {
                "urdf_path": args.urdf,
                "base_link": data["base_link"],
                # KinematicsLoaderCfg takes `tool_frames` (a list), not
                # `ee_link` -- the older single-EE field does not exist here
                "tool_frames": [data["tool_frame"]],
                "collision_link_names": [],
                "collision_spheres": {},
                "self_collision_ignore": {},
                "self_collision_buffer": {},
                # No lock_joints for the fingers: `tool_frames` prunes the
                # tree to the chain reaching that frame, so the gripper
                # subtree is not in the model at all and locking joints it
                # does not contain is an error, not a no-op.
            }
        }
    }

    cfg = InverseKinematicsCfg.create(
        robot=robot, num_seeds=args.num_seeds,
        self_collision_check=False, load_collision_spheres=False,
        position_tolerance=0.002, orientation_tolerance=0.02,
        max_batch_size=len(json.load(open(args.waypoints))['waypoints']),
    )
    ik = InverseKinematics(cfg)

    wps = data["waypoints"]
    goals = np.array([w["goal_base_flange"] for w in wps], dtype=np.float64)
    pos = torch.as_tensor(np.ascontiguousarray(goals[:, :3, 3]),
                          dtype=torch.float32, device="cuda").contiguous()
    quat = torch.as_tensor(
        np.ascontiguousarray(np.stack([mat_to_quat_wxyz(g[:3, :3]) for g in goals])),
        dtype=torch.float32, device="cuda").contiguous()

    t0 = time.time()
    # GoalToolPose, not Pose: solve_pose reorders by tool frame name, so it
    # needs the frame labels alongside the tensors. Shapes are
    # [batch, num_tool_frames, 3/4] for a single tool frame.
    # shapes are 5D: [batch, horizon, num_tool_frames, goalset, 3/4]
    n = pos.shape[0]
    goal = GoalToolPose(
        tool_frames=[data["tool_frame"]],
        position=pos.reshape(n, 1, 1, 1, 3).contiguous(),
        quaternion=quat.reshape(n, 1, 1, 1, 4).contiguous())
    result = ik.solve_pose(goal)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    success = result.success.detach().cpu().numpy().reshape(-1)
    perr = result.position_error.detach().cpu().numpy().reshape(-1) * 1000.0
    rerr = result.rotation_error.detach().cpu().numpy().reshape(-1)

    print(f"{data['robot']}: {len(wps)} waypoints, {args.num_seeds} seeds")
    print(f"  batch solve {elapsed * 1000:.1f} ms "
          f"({elapsed / len(wps) * 1000:.2f} ms/pose)")
    print()
    print(f"  {'#':>3} {'requested (world)':<26}{'OSC miss':>10}"
          f"{'IK':>6}{'IK pos err':>12}")
    n_ok = 0
    for i, w in enumerate(wps):
        ok = bool(success[i])
        n_ok += ok
        if w["error_mm"] > 20.0 or not ok:
            print(f"  {i:>3} {str(np.round(w['requested_pos'], 4)):<26}"
                  f"{w['error_mm']:>9.1f}mm{'  OK' if ok else ' FAIL':>6}"
                  f"{perr[i]:>10.3f}mm")
    print()
    print(f"  IK solved {n_ok}/{len(wps)} waypoints")

    missed = [i for i, w in enumerate(wps) if w["error_mm"] > 20.0]
    missed_ok = [i for i in missed if bool(success[i])]
    print(f"  of the {len(missed)} poses OSC missed by >20 mm, "
          f"IK solved {len(missed_ok)}")
    print()
    if missed and len(missed_ok) == len(missed):
        print("REACHABLE — every pose OSC missed is kinematically solvable.\n"
              "The UR5e failure is LOCAL DESCENT, not workspace. This is the\n"
              "case cuRobo exists to fix.")
    elif not missed_ok and missed:
        print("UNREACHABLE — the poses OSC missed are outside the UR5e's\n"
              "workspace. No planner recovers this; the task needs a different\n"
              "grasp, not a better executor.")
    else:
        print(f"MIXED — {len(missed_ok)}/{len(missed)} of the missed poses are\n"
              "reachable. Planning helps some of them; the rest are workspace.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
