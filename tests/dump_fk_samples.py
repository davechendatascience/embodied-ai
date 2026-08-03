"""Dump (joint config -> flange pose) samples from MuJoCo for the FK check.

Runs in `.venv`. `tests/test_curobo_fk.py` runs in `.venv-curobo`, reads this
JSON, and asks cuRobo for the same poses. The file on disk is the interface;
neither venv imports the other.

Poses are in the ROBOT BASE frame, because that is the frame cuRobo works in
and the frame `frames.goal_in_base()` produces. Comparing in the world frame
would fold `T_world_base` into the result and hide a base-mounting error inside
a kinematics error.
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

from rekep_libero.frames import homo, inv  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--robot", default="UR5e")
    ap.add_argument("--gripper", default="PandaGripper")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import mujoco

    import test_ur5e_scene as T
    from rekep_libero import fixtures as fx

    if args.robot == "Panda":
        env = T.build("Panda", gripper=args.gripper)
    else:
        ref_env = T.build("Panda", gripper="default")
        ref = fx.snapshot(ref_env.sim)
        ref_env.close()
        env = T.build(args.robot, gripper=args.gripper, fixture_ref=ref)

    sim = env.sim
    m, d = sim.model, sim.data
    robot = env.env.robots[0]
    arm_idx = np.asarray(robot._ref_joint_pos_indexes, dtype=int)
    grip_idx = np.asarray(robot.gripper.joints, dtype=object)
    lo, hi = m.jnt_range[robot._ref_joint_indexes].T

    # gripper joints too: the URDF carries them as prismatic, so cuRobo will
    # want values for them and a mismatch there moves the fingers, not the
    # flange -- silent unless the finger links are compared as well
    grip_jids = [m.joint_name2id(n) for n in robot.gripper.joints]
    grip_qadr = [int(m.jnt_qposadr[j]) for j in grip_jids]

    base = m.body_name2id("robot0_base")
    flange = m.body_name2id("robot0_right_hand")

    rng = np.random.default_rng(args.seed)
    rows = []
    for _ in range(args.samples):
        q = rng.uniform(lo + 0.05, hi - 0.05)
        d.qpos[arm_idx] = q
        mujoco.mj_forward(m._model, d._data)
        T_wb = homo(d.body_xpos[base], d.body_xmat[base].reshape(3, 3))
        T_wf = homo(d.body_xpos[flange], d.body_xmat[flange].reshape(3, 3))
        T_bf = inv(T_wb) @ T_wf
        rows.append({
            "arm_joints": [float(v) for v in q],
            "gripper_joints": [float(d.qpos[a]) for a in grip_qadr],
            "T_base_flange": T_bf.tolist(),
        })

    out = {
        "robot": args.robot,
        "gripper": args.gripper,
        "base_link": "robot0_base",
        "tool_frame": "robot0_right_hand",
        "arm_joint_names": [m.joint_id2name(j) for j in robot._ref_joint_indexes],
        "gripper_joint_names": list(robot.gripper.joints),
        "T_world_base": homo(d.body_xpos[base],
                             d.body_xmat[base].reshape(3, 3)).tolist(),
        "samples": rows,
    }
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"{args.robot}+{args.gripper}: {len(rows)} samples "
          f"({len(out['arm_joint_names'])} arm + "
          f"{len(out['gripper_joint_names'])} gripper joints) -> {args.out}")
    env.close()


if __name__ == "__main__":
    main()
