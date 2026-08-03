"""Does cuRobo's forward kinematics agree with MuJoCo's? Sub-millimetre or bust.

This is the non-negotiable check before any planning. cuRobo solves against the
URDF; LIBERO steps the MJCF. If the two disagree, cuRobo returns trajectories
that are collision-free and kinematically valid *for a robot that is not the
one in the simulator*, and the failure appears as "the planner is bad" rather
than as a model mismatch.

    # .venv
    MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/dump_fk_samples.py \
        --robot UR5e --out /tmp/fk_ur5e.json
    # .venv-curobo
    PYTHONPATH=src .venv-curobo/bin/python tests/test_curobo_fk.py \
        --samples /tmp/fk_ur5e.json --urdf configs/curobo/ur5e_pandagripper.urdf

Poses are compared in the ROBOT BASE frame so that a base-mounting error cannot
masquerade as a kinematics error, or cancel one.
"""

import argparse
import json
import sys

import numpy as np

POS_TOL_MM = 1.0
ROT_TOL_DEG = 0.1


def inv(T):
    R, p = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ p
    return out


def pose_delta(A, B):
    d = inv(A) @ B
    ang = np.degrees(np.arccos(np.clip((np.trace(d[:3, :3]) - 1.0) / 2.0, -1, 1)))
    return float(np.linalg.norm(d[:3, 3]) * 1000.0), float(ang)


def quat_wxyz_to_mat(q):
    w, x, y, z = [float(v) for v in q]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--urdf", required=True)
    args = ap.parse_args()

    import torch
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.types import JointState

    data = json.load(open(args.samples))
    tool = data["tool_frame"]
    cfg = KinematicsCfg.from_basic_urdf(
        urdf_path=args.urdf, base_link=data["base_link"], tool_frames=[tool])
    kin = Kinematics(cfg)

    print(f"{data['robot']} + {data['gripper']}")
    print(f"  cuRobo joint order: {list(kin.joint_names)}")
    print(f"  MuJoCo arm joints : {data['arm_joint_names']}")
    print(f"  MuJoCo grip joints: {data['gripper_joint_names']}")

    # cuRobo orders joints its own way; map by NAME rather than assuming the
    # URDF order survived. A silent permutation here produces plausible-looking
    # poses that are wrong for every configuration.
    mj_names = data["arm_joint_names"] + data["gripper_joint_names"]
    try:
        order = [mj_names.index(n) for n in kin.joint_names]
    except ValueError as exc:
        print(f"  FAIL: cuRobo wants a joint MuJoCo did not report: {exc}")
        return 1

    qs = np.array([s["arm_joints"] + s["gripper_joints"] for s in data["samples"]])
    # ascontiguousarray: fancy indexing leaves a strided view, and cuRobo's
    # CUDA kernel rejects non-contiguous input rather than copying it.
    q_curobo = torch.as_tensor(np.ascontiguousarray(qs[:, order]),
                               dtype=torch.float32, device="cuda").contiguous()
    js = JointState.from_position(q_curobo, joint_names=list(kin.joint_names))
    state = kin.compute_kinematics(js)

    # KinematicsState carries `tool_poses` [batch, horizon, num_links, 3/4],
    # indexed by the ORDER of tool_frames rather than keyed by link name.
    tp = state.tool_poses
    frames_list = state.tool_frames
    if callable(frames_list):
        frames_list = frames_list()
    frames_list = list(frames_list)
    ti = frames_list.index(tool)
    n = len(frames_list)
    pos = tp.position.detach().cpu().numpy().reshape(-1, n, 3)[:, ti]
    quat = tp.quaternion.detach().cpu().numpy().reshape(-1, n, 4)[:, ti]

    worst = (0.0, 0.0)
    worst_i = -1
    for i, s in enumerate(data["samples"]):
        T_mj = np.array(s["T_base_flange"], dtype=np.float64)
        T_cu = np.eye(4)
        T_cu[:3, :3] = quat_wxyz_to_mat(quat[i])
        T_cu[:3, 3] = pos[i]
        d = pose_delta(T_mj, T_cu)
        if d[0] > worst[0]:
            worst, worst_i = d, i

    print(f"  {len(data['samples'])} configurations compared in the base frame")
    print(f"  worst disagreement: {worst[0]:.4f} mm / {worst[1]:.4f} deg "
          f"(sample {worst_i})")

    ok = worst[0] <= POS_TOL_MM and worst[1] <= ROT_TOL_DEG
    print()
    print("cuRobo FK AGREES WITH MUJOCO — planning against this URDF is sound"
          if ok else
          f"cuRobo FK DISAGREES — {worst[0]:.3f} mm exceeds {POS_TOL_MM} mm; "
          f"do NOT plan with this URDF")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
