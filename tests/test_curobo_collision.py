"""Does the sphere model call a KNOWN-GOOD configuration a collision?

Coverage says the spheres do not leave holes. This asks the opposite question:
are they so fat, or the world so wrong, that configurations the arm genuinely
occupied come back as collisions? That failure is the expensive one, because
cuRobo then reports infeasible for every goal and it reads as a planner or
frame bug. The mount pedestal already produced exactly this once -- spheres
reaching 0.28 m above the table.

The configurations come from `dump_drawer_waypoints.py`: joint states the arm
really was in during the scripted drawer run. Free-space ones must be clear.
The grasp and pull are IN CONTACT with the handle by design, so a small
penetration there is physics, not a bug -- reported, not asserted on.

    PYTHONPATH=src .venv-curobo/bin/python tests/test_curobo_collision.py \
        --robot-yml configs/curobo/ur5e_pandagripper.yml \
        --world /tmp/world_goal0.json --waypoints /tmp/drawer_wp_ur5e.json
"""

import argparse
import json
import sys

import numpy as np

# Free-space configurations must clear every obstacle by this much.
FREE_CLEARANCE_TOL_MM = 0.0


def inv(T):
    R, p = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ p
    return out


def quat_wxyz_to_mat(q):
    w, x, y, z = [float(v) for v in q]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def sphere_box_penetration(centres, radii, box_pose, dims):
    """Positive = penetration depth, in metres. Exact for an oriented box."""
    R = quat_wxyz_to_mat(box_pose[3:])
    p = np.asarray(box_pose[:3], dtype=np.float64)
    half = np.asarray(dims, dtype=np.float64) * 0.5
    local = (centres - p) @ R                      # world -> box frame
    outside = np.maximum(np.abs(local) - half, 0.0)
    d = np.linalg.norm(outside, axis=1)            # 0 when the centre is inside
    return radii - d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--robot-yml", required=True)
    ap.add_argument("--world", required=True)
    ap.add_argument("--waypoints", required=True)
    args = ap.parse_args()

    import torch
    import yaml
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.types import JointState

    cfg_dict = yaml.safe_load(open(args.robot_yml))
    kin_cfg = cfg_dict["robot_cfg"]["kinematics"]
    cfg = KinematicsCfg.from_robot_yaml_file(
        cfg_dict, tool_frames=kin_cfg["tool_frames"],
        urdf_path=kin_cfg["urdf_path"])
    kin = Kinematics(cfg)

    wp = json.load(open(args.waypoints))
    world = json.load(open(args.world))
    T_wb = np.array(wp["T_world_base"], dtype=np.float64)
    T_bw = inv(T_wb)

    qs = np.array([w["arm_qpos"] for w in wp["waypoints"]], dtype=np.float64)
    mj_names = list(wp["joint_names"])
    # The FULL robot model includes the gripper subtree (the IK test pruned it
    # via tool_frames), so cuRobo wants finger joints the waypoint dump never
    # recorded. Fingers OPEN is the conservative choice: wider than closed, so
    # any collision it reports is real rather than an artefact of the pose.
    finger_open = {"gripper0_finger_joint1": 0.04,
                   "gripper0_finger_joint2": -0.04}
    for name, val in finger_open.items():
        if name in kin.joint_names and name not in mj_names:
            mj_names.append(name)
            qs = np.concatenate([qs, np.full((len(qs), 1), val)], axis=1)
    order = [mj_names.index(n) for n in kin.joint_names]
    q = torch.as_tensor(np.ascontiguousarray(qs[:, order]),
                        dtype=torch.float32, device="cuda").contiguous()
    state = kin.compute_kinematics(
        JointState.from_position(q, joint_names=list(kin.joint_names)))

    spheres = state.robot_spheres.detach().cpu().numpy().reshape(len(qs), -1, 4)
    print(f"{len(qs)} recorded configurations, "
          f"{spheres.shape[1]} robot spheres each")

    # obstacles: world frame in the JSON, base frame for the robot spheres
    boxes = []
    for name, spec in (world.get("cuboid") or {}).items():
        T_w = np.eye(4)
        T_w[:3, :3] = quat_wxyz_to_mat(spec["pose"][3:])
        T_w[:3, 3] = spec["pose"][:3]
        T_b = T_bw @ T_w
        qb = np.zeros(4)
        t = np.trace(T_b[:3, :3])
        if t > 0:
            s = np.sqrt(t + 1.0) * 2
            qb = np.array([0.25 * s, (T_b[2, 1] - T_b[1, 2]) / s,
                           (T_b[0, 2] - T_b[2, 0]) / s,
                           (T_b[1, 0] - T_b[0, 1]) / s])
        else:                                       # rare; fall back via numpy
            from numpy.linalg import eigh
            w_, v_ = eigh(T_b[:3, :3] + np.eye(3))
            qb = np.array([1.0, 0.0, 0.0, 0.0])
        boxes.append((name, [*T_b[:3, 3], *(qb / np.linalg.norm(qb))],
                      spec["dims"]))
    print(f"{len(boxes)} obstacles, expressed in the robot base frame\n")

    worst_per_config = []
    for i in range(len(qs)):
        c, r = spheres[i, :, :3], spheres[i, :, 3]
        keep = r > 0                                # cuRobo pads with r<=0
        c, r = c[keep], r[keep]
        worst, worst_name = -1e9, ""
        for name, pose, dims in boxes:
            pen = sphere_box_penetration(c, r, pose, dims)
            m = float(pen.max())
            if m > worst:
                worst, worst_name = m, name
        worst_per_config.append((worst, worst_name))

    pens = np.array([w for w, _ in worst_per_config])
    n_free = int((pens <= 0).sum())
    print(f"  configurations clear of every obstacle: {n_free}/{len(qs)}")
    print(f"  worst penetration overall: {pens.max() * 1000:.2f} mm "
          f"({worst_per_config[int(np.argmax(pens))][1]})")
    print(f"  median penetration: {np.median(pens) * 1000:+.2f} mm "
          f"(negative = clearance)")

    print("\n  first 6 configurations (approach, should be free):")
    for i in range(min(6, len(qs))):
        w, n = worst_per_config[i]
        print(f"    [{i}] {w * 1000:+8.2f} mm  {n}")

    approach = pens[:6]
    ok = bool((approach <= FREE_CLEARANCE_TOL_MM / 1000.0).all())
    print()
    if ok:
        print("COLLISION MODEL USABLE — the approach configurations the arm\n"
              "really occupied are collision-free, so cuRobo will not refuse\n"
              "every goal for reasons that look like a frame bug.")
        return 0
    print("COLLISION MODEL REJECTS REALITY — configurations the arm actually\n"
          "occupied report penetration. Planning against this returns\n"
          "infeasible for everything. Check sphere radii and the world frame.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
