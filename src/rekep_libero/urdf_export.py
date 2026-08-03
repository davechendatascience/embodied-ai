"""Emit a URDF for the arm MuJoCo is actually simulating.

cuRobo needs a URDF; robosuite ships MJCF. The obvious move is to fetch the
vendor's `ur_description` URDF -- and it is the wrong one, because nothing then
guarantees it matches the model LIBERO is stepping. A millimetre of
disagreement in a link length is invisible until cuRobo plans a collision-free
path that collides. This checkout of cuRobo also ships no UR5e config at all
(`ur10e`, `franka`, `dual_ur10e`, `unitree` only), so something has to be
authored either way.

Generating from the live MuJoCo model makes FK agreement true by construction,
and -- more usefully -- CHECKABLE: `tests/test_curobo_fk.py` compares cuRobo's
forward kinematics against MuJoCo's body poses over random configurations.

THE FRAME CONVERSION, which is where this would silently go wrong:

  MuJoCo   body B sits at (body_pos, body_quat) in its PARENT's body frame.
           B's joint lives at `jnt_pos` inside B's own frame, with axis
           `jnt_axis` also in B's frame. Rotating the joint rotates B.

  URDF     a joint connects parent link to child link; the joint origin is
           expressed in the PARENT LINK frame, and the CHILD LINK frame is
           coincident with the joint frame. The axis must pass through the
           child link origin.

So a MuJoCo body frame is NOT a URDF link frame whenever `jnt_pos != 0`. The
link frame emitted here is the body frame translated onto its joint:

    L_B  :=  B translated by jnt_pos(B),  same orientation
    T(L_P -> L_B)  =  Translate(-jnt_pos(P)) . T(P -> B) . Translate(jnt_pos(B))

Getting that wrong yields a URDF whose FK is right at the zero configuration
and drifts everywhere else, which is the most expensive way to be wrong.

The gripper fingers come out as PRISMATIC joints, because that is what they are
in the model. That is deliberate: the keypose interface carries a binary
gripper command executed separately, so the planner should not be free to
choose the opening -- but pinning it belongs in cuRobo's `lock_joints`, where
it is visible and adjustable, rather than baked irreversibly into the URDF at
whatever opening happened to be current at export time.
"""

import argparse
import xml.etree.ElementTree as ET

import numpy as np

HINGE, SLIDE = 3, 2
FREE, BALL = 0, 1


def _quat_wxyz_to_mat(q):
    w, x, y, z = np.asarray(q, dtype=np.float64)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _mat_to_rpy(R):
    """URDF rpy is fixed-axis XYZ, i.e. R = Rz(yaw) Ry(pitch) Rx(roll)."""
    sy = -R[2, 0]
    sy = np.clip(sy, -1.0, 1.0)
    pitch = np.arcsin(sy)
    if abs(abs(sy) - 1.0) < 1e-9:            # gimbal lock
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    return float(roll), float(pitch), float(yaw)


def _homo(pos, R):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def _chain(model, root_name, tip_name):
    """Body ids from root to tip inclusive, following body_parentid upward."""
    root = model.body_name2id(root_name)
    b = model.body_name2id(tip_name)
    out = []
    while b != -1:
        out.append(b)
        if b == root:
            break
        b = model.body_parentid[b]
    else:
        raise ValueError(f"{tip_name} is not a descendant of {root_name}")
    return list(reversed(out))


def _joints_of(model, body_id):
    n = int(model.body_jntnum[body_id])
    adr = int(model.body_jntadr[body_id])
    return list(range(adr, adr + n)) if n > 0 else []


def build_urdf(sim, root_body="robot0_base", tip_body="robot0_right_hand",
               extra_subtrees=("gripper0",), name="robot"):
    """URDF XML string for the chain root->tip, plus fixed gripper subtrees."""
    model = sim.model
    chain = _chain(model, root_body, tip_body)

    # every body we emit: the actuated chain, plus anything hanging off the tip
    # (the gripper) so its geometry reaches the collision model
    bodies = list(chain)
    for b in range(model.nbody):
        nm = model.body_id2name(b) or ""
        if b in bodies:
            continue
        if any(nm.startswith(p) for p in extra_subtrees):
            bodies.append(b)
    bodies_set = set(bodies)

    # joint offset per body, used to shift body frame -> link frame
    joint_of, jpos_of = {}, {}
    for b in bodies:
        js = [j for j in _joints_of(model, b)
              if int(model.jnt_type[j]) in (HINGE, SLIDE)]
        if len(js) > 1:
            raise ValueError(
                f"body {model.body_id2name(b)} has {len(js)} joints; URDF allows "
                f"one per joint element and this exporter does not split them")
        joint_of[b] = js[0] if js else None
        jpos_of[b] = (np.asarray(model.jnt_pos[js[0]], dtype=np.float64)
                      if js else np.zeros(3))

    robot = ET.Element("robot", {"name": name})
    ET.Comment("generated from the live MuJoCo model; do not hand-edit")

    for b in bodies:
        nm = model.body_id2name(b)
        link = ET.SubElement(robot, "link", {"name": nm})
        # Inertial is not used for kinematics but many parsers require it.
        # Values come from MuJoCo so the URDF is at least self-consistent.
        inertial = ET.SubElement(link, "inertial")
        ipos = np.asarray(model.body_ipos[b], dtype=np.float64) - jpos_of[b]
        ET.SubElement(inertial, "origin", {
            "xyz": " ".join(f"{v:.9g}" for v in ipos), "rpy": "0 0 0"})
        ET.SubElement(inertial, "mass", {"value": f"{float(model.body_mass[b]):.9g}"})
        ixx, iyy, izz = (float(v) for v in model.body_inertia[b])
        ET.SubElement(inertial, "inertia", {
            "ixx": f"{ixx:.9g}", "ixy": "0", "ixz": "0",
            "iyy": f"{iyy:.9g}", "iyz": "0", "izz": f"{izz:.9g}"})

    for b in bodies:
        if b == chain[0]:
            continue
        p = int(model.body_parentid[b])
        if p not in bodies_set:
            continue
        nm, pnm = model.body_id2name(b), model.body_id2name(p)

        # T(parent body -> child body) straight from the model
        T_pb_cb = _homo(np.asarray(model.body_pos[b], dtype=np.float64),
                        _quat_wxyz_to_mat(model.body_quat[b]))
        # shift both ends onto their joint frames
        T_lp_pb = _homo(-jpos_of[p], np.eye(3))
        T_cb_lc = _homo(jpos_of[b], np.eye(3))
        T = T_lp_pb @ T_pb_cb @ T_cb_lc

        j = joint_of[b]
        if j is None:
            jtype, axis, lo, hi = "fixed", None, None, None
        else:
            mj_type = int(model.jnt_type[j])
            jtype = "revolute" if mj_type == HINGE else "prismatic"
            # axis is expressed in the body frame, and the link frame differs
            # from it only by a translation, so the axis carries over unchanged
            axis = np.asarray(model.jnt_axis[j], dtype=np.float64)
            if model.jnt_limited[j]:
                lo, hi = (float(v) for v in model.jnt_range[j])
            else:
                # an unlimited revolute is `continuous` in URDF, but cuRobo
                # wants bounds; +-2pi is honest and finite
                jtype = "revolute"
                lo, hi = -2.0 * np.pi, 2.0 * np.pi

        joint = ET.SubElement(robot, "joint", {
            "name": (model.joint_id2name(j) if j is not None else f"{pnm}_to_{nm}"),
            "type": jtype})
        rpy = _mat_to_rpy(T[:3, :3])
        ET.SubElement(joint, "origin", {
            "xyz": " ".join(f"{v:.9g}" for v in T[:3, 3]),
            "rpy": " ".join(f"{v:.9g}" for v in rpy)})
        ET.SubElement(joint, "parent", {"link": pnm})
        ET.SubElement(joint, "child", {"link": nm})
        if axis is not None:
            ET.SubElement(joint, "axis", {
                "xyz": " ".join(f"{v:.9g}" for v in axis)})
            ET.SubElement(joint, "limit", {
                "lower": f"{lo:.9g}", "upper": f"{hi:.9g}",
                # cuRobo reads velocity/acceleration limits from its own yml;
                # these exist to satisfy the URDF schema, not to constrain
                "effort": "1000", "velocity": "3.14"})

    ET.indent(robot, space="  ")
    return ET.tostring(robot, encoding="unicode")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--robot", default="UR5e")
    ap.add_argument("--gripper", default="PandaGripper")
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "tests"))
    import test_ur5e_scene as T  # noqa: E402

    from . import fixtures as fx

    if args.robot == "Panda":
        env = T.build("Panda", gripper=args.gripper)
    else:
        ref_env = T.build("Panda", gripper="default")
        ref = fx.snapshot(ref_env.sim)
        ref_env.close()
        env = T.build(args.robot, gripper=args.gripper, fixture_ref=ref)

    xml = build_urdf(env.sim, name=f"{args.robot.lower()}_{args.gripper.lower()}")
    with open(args.out, "w") as f:
        f.write(xml)
    n_links = xml.count("<link ")
    n_joints = xml.count("<joint ")
    n_actuated = xml.count('type="revolute"') + xml.count('type="prismatic"')
    print(f"{args.robot} + {args.gripper}: {n_links} links, {n_joints} joints "
          f"({n_actuated} actuated) -> {args.out}")
    env.close()


if __name__ == "__main__":
    main()
