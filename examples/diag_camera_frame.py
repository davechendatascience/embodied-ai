"""What does the wrist camera actually see, geometrically?

A visual-servo policy has no access to the TCP. It learned to drive the CAMERA
along a trajectory, and the TCP followed because camera->TCP was a rigid
constant during training. Swap the gripper and that constant changes -- so the
quantity that has to be preserved is the full camera->TCP TRANSFORM, not its
length.

align_wrist_camera() matches only the length. This measures what is left over:
the vector in the CAMERA's own frame, where x/y are image-plane offsets (the
grasp point drifting across the field of view) and z is depth along the optical
axis (the object appearing nearer or further).

Run: MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python examples/diag_camera_frame.py
"""
import os, sys
import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R); sys.path.insert(0, R + "/examples")
sys.path.insert(0, R + "/third_party/LIBERO")
import libero_ur5e as U

SUITE, TASK = "libero_spatial", 0


def cam_to_tcp(env):
    """camera->TCP expressed in the CAMERA frame, metres."""
    m, d = env.sim.model, env.sim.data
    site = m.site_name2id(env.env.robots[0].controller.eef_name)
    cid = m.camera_name2id("robot0_eye_in_hand")
    Rc = d.cam_xmat[cid].reshape(3, 3)
    return Rc.T @ (d.site_xpos[site] - d.cam_xpos[cid])


def show(tag, v, ref=None):
    line = (f"  {tag:<26}[{v[0]*1000:+7.1f} {v[1]*1000:+7.1f} {v[2]*1000:+7.1f}]"
            f"   |v| {np.linalg.norm(v)*1000:6.1f} mm")
    if ref is not None:
        dv = v - ref
        ang = np.degrees(np.arccos(np.clip(
            np.dot(v, ref) / (np.linalg.norm(v) * np.linalg.norm(ref)), -1, 1)))
        line += (f"   delta {np.linalg.norm(dv)*1000:5.1f} mm"
                 f"   angle {ang:4.1f} deg")
    print(line)


def main():
    p, _, _ = U.build(SUITE, TASK)
    ref_fix = U.fixture_snapshot(p.sim)
    panda = cam_to_tcp(p); p.close()

    print("\ncamera->TCP in CAMERA frame  [x_img  y_img  z_depth]\n")
    show("Panda + PandaGripper", panda)

    for g in ("PandaGripper", "Robotiq85Gripper"):
        e, _, _ = U.build(SUITE, TASK, robot="UR5e", gripper=g, fixture_ref=ref_fix)
        show(f"UR5e + {g}", cam_to_tcp(e), panda)
        if g == "Robotiq85Gripper":
            U.align_wrist_camera(e.sim)
            show("UR5e + Robotiq85 ALIGNED", cam_to_tcp(e), panda)
        e.close()

    print("\n  align_wrist_camera() forces |v| to match. Any residual delta/angle\n"
          "  above is what it CANNOT fix: the grasp point sitting somewhere else\n"
          "  in the image than the policy was trained to expect.\n")


if __name__ == "__main__":
    main()
