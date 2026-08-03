"""The wrist image itself, once the geometry is exactly right.

With camera->TCP restored to the Panda's [0, -50, -97] mm, the kinematic chain
from camera to grasp point is identical -- and the policy still fails. So the
remaining difference has to be in the PIXELS, not the transform.

The wrist camera has a 75 deg field of view and the grasp point is 109 mm away,
so the gripper's own fingers occupy a large, permanent part of every training
frame. They are the one feature present in all of them. This measures how much
of the view they take up and how much it changes across grippers, at the SAME
aligned start pose.

Run: MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python examples/diag_wrist_view.py
"""
import os, sys
import numpy as np
import imageio

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R); sys.path.insert(0, R + "/examples")
sys.path.insert(0, R + "/third_party/LIBERO")
import libero_ur5e as U

SUITE, TASK, INIT = "libero_spatial", 0, 0


def at_start(env, start):
    """Drive to the reference start TCP so all views are from one pose."""
    m, d = env.sim.model, env.sim.data
    site = m.site_name2id(env.env.robots[0].controller.eef_name)
    obs = None
    for _ in range(250):
        dp = start - d.site_xpos[site]
        if np.linalg.norm(dp) < 0.004:
            break
        obs, _, _, _ = env.step(np.concatenate(
            [np.clip(dp / 0.05, -1, 1), np.zeros(3), [-1.]]).tolist())
    return env.env._get_observations()


def main():
    import torch as _t
    START = np.load(f"{R}/pairs/traj/traj_{SUITE}_Panda_raw_init{INIT}.npy")[0]

    p, suite, _ = U.build(SUITE, TASK)
    o = _t.load; _t.load = lambda *x, **k: o(*x, **{**k, "weights_only": False})
    init = suite.get_task_init_states(TASK); _t.load = o
    np.random.seed(0); p.reset(); p.set_init_state(U.remap_init_state(init[INIT], p.sim))
    for _ in range(10): p.step([0.] * (p.env.action_dim - 1) + [-1.])
    ref_img = at_start(p, START)["robot0_eye_in_hand_image"][::-1].copy()
    ref_fix = U.fixture_snapshot(p.sim); p.close()

    views = {"panda_gripper": ref_img}
    for tag, grip, cam in (("robotiq85", "Robotiq85Gripper", False),
                           ("robotiq85_aligned", "Robotiq85Gripper", True)):
        e, _, _ = U.build(SUITE, TASK, robot="UR5e", gripper=grip, fixture_ref=ref_fix)
        np.random.seed(0); e.reset()
        e.set_init_state(U.remap_init_state(init[INIT], e.sim))
        for _ in range(10): e.step([0.] * (e.env.action_dim - 1) + [-1.])
        obs = at_start(e, START)
        if cam:
            U.align_wrist_camera(e.sim)
            # _get_observations() alone returns a CACHED render; the camera move
            # only reaches the image after the renderer runs again.
            obs, _, _, _ = e.step([0.] * (e.env.action_dim - 1) + [-1.])
        views[tag] = obs["robot0_eye_in_hand_image"][::-1].copy()
        e.close()

    os.makedirs(f"{R}/videos", exist_ok=True)
    strip = np.concatenate(list(views.values()), axis=1)
    imageio.imwrite(f"{R}/videos/wrist_view_grippers.png", strip)

    print("\nwrist view at the SAME aligned start pose, vs the Panda's\n")
    print(f"  {'view':<22}{'mean |dpix|':>13}{'frac >10/255':>15}")
    for k, v in views.items():
        d = np.abs(v.astype(int) - ref_img.astype(int)).mean(2)
        print(f"  {k:<22}{d.mean():>13.1f}{100*(d > 10).mean():>14.1f}%")
    print(f"\n  -> videos/wrist_view_grippers.png "
          f"({' | '.join(views)})\n")


if __name__ == "__main__":
    main()
