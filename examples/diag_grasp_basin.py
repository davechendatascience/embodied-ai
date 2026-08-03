"""Is the Robotiq85 inherently a worse grasper, or just differently aligned?

No policy involved. Both grippers are scripted to the SAME known-good grasp pose
-- the point on the Panda's successful trajectory that came closest to the bowl
-- then told to close and lift. Sweeping a lateral/depth offset around that pose
maps each gripper's BASIN: how much positioning error it still converts into a
lift.

This separates two very different claims:
  "the Robotiq85 cannot grasp this object"      -> it fails even at offset 0
  "the Robotiq85 tolerates less error"          -> narrower basin, fine at 0

The second is what a visual-servo policy would feel as instability, because the
policy's residual positioning error is roughly fixed by its own precision.

Run: MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python examples/diag_grasp_basin.py
"""
import os, sys
import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R); sys.path.insert(0, R + "/examples")
sys.path.insert(0, R + "/third_party/LIBERO")
import libero_ur5e as U

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--axis", type=int, default=1, help="0=x 1=y(lateral) 2=z(height)")
_ap.add_argument("--only", default=None, help="substring filter on config name")
_A = _ap.parse_args()
AXIS_NAME = {0: "X", 1: "LATERAL", 2: "HEIGHT"}[_A.axis]

SUITE, TASK, INIT = "libero_spatial", 0, 0
BOWL = "akita_black_bowl_1_main"
OFFSETS = [-0.030, -0.020, -0.010, 0.0, 0.010, 0.020, 0.030]


def drive(env, target, steps=200, grip=-1.0, tol=0.003):
    m, d = env.sim.model, env.sim.data
    site = m.site_name2id(env.env.robots[0].controller.eef_name)
    for _ in range(steps):
        dp = target - d.site_xpos[site]
        if np.linalg.norm(dp) < tol:
            break
        env.step(np.concatenate([np.clip(dp / 0.05, -1, 1), np.zeros(3), [grip]]).tolist())
    return np.linalg.norm(target - d.site_xpos[site])


def trial(env, grasp, off_axis, off):
    """Return bowl lift in mm for one scripted grasp at grasp+offset."""
    m, d = env.sim.model, env.sim.data
    bid = m.body_name2id(BOWL)
    tgt = grasp.copy(); tgt[off_axis] += off
    drive(env, tgt + np.array([0, 0, 0.08]))          # above
    drive(env, tgt)                                    # descend
    z0 = float(d.body_xpos[bid][2])
    for _ in range(25):                                # close
        env.step([0.] * (env.env.action_dim - 1) + [1.])
    drive(env, tgt + np.array([0, 0, 0.15]), grip=1.0) # lift
    for _ in range(15):
        env.step([0.] * (env.env.action_dim - 1) + [1.])
    return (float(d.body_xpos[bid][2]) - z0) * 1000


def main():
    import torch as _t
    traj = np.load(f"{R}/pairs/traj/traj_{SUITE}_Panda_raw_init{INIT}.npy")

    p, suite, _ = U.build(SUITE, TASK)
    o = _t.load; _t.load = lambda *x, **k: o(*x, **{**k, "weights_only": False})
    init = suite.get_task_init_states(TASK); _t.load = o
    np.random.seed(0); p.reset(); p.set_init_state(U.remap_init_state(init[INIT], p.sim))
    for _ in range(10): p.step([0.] * (p.env.action_dim - 1) + [-1.])
    bowl = np.asarray(p.sim.data.body_xpos[p.sim.model.body_name2id(BOWL)], float).copy()
    ref_fix = U.fixture_snapshot(p.sim); p.close()

    # grasp pose = the point on the Panda's successful run closest to the bowl
    grasp = traj[int(np.argmin(np.linalg.norm(traj - bowl, axis=1)))].copy()
    print(f"\n  bowl {np.round(bowl,3)}   scripted grasp pose {np.round(grasp,3)}"
          f"   (dist {np.linalg.norm(grasp-bowl)*1000:.1f} mm)\n")

    rows = {}
    for tag, robot, grip in (("Panda+PandaGripper", "Panda", "default"),
                             ("UR5e+PandaGripper", "UR5e", "PandaGripper"),
                             ("UR5e+Robotiq85", "UR5e", "Robotiq85Gripper")):
        if _A.only and _A.only not in tag:
            continue
        lifts = []
        for off in OFFSETS:
            kw = {} if robot == "Panda" else {"robot": robot, "gripper": grip}
            e, _, _ = U.build(SUITE, TASK, fixture_ref=ref_fix, **kw)
            np.random.seed(0); e.reset()
            e.set_init_state(U.remap_init_state(init[INIT], e.sim))
            for _ in range(10): e.step([0.] * (e.env.action_dim - 1) + [-1.])
            try:
                lifts.append(trial(e, grasp, _A.axis, off))
            except Exception:
                lifts.append(float("nan"))
            e.close()
        rows[tag] = lifts

    print(f"  bowl lift (mm) vs {AXIS_NAME} offset from the known-good grasp pose\n")
    print("  " + " " * 22 + "".join(f"{o*1000:+7.0f}" for o in OFFSETS))
    for tag, lifts in rows.items():
        print(f"  {tag:<22}" + "".join(f"{v:7.0f}" for v in lifts))
    print("\n  a lift above ~30 mm is a real grasp; near 0 means it slipped or missed\n")


if __name__ == "__main__":
    main()
