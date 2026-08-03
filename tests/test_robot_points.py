"""Does joint-space FK match the simulator, and leave it exactly as it found it?

Two failure modes, both silent, both fatal to a planner:

  * WRONG POINTS. Every candidate would be scored at a gripper that is not
    where the planner thinks, and the resulting bad actions would look like a
    bad world model -- the confusion this project has paid for repeatedly.
  * MUTATED SIM. `at_config` writes `qpos` to evaluate a hypothetical. If the
    restore were imperfect, hundreds of evaluations per tick would walk the
    simulator away from the real state, and the drawer would drift for reasons
    no log would explain.

So: drive the arm somewhere, read the configuration it actually reached, and
compare FK-from-that-configuration against the geoms MuJoCo reports -- then
verify the full `qpos` is bit-identical afterwards.

    MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/test_robot_points.py
"""

import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from rekep_libero import add_rekep_to_path  # noqa: E402
from rekep_libero.config import load_config  # noqa: E402

add_rekep_to_path()

from rekep_libero.environment_libero import ReKepLiberoEnv  # noqa: E402
from robot_points import MujocoRobotPoints  # noqa: E402

# The bar is what the planner can resolve, not an arbitrary epsilon. The model
# scatters ~9 mm run to run on identical input (`NOTES.md` section 4), so FK
# error only matters as it approaches that. A sub-millimetre residual survives
# here because `live()` reads geoms as the last physics step left them while
# `at_config` re-runs `mj_forward`, which also re-solves constraints -- a
# difference in bookkeeping, not in kinematics.
MODEL_NOISE_MM = 9.0
TOL_MM = MODEL_NOISE_MM / 10.0


def main():
    config = load_config()
    ec = dict(config["env"])
    ec["bounds_min"], ec["bounds_max"] = config["workspace"]["bounds_min"], config["workspace"]["bounds_max"]
    ec["interpolate_pos_step_size"] = config["main"]["interpolate_pos_step_size"]
    ec["interpolate_rot_step_size"] = config["main"]["interpolate_rot_step_size"]
    env = ReKepLiberoEnv(ec, task_suite="libero_goal", task_id=0, robot="Panda",
                         resolution=config["libero"]["resolution"], reset_seed=0)

    rp = MujocoRobotPoints(env, 500)
    print(f"points  : {rp.n} over {len(rp.gids)} geoms")
    print(f"joints  : {len(rp.joint_names)} driving them -> {rp.joint_names}\n")

    worst = 0.0
    quat = env.get_ee_pose()[3:].copy()
    for label, delta in [("at reset", None),
                         ("+Y 60 mm", np.array([0.0, 0.06, 0.0])),
                         ("-Z 40 mm", np.array([0.0, 0.0, -0.04])),
                         ("+X 50 mm", np.array([0.05, 0.0, 0.0]))]:
        if delta is not None:
            t = env.get_ee_pose().copy()
            t[:3] += delta
            env.execute_action(np.concatenate(
                [t[:3], quat, [env.get_gripper_null_action()]]), precise=True)

        truth, truth_n = rp.live()
        before = np.asarray(env.sim.data.qpos).copy()
        pred, pred_n = rp.at_config(rp.current_config())
        after = np.asarray(env.sim.data.qpos).copy()

        err = np.linalg.norm(pred - truth, axis=1).max() * 1000
        ang = np.degrees(np.arccos(np.clip((pred_n * truth_n).sum(1), -1, 1))).max()
        drift = np.abs(after - before).max()
        worst = max(worst, err)
        print(f"  {label:12s} max {err:8.5f} mm   normal {ang:5.3f} deg   "
              f"qpos drift {drift:.2e}")
        assert drift == 0.0, f"at_config mutated the simulator by {drift:.2e}"

    # Closing the jaw is the case the rigid ee binding could not handle -- it
    # cost 37.6 mm of error there. In joint space the fingers are just more
    # joints, so it should cost nothing.
    env.close_gripper()
    truth, _ = rp.live()
    pred, _ = rp.at_config(rp.current_config())
    closed = np.linalg.norm(pred - truth, axis=1).max() * 1000
    print(f"  {'jaw closed':12s} max {closed:8.5f} mm   "
          f"(the rigid ee binding cost 37.6 mm here)")
    worst = max(worst, closed)

    # A configuration the robot is NOT in, which is the whole point.
    q = rp.current_config()
    q[0] += 0.15
    before = np.asarray(env.sim.data.qpos).copy()
    hypo, _ = rp.at_config(q)
    moved = np.linalg.norm(hypo - truth, axis=1).mean() * 1000
    assert np.abs(np.asarray(env.sim.data.qpos) - before).max() == 0.0
    print(f"\nhypothetical: joint 0 +0.15 rad moves the gripper {moved:.1f} mm, "
          f"sim untouched")

    t0 = time.perf_counter()
    for _ in range(20):
        rp.trajectory(np.repeat(rp.current_config()[None], 11, axis=0))
    dt = (time.perf_counter() - t0) / 20
    print(f"cost    : {dt*1e3:.2f} ms per 11-step candidate "
          f"-> {dt*1e3*32:.0f} ms for K=32")

    ok = worst < TOL_MM
    print(f"\nworst FK error {worst:.3f} mm = {worst / MODEL_NOISE_MM:.3f}x the model's "
          f"own {MODEL_NOISE_MM:.0f} mm noise")
    print(f"verdict : {'joint-space FK MATCHES the simulator, and never mutates it' if ok else f'DIVERGES by {worst:.3f} mm'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
