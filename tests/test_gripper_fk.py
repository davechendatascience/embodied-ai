"""Does gripper FK for a hypothetical pose match what the simulator does?

MPPI scores candidate trajectories it will never simulate, so the points it
scores them from have to be right without physics. If they drift, every
candidate is evaluated at a gripper that is not where the planner thinks it
is, and the resulting bad actions would look like a bad world model --
indefinitely, because nothing downstream could distinguish the two.

So this is the control for `at_ee_pose()`: drive the arm somewhere, read the
pose the controller ACTUALLY reached, and compare FK-from-that-pose against
the gripper points MuJoCo reports. Using the reached pose rather than the
commanded one is deliberate -- it isolates FK error from tracking error, which
are different bugs with different fixes.

It also checks the failure mode on purpose. The rigidity assumption is exactly
"the jaw has not moved since bind()", so closing the gripper MUST break it,
and `check_binding` must say so. A guard that never fires is untested.

    MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/test_gripper_fk.py
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
from rekep_libero.gripper_points import GripperPoints  # noqa: E402

# The model's own run-to-run scatter on identical input is ~9 mm (NOTES.md
# section 4). FK error only matters if it approaches that; below it, the
# planner cannot tell the difference no matter how exact we are.
MODEL_NOISE_MM = 9.0


def compare(gp, env, label):
    """max |FK(reached pose) - MuJoCo|, alongside how far the jaw has drifted.

    Reporting them together is the point. The rigid-body assumption is exactly
    "the jaw has not moved since bind()", so any FK error that matches the jaw
    drift is the assumption being violated, not the transform being wrong --
    and those need different fixes. A flat tolerance conflates them.
    """
    truth, truth_n = gp(env)
    ee = env.get_ee_pose()
    pred, pred_n = gp.at_ee_pose(ee[:3], ee[3:])
    err = np.linalg.norm(pred - truth, axis=1)
    # A normal that has rotated is as wrong as a point that has moved, and
    # normals are 3 of the 16 robot channels.
    ang = np.degrees(np.arccos(np.clip((pred_n * truth_n).sum(1), -1, 1)))
    drift = (gp.check_binding(env) or 0.0) * 1000
    print(f"  {label:34s} max {err.max()*1000:7.3f} mm  mean {err.mean()*1000:7.3f}  "
          f"normal {ang.max():4.2f} deg  jaw drift {drift:7.3f} mm")
    return err.max() * 1000, drift


def main():
    config = load_config()
    ec = dict(config["env"])
    ec["bounds_min"], ec["bounds_max"] = config["workspace"]["bounds_min"], config["workspace"]["bounds_max"]
    ec["interpolate_pos_step_size"] = config["main"]["interpolate_pos_step_size"]
    ec["interpolate_rot_step_size"] = config["main"]["interpolate_rot_step_size"]
    env = ReKepLiberoEnv(ec, task_suite="libero_goal", task_id=0, robot="Panda",
                         resolution=config["libero"]["resolution"], reset_seed=0)

    gp = GripperPoints(env, 500)
    gp.bind(env)
    print(f"gripper : {gp.n} points over {len(gp.gids)} geoms, bound at "
          f"ee {np.round(env.get_ee_pose()[:3], 3)}\n")

    worst, worst_drift = compare(gp, env, "at the bind pose")

    # Translate and rotate away from the bind pose. Rotation is the case that
    # would expose a transform composed in the wrong order, which pure
    # translation cannot catch.
    quat0 = env.get_ee_pose()[3:].copy()
    moves = [
        ("translated +Y 60 mm", np.array([0.0, 0.06, 0.0]), quat0),
        ("translated -Z 40 mm", np.array([0.0, 0.0, -0.04]), quat0),
        ("translated +X 50 mm", np.array([0.05, 0.0, 0.0]), quat0),
    ]
    for label, delta, quat in moves:
        target = env.get_ee_pose().copy()
        target[:3] += delta
        env.execute_action(np.concatenate([target[:3], quat,
                                           [env.get_gripper_null_action()]]), precise=True)
        e, d = compare(gp, env, label)
        worst, worst_drift = max(worst, e), max(worst_drift, d)

    # Rotate the wrist: composition order shows up here or nowhere.
    import transform_utils as T
    R = T.quat2mat(quat0) @ T.euler2mat(np.array([0.0, 0.0, 0.4]))
    target = env.get_ee_pose().copy()
    env.execute_action(np.concatenate([target[:3], T.mat2quat(R),
                                       [env.get_gripper_null_action()]]), precise=True)
    e, d = compare(gp, env, "wrist rotated 23 deg")
    worst, worst_drift = max(worst, e), max(worst_drift, d)

    # The transform is exact arithmetic, so it should be 0 at any pose where
    # the jaw has not moved. Everything above 0 should be the jaw.
    explained = worst <= worst_drift * 1.5 + 0.05
    print(f"\nFK across the moves: worst {worst:.3f} mm against a jaw drift of "
          f"{worst_drift:.3f} mm")
    print(f"          the error is {'fully explained by jaw creep — the transform itself is exact' if explained else 'LARGER than the jaw drift — the transform is wrong'}")
    print(f"          and {worst / MODEL_NOISE_MM:.2f}x the model's own {MODEL_NOISE_MM:.0f} mm "
          f"run-to-run noise, so it is "
          f"{'invisible to the planner' if worst < MODEL_NOISE_MM / 3 else 'big enough to matter'}")

    # ---- the guard, exercised on purpose --------------------------------
    env.close_gripper()
    drift_after = gp.check_binding(env)
    err_closed, _ = compare(gp, env, "AFTER closing the gripper")
    guard_works = drift_after > 2e-3 and err_closed > MODEL_NOISE_MM
    print(f"\nguard   : closing the jaw moved it {drift_after*1000:.1f} mm and cost "
          f"{err_closed:.1f} mm of FK error")
    print(f"          {'check_binding reports it, so a stale binding cannot pass silently' if guard_works else 'did NOT register — the guard is not protecting anything'}")

    # Re-binding must restore exactness, or the planner cannot recover from a
    # grasp mid-episode.
    gp.bind(env)
    err_rebound, _ = compare(gp, env, "after re-bind")

    # ---- cost, since this runs inside the planner's inner loop -----------
    ee = env.get_ee_pose()
    poses = np.repeat(ee[None], 11, axis=0)
    t0 = time.perf_counter()
    for _ in range(20):
        gp.trajectory(poses)
    dt = (time.perf_counter() - t0) / 20
    print(f"\ncost    : {dt*1e3:.2f} ms per 11-step candidate "
          f"-> {dt*1e3*32:.0f} ms for K=32 on CPU")
    print(f"          (model rollout is 29 ms/candidate, so this is "
          f"{'negligible' if dt*1e3 < 3 else 'NOT negligible — vectorise it'})")

    ok = explained and guard_works and err_rebound < 0.05 and worst < MODEL_NOISE_MM / 3
    print(f"\nverdict : {'FK MATCHES the simulator — safe to plan on' if ok else 'FK DIVERGES — do not plan on it'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
