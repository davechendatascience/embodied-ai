"""Why does a grasp with object geometry between the jaws still close on air?

`diagnose_grasp_geometry.py` established that the proposals are fine: every CGN
grasp and the analytic grasp both contain object points inside the jaw volume,
yet both close on nothing. So the fault is downstream of the proposer. Two
suspects, tested here rather than argued about:

  A. THE JAWS ARE NOT WHERE WE THINK. `finger_offset()` currently measures
     -0.0036 m -- the fingertips BEHIND the ee site -- where earlier notes on
     this env recorded roughly +0.02 m. If that number is wrong, every proposer
     is handed a bad correction and commands the wrist to the wrong height. This
     dumps the actual gripper geometry so the number can be checked against it.

  B. THE OBJECT IS GONE BEFORE THE JAWS CLOSE. The approach descends 0.05 m with
     the fingers open; a light object can be knocked aside on the way in. Then
     the grasp was correct and the object simply is not there any more. This
     tracks the object's pose at every phase.

And separately, for the objects that were never reached at all:

  C. IS IT IK OR IS IT REACH? The bowls sit 0.24-0.38 m in y from the ee home
     and the arm got 0.33 m short. Either the DLS IK failed to converge, or the
     pose is genuinely outside the Panda's workspace. The solver reports both
     residual and feasibility, so this just asks it.

    MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/diagnose_execution.py
"""

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from rekep_libero import add_rekep_to_path  # noqa: E402
from rekep_libero.config import load_config  # noqa: E402

add_rekep_to_path()

from rekep_libero.environment_libero import ReKepLiberoEnv  # noqa: E402
from rekep_libero.grasp import AnalyticGraspProposer  # noqa: E402
import transform_utils as T  # noqa: E402


def dump_gripper(env):
    """A: what the finger geoms actually are, and where, relative to the ee site."""
    model, data = env.sim.model, env.sim.data
    ee = env.get_ee_pos()
    R = T.quat2mat(env.get_ee_pose()[3:])
    approach_world = R @ env.GRASP_APPROACH_AXIS

    print("=== A. gripper geometry (local frame, +approach is where the fingers point)")
    print(f"{'geom':34s} {'body':26s} {'along approach':>15s} {'|lateral|':>10s}")
    tips = []
    for g in env._gripper_geom_ids:
        gname = model.geom_id2name(g) or f"geom{g}"
        bname = model.body_id2name(model.geom_bodyid[g]) or ""
        rel = data.geom_xpos[g] - ee
        along = float(rel @ approach_world)
        lateral = float(np.linalg.norm(rel - along * approach_world))
        star = ""
        if "tip" in bname:
            tips.append(along)
            star = "   <- matched by finger_offset()"
        print(f"{gname:34s} {bname:26s} {along:+14.4f}m {lateral:9.4f}m{star}")

    print(f"\nfinger_offset() returns {env.finger_offset():+.4f} m "
          f"(mean of {len(tips)} geoms whose BODY name contains 'tip')")
    if tips:
        print(f"  those geoms lie at {np.round(tips, 4)} along the approach")
    print("  a NEGATIVE offset means the tips sit behind the site, so a proposer")
    print("  pushes the wrist FORWARD to compensate -- deeper into the object.\n")


def track_grasp(env, name, grasp_depth):
    """B: where the object is at every phase of the approach."""
    env.reset()
    proposer = AnalyticGraspProposer(finger_offset=env.finger_offset())
    points = env.object_points(name)
    pos, quat, width = proposer.propose(points, env.GRASP_APPROACH_AXIS,
                                        env.gripper_closing_axis_idx())
    if not proposer.fits(width):
        print(f"=== B. {name}: analytic says too wide ({width:.3f} m), skipping")
        return

    def snap(tag):
        obj = np.asarray(env._object_poses()[name][0])
        qpos = np.round(env._last_obs["robot0_gripper_qpos"], 4)
        print(f"  {tag:12s} obj {np.round(obj, 4)}  ee {np.round(env.get_ee_pos(), 4)}  "
              f"jaw {qpos}  contacts {env._contacting_objects() or '{}'}")
        return obj

    print(f"=== B. {name}: object pose through the approach")
    R = T.quat2mat(quat)
    approach_world = R @ env.GRASP_APPROACH_AXIS
    pre = pos - approach_world * grasp_depth

    start = snap("at reset")
    env.execute_action(np.concatenate([pre, quat, [env.get_gripper_null_action()]]), precise=True)
    at_pre = snap("at pre-grasp")
    env.execute_action(np.concatenate([pos, quat, [env.get_gripper_null_action()]]), precise=True)
    at_desc = snap("descended")
    env.execute_action(np.concatenate([pos, quat, [env.get_gripper_close_action()]]), precise=True)
    snap("closed")

    moved_pre = float(np.linalg.norm(at_pre - start))
    moved_desc = float(np.linalg.norm(at_desc - at_pre))
    print(f"  object moved {moved_pre * 1000:.1f} mm reaching pre-grasp, "
          f"{moved_desc * 1000:.1f} mm during the descent")
    if moved_desc > 0.01:
        print("  -> knocked aside on the way in: the grasp was fine, the object left\n")
    else:
        print("  -> object stayed put, so the jaws simply missed it\n")


def probe_reach(env, name):
    """C: can the IK reach this object's grasp at all?"""
    proposer = AnalyticGraspProposer(finger_offset=env.finger_offset())
    points = env.object_points(name)
    if len(points) < 10:
        return
    centre = points.mean(axis=0)
    # ask for a plain top-down pose over the object, so this measures reach and
    # not whatever orientation a proposer happened to pick
    target = np.concatenate([centre + [0, 0, 0.05], env.get_ee_pose()[3:]])
    result = env.ik_solver.solve(T.convert_pose_quat2mat(target))
    pos_err = getattr(result, "position_error", None)
    ok = getattr(result, "success", None)
    dy = float(centre[1] - env.get_ee_pos()[1])
    print(f"  {name:32s} dy {dy:+.3f}m  ik_success {ok}  pos_err "
          f"{pos_err if pos_err is None else round(float(pos_err), 5)}")


def main():
    config = load_config()
    ec = dict(config["env"])
    ec["bounds_min"], ec["bounds_max"] = config["workspace"]["bounds_min"], config["workspace"]["bounds_max"]
    ec["interpolate_pos_step_size"] = config["main"]["interpolate_pos_step_size"]
    ec["interpolate_rot_step_size"] = config["main"]["interpolate_rot_step_size"]
    env = ReKepLiberoEnv(ec, task_id=config["libero"]["task_id"])

    dump_gripper(env)
    track_grasp(env, "cookies_1", config["main"]["grasp_depth"])

    print("=== C. reachability of each object (plain top-down pose, IK only)")
    for name in env._object_geom_ids:
        probe_reach(env, name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
