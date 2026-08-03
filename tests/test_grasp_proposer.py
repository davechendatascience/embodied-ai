"""Does an orientation-aware grasp actually pick objects up?

L3 of diagnose_layers.py failed with ReKep's orientation-blind heuristic. This
tests the hypothesis that orientation was the missing piece, using the analytic
PCA proposer -- no learned model, so a pass proves the diagnosis rather than
the model.

    MUJOCO_GL=egl .venv/bin/python tests/test_grasp_proposer.py [object ...]
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

import transform_utils as T  # noqa: E402
from rekep_libero.environment_libero import ReKepLiberoEnv, AGENTVIEW  # noqa: E402
from rekep_libero.grasp import AnalyticGraspProposer  # noqa: E402


def gripper_closing_axis(env):
    """Which ee-frame axis the fingers separate along, measured not assumed."""
    model, data = env.sim.model, env.sim.data
    finger_pos = [data.geom_xpos[g] for g in env._gripper_geom_ids
                  if "finger" in (model.body_id2name(model.geom_bodyid[g]) or "")]
    finger_pos = np.array(finger_pos)
    spread = finger_pos.max(axis=0) - finger_pos.min(axis=0)
    R = T.quat2mat(env.get_ee_pose()[3:])
    local = R.T @ spread          # express the spread in the ee frame
    return int(np.argmax(np.abs(local))), np.round(local, 4)


def object_points(env, name):
    """Delegate to the env rather than reimplementing the predicate.

    This used to carry its own copy of the geom-bounding-sphere test. When the
    env switched to exact collision boxes, the copy did not, so this test kept
    measuring contaminated clouds (cookies_1 at 830 points and 0.092 m wide
    instead of 642 and 0.060 m) and reported TOO WIDE for objects the pipeline
    now grasps. One definition, in the env.
    """
    return env.object_points(name)


def try_grasp(env, proposer, name, grasp_depth, closing_idx=1):
    """Propose a grasp, execute it, and report whether the object came up.

    Both proposers run through this same body on purpose. The interesting
    comparison is physical -- does the gripper end up holding the object -- and
    that is only meaningful if the approach, descent, closing and lift are
    byte-identical between them. Only the `propose` call differs.
    """
    env.reset()
    start_z = env._object_poses()[name][0][2]
    pts = object_points(env, name)

    print(f"\n--- {name} ---")
    if isinstance(proposer, AnalyticGraspProposer):
        pos, quat, width = proposer.propose(pts, env.GRASP_APPROACH_AXIS, closing_idx)
        fits = proposer.fits(width)
        print(f"  {len(pts)} points, narrowest width {width:.3f} m "
              f"(opening {proposer.max_opening:.3f}) -> {'fits' if fits else 'TOO WIDE'}")
        if not fits:
            return False
    else:
        result = proposer.propose_for_object(env, name)
        if result is None:
            print(f"  {len(pts)} points, {len(proposer.last_candidates)} candidates "
                  f"-> NONE USABLE")
            return False
        pos, quat, width = result
        print(f"  {len(pts)} points, {len(proposer.last_candidates)} candidates, "
              f"best opening {width:.3f} m (max {proposer.max_opening:.3f}) -> fits")

    R = T.quat2mat(quat)
    approach_world = R @ env.GRASP_APPROACH_AXIS
    pre = pos - approach_world * grasp_depth
    env.execute_action(np.concatenate([pre, quat, [env.get_gripper_null_action()]]), precise=True)
    print(f"  pre-grasp -> requested {np.round(pre,3)} reached {np.round(env.get_ee_pos(),3)}")

    env.execute_action(np.concatenate([pos, quat, [env.get_gripper_close_action()]]), precise=True)
    print(f"  grasp     -> requested {np.round(pos,3)} reached {np.round(env.get_ee_pos(),3)}")

    contacts = env._contacting_objects()
    print(f"  contacts: {contacts or '{}'}  qpos {np.round(env._last_obs['robot0_gripper_qpos'],4)}")
    if name not in contacts:
        print("  FAIL — nothing in the gripper")
        return False

    lift = env.get_ee_pose().copy()
    lift[2] += 0.12
    env.execute_action(np.concatenate([lift, [env.get_gripper_null_action()]]), precise=False)
    dz = env._object_poses()[name][0][2] - start_z
    lifted = dz > 0.03
    print(f"  lift      -> object rose {dz:+.3f} m, still held: {name in env._contacting_objects()}")
    print(f"  {'PASS — picked up' if lifted else 'FAIL — grasped but not lifted'}")
    return lifted


def main():
    config = load_config()
    ws, ec = config["workspace"], dict(config["env"])
    ec["bounds_min"], ec["bounds_max"] = ws["bounds_min"], ws["bounds_max"]
    ec["interpolate_pos_step_size"] = config["main"]["interpolate_pos_step_size"]
    ec["interpolate_rot_step_size"] = config["main"]["interpolate_rot_step_size"]
    env = ReKepLiberoEnv(ec, task_id=config["libero"]["task_id"])

    axis, local = gripper_closing_axis(env)
    print(f"gripper closing axis (measured): local {'XYZ'[axis]}  spread {local}")
    print(f"approach axis (configured)     : local {'XYZ'[int(np.argmax(np.abs(env.GRASP_APPROACH_AXIS)))]}")

    off = env.finger_offset()
    print(f'finger offset beyond ee site   : {off:+.4f} m')

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    which = "contact_graspnet" if "--cgn" in sys.argv else "analytic"
    if which == "contact_graspnet":
        from rekep_libero import grasp_cgn
        if not grasp_cgn.available():
            print(f"no Contact-GraspNet checkpoint at {grasp_cgn.CHECKPOINT_DIR}")
            return 1
        proposer = grasp_cgn.ContactGraspNetProposer(finger_offset=off)
    else:
        proposer = AnalyticGraspProposer(finger_offset=off)
    print(f"proposer                       : {which}")

    targets = args or ["cookies_1", "akita_black_bowl_1"]
    results = {n: try_grasp(env, proposer, n, config["main"]["grasp_depth"], axis) for n in targets}

    print("\n" + "=" * 56)
    print(f"  proposer: {which}")
    for n, ok in results.items():
        print(f"  {n:34s} {'PASS' if ok else 'FAIL'}")
    return 0 if any(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
