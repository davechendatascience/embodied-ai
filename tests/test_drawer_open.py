"""End-to-end: see the handle, grasp it, pull, and check the drawer opened.

Everything before this measured grasp POSES. A pose near the handle pointing the
right way is necessary but not sufficient -- the gripper still has to close on a
26 mm bar and drag a jointed body 0.16 m without slipping off. This is the first
test in the project that exercises articulation at all.

The chain being tested:
  1. aim the WRIST camera at the handle (env.look_at) -- the fixed agentview
     sees that face at grazing incidence and yields zero grasps
  2. Contact-GraspNet proposes from that view
  3. approach, close, pull +Y
  4. read the drawer's own slide joint, and LIBERO's success check

Ground truth is `wooden_cabinet_1_middle_level`, a SLIDE joint on Y with range
[-0.16, 0.01]. Negative qpos moves the handle from y=-0.147 to y=+0.013, i.e.
TOWARD the robot, and LIBERO reports success at -0.16. So "open" is a 0.16 m
pull in +Y, despite the joint value going negative.

    MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/test_drawer_open.py
    MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/test_drawer_open.py --video
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

from rekep_libero.environment_libero import ReKepLiberoEnv, WRISTVIEW, EpisodeFinished  # noqa: E402
from rekep_libero import grasp_cgn  # noqa: E402
import transform_utils as T  # noqa: E402

DRAWER = "cabinet_middle"
JOINT = "wooden_cabinet_1_middle_level"
# The drawer's slide joint has damping=50 and its qpos trails the wrist by ~18%
# through slip, so commanding exactly 0.16 m stops short of LIBERO's threshold.
PULL = 0.24
STEP = 0.02


def drawer_geoms(env, drawer=DRAWER):
    model = env.sim.model
    return [g for g in range(model.ngeom)
            if (model.body_id2name(model.geom_bodyid[g]) or "").endswith(drawer)]


def handle_geoms(env, drawer=DRAWER):
    data = env.sim.data
    return [g for g in drawer_geoms(env, drawer) if data.geom_xpos[g][1] > -0.160]


def handle_pos(env):
    data = env.sim.data
    return np.mean([data.geom_xpos[g] for g in handle_geoms(env)], axis=0)


def drawer_qpos(env):
    model, data = env.sim.model, env.sim.data
    return float(data.qpos[model.jnt_qposadr[model.joint_name2id(JOINT)]])


def touching_drawer(env):
    """Gripper geoms in contact with the drawer, from MuJoCo's contact list.

    `_contacting_objects` only knows about movable objects; a cabinet is a
    fixture and never appears there.
    """
    data = env.sim.data
    wanted = set(int(g) for g in drawer_geoms(env))
    n = 0
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 in env._gripper_geom_set and g2 in wanted) or \
           (g2 in env._gripper_geom_set and g1 in wanted):
            n += 1
    return n


def main():
    if not grasp_cgn.available():
        print("no Contact-GraspNet checkpoint")
        return 1

    config = load_config()
    ec = dict(config["env"])
    ec["bounds_min"], ec["bounds_max"] = config["workspace"]["bounds_min"], config["workspace"]["bounds_max"]
    ec["interpolate_pos_step_size"] = config["main"]["interpolate_pos_step_size"]
    ec["interpolate_rot_step_size"] = config["main"]["interpolate_rot_step_size"]
    # --robot UR5e runs the SAME scripted grasp-and-pull on the second
    # embodiment. The gripper is matched (PandaGripper) so the TCP is identical
    # -- NOTES.md section 7 -- which means every pose in this test means the
    # same thing on both arms and only the kinematics differ.
    robot = "Panda"
    if "--robot" in sys.argv:
        robot = sys.argv[sys.argv.index("--robot") + 1]
    gripper = "default" if robot == "Panda" else "PandaGripper"
    # JOINT_POSITION routes waypoints through IK + joint control instead of
    # OSC's local task-space descent. cuRobo IK showed all 45 of this test's
    # waypoints are reachable on the UR5e (45/45), so the misses were descent.
    controller = "JOINT_POSITION" if "--joint-control" in sys.argv else "OSC_POSE"
    env = ReKepLiberoEnv(ec, task_suite="libero_goal", task_id=0,
                         robot=robot, gripper=gripper, controller=controller,
                         resolution=config["libero"]["resolution"])
    print(f"robot   : {robot} + {gripper} + {controller}")

    truth = handle_pos(env)
    print(f"task    : {env.instruction}")
    print(f"handle  : {np.round(truth, 4)}   drawer qpos {drawer_qpos(env):+.4f}\n")

    proposer = grasp_cgn.ContactGraspNetProposer(finger_offset=env.finger_offset(), top_k=64)
    grasp_cgn.estimator()

    # 1-2. aim the wrist at the handle from several directions around the +Y
    # face it presents, pool the candidates, take the actionable one nearest the
    # handle. One view is not a stable input -- see propose_multiview.
    #
    # The drawer opens by travelling +Y, so the gripper must approach along -Y,
    # into the face. A grasp that merely touches the handle from the side holds
    # nothing when pulled: measured, alignment 0.49 stalled, 0.84 slipped after
    # 2.3 mm, 0.97 pulled 125 mm.
    def mask_fn(cam_id):
        return env.points_in_geoms(env.get_cam_obs()[cam_id]["points"],
                                   handle_geoms(env), margin=0.01)

    directions = [(0.0, 1.0, 0.0), (0.0, 1.0, 0.35), (0.3, 1.0, 0.1), (-0.3, 1.0, 0.1)]
    result = proposer.propose_multiview(
        env, mask_fn, truth, directions, cam_id=WRISTVIEW, keypoint=truth,
        action_axis=np.array([0.0, -1.0, 0.0]), min_action_alignment=0.9)
    print(f"looked  : {getattr(proposer, 'last_views', 0)}/{len(directions)} views usable, "
          f"ee {np.round(env.get_ee_pos(), 3)}")
    if result is None:
        print(f"no usable grasp ({len(proposer.last_candidates)} pooled, "
              f"{proposer.last_rejected} unactionable, "
              f"{getattr(proposer, 'last_empty_jaws', 0)} empty jaws, "
              f"{getattr(proposer, 'last_too_wide', 0)} too wide, "
              f"{getattr(proposer, 'last_degenerate', 0)} degenerate frame)")
        if proposer.last_candidates:
            widths = sorted(c["width"] for c in proposer.last_candidates)
            print(f"          survivor widths {np.round(widths, 4)} "
                  f"(usable <= {proposer.max_opening * proposer.clearance:.4f})")
        return 1
    pos, quat, width = result
    approach = T.quat2mat(quat) @ env.GRASP_APPROACH_AXIS
    print(f"grasp   : {np.round(pos, 4)}  opening {width * 1000:.1f} mm  "
          f"{np.linalg.norm(pos - truth) * 1000:.1f} mm from handle  "
          f"approach.-Y {float(approach @ [0, -1, 0]):+.2f}  "
          f"({len(proposer.last_candidates)} candidates, "
          f"{proposer.last_rejected} unactionable, "
          f"{getattr(proposer, 'last_empty_jaws', 0)} empty jaws)")

    # 3. approach along the grasp's own approach axis, then close
    pre = pos - approach * 0.08
    env.execute_action(np.concatenate([pre, quat, [env.get_gripper_null_action()]]), precise=True)
    print(f"pre     : requested {np.round(pre, 3)} reached {np.round(env.get_ee_pos(), 3)}")
    env.execute_action(np.concatenate([pos, quat, [env.get_gripper_null_action()]]), precise=True)
    print(f"at grasp: requested {np.round(pos, 3)} reached {np.round(env.get_ee_pos(), 3)}")
    env.close_gripper()
    print(f"closed  : {touching_drawer(env)} gripper-drawer contacts, "
          f"jaw {np.round(env._last_obs['robot0_gripper_qpos'], 4)}")

    # 4. pull +Y in small steps, holding orientation and grip.
    #
    # EpisodeFinished is a SUCCESS path, not an error: LIBERO sets done=True the
    # moment the task is solved, and the next _step raises. Letting it propagate
    # killed the script before it printed anything, so two runs that had pulled
    # the drawer to qpos -0.125 and -0.130 looked like they had produced no
    # result at all.
    start_q = drawer_qpos(env)
    finished = False
    for i in range(int(PULL / STEP)):
        if finished:
            break
        target = env.get_ee_pose().copy()
        target[1] += STEP
        # precise=True matters here. Coarse mode allows 30 OSC iterations against
        # a 0.02 m tolerance, which is not enough to drag a joint with damping=50
        # -- the wrist simply stalls and the drawer never moves. The gripper
        # action is `null` because `close_gripper()` early-returns once
        # `last_gripper_action` is already 1.0, and `_move_to_waypoint` keeps
        # commanding that held value throughout the motion anyway.
        try:
            env.execute_action(np.concatenate([target[:3], quat, [env.get_gripper_null_action()]]),
                               precise=True)
        except EpisodeFinished as exc:
            print(f"  LIBERO ended the episode at pull {(i + 1) * STEP:.2f} m ({exc})")
            finished = True
        if (i + 1) % 2 == 0:
            print(f"  pull {(i + 1) * STEP:.2f} m -> qpos {drawer_qpos(env):+.4f}  "
                  f"contacts {touching_drawer(env)}  ee y {env.get_ee_pos()[1]:+.3f}")

    end_q = drawer_qpos(env)
    opened = abs(end_q - start_q)
    print(f"\ndrawer  : qpos {start_q:+.4f} -> {end_q:+.4f}  (opened {opened * 1000:.1f} mm)")
    try:
        success = bool(env.env.env._check_success())
    except Exception:  # noqa: BLE001 - the joint reading is the real measure
        success = None
    print(f"success : {success}   (LIBERO reports True at qpos -0.16)")
    print(f"verdict : {'OPENED' if opened > 0.02 else 'DID NOT OPEN'}")

    # Frames are captured in `_step` for the whole run, so this costs one
    # encode and nothing else. It is off by default because the pass/fail is
    # the joint reading, not the picture.
    if "--video" in sys.argv:
        # name by arm, or the second embodiment silently overwrites the first
        tag = f"_{robot}" + ("_joint" if controller == "JOINT_POSITION" else "")
        outcome = "opened" if opened > 0.02 else "failed"
        path = env.save_video(
            os.path.join("videos", f"drawer_open{tag}_{outcome}.mp4"))
        print(f"video   : {path} ({len(env._frames)} frames)")
    return 0 if opened > 0.02 else 1


if __name__ == "__main__":
    sys.exit(main())
