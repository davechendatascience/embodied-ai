"""Close the loop: PointWorld chooses the actions that open the drawer.

Everything before this SCORED the model. This is the first thing that lets it
DRIVE. Each tick: observe the scene, sample candidate gripper trajectories,
roll them out through PointWorld over the socket, take the best candidate's
FIRST step, execute it, and re-observe.

Receding horizon is not a style choice. The model under-predicts displacement
3.6x, so the argmin of a 10-step rollout asks for roughly 2x the travel the
goal needs (`NOTES.md` section 4). Executing one step and re-planning turns
that bias into a harmless overshoot on a step that gets corrected; running the
whole plan open-loop would sail past the target.

NOTHING HERE READS LANGUAGE. The target -- which points, and where they should
go -- comes from the simulator's own geometry, deliberately. A VLM would
produce the same two arrays from "open the middle drawer", and that swap is
the only place language enters the system. Building it this way means a
planner bug can never be mistaken for a grounding failure, which matters
because our VLM grounding has failed once already.

The grasp is scripted, not planned. Whether PointWorld could discover grasps
by itself is a separate and unmeasured question: the margin that decides grasp
success here is ~7 mm (`NOTES.md` section 3) and the model's error floor is
10-15 mm, so the prior is that it cannot. Contact-GraspNet or a scripted grasp
places the fingers; PointWorld plans what happens after.

    scripts/pointworld_serve.sh &                     # in its own shell
    MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/plan_drawer_pointworld.py
"""

import argparse
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from rekep_libero import add_rekep_to_path  # noqa: E402
from rekep_libero.config import load_config  # noqa: E402

add_rekep_to_path()

from rekep_libero.environment_libero import (  # noqa: E402
    ReKepLiberoEnv, EpisodeFinished,
)
from rekep_libero.gripper_points import GripperPoints  # noqa: E402
from rekep_libero.pw_observation import NR, T_LEN, live_observation  # noqa: E402
from pointworld_bridge.client import PointWorldClient  # noqa: E402
from pointworld_bridge.protocol import DEFAULT_SOCKET  # noqa: E402
import transform_utils as T  # noqa: E402

DRAWER = "cabinet_middle"
JOINT = "wooden_cabinet_1_middle_level"
OPEN_QPOS = -0.16          # LIBERO reports success here
RATES_MM = (8.0, 16.0, 26.0)


def handle_geoms(env):
    """The drawer FACE geoms — the part a grasp and a pull actually act on."""
    model, data = env.sim.model, env.sim.data
    return [g for g in range(model.ngeom)
            if (model.body_id2name(model.geom_bodyid[g]) or "").endswith(DRAWER)
            and data.geom_xpos[g][1] > -0.160]


def drawer_qpos(env):
    model, data = env.sim.model, env.sim.data
    return float(data.qpos[model.jnt_qposadr[model.joint_name2id(JOINT)]])


def scripted_grasp(env):
    """The ideal grasp: approach -Y into the face, jaws closing across the bar.

    This is the control experiment from `NOTES.md` section 3 that proved the
    mechanism, not a planned action. It exists so the planner is measured on
    what it is for -- the post-grasp motion.
    """
    from rekep_libero.grasp import ee_rotation

    hg = handle_geoms(env)
    if not hg:
        raise SystemExit("no drawer handle in this task")
    truth = np.mean([env.sim.data.geom_xpos[g] for g in hg], axis=0)

    approach = np.array([0.0, -1.0, 0.0])
    R = ee_rotation(approach, np.array([0.0, 0.0, 1.0]),
                    env.GRASP_APPROACH_AXIS, env.gripper_closing_axis_idx())
    quat = T.mat2quat(R)
    pos = truth - approach * env.finger_offset()
    env.execute_action(np.concatenate([pos - approach * 0.10, quat,
                                       [env.get_gripper_open_action()]]), precise=True)
    env.execute_action(np.concatenate([pos, quat, [env.get_gripper_null_action()]]),
                       precise=True)
    env.close_gripper()
    return quat


def candidate_directions():
    """Six axes plus the eight horizontal diagonals.

    A structured set rather than Gaussian samples, because the model scatters
    ~9 mm run to run: a candidate must beat its neighbours by more than that
    to be chosen for a reason rather than by chance. The validated margins are
    tens of millimetres between directions, so a coarse set is the honest
    resolution to plan at.
    """
    dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for dx in (-1, 1):
        for dy in (-1, 1):
            dirs.append((dx * 0.7071, dy * 0.7071, 0.0))
            dirs.append((dx * 0.7071, 0.0, dy * 0.7071))
    return np.array(dirs, dtype=np.float64)


def build_candidates(gripper, ee_pose, steps):
    """(K,T,Nr,3) world-frame gripper flows, plus the per-step ee delta of each."""
    flows, deltas, labels = [], [], []
    for d in candidate_directions():
        d = d / np.linalg.norm(d)
        for rate in RATES_MM:
            step = d * rate / 1000.0
            poses = np.array([np.concatenate([ee_pose[:3] + step * t, ee_pose[3:]])
                              for t in range(steps)])
            flows.append(gripper.trajectory(poses))
            deltas.append(step)
            labels.append(f"[{d[0]:+.1f},{d[1]:+.1f},{d[2]:+.1f}] @ {rate:.0f}mm")
    # Standing still must be on the menu, or "do nothing" can never win and the
    # planner will always move, however bad every option is.
    poses = np.repeat(ee_pose[None], steps, axis=0)
    flows.append(gripper.trajectory(poses))
    deltas.append(np.zeros(3))
    labels.append("still")
    return np.stack(flows), np.array(deltas), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default=DEFAULT_SOCKET)
    ap.add_argument("--ticks", type=int, default=25)
    ap.add_argument("--rebind-mm", type=float, default=3.0,
                    help="re-bind the gripper FK once the jaw has drifted this far")
    cli = ap.parse_args()

    config = load_config()
    ec = dict(config["env"])
    ec["bounds_min"], ec["bounds_max"] = config["workspace"]["bounds_min"], config["workspace"]["bounds_max"]
    ec["interpolate_pos_step_size"] = config["main"]["interpolate_pos_step_size"]
    ec["interpolate_rot_step_size"] = config["main"]["interpolate_rot_step_size"]
    env = ReKepLiberoEnv(ec, task_suite="libero_goal", task_id=0, robot="Panda",
                         resolution=config["libero"]["resolution"], reset_seed=0)
    print(f"task    : {env.instruction}")

    quat = scripted_grasp(env)
    q0 = drawer_qpos(env)
    print(f"grasped : drawer qpos {q0:+.4f}, ee {np.round(env.get_ee_pos(), 3)}")

    # Bind AFTER closing: the rigid-body assumption is "the jaw has not moved
    # since bind()", and closing it costs 37.6 mm of FK error.
    gripper = GripperPoints(env, NR)
    gripper.bind(env)

    with PointWorldClient(cli.socket) as pw:
        print(f"service : {pw.ping()['torch']}\n")
        history = [q0]
        rebinds = 0
        for tick in range(cli.ticks):
            obs = live_observation(env, gripper, steps=T_LEN)
            points0 = obs["scene_flows"][0, 0]

            # ---- the target spec, from geometry -------------------------
            # WHICH points: the drawer face. WHERE they go: far enough in +Y
            # to carry the slide joint to LIBERO's success threshold. Exactly
            # the mask-plus-targets a VLM would have to produce.
            mask = env.points_in_geoms(points0, handle_geoms(env), margin=0.01)
            goal_idx = np.flatnonzero(mask)
            if len(goal_idx) == 0:
                print(f"tick {tick:2d}: no drawer points visible — stopping")
                break
            # SIGN TRAP (`NOTES.md` section 5): this drawer OPENS by travelling
            # +Y while its slide joint runs NEGATIVE, from +0.001 to -0.16. So
            # the travel still owed is `qpos - OPEN_QPOS`, a positive number,
            # applied in +Y. Getting this backwards asks the drawer to stay
            # exactly where it is -- and the planner then correctly picks
            # "still" forever, which looks like a broken planner and is not.
            remaining = max(0.0, drawer_qpos(env) - OPEN_QPOS)
            goal_pos = points0[goal_idx] + np.array([0.0, remaining, 0.0], dtype=np.float32)

            pw.observe(obs)
            ee = env.get_ee_pose()
            flows, deltas, labels = build_candidates(gripper, ee, T_LEN)
            head, out = pw.rollout(flows, goal_idx, goal_pos)
            best = head["best"]
            cost = out["cost"]

            # Margin over the runner-up, against the model's own noise. A
            # choice inside the noise is a coin flip, and saying so is more
            # useful than pretending the planner decided.
            order = np.argsort(cost)
            margin = (cost[order[1]] - cost[order[0]]) * 1000

            step = deltas[best]
            if np.linalg.norm(step) > 0:
                target = ee.copy()
                target[:3] += step
                try:
                    env.execute_action(np.concatenate(
                        [target[:3], quat, [env.get_gripper_null_action()]]), precise=True)
                except EpisodeFinished:
                    print(f"tick {tick:2d}: LIBERO ended the episode — task solved")
                    history.append(drawer_qpos(env))
                    break

            q = drawer_qpos(env)
            history.append(q)

            # The jaw creeps while it holds a load, and the FK binding is only
            # valid while the jaw has not moved. Measured on the first run:
            # drift reached 23 mm, which is well above the model's own 9 mm
            # noise, so the planner was scoring candidates at a gripper that
            # was not where it thought. Re-binding is cheap and exact.
            drift = gripper.check_binding(env)
            if drift is not None and drift * 1000 > cli.rebind_mm:
                gripper.bind(env)
                rebinds += 1
            print(f"tick {tick:2d}: chose {labels[best]:26s} cost {cost[best]*1000:6.1f} mm "
                  f"(margin {margin:5.1f}) | {len(goal_idx):3d} goal pts | "
                  f"qpos {q:+.4f} | still owed {remaining*1000:5.1f} mm"
                  + (f" | rebind ({drift*1000:.1f} mm jaw drift)" if drift and drift * 1000 > cli.rebind_mm else ""))
            if q <= OPEN_QPOS:
                print(f"tick {tick:2d}: reached the success threshold")
                break

    opened = abs(history[-1] - history[0])
    try:
        success = bool(env.env.env._check_success())
    except Exception:  # noqa: BLE001 - the joint reading is the real measure
        success = None
    print(f"\ndrawer  : qpos {history[0]:+.4f} -> {history[-1]:+.4f} "
          f"(opened {opened * 1000:.1f} mm)")
    print(f"success : {success}   (LIBERO reports True at qpos {OPEN_QPOS})")
    print(f"rebinds : {rebinds} — the jaw creeps under load and stale FK is worse than the model's noise")
    print(f"verdict : PointWorld {'OPENED the drawer' if opened > 0.02 else 'DID NOT open the drawer'} "
          f"by planning, with a scripted grasp and no language input")
    return 0 if opened > 0.02 else 1


if __name__ == "__main__":
    sys.exit(main())
