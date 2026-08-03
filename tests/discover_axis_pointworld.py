"""Can PointWorld reveal which way an articulated part is FREE to move?

If it can, a planner does not need `model.jnt_axis` -- privileged simulator
knowledge a real robot cannot look up. The world model would supply the
constraint direction by being asked what happens, which is the thing a world
model is for.

HOW NOT TO TEST THIS. Scoring candidates against a goal in +Y proves nothing:
a model that believes "a grasped object follows the hand" already ranks +Y
first, because -Y sends the object away from the goal. The asymmetry appears
whether or not the model knows a drawer exists.

So the measurement is DIRECTION-AGNOSTIC: how far does the model think the
target's points move, regardless of where they go? A model that has learnt the
constraint predicts large motion along the slide and little across it. A model
that has only learnt "grasped things follow the hand" predicts the same
displacement in every direction. Those two are distinguishable, and only the
first is useful.

Conveniently that costs nothing extra to ask: set the goal to "stay exactly
where you are", and the returned cost IS the predicted displacement.

The ground truth is measured, not assumed. Every candidate direction is also
EXECUTED in the simulator from the same saved state, so the model's ordering
is compared against what actually happens rather than against the joint axis
alone. `jnt_axis` is used only to score the answer, never as an input.

    scripts/run_axis_discovery.sh
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

from rekep_libero.environment_libero import ReKepLiberoEnv, EpisodeFinished  # noqa: E402
from rekep_libero.gripper_points import GripperPoints  # noqa: E402
from rekep_libero.pw_observation import NR, T_LEN, live_observation  # noqa: E402
from rekep_libero import task_spec as specs  # noqa: E402
from pointworld_bridge.client import PointWorldClient  # noqa: E402
from pointworld_bridge.protocol import DEFAULT_SOCKET  # noqa: E402
import transform_utils as T  # noqa: E402

PART = "wooden_cabinet_1_cabinet_middle"
JOINT = "wooden_cabinet_1_middle_level"
RATE_MM = 20.0
PROBE_STEPS = 4
# The simulator control is EXPENSIVE and does not need to cover the sphere.
# Every `execute_action(precise=True)` runs OSC to a tight tolerance while
# LIBERO renders two cameras with depth and segmentation on each physics step,
# and `precise` is not optional here -- coarse mode stalls the wrist against a
# damping=50 joint (`NOTES.md` section 5), which would report "did not move"
# for every direction and fake the very result being tested. So the model is
# asked about all 18 directions, and only the few that decide the question are
# executed: the two along the axis and the two across it.
SIM_PROBES = 4


def probe_directions():
    """Six axes plus the twelve edge diagonals — a coarse cover of the sphere."""
    dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for a in (-1, 1):
        for b in (-1, 1):
            dirs += [(a * 0.7071, b * 0.7071, 0.0),
                     (a * 0.7071, 0.0, b * 0.7071),
                     (0.0, a * 0.7071, b * 0.7071)]
    return np.array(dirs, dtype=np.float64)


def true_axis(env):
    """The joint's world-frame axis. ANSWER KEY ONLY — never an input."""
    model, data = env.sim.model, env.sim.data
    jid = model.joint_name2id(JOINT)
    bid = int(model.jnt_bodyid[jid])
    axis = np.asarray(data.xmat[bid]).reshape(3, 3) @ np.asarray(model.jnt_axis[jid])
    adr = model.jnt_qposadr[jid]
    lo, hi = model.jnt_range[jid]
    q = float(data.qpos[adr])
    # The direction with travel available is the one the drawer can actually go.
    stop = lo if abs(q - lo) > abs(q - hi) else hi
    return axis * np.sign(stop - q), adr


def scripted_grasp(env):
    from rekep_libero.grasp import ee_rotation

    model, data = env.sim.model, env.sim.data
    hg = [g for g in range(model.ngeom)
          if (model.body_id2name(model.geom_bodyid[g]) or "").endswith("cabinet_middle")
          and data.geom_xpos[g][1] > -0.160]
    truth = np.mean([data.geom_xpos[g] for g in hg], axis=0)
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


def sim_truth(env, quat, direction, adr, steps=PROBE_STEPS):
    """Actually push this way and measure the joint's response, then rewind.

    The simulator is the arbiter. Comparing the model's ordering against the
    joint axis alone would only show whether it agrees with a vector; comparing
    it against executed motion shows whether it agrees with the world.
    """
    state = env.sim.get_state()
    q0 = float(env.sim.data.qpos[adr])
    try:
        for _ in range(steps):
            target = env.get_ee_pose().copy()
            target[:3] += direction * RATE_MM / 1000.0
            try:
                env.execute_action(np.concatenate(
                    [target[:3], quat, [env.get_gripper_null_action()]]), precise=True)
            except EpisodeFinished:
                break
        moved = abs(float(env.sim.data.qpos[adr]) - q0)
    finally:
        env.sim.set_state(state)
        env.sim.forward()
    return moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default=DEFAULT_SOCKET)
    ap.add_argument("--skip-sim", action="store_true",
                    help="model predictions only; skip the executed control")
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
    axis, adr = true_axis(env)
    print(f"grasped : ee {np.round(env.get_ee_pos(), 3)}")
    print(f"truth   : joint axis {np.round(axis, 3)} (answer key — not an input)\n")

    gripper = GripperPoints(env, NR)
    gripper.bind(env)
    dirs = probe_directions()

    with PointWorldClient(cli.socket) as pw:
        obs = live_observation(env, gripper, steps=T_LEN)
        points0 = obs["scene_flows"][0, 0]
        mask = env.points_in_geoms(points0, specs.object_geoms(env, PART), margin=0.01)
        goal_idx = np.flatnonzero(mask)
        print(f"target  : {len(goal_idx)} points on {PART}")

        # Goal = "stay exactly where you are", so the cost the service returns
        # IS the predicted displacement of those points. Direction-agnostic by
        # construction: it cannot reward +Y for being toward anything.
        goal_pos = points0[goal_idx].copy()

        ee = env.get_ee_pose()
        flows = []
        for d in dirs:
            step = d * RATE_MM / 1000.0
            poses = np.array([np.concatenate([ee[:3] + step * t, ee[3:]])
                              for t in range(T_LEN)])
            flows.append(gripper.trajectory(poses))
        pw.observe(obs)
        _, out = pw.rollout(np.stack(flows), goal_idx, goal_pos)
        predicted = out["cost"] * 1000.0        # mm of predicted point motion

    truth_mm = np.full(len(dirs), np.nan)
    if not cli.skip_sim:
        # Two most-along and two most-across, which is all the control needs to
        # separate "knows the joint" from "thinks grasped things follow".
        al = dirs @ axis
        probe_idx = [int(np.argmax(al)), int(np.argmin(al)),          # +axis, -axis
                     int(np.argmin(np.abs(al)))]                      # across
        print(f"probing the simulator on {len(probe_idx)} of {len(dirs)} directions "
              f"(state saved and rewound each time; precise OSC + rendering is "
              f"the slow part)...")
        for i in probe_idx:
            truth_mm[i] = sim_truth(env, quat, dirs[i], adr) * 1000.0
            print(f"  executed [{dirs[i][0]:+.2f},{dirs[i][1]:+.2f},{dirs[i][2]:+.2f}] "
                  f"-> joint moved {truth_mm[i]:6.1f} mm")

    order = np.argsort(-predicted)
    print(f"\n{'':22s} {'PointWorld':>11s} {'sim: joint':>11s}")
    print(f"{'direction':22s} {'says (mm)':>11s} {'moved (mm)':>11s}   alignment to true axis")
    for i in order:
        d = dirs[i]
        al = float(d @ axis)
        t = "  n/a" if np.isnan(truth_mm[i]) else f"{truth_mm[i]:7.1f}"
        print(f"[{d[0]:+.2f},{d[1]:+.2f},{d[2]:+.2f}]  {predicted[i]:10.1f} {t:>11s}"
              f"   {al:+.2f}")

    best = dirs[order[0]]
    angle = np.degrees(np.arccos(np.clip(best @ axis, -1, 1)))
    print(f"\nmodel's best direction : {np.round(best, 3)}  "
          f"({angle:.1f} deg from the true joint axis)")

    # Discrimination is the whole question, and the comparison has to be
    # SIGNED. Averaging |d . axis| lumps the free direction together with the
    # blocked one, so a model predicting lots of motion INTO the closed drawer
    # scores as if it had found the axis. Free, blocked and across are three
    # different predictions and only one ordering means anything.
    free = predicted[(dirs @ axis) > 0.9].mean()
    blocked = predicted[(dirs @ axis) < -0.9].mean()
    across = predicted[np.abs(dirs @ axis) < 0.2].mean()
    print(f"predicted motion: free (+axis) {free:.1f} mm | blocked (-axis) "
          f"{blocked:.1f} mm | across {across:.1f} mm")
    print(f"          a model that knows the joint puts free >> blocked; one that "
          f"believes “grasped things follow the hand” puts them equal; this puts "
          f"blocked {blocked / max(free, 1e-6):.2f}x the free direction")

    if not cli.skip_sim:
        valid = ~np.isnan(truth_mm)
        if valid.sum() > 2 and np.std(truth_mm[valid]) > 1e-9:
            rp = np.corrcoef(np.argsort(np.argsort(-predicted[valid])),
                             np.argsort(np.argsort(-truth_mm[valid])))[0, 1]
            print(f"rank correlation with what the simulator actually does: "
                  f"{rp:+.2f}")
        print(f"simulator's best direction: {np.round(dirs[int(np.nanargmax(truth_mm))], 3)}")

    discriminates = free > 1.5 * blocked
    recovered = angle < 30.0
    print(f"\nverdict : the model {'DISCRIMINATES the free axis' if discriminates else 'does NOT discriminate — it predicts motion whichever way you push, which is “grasped things follow the hand”, not knowledge of the joint'}")
    print(f"          argmax is {angle:.0f} deg from truth, so the axis is "
          f"{'RECOVERABLE without jnt_axis' if recovered else 'NOT recoverable this way'}")
    return 0 if (discriminates and recovered) else 1


if __name__ == "__main__":
    sys.exit(main())
