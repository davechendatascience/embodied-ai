"""Can PointWorld RANK grasps? The question Contact-GraspNet is standing in for.

`grasp_target` uses Contact-GraspNet, and its stated reason is:

    the margin deciding grasp success is ~7 mm and the model's error floor is
    10-15 mm

**That error floor is retracted.** It was measured on `small-droid` with
`domains=droid` and the 16/31 layout; on `large-droid+behavior` the error on
moved points is 3.92 mm with `cos 0.999`. So the prior against planning grasps
through PointWorld rests on a number this project already withdrew, and it has
never been tested directly. This tests it.

HOW TO ASK THE QUESTION. A grasp cannot be scored by what happens DURING the
approach -- a good approach moves nothing. Its value appears only after the jaw
closes, when the hand moves and the object either comes or does not. So each
candidate is scored by a two-phase trajectory: put the gripper at the candidate
pose, then LIFT, and ask the model how far the object's points travel. A grasp
that holds predicts the object following the hand; one that closes on air
predicts it staying put.

That is also the one thing every measurement agrees this model is confident
about, which is what makes the test fair rather than rigged against it.

THE CONTROL IS THE POINT. Every candidate is also EXECUTED in the simulator
from the same saved state -- approach, close, lift -- and the object's real
displacement measured. Without that, a broken candidate generator is
indistinguishable from a bad model and produces a confident, wrong verdict
(`NOTES.md` section 4). The model's ranking is compared against what actually
happens, not against an assumption about which grasp is good.

    scripts/run_grasp_ranking.sh --suite libero_10 --task-id 0
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
from robot_points import MujocoRobotPoints  # noqa: E402
from pointworld_bridge.client import PointWorldClient  # noqa: E402
from pointworld_bridge.protocol import DEFAULT_SOCKET  # noqa: E402
import transform_utils as T  # noqa: E402

LIFT_MM = 12.0          # per step; 10 steps is a 120 mm lift, unambiguous
UP = np.array([0.0, 0.0, 1.0])


def candidates(env, target, lateral=(0.0, 0.025, 0.05), approaches=None):
    """Grasp poses around the object, DELIBERATELY including bad ones.

    A ranking test needs candidates that differ in quality by more than the
    model's noise, so lateral offsets are swept: 0 mm should hold, 50 mm should
    close on air. If the model cannot separate those it cannot separate
    anything finer either, and no amount of tuning changes that.
    """
    from rekep_libero.grasp import ee_rotation

    lo, hi = specs.object_aabb(env, target)
    c = (lo + hi) * 0.5
    if approaches is None:
        approaches = [(0.0, -1.0, 0.0), (0.0, 1.0, 0.0),
                      (-1.0, 0.0, 0.0), (0.0, 0.0, -1.0)]
    out = []
    for a in approaches:
        a = np.asarray(a, dtype=float)
        # a perpendicular direction to slide the grasp off-centre along
        perp = np.cross(a, UP)
        if np.linalg.norm(perp) < 1e-6:
            perp = np.array([1.0, 0.0, 0.0])
        perp /= np.linalg.norm(perp)
        half = float(np.abs(a) @ ((hi - lo) * 0.5))
        # The jaw must close ACROSS the approach, so the reference axis cannot
        # be parallel to it -- a top-down grasp closes horizontally, not
        # vertically, and `ee_rotation` rejects the degenerate pair outright
        # rather than silently producing a rotation nobody meant.
        sec = UP if abs(float(a @ UP)) < 0.9 else np.array([1.0, 0.0, 0.0])
        R = ee_rotation(a, sec, env.GRASP_APPROACH_AXIS, env.gripper_closing_axis_idx())
        quat = T.mat2quat(R)
        for d in lateral:
            pos = c - a * (half + env.finger_offset()) + perp * d
            out.append({"pos": pos, "quat": quat, "approach": a, "off_mm": d * 1000,
                        "label": f"[{a[0]:+.0f},{a[1]:+.0f},{a[2]:+.0f}] off {d*1000:4.0f}mm"})
    return out


def execute(env, cand, target, steps=8):
    """Approach, close, lift -- and report how far the object really went.

    The simulator state is saved and restored, so every candidate starts from
    the same scene. That is what makes the ranking comparable at all.
    """
    state = env.sim.get_state()
    start = specs.object_center(env, target).copy()
    moved = 0.0
    try:
        env.execute_action(np.concatenate([cand["pos"] - cand["approach"] * 0.10,
                                           cand["quat"],
                                           [env.get_gripper_open_action()]]), precise=True)
        env.execute_action(np.concatenate([cand["pos"], cand["quat"],
                                           [env.get_gripper_null_action()]]), precise=True)
        env.close_gripper()
        for _ in range(steps):
            tgt = env.get_ee_pose().copy()
            tgt[:3] += UP * LIFT_MM / 1000.0
            env.execute_action(np.concatenate([tgt[:3], cand["quat"],
                                               [env.get_gripper_null_action()]]), precise=True)
        moved = float(np.linalg.norm(specs.object_center(env, target) - start)) * 1000
    except EpisodeFinished:
        moved = float(np.linalg.norm(specs.object_center(env, target) - start)) * 1000
    finally:
        env.sim.set_state(state)
        env.sim.forward()
    return moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default=DEFAULT_SOCKET)
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--target", default=None)
    ap.add_argument("--no-sim", action="store_true",
                    help="model only; skip the executed control (fast, and "
                         "unable to conclude anything on its own)")
    cli = ap.parse_args()

    config = load_config()
    ec = dict(config["env"])
    ec["bounds_min"], ec["bounds_max"] = config["workspace"]["bounds_min"], config["workspace"]["bounds_max"]
    ec["interpolate_pos_step_size"] = config["main"]["interpolate_pos_step_size"]
    ec["interpolate_rot_step_size"] = config["main"]["interpolate_rot_step_size"]
    env = ReKepLiberoEnv(ec, task_suite=cli.suite, task_id=cli.task_id, robot="Panda",
                         resolution=config["libero"]["resolution"], reset_seed=0)

    stages = specs.stages_from_bddl(env)
    target = cli.target or stages[0].target
    print(f"task    : {cli.suite}/{cli.task_id} — {env.instruction}")
    print(f"target  : {target}")

    cands = candidates(env, target)
    print(f"probing : {len(cands)} grasp candidates\n")

    # The gripper points are frozen with the jaw CLOSED, because that is the
    # configuration a grasp is evaluated in. Binding open would model a hand
    # that never shuts.
    env.close_gripper()
    gp = GripperPoints(env, NR)
    gp.bind(env)
    rp = MujocoRobotPoints(env, NR)

    with PointWorldClient(cli.socket) as pw:
        obs = live_observation(env, rp, steps=T_LEN)
        points0 = obs["scene_flows"][0, 0]
        mask = env.points_in_geoms(points0, specs.object_geoms(env, target), margin=0.01)
        goal_idx = np.flatnonzero(mask)
        if len(goal_idx) == 0:
            raise SystemExit(f"no visible points on {target}")
        print(f"target  : {len(goal_idx)} visible points\n")

        # Goal = "stay exactly where you are", so the returned cost IS the
        # predicted displacement of the object. Direction-agnostic, the same
        # trick `discover_axis_pointworld.py` uses: a grasp is good when the
        # object MOVES with the hand, whichever way that is.
        goal_pos = points0[goal_idx].copy()

        flows = []
        for c in cands:
            poses = np.array([np.concatenate([c["pos"] + UP * LIFT_MM / 1000.0 * t,
                                              c["quat"]]) for t in range(T_LEN)])
            flows.append(gp.trajectory(poses))
        pw.observe(obs)
        _, out = pw.rollout(np.stack(flows), goal_idx, goal_pos)
        pred = out["cost"] * 1000.0

    truth = np.full(len(cands), np.nan)
    if not cli.no_sim:
        print("executing every candidate in the simulator (saved and rewound)...")
        for i, c in enumerate(cands):
            truth[i] = execute(env, c, target)
            print(f"  {c['label']:24s} model {pred[i]:7.1f} mm | real {truth[i]:7.1f} mm")

    print(f"\n{'candidate':24s} {'model says':>11s} {'sim does':>10s}")
    order = np.argsort(-pred)
    for i in order:
        t = "n/a" if np.isnan(truth[i]) else f"{truth[i]:.1f}"
        print(f"{cands[i]['label']:24s} {pred[i]:10.1f} mm {t:>10s}")

    if not cli.no_sim and np.isfinite(truth).sum() > 2:
        ok = np.isfinite(truth)
        rp_ = np.argsort(np.argsort(pred[ok]))
        rt_ = np.argsort(np.argsort(truth[ok]))
        rho = float(np.corrcoef(rp_, rt_)[0, 1])
        best_model = int(np.argmax(np.where(ok, pred, -np.inf)))
        held = truth[ok] > 20.0
        print(f"\nrank correlation model vs simulator : {rho:+.2f}")
        print(f"model's top pick actually moved      : {truth[best_model]:.1f} mm")
        print(f"candidates that really held          : {int(held.sum())}/{int(ok.sum())}")
        good = rho > 0.5 and truth[best_model] > 20.0
        print(f"\nverdict : PointWorld {'CAN' if good else 'CANNOT'} rank grasps here — "
              f"{'Contact-GraspNet is replaceable' if good else 'the detector earns its place'}")
        return 0 if good else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
