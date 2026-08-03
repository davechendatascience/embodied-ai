"""How well does each arm actually follow an OSC_POSE delta command?

E2 -- dense end-effector deltas replayed on a UR5e -- is the baseline that
keypose + cuRobo has to beat. It is only worth beating if it was given a fair
controller, and measured on the VLA-JEPA rollout it was not:

    cos(commanded, achieved)   Panda 0.992    UR5e 0.701

That gap is a property of the CONTROLLER, not of the policy, and it would
otherwise be silently credited to cuRobo. Worse, robosuite has no UR5e OSC
tuning to fall back on: `default_ur5e.json` is a **JOINT_VELOCITY** config, so
LIBERO's generic `osc_pose.json` (kp=150) is not a UR5e default being
overridden -- it is the only thing anyone has ever used here.

So the gain has to be chosen, and chosen by measurement. This drives a fixed
scripted delta sequence -- no policy, no perception, nothing learned -- and
reports how faithfully each arm follows it across a sweep of kp:

    MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/tune_ur5e_osc.py

The sequence is deliberately dull: constant-direction pushes along each world
axis, at the magnitude the policy actually commands. A controller that tracks
those badly cannot track anything.
"""

import argparse
import os
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from rekep_libero import add_rekep_to_path  # noqa: E402

add_rekep_to_path()

SUITE = "libero_goal"
TASK_ID = 0
STEPS_PER_LEG = 25

# The typical |world_vector| VLA-JEPA emits, taken from the recorded traces
# rather than guessed: median commanded magnitude was ~0.4 in normalised units.
LEG_MAGNITUDE = 0.4
LEGS = [np.array(v, dtype=float) for v in
        ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1])]


def build(robot, gripper, resolution=256):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    if robot != "Panda":
        from rekep_libero import robots_ur5e
        assert robots_ur5e.registered()

    suite = benchmark.get_benchmark_dict()[SUITE]()
    task = suite.get_task(TASK_ID)
    bddl = os.path.join(get_libero_path("bddl_files"),
                        task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl, robots=[robot], gripper_types=gripper,
        camera_heights=resolution, camera_widths=resolution,
        controller="OSC_POSE", camera_depths=False,
        camera_names=["agentview"], horizon=100000,
    )
    np.random.seed(0)
    env.reset()
    return env


def set_kp(env, kp):
    """Retune the live OSC controller.

    robosuite computes kd from kp at construction, so setting kp alone leaves a
    damping ratio that no longer matches and the arm rings. Both are set here.
    """
    c = env.env.robots[0].controller
    ratio = getattr(c, "damping_ratio", 1.0)
    c.kp = np.ones(6) * kp
    c.kd = 2.0 * np.sqrt(c.kp) * ratio
    return c


def track(env, kp):
    """Drive the scripted legs and report tracking fidelity."""
    set_kp(env, kp)
    cosines, ratios = [], []
    for leg in LEGS:
        cmd = np.concatenate([leg * LEG_MAGNITUDE, np.zeros(3), [-1.0]])
        for _ in range(STEPS_PER_LEG):
            before = env.sim.data.site_xpos[
                env.sim.model.site_name2id(
                    env.env.robots[0].controller.eef_name)].copy()
            env.step(cmd.tolist())
            after = env.sim.data.site_xpos[
                env.sim.model.site_name2id(
                    env.env.robots[0].controller.eef_name)].copy()
            d = after - before
            n = np.linalg.norm(d)
            if n < 1e-9:
                continue
            cosines.append(float(np.dot(d / n, leg)))
            ratios.append(float(n / LEG_MAGNITUDE))
    return (float(np.median(cosines)), float(np.mean(cosines)),
            float(np.median(ratios)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kps", type=float, nargs="+",
                    default=[50, 100, 150, 250, 400, 600, 900])
    args = ap.parse_args()

    print(f"scripted OSC tracking, {SUITE}/{TASK_ID}, "
          f"{len(LEGS)} legs x {STEPS_PER_LEG} steps, |cmd|={LEG_MAGNITUDE}")
    print(f"{'arm':<22}{'kp':>7}{'cos median':>13}{'cos mean':>11}{'mm/step':>10}")

    rows = {}
    for robot, gripper in (("Panda", "default"), ("UR5e", "PandaGripper")):
        env = build(robot, gripper)
        for kp in args.kps:
            med, mean, ratio = track(env, kp)
            rows.setdefault(robot, []).append((kp, med, mean, ratio))
            print(f"{robot + '/' + gripper:<22}{kp:>7.0f}{med:>13.3f}"
                  f"{mean:>11.3f}{ratio * 1000 * LEG_MAGNITUDE:>10.2f}")
            # a fresh scene per gain: a badly tuned gain can leave the arm
            # somewhere the next gain cannot recover from, which would make the
            # sweep measure history rather than the gain
            env.close()
            env = build(robot, gripper)
        env.close()

    print()
    for robot, rs in rows.items():
        best = max(rs, key=lambda r: r[1])
        print(f"best for {robot}: kp={best[0]:.0f}  cos median {best[1]:.3f}")
    print("\nLIBERO's stock generic OSC config is kp=150 for BOTH arms.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
