"""E1 — does the keypose abstraction preserve the task, on the PANDA?

The kill-test for the whole GAM-keypose + cuRobo thesis. GAM emits dense 8-step
chunks of end-effector deltas; the thesis is that those can be compressed to a
handful of absolute SE(3) poses and executed by a planner without losing the
task. If that is false on the arm GAM was TRAINED on, it is false everywhere,
and no UR5e or cuRobo work is warranted.

There is a confound to clear first, and it is not optional. GAM runs in
`.venv-gam` against **its own LIBERO checkout on mujoco 3.6.0**; this replay
runs in `.venv` against `third_party/LIBERO` on **mujoco 3.1.6**. A keypose
replay that failed could mean the abstraction lost the task, or merely that the
two simulators disagree. So there are two modes and the first gates the second:

    --mode dense     replay the trace's own per-step `env_action` open-loop.
                     Agreement here means the two builds are the same world and
                     a keypose failure can be attributed to the abstraction.
                     Disagreement means E1 must move into .venv-gam instead.

    --mode keypose   E1 proper: drive only the extracted keyposes, absolute,
                     through the OSC waypoint controller.

The two venvs never meet. GAM writes `.jsonl`, this reads it. Same rule as the
PointWorld bridge — the file on disk is the whole interface.

    # in .venv-gam, once GAM has run with --trace-actions
    .venv-gam/bin/python -m rekep_libero.keypose <trace.jsonl> --out kp.npz

    # here, in .venv
    MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/test_e1_keypose_replay.py \
        --trace <trace.jsonl> --mode dense
    MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/test_e1_keypose_replay.py \
        --trace <trace.jsonl> --mode keypose
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from rekep_libero import add_rekep_to_path  # noqa: E402

add_rekep_to_path()

from rekep_libero.config import load_config  # noqa: E402
from rekep_libero.environment_libero import ReKepLiberoEnv  # noqa: E402
from rekep_libero.keypose import keyposes, load_trace, summarize  # noqa: E402


def build_env(suite, task_id, init_state_id, seed):
    # ReKepLiberoEnv wants the `env` section with a few keys hoisted out of
    # `workspace`/`main`, not the whole config -- same shape
    # tests/test_drawer_open.py builds.
    cfg = load_config()
    ec = dict(cfg["env"])
    ec["bounds_min"] = cfg["workspace"]["bounds_min"]
    ec["bounds_max"] = cfg["workspace"]["bounds_max"]
    ec["interpolate_pos_step_size"] = cfg["main"]["interpolate_pos_step_size"]
    ec["interpolate_rot_step_size"] = cfg["main"]["interpolate_rot_step_size"]
    return ReKepLiberoEnv(
        ec, task_suite=suite, task_id=task_id,
        init_state_id=init_state_id, reset_seed=seed, verbose=False,
    )


def replay_dense(env, steps, report_every=50):
    """Feed the trace's own actions back, open-loop, and watch the EE diverge.

    Divergence is the measurement. If the two simulator builds agree, the arm
    should track GAM's recorded end-effector positions closely; a growing gap
    says they do not, and says it in millimetres rather than as a pass/fail.
    """
    errs = []
    for i, s in enumerate(steps):
        env._step(np.asarray(s["env_action"], dtype=np.float64))
        here = env.get_ee_pose()[:3]
        errs.append(float(np.linalg.norm(here - s["pos"])) * 1000.0)
        if report_every and (i + 1) % report_every == 0:
            print(f"    step {i + 1:>4}: divergence {errs[-1]:8.2f} mm")
        if env.is_success():
            break
    return np.asarray(errs)


def replay_keyposes(env, steps, marks, pos_tol=0.005, rot_tol=2.0, max_steps=60):
    """E1 proper: absolute pose targets only, nothing in between."""
    from rekep_libero.environment_libero import EpisodeFinished

    reached = []
    for idx, why in marks:
        s = steps[idx]
        target = np.concatenate([s["pos"], s["quat"]])
        # LIBERO raises EpisodeFinished the moment it reports `done`, and for a
        # PLACE task `done` fires when the object lands -- i.e. on SUCCESS,
        # inside the very open_gripper() that completes the task. Letting that
        # propagate turns a pass into a traceback. Catch it and let
        # is_success() be the judge; NOTES.md section 1's rule, in the harness
        # rather than the model.
        try:
            env._move_to_waypoint(target, pos_threshold=pos_tol,
                                  rot_threshold=rot_tol, max_steps=max_steps)
            here = env.get_ee_pose()
            err = float(np.linalg.norm(here[:3] - s["pos"])) * 1000.0
            # The gripper command is the OTHER half of a keypose and the half
            # that is easy to forget. LIBERO's convention is +1 close / -1 open.
            if s["grip_cmd"] > 0:
                env.close_gripper()
            else:
                env.open_gripper()
        except EpisodeFinished as exc:
            here = env.get_ee_pose()
            err = float(np.linalg.norm(here[:3] - s["pos"])) * 1000.0
            reached.append((s["step"], why, err, s["grip_cmd"], env.is_success()))
            print(f"    keypose step {s['step']:>4} {why:<8} "
                  f"reached within {err:6.2f} mm  grip {s['grip_cmd']:+.0f}  "
                  f"success={reached[-1][4]}  [{exc}]")
            return reached
        reached.append((s["step"], why, err, s["grip_cmd"], env.is_success()))
        print(f"    keypose step {s['step']:>4} {why:<8} "
              f"reached within {err:6.2f} mm  grip {s['grip_cmd']:+.0f}  "
              f"success={reached[-1][4]}")
        if env.is_success():
            break
    return reached


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--mode", choices=["dense", "keypose"], default="keypose")
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--task-id", type=int, default=0)
    # GAM's eval walks init states in episode order, so episode N is init state
    # N. Passing it explicitly rather than assuming: a mismatched init state is
    # a different scene, and this whole repo has been bitten by that once.
    ap.add_argument("--init-state-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vel-eps-mm", type=float, default=2.0)
    ap.add_argument("--video", type=str, default=None,
                    help="write the replay to this mp4")
    ap.add_argument("--turn-deg", type=float, default=20.0,
                    help="corner threshold; the gripper criterion never "
                         "fires on a task done with the gripper open")
    args = ap.parse_args()

    steps = load_trace(args.trace)
    if not steps:
        print(f"no env_step records in {args.trace}")
        return 1

    # env_action is needed for dense replay and load_trace does not keep it,
    # so pull it here rather than widening the parser's contract.
    import json
    actions = []
    with open(args.trace) as f:
        for line in f:
            rec = json.loads(line) if line.strip() else {}
            if rec.get("record_type") == "env_step":
                actions.append(np.asarray(rec["env_action"], dtype=np.float64))
    for s, a in zip(steps, actions):
        s["env_action"] = a

    marks = keyposes(steps, args.vel_eps_mm, turn_deg=args.turn_deg)
    s = summarize(steps, marks)
    print(f"trace: {s['steps']} steps, policy success={s['succeeded']}")
    print(f"       {s['keyposes']} keyposes ({s['compression']:.1f}x), "
          f"path/chord {s['path_over_chord']:.2f}")

    env = build_env(args.suite, args.task_id, args.init_state_id, args.seed)
    if args.video:
        env.FRAME_EVERY = 1   # real time, not 5x
        env._frames = []
    start = env.get_ee_pose()[:3]
    print(f"       our start EE {np.round(start, 4)}  "
          f"GAM step-0 EE {np.round(steps[0]['pos'], 4)}  "
          f"delta {np.linalg.norm(start - steps[0]['pos']) * 1000:.1f} mm")

    print(f"\n=== {args.mode.upper()} REPLAY ===")
    if args.mode == "dense":
        errs = replay_dense(env, steps)
        ok = env.is_success()
        print(f"\n  divergence: final {errs[-1]:.2f} mm, max {errs.max():.2f} mm, "
              f"median {np.median(errs):.2f} mm")
        print(f"  LIBERO _check_success() = {ok}   (dense recorded {s['succeeded']})")
        # A dense open-loop replay that reproduces GAM's outcome licenses the
        # keypose run below. One that does not means the sims differ and E1
        # has to move into .venv-gam.
        if args.video:
            print(f"  video: {env.save_video(args.video)} ({len(env._frames)} frames)")
        print("\n" + ("SIMS AGREE — keypose mode is interpretable"
                      if ok == s["succeeded"] else
                      "SIMS DISAGREE — do NOT interpret keypose mode; "
                      "move E1 into .venv-gam"))
        return 0 if ok == s["succeeded"] else 1

    reached = replay_keyposes(env, steps, marks)
    if args.video:
        print(f"  video: {env.save_video(args.video)} ({len(env._frames)} frames)")
    ok = reached[-1][-1] if reached else False
    worst = max(r[2] for r in reached) if reached else float("nan")
    print(f"\n  worst keypose reach error {worst:.2f} mm")
    print(f"  LIBERO _check_success() = {ok}   (dense recorded {s['succeeded']})")
    print("\n" + ("E1 PASSES — the abstraction keeps the task on the Panda"
                  if ok else
                  "E1 FAILS — the dense path carried something the keyposes do not"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
