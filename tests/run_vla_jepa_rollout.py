"""E0 — drive VLA-JEPA on LIBERO and record the dense end-effector trace.

The sim half of VLA-JEPA's own client/server split. The policy lives in
`.venv-jepa` behind a websocket; this runs in `.venv`, which already has
LIBERO on mujoco 3.1.6. Neither venv imports the other.

    # .venv-jepa
    scripts/vla_jepa_serve.sh
    # .venv
    MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/run_vla_jepa_rollout.py \
        --suite libero_goal --task-id 0 --episodes 5

**Their client code is reused deliberately, not reimplemented.**
`M1Inference` carries adaptive action ensembling, sticky-gripper logic and the
unnormalisation statistics; rewriting any of it would put a bug of ours between
the checkpoint and its published number, and E0 exists precisely to reproduce
that number. We supply the loop, the observation plumbing and the recording.

The observation contract is copied from `examples/LIBERO/eval_libero.py`:

    images  [agentview[::-1, ::-1], wrist[::-1, ::-1]]   BOTH axes flipped
    state   eef_pos(3) + quat->axisangle(3) + gripper_qpos(2)   = 8, not 7
    action  world_vector(3) + rotation_delta(3) + gripper(1)

The 180-degree image flip is not cosmetic. LIBERO renders agentview upside
down relative to what the policy was trained on, and getting it wrong yields a
policy that runs, emits plausible actions, and never completes anything --
which reads as a bad checkpoint rather than a bad transform.

The trace is written in the same shape `rekep_libero.keypose` already reads
from GAM (`record_type: env_step`, `obs_after_step`, `env_action`), so one
extractor serves both policies and E1 does not care which produced the trace.
"""

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "third_party" / "VLA_JEPA"))

from rekep_libero import add_rekep_to_path  # noqa: E402

add_rekep_to_path()

VLA_JEPA_CKPT = (REPO / "third_party" / "VLA_JEPA" / "checkpoints_hf" /
                 "LIBERO" / "checkpoints" / "VLA-JEPA-LIBERO.pt")

# LIBERO settles the scene for a few steps before the policy takes over; their
# eval uses a zero action with the gripper open.
DUMMY_ACTION = [0.0] * 6 + [-1.0]
NUM_STEPS_WAIT = 10


def quat2axisangle(quat):
    """xyzw quaternion -> axis-angle. Copied from their `_quat2axisangle`."""
    q = np.asarray(quat, dtype=np.float64).copy()
    if q[3] > 1.0:
        q[3] = 1.0
    elif q[3] < -1.0:
        q[3] = -1.0
    den = np.sqrt(1.0 - q[3] * q[3])
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * np.arccos(q[3])) / den


def build_env(suite_name, task_id, resolution, robot="Panda",
              gripper="default", init_states=None, episodes=0):
    """One LIBERO scene. `robot="UR5e"` is E2/E3; everything else is E0/E1.

    A non-Panda arm needs three things that are NOT automatic, all established
    in NOTES.md section 7: MountedUR5e has to be registered (LIBERO's registry
    knows two Pandas), the recorded init state has to be remapped because it is
    Panda-shaped and read positionally, and the fixtures have to be pinned
    against a Panda scene because robosuite's zero-magnitude init noise draw is
    one number shorter for a 6-DOF arm and shifts every sampled placement.
    """
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    if robot != "Panda":
        from rekep_libero import robots_ur5e
        assert robots_ur5e.registered(), "MountedUR5e failed to register"

    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"),
                        task.problem_folder, task.bddl_file)

    def make(which, grip):
        return OffScreenRenderEnv(
            bddl_file_name=bddl, robots=[which], gripper_types=grip,
            camera_heights=resolution, camera_widths=resolution,
            controller="OSC_POSE", camera_depths=False,
            camera_names=["agentview", "robot0_eye_in_hand"],
            horizon=10000,
        )

    ref = None
    if robot != "Panda":
        from rekep_libero import fixtures as fx
        ref_env = make("Panda", "default")
        np.random.seed(0)
        ref_env.reset()
        fixtures = fx.snapshot(ref_env.sim)
        # The Panda's START END-EFFECTOR POSE, per init state. A closed-loop
        # policy is being asked a different question if the arm begins 181 mm
        # from where it began in training, so the UR5e is IK'd onto these.
        # Captured per episode because the recorded arm pose varies ~0.11 rad
        # across the 50 init states -- one reference pose would be wrong for
        # every episode but the first.
        sid = ref_env.sim.model.site_name2id(
            ref_env.env.robots[0].controller.eef_name)
        starts = {}
        if init_states is not None:
            for ep in range(max(episodes, 1)):
                ref_env.set_init_state(init_states[ep % len(init_states)])
                starts[ep] = (ref_env.sim.data.site_xpos[sid].copy(),
                              ref_env.sim.data.site_xmat[sid].reshape(3, 3).copy())
        ref_env.close()
        ref = {"fixtures": fixtures, "starts": starts}

    return make(robot, gripper), suite, task, ref


def write_video(frames, path, fps=20):
    """20 fps because that is LIBERO's control_freq — one frame per env step,
    so the video runs at the same rate the policy actually acted."""
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), frames, fps=fps, macro_block_size=1)
    return path


def run_episode(env, model, task_desc, init_state, max_steps, trace_path,
                seed, frames=None, fixture_ref=None, start_pose=None):
    import torch  # noqa: F401  (only to confirm .venv's torch is untouched)

    # Seed BEFORE reset. NOTES.md section 1: fixtures are placed by the
    # sampler during reset() and a pinned init state does not cover them.
    # Their own eval_libero.py:289 notes the same thing independently.
    np.random.seed(seed)
    env.reset()
    if fixture_ref is not None:
        from rekep_libero import fixtures as fx
        fx.pin(env.sim, fixture_ref["fixtures"], warn=None)
    from rekep_libero.init_state import remap_panda_init_state
    obs = env.set_init_state(remap_panda_init_state(init_state, env.sim))

    if start_pose is not None:
        from rekep_libero.arm_ik import place_arm_at
        pos_err, rot_err, iters = place_arm_at(env, start_pose[0], start_pose[1])
        if pos_err > 0.005:
            print(f"    WARNING: start-pose IK left {pos_err * 1000:.1f} mm of "
                  f"error after {iters} iters -- this episode does NOT start "
                  f"where the Panda started")
        else:
            print(f"    start-pose IK: {pos_err * 1000:.2f} mm, "
                  f"{np.degrees(rot_err):.2f} deg, {iters} iters")
        obs = env.env._get_observations()
    model.reset(task_desc)

    f = open(trace_path, "w", buffering=1)
    f.write(json.dumps({"record_type": "episode_start",
                        "task_desc": task_desc,
                        "num_steps_wait": NUM_STEPS_WAIT}) + "\n")

    done = False
    t = 0
    steps_run = 0
    while t < max_steps + NUM_STEPS_WAIT:
        if t < NUM_STEPS_WAIT:
            obs, _reward, done, _info = env.step(DUMMY_ACTION)
            t += 1
            continue

        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        state = np.concatenate((obs["robot0_eef_pos"],
                                quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"]))

        response = model.step(images=[img, wrist],
                              task_description=task_desc,
                              step=steps_run,
                              state=np.expand_dims(state, axis=0))
        raw = response["raw_action"]
        wv = np.asarray(raw["world_vector"], dtype=np.float32).reshape(-1)
        rot = np.asarray(raw["rotation_delta"], dtype=np.float32).reshape(-1)
        grip_raw = np.asarray(raw["open_gripper"], dtype=np.float32).reshape(-1)
        # VERBATIM from their `_binarize_gripper_open`, and it is worth the
        # comment: the model's `open_gripper` is [0, 1] with 1 = OPEN, while
        # LIBERO's action convention is +1 = CLOSE. The mapping is therefore
        # INVERTED, and paraphrasing it as "close when > 0.5" cost a 0/2 E0
        # that looked exactly like a checkpoint that does not work.
        grip = np.asarray([1.0 - 2.0 * (float(grip_raw[0]) > 0.5)],
                          dtype=np.float32)
        action = np.concatenate([wv, rot, grip])

        obs, _reward, done, _info = env.step(action.tolist())
        steps_run += 1
        t += 1

        if frames is not None:
            # The policy's own view, side by side: agentview left, wrist right.
            # These are the FLIPPED images -- the ones the model sees and the
            # ones that are right-side up to a human. LIBERO renders agentview
            # upside down, so an unflipped video would look wrong while the
            # policy was fine, or vice versa.
            a = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            w = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            frames.append(np.concatenate([a, w], axis=1))

        f.write(json.dumps({
            "record_type": "env_step",
            "executed_step": steps_run,
            "env_action": action.tolist(),
            "env_gripper": float(grip[0]),
            "success_after_step": bool(done),
            "obs_after_step": {
                "robot0_eef_pos": np.asarray(obs["robot0_eef_pos"]).tolist(),
                "robot0_eef_quat": np.asarray(obs["robot0_eef_quat"]).tolist(),
                "robot0_gripper_width": float(
                    obs["robot0_gripper_qpos"][0] - obs["robot0_gripper_qpos"][1]),
            },
        }) + "\n")

        if done:
            break

    f.close()
    return bool(done), steps_run


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=520)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=15084)
    ap.add_argument("--unnorm-key", default="franka")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="/tmp/vla_jepa_traces")
    ap.add_argument("--robot", default="Panda",
                    help="Panda for E0/E1; UR5e for E2")
    ap.add_argument("--gripper", default="default",
                    help='"PandaGripper" keeps the tool matched across arms')
    ap.add_argument("--video", action="store_true")
    args = ap.parse_args()

    from examples.LIBERO.model2libero_interface import M1Inference

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from libero.libero import benchmark
    import torch  # noqa: F401
    from rekep_libero.grasp_cgn import allow_numpy_unpickling
    allow_numpy_unpickling(torch)
    _suite = benchmark.get_benchmark_dict()[args.suite]()
    init_states = _suite.get_task_init_states(args.task_id)

    env, suite, task, ref = build_env(
        args.suite, args.task_id, args.resolution,
        robot=args.robot, gripper=args.gripper,
        init_states=init_states, episodes=args.episodes)
    fixture_ref = ref
    task_desc = task.language
    print(f"task: {args.suite}/{args.task_id}  \"{task_desc}\"  "
          f"robot={args.robot} gripper={args.gripper}")

    model = M1Inference(policy_ckpt_path=str(VLA_JEPA_CKPT),
                        unnorm_key=args.unnorm_key,
                        policy_setup="franka",
                        host=args.host, port=args.port)

    successes, rows = 0, []
    for ep in range(args.episodes):
        trace = out / f"{args.suite}_task{args.task_id}_ep{ep}.jsonl"
        t0 = time.time()
        frames = [] if args.video else None
        # episode N uses init state N -- their eval_libero.py indexes
        # initial_states[episode_idx], so E1's replay must use the same index.
        ok, steps = run_episode(env, model, task_desc,
                                init_states[ep % len(init_states)],
                                args.max_steps, trace, args.seed, frames,
                                fixture_ref,
                                (ref or {}).get("starts", {}).get(ep))
        successes += int(ok)
        rows.append((ep, ok, steps, time.time() - t0))
        msg = (f"  ep{ep}: success={ok}  steps={steps}  "
               f"{time.time() - t0:.1f}s  -> {trace.name}")
        if frames:
            vid = write_video(
                frames,
                REPO / "videos" / f"vla_jepa_{args.robot}_{args.suite}"
                                  f"_task{args.task_id}_ep{ep}"
                                  f"_{'ok' if ok else 'fail'}.mp4")
            msg += f"  video {vid.name}"
        print(msg)

    print(f"\nE0: {successes}/{args.episodes} = "
          f"{100.0 * successes / args.episodes:.1f}%  on {args.suite}/{args.task_id}")
    print(f"traces in {out}")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
