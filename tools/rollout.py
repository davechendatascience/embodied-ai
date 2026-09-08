#!/usr/bin/env python
"""Run a trained head in LIBERO and emit one trial per episode.

Runs in .venv-libero (LIBERO pins an old stack); the geometry code is imported
by path, not by sharing an environment.

Both policies drive JOINT_POSITION control, not OSC. That is the whole point:
if OSC executes the action, OSC does the inverse kinematics and an embodiment
swap is absorbed by the controller rather than by the policy, which is the
confound that makes the usual LIBERO transfer result uninformative.

  baseline    predicts delta-q padded to MAX_DOF. On a new arm there is nothing
              to do but slice it to that arm's joint count -- the charitable
              reading of zero-shot transfer for a padded-vector policy.
  screwhead   predicts a body twist, decoded through the target arm's own
              Jacobian. The joint count never appears in the learned output.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party" / "LIBERO"))

JOINT_ACTION_SCALE = 0.05          # robosuite joint_position.json output_max
# The four cells of the swap, all tendon-free. Only PandaGripper and
# RethinkGripper are: every other robosuite gripper (Robotiq 85/140/S, Jaco
# three-finger) couples its fingers with a <tendon> spring, and the Robotiq85
# leaves its declared joint range by 1.358 rad under that spring, which puts it
# beyond kinematic reasoning entirely. Measured violations here are 0.00000
# (PandaGripper) and 0.00034 (RethinkGripper).
CELLS = {
    "source":       dict(robot="Panda", mjcf="panda", gripper="PandaGripper",
                         dof=7, arm_swapped=False, gripper_swapped=False),
    "arm_only":     dict(robot="UR5e", mjcf="ur5e", gripper="PandaGripper",
                         dof=6, arm_swapped=True, gripper_swapped=False),
    "gripper_only": dict(robot="Panda", mjcf="panda", gripper="RethinkGripper",
                         dof=7, arm_swapped=False, gripper_swapped=True),
    "both":         dict(robot="UR5e", mjcf="ur5e", gripper="RethinkGripper",
                         dof=6, arm_swapped=True, gripper_swapped=True),
}
ARMS = CELLS      # backwards-compatible name for the --arm flag


def clip_encoder(device):
    from transformers import CLIPModel, CLIPTokenizer
    m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)

    def images(*frames):
        x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().to(device) / 255.0
        x = torch.nn.functional.interpolate(x, size=224, mode="bilinear", align_corners=False)
        with torch.no_grad():
            return m.vision_model(pixel_values=(x - mean) / std).pooler_output

    def text(s):
        ids = tok([s], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            return m.text_projection(m.text_model(**ids).pooler_output)[0]
    return images, text


def build_chain(mjcf_name: str, tool_z: float):
    """Chain with the tool frame at THIS arm's grip site.

    tool_z is measured from the live model, never assumed. The offset is a
    property of the GRIPPER, not the arm: PandaGripper puts the grip site
    0.0970 m beyond the flange, Robotiq85Gripper 0.1450 m. LIBERO gives the
    Panda the former and the UR5e the latter, so a single hardcoded constant is
    48 mm wrong on the held-out arm -- and a 48 mm tool-frame error makes every
    decoded twist reference the wrong point while looking like a kinematic
    transfer failure, which is the opposite of what it is.
    """
    from screwhead.libero import ROBOSUITE_ROBOTS
    from screwhead.mjcf import from_mjcf
    base = from_mjcf(ROBOSUITE_ROBOTS / mjcf_name / "robot.xml", angle="radian", name=mjcf_name)
    off = torch.eye(4, dtype=base.M.dtype)
    off[2, 3] = float(tool_z)
    return base.with_tool(off)


def write_video(path: Path, frames, fps: int = 20) -> None:
    """Agentview and wrist side by side, at the control rate.

    Written at 20 fps so the video runs in real time against the control loop --
    a clip that plays faster than the robot moved makes hesitation look like
    decisiveness.
    """
    import imageio.v2 as imageio
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, macro_block_size=1) as w:
        for f in frames:
            w.append_data(np.ascontiguousarray(f))


def set_joint_gains(env, kp: float) -> None:
    """Stiffen the joint controller. Re-fetched every call on purpose: reset()
    rebuilds the controller object, so a handle captured once goes stale and
    silently leaves the gain at its default."""
    c = env.env.robots[0].controller
    n = len(np.atleast_1d(c.kp))
    c.kp = np.ones(n) * kp
    c.kd = 2 * np.sqrt(c.kp)


def replay(args) -> int:
    """Execute recorded demonstrations through the policy's own control path."""
    import h5py
    from screwhead.libero import task_files
    from screwhead.libero_env import register_ur5e, remap_init_state
    register_ur5e()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    arm = ARMS[args.arm]
    bm = benchmark.get_benchmark_dict()[args.suite]()
    files = {f.stem.replace("_demo", ""): f for f in task_files(args.suite)}
    total = ok = 0
    for ti in range(bm.n_tasks):
        task = bm.get_task(ti)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        f = files.get(task.name)
        if f is None:
            print(f"  [{ti}] no hdf5 for {task.name}", flush=True)
            continue
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128,
                                 robots=[arm["robot"]], gripper_types=arm["gripper"],
                                 controller="JOINT_POSITION")
        with h5py.File(f, "r") as h:
            keys = list(h["data"].keys())[: args.replay]
            for k in keys:
                g = h["data"][k]
                q = np.asarray(g["obs"]["joint_states"][:], np.float64)
                grip = np.asarray(g["actions"][:, 6], np.float64)
                env.reset()
                env.set_init_state(remap_init_state(np.asarray(g["states"][0]), env.sim))
                set_joint_gains(env, args.kp)
                for _ in range(3):
                    env.step(np.zeros(env.env.action_dim))
                success = False
                for t in range(len(q) - 1):
                    set_joint_gains(env, args.kp)
                    cur = env.env._get_observations()["robot0_joint_pos"]
                    dq = q[t + 1][: arm["dof"]] - cur[: arm["dof"]]
                    a = np.zeros(env.env.action_dim)
                    a[: arm["dof"]] = np.clip(dq / JOINT_ACTION_SCALE, -1, 1)
                    a[-1] = np.clip(grip[t], -1, 1)
                    _, _, done, _ = env.step(a)
                    if done:
                        success = True
                        break
                total += 1; ok += int(success)
        env.env.close()
        print(f"  [{ti}] {task.name[:44]:44s} replay {ok}/{total}", flush=True)
    print(f"REPLAY {args.arm}: {ok}/{total} demonstrations reached done")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", choices=["baseline", "screwhead"],
                    help="resolve --checkpoint from checkpoints/<policy>_<suite>.pt. The "
                         "declared tests use this so the run line names the policy under "
                         "test rather than a path that could point anywhere.")
    ap.add_argument("--checkpoint")
    ap.add_argument("--cell", "--arm", dest="arm", default="source", choices=sorted(CELLS),
                    help="which cell of the arm x gripper factorial to roll out")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--episodes-per-task", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--align-camera", action="store_true",
                    help="pin the wrist camera at the Panda's relative offset, so a "
                         "failure cannot be blamed on kinematics when it is visual")
    ap.add_argument("--kp", type=float, default=4000.0,
                    help="joint-position gain. robosuite's default of 50 cannot close a "
                         "50 ms step: demonstration replay tracks to 0.52 rad and reaches "
                         "done 0/15. At 4000 it tracks to 0.007 rad and reaches done 15/15.")
    ap.add_argument("--video", metavar="DIR",
                    help="write one mp4 per episode. The frames are already being "
                         "rendered for the policy and thrown away, so this costs "
                         "only the encode -- and a rollout scoring 0 is far easier "
                         "to diagnose by watching than by reading numbers.")
    ap.add_argument("--video-every", type=int, default=1,
                    help="record every Nth episode")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--replay", type=int, default=0, metavar="N",
                    help="ignore the checkpoint and replay N recorded demonstrations per "
                         "task through the same control path. Validates the harness: if a "
                         "demonstration cannot reach `done` here, a policy scoring 0 says "
                         "nothing about the policy.")
    args = ap.parse_args()
    if not args.checkpoint:
        if not args.policy:
            ap.error("give --checkpoint or --policy")
        args.checkpoint = f"checkpoints/{args.policy}_{args.suite}.pt"
    if args.policy and not args.replay:
        # The checkpoint records what it is; disagreeing with the flag means the
        # run would silently measure a different policy than the test declares.
        _peek = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if _peek.get("policy") != args.policy:
            ap.error(f"--policy {args.policy} but {args.checkpoint} holds "
                     f"{_peek.get('policy')!r}")

    os.environ.setdefault("MUJOCO_GL", "egl")
    torch.set_default_dtype(torch.float32)
    # LIBERO's get_task_init_states calls torch.load without weights_only=False,
    # and torch >= 2.6 defaults it to True. These are LIBERO's own bundled init
    # state files, so allowlist the numpy reconstructor rather than disabling the
    # check globally.
    _torch_load = torch.load

    def _load_trusted(*a, **kw):
        kw.setdefault("weights_only", False)
        return _torch_load(*a, **kw)

    from screwhead.libero_env import (PANDA_CAM_TO_TCP_VEC, align_wrist_camera,
                                      gripper_geom, register_ur5e, remap_init_state)
    register_ur5e()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from screwhead.ik import decode_twist
    from screwhead.interface import ActionSpec
    from screwhead.policy import MAX_DOF, BaselineHead, ScrewHead
    from screwhead.spec import encode

    if args.replay:
        return replay(args)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    policy_kind, chunk = ck["policy"], ck["chunk"]
    model = (BaselineHead(chunk=chunk) if policy_kind == "baseline" else ScrewHead(chunk=chunk))
    model.load_state_dict(ck["state_dict"]); model.to(args.device).eval()
    act_std = ck.get("act_std")
    if act_std is None:
        raise SystemExit(f"{args.checkpoint} predates action normalisation and carries no "
                         "act_std. Retrain it: its loss was dominated by the gripper channel.")
    act_std = torch.as_tensor(act_std, dtype=torch.float32)

    arm = ARMS[args.arm]
    aspec = ActionSpec()
    chain = spec_tokens = spec_mask = None      # built once the live model is up
    twist_scale = torch.tensor(
        [aspec.rot_scale * aspec.control_hz] * 3 + [aspec.pos_scale * aspec.control_hz] * 3)

    enc_img, enc_txt = clip_encoder(args.device)
    bm = benchmark.get_benchmark_dict()[args.suite]()
    trials = []
    for ti in range(bm.n_tasks):
        task = bm.get_task(ti)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        torch.load = _load_trusted          # LIBERO's own bundled init states
        try:
            init_states = bm.get_task_init_states(ti)
        finally:
            torch.load = _torch_load
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128,
                                 robots=[arm["robot"]], gripper_types=arm["gripper"],
                                 controller="JOINT_POSITION")
        if chain is None:
            env.reset()
            flange_to_tcp, _cam = gripper_geom(env)
            chain = build_chain(arm["mjcf"], flange_to_tcp)
            tk, mk = encode(chain).padded(MAX_DOF)
            spec_tokens = tk.float().to(args.device)[None]
            spec_mask = mk.to(args.device)[None]
            print(f"  {arm['robot']}: dof={chain.n} flange->TCP {flange_to_tcp:.4f} m "
                  f"(gripper-dependent, measured)", flush=True)
        tfeat = enc_txt(task.language)[None]
        for ep in range(args.episodes_per_task):
            env.reset()
            env.set_init_state(remap_init_state(init_states[ep % len(init_states)], env.sim))
            if args.align_camera:
                align_wrist_camera(env.sim, PANDA_CAM_TO_TCP_VEC)
            frames = []
            record = bool(args.video) and (ep % max(args.video_every, 1) == 0)
            obs, success = None, False
            set_joint_gains(env, args.kp)
            for _ in range(3):
                obs, _, _, _ = env.step(np.zeros(env.env.action_dim))
            q = torch.zeros(1, MAX_DOF + 1)
            for step in range(0, args.max_steps, chunk):
                f = enc_img(obs["agentview_image"], obs["robot0_eye_in_hand_image"])
                qpos = torch.tensor(obs["robot0_joint_pos"], dtype=torch.float32)
                q[0, :arm["dof"]] = qpos
                with torch.no_grad():
                    if policy_kind == "baseline":
                        pred = model(f[:1], f[1:], tfeat, q.to(args.device),
                                     torch.zeros(1, dtype=torch.long, device=args.device))[0].cpu() * act_std
                    else:
                        pred = model(f[:1], f[1:], tfeat, q.to(args.device),
                                     spec_tokens, spec_mask)[0].cpu() * act_std
                for h in range(chunk):
                    if policy_kind == "baseline":
                        # Already in action units: the head is trained on
                        # dq / JOINT_ACTION_SCALE so the joints and the gripper
                        # share one scale in the loss.
                        dq = pred[h, :arm["dof"]] * JOINT_ACTION_SCALE
                        grip = float(pred[h, MAX_DOF])
                    else:
                        V = (pred[h, :6] * twist_scale)[None].double()
                        th = torch.tensor(obs["robot0_joint_pos"], dtype=torch.float64)[None]
                        dq = (decode_twist(chain, th, V, dt=aspec.dt, lam=0.05).delta[0]).float()
                        grip = float(pred[h, 6])
                    a = np.zeros(env.env.action_dim)
                    a[:arm["dof"]] = (dq / JOINT_ACTION_SCALE).numpy().clip(-1, 1)
                    a[-1] = np.clip(grip, -1, 1)
                    set_joint_gains(env, args.kp)
                    obs, _, done, _ = env.step(a)
                    if record:
                        frames.append(np.concatenate(
                            [obs["agentview_image"][::-1],
                             obs["robot0_eye_in_hand_image"][::-1]], axis=1))
                    if done:
                        success = True
                        break
                if success:
                    break
            if record and frames:
                write_video(Path(args.video) / f"{args.arm}_{policy_kind}_t{ti:02d}_e{ep:02d}"
                            f"_{'ok' if success else 'fail'}.mp4", frames)
            trials.append({
                "metrics": {"success": bool(success)},
                "conditions": {"robot": arm["robot"], "gripper": arm["gripper"],
                               "dof": arm["dof"], "task_suite": args.suite,
                               "policy_revision": policy_kind,
                               "heldout": arm["arm_swapped"] or arm["gripper_swapped"],
                               "arm_swapped": arm["arm_swapped"],
                               "gripper_swapped": arm["gripper_swapped"],
                               "camera_moved": not args.align_camera, "task": task.name},
                "repro": {"robot": arm["robot"], "gripper": arm["gripper"],
                          "task_suite": args.suite, "policy_revision": policy_kind},
            })
        env.env.close()
        n_ok = sum(t["metrics"]["success"] for t in trials[-args.episodes_per_task:])
        print(f"  [{ti}] {task.name[:44]:44s} {n_ok}/{args.episodes_per_task}", flush=True)

    Path(args.out).write_text(json.dumps({"trials": trials}, indent=2))
    total = sum(t["metrics"]["success"] for t in trials)
    print(f"{policy_kind} on {args.arm}: {total}/{len(trials)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
