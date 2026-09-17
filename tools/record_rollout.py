#!/usr/bin/env python
"""Record a student VLA (or the teacher) driving, with the evaluation's exact decode.

  record_rollout.py CKPT --task 5 --episodes 0 3 7          student, eval episodes 0, 3, 7 of task 5
  record_rollout.py teacher --task 4 --episodes 0 1         the scripted program through the same servos

The environment is built as tools/distill.py's evaluation builds it (seed 555 * 100 + task,
the declared randomization, the program's layout veto), and resets are replayed without
stepping to reach episode k, so episode k shows the layout and start pose that evaluation
episode k had. The student's action goes through distill.decode_student -- the decode
evaluation and DAgger use -- and the gripper mode comes from the checkpoint.

Writes videos/<name>/task<T>_ep<K>_<ok|fail>.mp4 (agent view | wrist, overlay: step,
teacher phase, aperture, the student's gripper target and the teacher's) and a per-step
.csv next to it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))

RAND = dict(horizon=400, layout_radius=0.08, start_xy_m=0.10, start_z_m=0.05, start_yaw_deg=30.0,
            start_tilt_deg=10.0, start_null_rad=0.3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("policy", help="student checkpoint, or 'teacher'")
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--episodes", type=int, nargs="+", default=[0])
    ap.add_argument("--seed", type=int, default=555, help="evaluation seed; the env seed is seed*100 + task")
    ap.add_argument("--res", type=int, default=320)
    ap.add_argument("--out", default="videos")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import cv2
    import imageio.v2 as imageio
    import torch
    from distill import decode_student, load_student, spec_tokens
    from screwhead.clip_features import clip_encoder
    from screwhead.dino_features import DinoFeatures
    from screwhead.gripper_servo import channel_to_target, program_target, target_to_channel
    from screwhead.scripted_teacher import ScriptedTeacher
    from screwhead.teacher_env import PrivilegedEnv

    teacher_only = args.policy == "teacher"
    if teacher_only:
        gt, name, zero = True, "teacher", "none"
    else:
        model, act_std, ck = load_student(args.policy, args.device)
        gt, zero, name = bool(ck.get("gripper_target")), ck.get("zero", "none"), Path(args.policy).stem
        dino = DinoFeatures(args.device)
        tok, tmask = spec_tokens(args.device)
    env = PrivilegedEnv(args.task, seed=args.seed * 100 + args.task, render=True,
                        gripper_mode="target" if gt else "command", **RAND)
    prog = ScriptedTeacher(env)
    env.layout_check = prog.layout_feasible
    if not teacher_only:
        _, enc_txt = clip_encoder(args.device)
        text = enc_txt(env.language).float()[None]
        if zero == "text":
            text = torch.zeros_like(text)
    outdir = ROOT / args.out / name
    outdir.mkdir(parents=True, exist_ok=True)

    R = args.res
    for k in range(max(args.episodes) + 1):
        env.reset()
        if k not in args.episodes:
            continue                               # a reset consumes the env RNG exactly as evaluation did
        frames, rows, done, info = [], [], False, {}
        while not done:
            label = prog.act().astype(np.float32)   # the teacher's action at this state (and its phase)
            t_ap = program_target(prog.phase, float(label[6]), prog.k.preshape_aperture) if gt else None
            if gt:
                label[6] = target_to_channel(t_ap)
            if teacher_only:
                act = label
            else:
                a_img, w_img = env.images()
                f = dino([a_img, w_img])
                a, w = f[0:1], f[1:2]
                if zero == "image":
                    a, w = torch.zeros_like(a), torch.zeros_like(w)
                st = torch.as_tensor(env.student_state()[None], device=args.device)
                if "state_mean" in ck:
                    st = (st - torch.as_tensor(np.asarray(ck["state_mean"]), device=args.device)) / \
                        torch.as_tensor(np.asarray(ck["state_std"]), device=args.device)
                with torch.no_grad():
                    out = model(a, w, text, st.float(), tok, tmask)[:, 0]
                act = decode_student(out, act_std, gt, ck.get("gripper_levels"), ck.get("gripper_classes"))[0]
            sn = env.snapshot()
            sim = env.env.sim
            img = np.ascontiguousarray(np.concatenate(
                [sim.render(camera_name="agentview", width=R, height=R)[::-1],
                 sim.render(camera_name="robot0_eye_in_hand", width=R, height=R)[::-1]], 1))
            who = "teacher" if teacher_only else f"VLA{' (blind)' if zero == 'image' else ''}"
            cv2.putText(img, f"{who}: {env.language[:60]}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255, 255, 255), 1, cv2.LINE_AA)
            grip = (f"target {float(channel_to_target(act[6])) * 1000:4.1f} mm" if gt else f"grip {act[6]:+.0f}")
            teach = f"{prog.phase} {t_ap * 1000:4.1f} mm" if gt else f"{prog.phase} {label[6]:+.0f}"
            cv2.putText(img, f"t={env.t:3d}  ap {sn['aperture'] * 1000:4.1f} mm  {grip}  | teacher {teach}",
                        (6, R - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 0), 1, cv2.LINE_AA)
            frames.append(img)
            rows.append([env.t, prog.phase, sn["aperture"] * 1000, float(channel_to_target(act[6])) * 1000 if gt else act[6],
                         (t_ap * 1000) if gt else label[6], *np.round(act[:6], 4), *np.round(label[:6], 4)])
            _, _, done, info = env.step(act)
        tag = "ok" if info["success"] else "fail"
        stem = outdir / f"task{args.task}_ep{k}_{tag}"
        imageio.mimsave(f"{stem}.mp4", frames, fps=20, macro_block_size=1)
        with open(f"{stem}.csv", "w") as fh:
            fh.write("t,teacher_phase,aperture_mm,policy_gripper,teacher_gripper,"
                     + ",".join(f"policy_v{i}" for i in range(6)) + "," + ",".join(f"teacher_v{i}" for i in range(6)) + "\n")
            for r in rows:
                fh.write(",".join(str(x) for x in r) + "\n")
        print(f"{stem}.mp4  {tag}  steps {env.t}", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
