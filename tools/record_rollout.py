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


def _parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy", help="student checkpoint, or 'teacher'")
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--episodes", type=int, nargs="+", default=[0])
    ap.add_argument("--seed", type=int, default=555, help="evaluation seed; the env seed is seed*100 + task")
    ap.add_argument("--res", type=int, default=320)
    ap.add_argument("--out", default="videos")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


class _StudentPolicy:
    """A student checkpoint acting through the evaluation's decode (distill.decode_student)."""

    def __init__(self, ckpt, device):
        from distill import load_student, spec_tokens
        from screwhead.dino_features import DinoFeatures
        self.model, self.act_std, self.ck = load_student(ckpt, device)
        self.gt, self.zero = bool(self.ck.get("gripper_target")), self.ck.get("zero", "none")
        self.dino = DinoFeatures(device)
        self.tok, self.tmask = spec_tokens(device)
        self.device, self.text = device, None

    def set_instruction(self, language):
        import torch
        from screwhead.clip_features import clip_encoder
        _, enc_txt = clip_encoder(self.device)
        self.text = enc_txt(language).float()[None]
        if self.zero == "text":
            self.text = torch.zeros_like(self.text)

    def act(self, env):
        import torch
        from distill import decode_student
        dev, ck = self.device, self.ck
        a_img, w_img = env.images()
        f = self.dino([a_img, w_img])
        a, w = f[0:1], f[1:2]
        if self.zero == "image":
            a, w = torch.zeros_like(a), torch.zeros_like(w)
        st = torch.as_tensor(env.student_state()[None], device=dev)
        if "state_mean" in ck:
            st = (st - torch.as_tensor(np.asarray(ck["state_mean"]), device=dev)) / \
                torch.as_tensor(np.asarray(ck["state_std"]), device=dev)
        with torch.no_grad():
            out = self.model(a, w, self.text, st.float(), self.tok, self.tmask)[:, 0]
        return decode_student(out, self.act_std, self.gt, ck.get("gripper_levels"), ck.get("gripper_classes"))[0]


def _frame(env, R, captions):
    """Agent view | wrist at R px, captioned top (who and the instruction) and bottom (the step)."""
    import cv2
    sim = env.env.sim
    img = np.ascontiguousarray(np.concatenate(
        [sim.render(camera_name="agentview", width=R, height=R)[::-1],
         sim.render(camera_name="robot0_eye_in_hand", width=R, height=R)[::-1]], 1))
    top, bottom = captions
    cv2.putText(img, top, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, bottom, (6, R - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 0), 1, cv2.LINE_AA)
    return img


def _write_csv(path, rows):
    with open(path, "w") as fh:
        fh.write("t,teacher_phase,aperture_mm,policy_gripper,teacher_gripper,"
                 + ",".join(f"policy_v{i}" for i in range(6)) + "," + ",".join(f"teacher_v{i}" for i in range(6)) + "\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")


def _episode(env, prog, student, gt, R):
    """One episode from the current reset: the video frames, the per-step rows, and the final info."""
    from screwhead.gripper_servo import channel_to_target, program_target, target_to_channel
    frames, rows, done, info = [], [], False, {}
    who = "teacher" if student is None else f"VLA{' (blind)' if student.zero == 'image' else ''}"
    while not done:
        label = prog.act().astype(np.float32)   # the teacher's action at this state (and its phase)
        t_ap = program_target(prog.phase, float(label[6]), prog.k.preshape_aperture) if gt else None
        if gt:
            label[6] = target_to_channel(t_ap)
        act = label if student is None else student.act(env)
        sn = env.snapshot()
        grip = (f"target {float(channel_to_target(act[6])) * 1000:4.1f} mm" if gt else f"grip {act[6]:+.0f}")
        teach = f"{prog.phase} {t_ap * 1000:4.1f} mm" if gt else f"{prog.phase} {label[6]:+.0f}"
        frames.append(_frame(env, R, (f"{who}: {env.language[:60]}",
                                      f"t={env.t:3d}  ap {sn['aperture'] * 1000:4.1f} mm  {grip}  | teacher {teach}")))
        rows.append([env.t, prog.phase, sn["aperture"] * 1000, float(channel_to_target(act[6])) * 1000 if gt else act[6],
                     (t_ap * 1000) if gt else label[6], *np.round(act[:6], 4), *np.round(label[:6], 4)])
        _, _, done, info = env.step(act)
    return frames, rows, info


def main() -> int:
    args = _parse()

    import imageio.v2 as imageio
    from screwhead.scripted_teacher import ScriptedTeacher
    from screwhead.teacher_env import PrivilegedEnv

    if args.policy == "teacher":
        student, gt, name = None, True, "teacher"
    else:
        student = _StudentPolicy(args.policy, args.device)
        gt, name = student.gt, Path(args.policy).stem
    env = PrivilegedEnv(args.task, seed=args.seed * 100 + args.task, render=True,
                        gripper_mode="target" if gt else "command", **RAND)
    prog = ScriptedTeacher(env)
    env.layout_check = prog.layout_feasible
    if student is not None:
        student.set_instruction(env.language)
    outdir = ROOT / args.out / name
    outdir.mkdir(parents=True, exist_ok=True)

    for k in range(max(args.episodes) + 1):
        env.reset()
        if k not in args.episodes:
            continue                               # a reset consumes the env RNG exactly as evaluation did
        frames, rows, info = _episode(env, prog, student, gt, args.res)
        tag = "ok" if info["success"] else "fail"
        stem = outdir / f"task{args.task}_ep{k}_{tag}"
        imageio.mimsave(f"{stem}.mp4", frames, fps=20, macro_block_size=1)
        _write_csv(f"{stem}.csv", rows)
        print(f"{stem}.mp4  {tag}  steps {env.t}", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
