#!/usr/bin/env python
"""Displacement probe ALONG the trajectory, teacher and student at the same states.

tools/probe_sensitivity.py asks only at the post-reset start pose, 250 mm above
the table, where the wrist camera may not see the bowl at all. A student that
consults vision near the bowl -- or only for whether the grasp took -- looks
blind there. This drives the TEACHER for K steps (so the states are the ones a
competent policy visits, regardless of how good the student is), freezes the
simulator, moves the bowl over a grid, re-renders, and asks both policies for
their next action. The teacher's response at the same state is the reference
for what aligned looks like.

Commanded linear velocity is rotated from the tool frame into the base frame by
FK on the live joints before it is compared with the displacement.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", default="checkpoints/teacher_bc_nojoints.pt")
    ap.add_argument("--zero", default="none", choices=["none", "image"])
    ap.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--advance", type=int, nargs="+", default=[0, 20, 40, 60])
    ap.add_argument("--radius", type=float, default=0.03)
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--inits", type=int, default=2, help="init states per task")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
    from distill import load_student, spec_tokens
    from rollout import clip_encoder
    from screwhead.interface import ActionSpec
    from screwhead.kinematics import fk
    from screwhead.teacher_env import TARGET, PrivilegedEnv
    from teacher_rl import build
    dev = args.device
    enc_img, enc_txt = clip_encoder(dev)
    student, s_std, _ = load_student(args.student, dev)
    teacher, _, tst, _ = build(args.teacher, "cpu")
    tok, tmask = spec_tokens(dev)
    spec = ActionSpec()
    lin = spec.pos_scale * spec.control_hz
    offs = np.linspace(-args.radius, args.radius, args.grid)
    D = np.array([[dx, dy, 0.0] for dx in offs for dy in offs])

    def teacher_act(o):
        x = (torch.as_tensor(o) - tst["obs_mu"]) / tst["obs_sd"] * tst["obs_mask"]
        with torch.no_grad():
            return np.clip(teacher(x).numpy() * tst["act_sd"].numpy(), -1, 1)

    def gain_align(V):
        Dc, Vc = D - D.mean(0), V - V.mean(0)
        G = np.linalg.lstsq(Dc[:, :2], Vc[:, :2], rcond=None)[0]
        den = np.linalg.norm(Dc[:, :2]) * np.linalg.norm(Vc[:, :2])
        return float(np.linalg.norm(G, 2)), (float((Dc[:, :2] * Vc[:, :2]).sum() / den) if den > 1e-12 else 0.0)

    rows = []
    for ti in args.tasks:
        env = PrivilegedEnv(ti, radius_m=0.0, render=True, horizon=400)
        text = enc_txt(env.language)[None]
        for init in range(args.inits):
            o = env.reset(init_index=init)
            t, done = 0, False
            for K in args.advance:
                while t < K:
                    o, _, done, _ = env.step(teacher_act(o)); t += 1
                    if done: break
                if done: break
                sim = env.env.sim
                saved = np.asarray(sim.get_state().flatten()).copy()
                jid = sim.model.joint_name2id(f"{TARGET}_joint0")
                qadr = int(sim.model.jnt_qposadr[jid])
                base = sim.data.qpos[qadr:qadr + 3].copy()
                q = torch.tensor(np.asarray(env.raw["robot0_joint_pos"]), dtype=torch.float64)
                R = fk(env.chain, q[None])[0, :3, :3].numpy()
                o_bt = o[29:32]
                Vt, Vs = [], []
                for d in D:
                    sim.set_state_from_flattened(saved)
                    sim.data.qpos[qadr:qadr + 3] = base + d
                    sim.forward()
                    od = env.obs()                                  # force_update: re-renders
                    Vt.append(R @ (teacher_act(od)[3:6] * lin))
                    a, w = env.images()
                    f = enc_img(a, w).float()
                    fa, fw = (torch.zeros_like(f[:1]), torch.zeros_like(f[1:])) if args.zero == "image" else (f[:1], f[1:])
                    st = torch.as_tensor(env.student_state(), device=dev)[None]
                    with torch.no_grad():
                        p = student(fa, fw, text, st, tok, tmask)[0, 0] * s_std
                    Vs.append(R @ (p.clamp(-1, 1).cpu().numpy()[3:6] * lin))
                sim.set_state_from_flattened(saved); sim.forward(); o = env.obs()
                gt, at = gain_align(np.array(Vt)); gs, as_ = gain_align(np.array(Vs))
                rows.append(dict(task=ti, init=init, K=K, dist_mm=float(np.linalg.norm(o_bt) * 1000),
                                 t_gain=gt, t_align=at, s_gain=gs, s_align=as_))
        env.close()
        print(f"  task {ti} done", flush=True)

    import collections
    by = collections.defaultdict(list)
    for r in rows:
        by[r["K"]].append(r)
    print(f"\n{'step K':>6} {'n':>3} {'tool-bowl mm':>12} | {'teacher gain':>12} {'align':>6} | {'student gain':>12} {'align':>6}")
    for K in sorted(by):
        g = by[K]; m = lambda k: np.median([r[k] for r in g])
        print(f"{K:6d} {len(g):3d} {m('dist_mm'):12.0f} | {m('t_gain'):12.3f} {m('t_align'):+6.2f} | "
              f"{m('s_gain'):12.3f} {m('s_align'):+6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
