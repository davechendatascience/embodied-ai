#!/usr/bin/env python
"""Does the policy's action change when the target object moves?

The cleanest counterfactual available. Hold the robot pose and the instruction
fixed, move only the target object over a grid, and ask the policy for its next
action at each placement. A trajectory prior answers identically everywhere; a
visual controller's commanded direction follows the object.

Reports a gain in (m/s) of commanded linear velocity per metre of object
displacement, and the cosine between the commanded direction change and the
direction the object actually moved. A blind checkpoint must score exactly zero
on both, which is the control that says the measurement works.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from screwhead.interface import ActionSpec                      # noqa: E402
from screwhead.libero_env import gripper_geom, register_ur5e, remap_init_state  # noqa: E402
from screwhead.policy import MAX_DOF, BaselineHead, ScrewHead    # noqa: E402
from screwhead.spec import encode                                # noqa: E402
from screwhead.state import tool_state                           # noqa: E402


def object_free_joints(sim):
    """(name, qpos address) for every free joint that is not part of the robot."""
    import mujoco
    out = []
    for j in range(sim.model.njnt):
        if sim.model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = sim.model.joint_id2name(j)
        if name is None or name.startswith("robot") or "gripper" in name:
            continue
        out.append((name, int(sim.model.jnt_qposadr[j])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--policy", choices=["baseline", "screwhead"], default="screwhead")
    ap.add_argument("--zero", default="none", choices=["none", "image", "text", "image+text"])
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--radius", type=float, default=0.06, help="grid half-width, metres")
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--object", default="", help="substring of the joint to move; "
                                                 "default is the first non-robot free joint")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from rollout import build_chain, clip_encoder, set_joint_gains

    register_ur5e()
    _torch_load = torch.load

    def _load_trusted(*a, **kw):
        kw["weights_only"] = False
        return _torch_load(*a, **kw)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    act_std = ck["act_std"]
    model = (BaselineHead(chunk=ck["chunk"]) if args.policy == "baseline"
             else ScrewHead(chunk=ck["chunk"]))
    model.load_state_dict(ck["state_dict"]); model.to(args.device).eval()
    enc_img, enc_txt = clip_encoder(args.device)
    aspec = ActionSpec()
    twist_scale = torch.tensor([aspec.rot_scale * aspec.control_hz] * 3 +
                               [aspec.pos_scale * aspec.control_hz] * 3)

    bm = benchmark.get_benchmark_dict()[args.suite]()
    offs = np.linspace(-args.radius, args.radius, args.grid)
    gains, coss = [], []
    for ti in args.tasks:
        task = bm.get_task(ti)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        torch.load = _load_trusted
        try:
            init_states = bm.get_task_init_states(ti)
        finally:
            torch.load = _torch_load
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128,
                                 robots=["Panda"], gripper_types="PandaGripper",
                                 controller="JOINT_POSITION")
        env.reset()
        flange_to_tcp, _ = gripper_geom(env)
        chain = build_chain("panda", flange_to_tcp)
        tk, mk = encode(chain).padded(MAX_DOF)
        spec_tokens, spec_mask = tk.float().to(args.device)[None], mk.to(args.device)[None]

        env.reset()
        env.set_init_state(remap_init_state(init_states[0], env.sim))
        set_joint_gains(env, 4000.0)
        for _ in range(3):
            env.step(np.zeros(env.env.action_dim))

        cands = object_free_joints(env.sim)
        pick = [c for c in cands if args.object.lower() in c[0].lower()] if args.object else cands
        if not pick:
            print(f"[{ti}] no object joint matching {args.object!r}; have "
                  f"{[c[0] for c in cands]}"); env.close(); continue
        name, adr = pick[0]
        base = env.sim.data.qpos[adr:adr + 3].copy()

        tfeat = enc_txt(task.language)[None]
        if args.zero in ("text", "image+text"):
            tfeat = torch.zeros_like(tfeat)
        q = torch.tensor(env.env._get_observations(force_update=True)["robot0_joint_pos"], dtype=torch.float64)
        from screwhead.kinematics import fk as _fk
        R_tool = _fk(chain, q[None])[0, :3, :3].float()

        rows = []
        for dx in offs:
            for dy in offs:
                env.sim.data.qpos[adr:adr + 3] = base + np.array([dx, dy, 0.0])
                env.sim.forward()
                obs = env.env._get_observations(force_update=True)
                f = enc_img(obs["agentview_image"], obs["robot0_eye_in_hand_image"])
                if args.zero in ("image", "image+text"):
                    f = torch.zeros_like(f)
                gs = obs.get("robot0_gripper_qpos")
                gp = None if gs is None else float(gs[0] - gs[1])
                st = tool_state(chain, q[None],
                                None if gp is None else torch.tensor([gp], dtype=torch.float64)
                                ).float()
                with torch.no_grad():
                    pred = (model(f[:1], f[1:], tfeat, st.to(args.device),
                                  spec_tokens, spec_mask)[0].cpu() * act_std)
                # The head emits a BODY twist: its linear part is R^T p_dot,
                # expressed in the tool frame. The object moved in world
                # coordinates, so compare like with like or the alignment is
                # wrong by a rotation -- which, with the arm held still, is a
                # constant one that leaves the gain intact and the sign not.
                v_b = (pred[0, :6] * twist_scale)[3:]
                v = (R_tool @ v_b).numpy()
                rows.append((dx, dy, v))
        env.close()

        D = np.array([[r[0], r[1], 0.0] for r in rows])
        V = np.stack([r[2] for r in rows])
        Dc, Vc = D - D.mean(0), V - V.mean(0)
        # least-squares gain of commanded velocity on object displacement
        G = np.linalg.lstsq(Dc[:, :2], Vc[:, :2], rcond=None)[0]
        gain = float(np.linalg.norm(G, 2))
        num = float((Dc[:, :2] * Vc[:, :2]).sum())
        den = float(np.linalg.norm(Dc[:, :2]) * np.linalg.norm(Vc[:, :2]) + 1e-12)
        cos = num / den
        spread = float(np.linalg.norm(Vc, axis=1).mean())
        print(f"[{ti}] object={name}  gain={gain:6.3f} (m/s per m)  "
              f"alignment cos={cos:+.3f}  mean |dv|={spread*1000:6.2f} mm/s", flush=True)
        gains.append(gain); coss.append(cos)
    if gains:
        print(f"\nacross {len(gains)} tasks: median gain {np.median(gains):.3f}, "
              f"median alignment {np.median(coss):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
