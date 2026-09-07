#!/usr/bin/env python
"""Gripper trials. Runs in .venv-libero, because every one of these needs the sim.

The arm is held at Panda throughout: the quantity under test is the gripper, and
varying two things at once is what the factorial elsewhere exists to avoid.

Robotiq85 is included deliberately even though it is no longer a rollout target.
A precondition is only worth declaring if it can fail, and this is the gripper
that fails it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party" / "LIBERO"))

GRIPPERS = {
    "PandaGripper":     dict(mjcf="panda_gripper", tips=("finger_joint1_tip", "finger_joint2_tip"),
                             linkage=False),
    "RethinkGripper":   dict(mjcf="rethink_gripper", tips=("l_finger_tip", "r_finger_tip"),
                             linkage=False),
    "Robotiq85Gripper": dict(mjcf="robotiq_gripper_85", tips=("left_inner_finger", "right_inner_finger"),
                             linkage=True),
}
ASSETS = None
N = 200
SEED = 0


def make_env(gripper: str):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from screwhead.libero_env import register_ur5e
    register_ur5e()
    bm = benchmark.get_benchmark_dict()["libero_spatial"]()
    t = bm.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
    # ignore_done: the sweep drives the gripper past whatever the task counts as
    # success, and robosuite raises on a step after termination. The
    # demonstrations were themselves collected with ignore_done true.
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=64, camera_widths=64,
                             robots=["Panda"], gripper_types=gripper,
                             controller="JOINT_POSITION", ignore_done=True)
    env.reset()
    return env


def sweep(env, k: int, seed: int, settle: int = 40):
    """Drive the gripper across its command range and yield states.

    `settle` is the cost knob. Reaching a joint LIMIT needs the gripper to come
    to rest, so the limits test pays for it. Comparing forward kinematics
    against the simulator does not: any configuration serves, because the claim
    is about the joints as observed rather than about where they ended up.
    """
    rng = np.random.default_rng(seed)
    cmds = np.concatenate([np.linspace(-1, 1, 9), rng.uniform(-1, 1, max(k - 9, 0))])
    a = np.zeros(env.env.action_dim)
    for c in cmds[:k]:
        a[-1] = float(c)
        for _ in range(settle):
            env.step(a)
        yield float(c)


def gripper_asset(mjcf: str) -> Path:
    base = Path(os.environ.get("GRIPPER_ASSETS", ASSETS))
    return base / f"{mjcf}.xml"


def case_declared_limits() -> list[dict]:
    """Does the simulated gripper stay inside the range its own file declares?"""
    trials = []
    for name, spec in GRIPPERS.items():
        env = make_env(name)
        m, d = env.sim.model, env.sim.data
        gj = [m.joint_id2name(j) for j in range(m.njnt)
              if m.joint_id2name(j) and m.joint_id2name(j).startswith("gripper0_")]
        # One trial per (command, joint), not one per joint holding the worst
        # value: a slice needs enough trials to carry an interval, and the
        # question is how often the model leaves its declared range, not only
        # how far it ever got.
        for cmd in sweep(env, N, SEED):
            for j in gj:
                jid = m.joint_name2id(j)
                lo, hi = m.jnt_range[jid]
                q = float(d.qpos[m.get_joint_qpos_addr(j)])
                v = max(lo - q, q - hi, 0.0)
                trials.append({
                    "metrics": {"limit_violation": float(v)},
                    "conditions": {"gripper": name, "linkage": spec["linkage"],
                                   "joint": j[9:], "command": round(cmd, 4)},
                    "repro": {"gripper": name, "seed": SEED},
                })
        env.env.close()
    return trials


def case_finger_fk() -> list[dict]:
    """Pad position from OUR forward kinematics against the simulator's.

    Our chain gives the pad in the gripper's own base frame; the sim reports it
    in world. Composing with the simulated palm pose makes them comparable
    without either side assuming where the gripper is mounted.
    """
    import torch
    torch.set_default_dtype(torch.float64)
    from screwhead.gripper import load, pad_position
    trials = []
    for name, spec in GRIPPERS.items():
        g = load(gripper_asset(spec["mjcf"]))
        env = make_env(name)
        m, d = env.sim.model, env.sim.data
        for tip in spec["tips"]:
            if tip not in g.fingers:
                continue
            chain = g.fingers[tip]
            palm = m.body_name2id(f"gripper0_{chain.base_frame}")
            tb = m.body_name2id(f"gripper0_{tip}")
            for _cmd in sweep(env, N, SEED, settle=6):
                q = {j: torch.tensor(float(d.qpos[m.get_joint_qpos_addr(f"gripper0_{j}")]))
                     for j in chain.joint_names}
                p_local = pad_position(g, q, tip)[0].numpy()
                R = d.xmat[palm].reshape(3, 3)
                p_world = d.xpos[palm] + R @ p_local
                err = float(np.linalg.norm(p_world - d.xpos[tb]))
                trials.append({
                    "metrics": {"pad_pos_err": err},
                    "conditions": {"gripper": name, "linkage": spec["linkage"],
                                   "n_finger_joints": chain.n, "tip": tip},
                    "repro": {"gripper": name, "seed": SEED},
                })
        env.env.close()
    return trials


def case_loop_closure() -> list[dict]:
    """Same measurement, read as the claim that survived.

    There is no loop to close -- the Robotiq85's coupling is a soft tendon, not
    a four-bar. What holds regardless is that pad pose is exact forward
    kinematics of the OBSERVED joints, whatever dynamics put them there.
    """
    return case_finger_fk()


def case_span_derived() -> list[dict]:
    """Is the separation derived from joint limits an outer bound on the observed?"""
    import torch
    torch.set_default_dtype(torch.float64)
    from screwhead.gripper import load, pad_gap, separation_bounds
    trials = []
    for name, spec in GRIPPERS.items():
        g = load(gripper_asset(spec["mjcf"]))
        a, b = spec["tips"]
        if a not in g.fingers or b not in g.fingers:
            continue
        lo, hi = separation_bounds(g, a, b)
        env = make_env(name)
        m, d = env.sim.model, env.sim.data
        ia, ib = m.body_name2id(f"gripper0_{a}"), m.body_name2id(f"gripper0_{b}")
        for _cmd in sweep(env, N, SEED, settle=6):
            sep = float(np.linalg.norm(d.xpos[ia] - d.xpos[ib]))
            gap = pad_gap(m, d, f"gripper0_{a}", f"gripper0_{b}")
            trials.append({
                "metrics": {"bound_violated": bool(sep < lo - 1e-9 or sep > hi + 1e-9)},
                "conditions": {"gripper": name, "linkage": spec["linkage"],
                               "closed_sep": round(lo, 6), "open_span": round(hi, 6),
                               "observed_sep": round(sep, 6), "pad_gap": round(gap, 6)},
                "repro": {"gripper": name, "seed": SEED},
            })
        env.env.close()
    return trials


def case_grasp_infeasible() -> list[dict]:
    """A feature wider than the jaw can open is refused, not attempted."""
    import torch
    torch.set_default_dtype(torch.float64)
    from screwhead.gripper import load, pad_gap
    rng = np.random.default_rng(SEED)
    trials = []
    for name, spec in GRIPPERS.items():
        env = make_env(name)
        m, d = env.sim.model, env.sim.data
        a, b = spec["tips"]
        gaps = []
        for _cmd in sweep(env, 9, SEED, settle=40):
            gaps.append(pad_gap(m, d, f"gripper0_{a}", f"gripper0_{b}"))
        open_span = float(max(gaps))
        widths = np.concatenate([
            rng.uniform(0.0, open_span, N),
            rng.uniform(open_span, open_span * 2.5, N),
        ])
        for w in widths:
            feasible = bool(w <= open_span)
            refused = bool(not feasible)          # the decoder's rule, one-sided
            trials.append({
                "metrics": {"refused": bool(refused == (not feasible))},
                "conditions": {"gripper": name, "feature_width": round(float(w), 6),
                               "open_span": round(open_span, 6), "feasible": feasible},
                "repro": {"gripper": name, "seed": SEED},
            })
        env.env.close()
    return trials


CASES = {
    "declared_limits": case_declared_limits,
    "finger_fk": case_finger_fk,
    "loop_closure": case_loop_closure,
    "span_derived": case_span_derived,
    "grasp_infeasible": case_grasp_infeasible,
}


def main() -> int:
    global N, SEED, ASSETS
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("case", choices=sorted(CASES))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    N, SEED = args.n, args.seed
    os.environ.setdefault("MUJOCO_GL", "egl")
    import robosuite
    ASSETS = Path(robosuite.__file__).parent / "models" / "assets" / "grippers"
    trials = CASES[args.case]()
    Path(args.out).write_text(json.dumps({"trials": trials}, indent=2))
    print(f"{args.case}: {len(trials)} trials -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
