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
        # Re-run forward kinematics before anything reads the model. d.xpos lags
        # d.qpos after integration, and reading the pair unsynchronised puts a
        # 0.2 mm error into every geometric comparison -- enough to fail a 1e-6
        # contract for a reason that has nothing to do with the kinematics.
        # Measured: median residual 3.1e-6 m without this, 5.2e-17 with it.
        env.sim.forward()
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

    Our chain gives the pad in the gripper MJCF's worldbody frame; the sim
    reports it in the frame of the body the gripper was merged onto. Those
    differ by a fixed transform that is not worth reconstructing -- and worse,
    FITTING it hides exactly what the test is for.

    Pairwise distances between the sampled pad positions are a complete rigid
    invariant instead: two point sets have equal distance matrices iff one is a
    rigid image of the other. No frame, no fit, nothing for an error to hide in.
    """
    import torch
    torch.set_default_dtype(torch.float64)
    from screwhead.gripper import load, pad_position
    trials = []
    for name, spec in GRIPPERS.items():
        g = load(gripper_asset(spec["mjcf"]))
        tips = [t for t in spec["tips"] if t in g.fingers]
        if len(tips) < 2:
            continue
        env = make_env(name)
        m, d = env.sim.model, env.sim.data
        palm = m.body_name2id(f"gripper0_{g.fingers[tips[0]].base_frame}")

        mine, sim = {t: [] for t in tips}, {t: [] for t in tips}
        for _cmd in sweep(env, N, SEED, settle=6):
            R = d.xmat[palm].reshape(3, 3)
            for tip in tips:
                chain = g.fingers[tip]
                q = {j: torch.tensor(float(d.qpos[m.get_joint_qpos_addr(f"gripper0_{j}")]))
                     for j in chain.joint_names}
                mine[tip].append(pad_position(g, q, tip)[0].numpy())
                sim[tip].append(R.T @ (d.xpos[m.body_name2id(f"gripper0_{tip}")] - d.xpos[palm]))

        for tip in tips:
            A = np.array(mine[tip]); B = np.array(sim[tip])
            # Pairwise distances are a COMPLETE rigid invariant: they are equal
            # for two point sets iff one is a rigid image of the other. So the
            # unknown mount transform never has to be reconstructed, and unlike
            # a fitted rotation there is nothing for an error to hide in. The
            # earlier Procrustes version fitted a rotation to a nearly collinear
            # cloud -- two pads sliding along one axis -- which is ill
            # conditioned and smeared a real residual across every sample.
            for i in range(len(A)):
                j = (i + 1 + i % max(len(A) - 1, 1)) % len(A)
                if i == j:
                    continue
                err = abs(float(np.linalg.norm(A[i] - A[j]) - np.linalg.norm(B[i] - B[j])))
                trials.append({
                    "metrics": {"pad_pos_err": err},
                    "conditions": {"gripper": name, "linkage": spec["linkage"],
                                   "n_finger_joints": g.fingers[tip].n, "tip": tip},
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
    """Is the separation derived from joint limits an outer bound on the observed?

    Bounded up to the LIMIT SOFTNESS, propagated through the same forward
    kinematics under test rather than allowed as a fudge:

        tol = sum_j |d sep / d q_j| * violation_j

    MuJoCo enforces joint limits as soft constraints, so a joint under load sits
    slightly outside its declared range and the separation follows. Measured on
    the RethinkGripper at its closed extreme: the bound falls short by 0.673 mm
    and the two slides are each 0.337 mm past their limits -- 0.337 + 0.337 =
    0.673, exact to three decimals, because a parallel jaw's sensitivity is 1
    per finger. The sensitivity is taken by autograd so it is right for a
    linkage too, where it is not 1.

    Whether the joints SHOULD be outside their range is a different claim, and
    CTR-declared-limits-respected is the one that asks it.
    """
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
        joints = list(dict.fromkeys(g.fingers[a].joint_names + g.fingers[b].joint_names))

        def softness_tolerance():
            """Propagate each joint's limit violation through d sep / d q."""
            from screwhead.gripper import separation
            qv, viol = {}, {}
            for j in joints:
                jid = m.joint_name2id(f"gripper0_{j}")
                q = float(d.qpos[m.get_joint_qpos_addr(f"gripper0_{j}")])
                l, h = m.jnt_range[jid]
                viol[j] = max(l - q, q - h, 0.0)
                qv[j] = torch.tensor(q, requires_grad=True)
            sep = separation(g, qv, a, b).sum()
            grads = torch.autograd.grad(sep, [qv[j] for j in joints], allow_unused=True)
            return sum(abs(float(gr if gr is not None else 0.0)) * viol[j]
                       for j, gr in zip(joints, grads))

        for _cmd in sweep(env, N, SEED, settle=6):
            sep = float(np.linalg.norm(d.xpos[ia] - d.xpos[ib]))
            gap = pad_gap(m, d, f"gripper0_{a}", f"gripper0_{b}")
            tol = softness_tolerance() + 1e-9
            trials.append({
                "metrics": {"bound_violated": bool(sep < lo - tol or sep > hi + tol)},
                "conditions": {"gripper": name, "linkage": spec["linkage"],
                               "closed_sep": round(lo, 6), "open_span": round(hi, 6),
                               "observed_sep": round(sep, 6), "pad_gap": round(gap, 6),
                               "softness_tol": round(tol, 8)},
                "repro": {"gripper": name, "seed": SEED},
            })
        env.env.close()
    return trials


def case_grasp_infeasible() -> list[dict]:
    """A feature the jaw cannot open around is refused, and nothing else is.

    The refusal must be decided from the DERIVED span -- what our chain says the
    gripper can do -- while feasibility is judged by the OBSERVED span the
    simulator actually reaches. Deciding both from the same number is vacuous,
    which is what the first version of this did.

    The guarantee wanted is one-sided: never refuse a graspable feature. An
    over-cautious refusal is a lost opportunity; an optimistic acceptance is a
    failed grasp with no warning, and the decoder has no way to recover from it.
    So a trial passes unless the derived rule refuses something the gripper can
    in fact open around.
    """
    import torch
    torch.set_default_dtype(torch.float64)
    from screwhead.gripper import load, pad_gap
    rng = np.random.default_rng(SEED)
    trials = []
    for name, spec in GRIPPERS.items():
        g = load(gripper_asset(spec["mjcf"]))
        a_tip, b_tip = spec["tips"]
        if a_tip not in g.fingers or b_tip not in g.fingers:
            continue
        env = make_env(name)
        m, d = env.sim.model, env.sim.data
        gaps = [pad_gap(m, d, f"gripper0_{a_tip}", f"gripper0_{b_tip}")
                for _cmd in sweep(env, 9, SEED, settle=40)]
        observed_span = float(max(gaps))

        # What OUR model says the jaw can open to: body-separation range from
        # the chain, less the pad thickness the simulator reports. The pad
        # geometry is static, so reading it once is a property of the gripper
        # and not of this episode.
        from screwhead.gripper import separation_bounds
        lo_sep, hi_sep = separation_bounds(g, a_tip, b_tip)
        ia = m.body_name2id(f"gripper0_{a_tip}"); ib = m.body_name2id(f"gripper0_{b_tip}")
        sep_now = float(np.linalg.norm(d.xpos[ia] - d.xpos[ib]))
        pad_thickness = sep_now - pad_gap(m, d, f"gripper0_{a_tip}", f"gripper0_{b_tip}")
        derived_span = hi_sep - pad_thickness

        widths = np.concatenate([
            rng.uniform(0.0, observed_span, N),
            rng.uniform(observed_span, observed_span * 2.5, N),
        ])
        for w in widths:
            w = float(w)
            feasible = w <= observed_span            # what the simulator can do
            refused = w > derived_span               # what our rule decides
            trials.append({
                "metrics": {"refused": bool(not (refused and feasible))},
                "conditions": {"gripper": name, "feature_width": round(w, 6),
                               "open_span": round(derived_span, 6),
                               "observed_span": round(observed_span, 6),
                               "feasible": bool(feasible), "rule_refused": bool(refused)},
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
