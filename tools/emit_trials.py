#!/usr/bin/env python
"""Emit trials for the component-belief MCP.

Usage: python tools/emit_trials.py $OUT <case> [--n PER_ROBOT]

One JSON object on stdout is not the contract -- the server reads $OUT. Each
trial is one sampled configuration, not one suite: "the suite passed" throws
away the evidence every later question needs.

A robot that cannot be loaded emits NO trials and is reported on stderr. A
missing measurement is missing; it is never a passing trial.
"""
from __future__ import annotations

import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_default_dtype(torch.float64)

from screwhead.ik import decode_twist, dls, nullspace_projector, sigma_min, solve_ik  # noqa: E402
from screwhead.kinematics import body_jacobian, fk, space_jacobian  # noqa: E402
from screwhead.mjcf import from_mjcf  # noqa: E402
from screwhead.se3 import adjoint, inverse, log_se3  # noqa: E402

ASSETS = Path(os.environ.get(
    "ROBOT_ASSETS",
    "/home/edge-host/Documents/GitHub/vla_jepa/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots",
))
ROBOTS = ("panda", "ur5e", "iiwa", "kinova3", "jaco")
# Sample count is a CLI argument, not just an env var, because the declared
# `run` line is what mints a test version. Changing how much a test measures
# without changing its version would pool old and new trials silently.
N_PER_ROBOT = 200
# Declared on the run line and recorded in repro.seed. hash() is salted per
# process, so deriving a seed from it makes every run sample different
# configurations and none of them reproducible.
BASE_SEED = 0
LAM = 0.02
MAX_ITERS = 200
# Separates the two regimes cleanly: failures cluster at sigma_min 0.003-0.008,
# successes at 0.085-0.124. Declared on the run line so the bucket boundary is
# versioned with the test rather than floating in the code.
SIGMA_THRESH = 0.02


def geodesic(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Rotation distance without arccos blowing up near identity.

    arccos((tr-1)/2) has a sqrt singularity at 0, so a machine-precision-exact
    rotation reads as 1e-7 rad of "error". This does not.
    """
    d = Ra.T @ Rb
    return float(2 * np.arcsin(min(1.0, np.linalg.norm(d - np.eye(3), "fro") / (2 * np.sqrt(2)))))


def arm(robot: str):
    """Chain with the angle convention robosuite actually compiles under."""
    return from_mjcf(ASSETS / robot / "robot.xml", name=robot, angle="radian")


def regime_pair(chain, k: int, seed: int, near_singular: bool, span: float = 0.15):
    """Reachable pairs restricted to one singularity regime.

    The two regimes need separate contracts because their honest bars differ by
    a wide margin -- ~0.997 in the interior against ~0.92 near a singularity.
    Pooling them under one target_rate would let an interior regression from
    0.997 to 0.85 pass unnoticed, which is the opposite of what the gate is for.
    """
    g = torch.Generator().manual_seed(seed)
    lo, hi = chain.limits[:, 0], chain.limits[:, 1]
    picked_q0, picked_q1 = [], []
    for _ in range(200):                      # rejection sampling; ~10% qualify
        pool = chain.sample(4 * k, g)
        keep = (sigma_min(chain, pool) < SIGMA_THRESH) == near_singular
        if keep.any():
            q0 = pool[keep]
            dq = (torch.rand(q0.shape[0], chain.n, generator=g) * 2 - 1) * span
            picked_q0.append(q0)
            picked_q1.append(torch.clamp(q0 + dq, lo, hi))
        if sum(x.shape[0] for x in picked_q0) >= k:
            break
    q0 = torch.cat(picked_q0)[:k]
    q1 = torch.cat(picked_q1)[:k]
    return q0, q1, g


def reachable_pair(chain, k: int, seed: int, span: float = 0.15):
    """(start, target) where the target is reachable from the start IN LIMITS.

    Built by perturbing a sampled configuration and clamping back into the box,
    so a within-limits solution provably exists. Random start to random target
    would conflate solver failure with genuine infeasibility -- with limits on,
    only ~50% of such pairs are locally reachable at all.
    """
    g = torch.Generator().manual_seed(seed)
    q0 = chain.sample(k, g)
    lo, hi = chain.limits[:, 0], chain.limits[:, 1]
    dq = (torch.rand(k, chain.n, generator=g) * 2 - 1) * span
    return q0, torch.clamp(q0 + dq, lo, hi), g


def load(robot: str):
    import mujoco
    f = ASSETS / robot / "robot.xml"
    model = mujoco.MjModel.from_xml_path(str(f))
    return model, mujoco.MjData(model), from_mjcf(f, name=robot)


def robot_seed(robot: str) -> int:
    """Stable across processes, unlike hash() on a str."""
    return (BASE_SEED * 1_000_003 + zlib.crc32(robot.encode())) % 2**31


def wide_sample(chain, k: int, seed: int) -> torch.Tensor:
    """Full revolute span, not the declared limits.

    robosuite's robot.xml omits <compiler angle>, so a standalone parse reads
    limits as degrees (panda joint1 becomes +/-0.05 rad). Sampling those would
    exercise only the neighbourhood of home, where a wrong screw axis still
    looks right. FK correctness is a claim about the configuration space.
    """
    g = torch.Generator().manual_seed(seed)
    q = (torch.rand(k, chain.n, generator=g) * 2 - 1) * torch.pi
    for i, t in enumerate(chain.joint_types):
        if t == "prismatic":
            q[:, i] = q[:, i] * 0.1 / torch.pi
    return q


def case_poe_fk() -> list[dict]:
    import mujoco
    trials = []
    for robot in ROBOTS:
        try:
            model, data, chain = load(robot)
        except Exception as exc:  # asset problems are not evidence about the converter
            print(f"skip {robot}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, chain.tool_frame)
        q = wide_sample(chain, N_PER_ROBOT, seed=robot_seed(robot))
        T = fk(chain, q)
        for k in range(len(q)):
            data.qpos[:chain.n] = q[k].numpy()
            mujoco.mj_kinematics(model, data)
            trials.append({
                "metrics": {
                    "fk_pos_err": float(np.linalg.norm(T[k, :3, 3].numpy() - data.xpos[bid])),
                    "fk_rot_err": geodesic(T[k, :3, :3].numpy(), data.xmat[bid].reshape(3, 3)),
                },
                "conditions": {"robot": robot, "dof": chain.n, "urdf_source": "robosuite-mjcf"},
                # compatibility_key reads from repro, not conditions: a
                # different arm is a hardware swap, and must never pool.
                "repro": {"robot": robot, "seed": robot_seed(robot)},
            })
    return trials


def case_jacobian_fd() -> list[dict]:
    """Jacobian against finite differences of this module's own FK.

    Also asserts the space and body forms agree, which is a second, independent
    way to be wrong about the Adjoint order.
    """
    trials = []
    eps = 1e-6
    for robot in ROBOTS:
        try:
            _, _, chain = load(robot)
        except Exception as exc:
            print(f"skip {robot}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        q = wide_sample(chain, N_PER_ROBOT, seed=robot_seed(robot))
        Js = space_jacobian(chain, q)
        Jb = body_jacobian(chain, q)
        T = fk(chain, q)
        Ad = adjoint(inverse(T))
        for k in range(len(q)):
            cols = []
            for i in range(chain.n):
                dq = torch.zeros(chain.n, dtype=q.dtype)
                dq[i] = eps
                # d/dtheta_i of T, expressed as a space twist: (T' T^-1) unhatted
                Tp = fk(chain, (q[k] + dq)[None])[0]
                Tm = fk(chain, (q[k] - dq)[None])[0]
                V = log_se3(Tp @ inverse(Tm)[None])[0] / (2 * eps)
                cols.append(V)
            fd = torch.stack(cols, dim=-1)
            scale = max(1.0, float(Js[k].abs().max()))
            trials.append({
                "metrics": {
                    "jac_max_err": float(max(
                        (Js[k] - fd).abs().max() / scale,
                        (Jb[k] - Ad[k] @ Js[k]).abs().max(),
                    )),
                },
                "conditions": {"robot": robot, "dof": chain.n},
                "repro": {"robot": robot, "seed": robot_seed(robot)},
            })
    return trials


def _convergence(near_singular: bool) -> list[dict]:
    trials = []
    for robot in ROBOTS:
        try:
            chain = arm(robot)
        except Exception as exc:
            print(f"skip {robot}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        q0, q1, _ = regime_pair(chain, N_PER_ROBOT, robot_seed(robot), near_singular)
        res = solve_ik(chain, fk(chain, q1), q0, lam=LAM, max_iters=MAX_ITERS, trust=0.2)
        sig = sigma_min(chain, q0)
        for k in range(len(q0)):
            trials.append({
                "metrics": {
                    "converged": bool(res["converged"][k]),
                    "final_pos_err": float(res["pos_err"][k]),
                },
                "conditions": {
                    "robot": robot, "dof": chain.n, "damping": LAM,
                    "near_singular": bool(sig[k] < SIGMA_THRESH),
                    "sigma_min": round(float(sig[k]), 6),
                },
                "repro": {"robot": robot, "seed": robot_seed(robot), "damping": LAM, "iters": MAX_ITERS},
            })
    return trials


def case_ik_convergence() -> list[dict]:
    """The operating regime: away from singularities."""
    return _convergence(near_singular=False)


def case_ik_near_singular() -> list[dict]:
    """The hard regime, measured separately and held to an honest bar.

    Failures here are not slow convergence -- ur5e scores 35/38 at both 200 and
    600 iterations, so the stragglers are stuck, not starved. Retargeting must
    detect and drop these frames rather than expect the solver to rescue them.
    """
    return _convergence(near_singular=True)


def case_ik_step_bound() -> list[dict]:
    """||dtheta|| <= ||e|| / (2 lambda), the guarantee damping buys.

    Checked on the raw damped step, because that is where the bound is proved
    (the singular values of J^T(JJ^T+l^2 I)^-1 peak at 1/(2l)). Clamping is
    measured separately: it can only shorten the step, never lengthen it.
    """
    trials = []
    for robot in ROBOTS:
        try:
            chain = arm(robot)
        except Exception as exc:
            print(f"skip {robot}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        g = torch.Generator().manual_seed(robot_seed(robot))
        q = chain.sample(N_PER_ROBOT, g)
        e = torch.randn(N_PER_ROBOT, 6, generator=g) * 0.2
        J = body_jacobian(chain, q)
        raw = dls(J, e, LAM)
        clamped = decode_twist(chain, q, e, dt=1.0, lam=LAM, respect_limits=True)
        bound = torch.linalg.norm(e, dim=-1) / (2 * LAM)
        sig = sigma_min(chain, q)
        for k in range(len(q)):
            in_limits = bool(((clamped.theta[k] >= chain.limits[:, 0] - 1e-9)
                              & (clamped.theta[k] <= chain.limits[:, 1] + 1e-9)).all())
            trials.append({
                "metrics": {"bound_ok": bool(
                    float(torch.linalg.norm(raw[k])) <= float(bound[k]) + 1e-12
                    and float(torch.linalg.norm(clamped.delta[k])) <= float(bound[k]) + 1e-12
                    and in_limits
                )},
                "conditions": {
                    "robot": robot, "dof": chain.n, "damping": LAM,
                    "near_singular": bool(sig[k] < SIGMA_THRESH),
                },
                "repro": {"robot": robot, "seed": robot_seed(robot), "damping": LAM},
            })
    return trials


def case_ik_unreachable() -> list[dict]:
    """A target outside the workspace must come back as a residual, not a step.

    Built by pushing the tool position radially to three times the arm's reach,
    which no configuration can satisfy. The decoder passes only if it reports a
    non-zero residual AND does not claim convergence -- a confident step toward
    an impossible pose is FM-silent-infeasible.
    """
    trials = []
    for robot in ROBOTS:
        try:
            chain = arm(robot)
        except Exception as exc:
            print(f"skip {robot}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        g = torch.Generator().manual_seed(robot_seed(robot))
        q0 = chain.sample(N_PER_ROBOT, g)
        T = fk(chain, q0)
        # Unreachability has to be constructed from the arm's actual reach, not
        # from a scaling of the home pose: fk(zeros) is the home TOOL position,
        # and scaling outward from it can land back inside the workspace. Bound
        # the reachable set empirically, then step well outside it.
        cloud = fk(chain, chain.sample(2000, torch.Generator().manual_seed(1)))[:, :3, 3]
        centre = cloud.mean(0)
        radius = float(torch.linalg.norm(cloud - centre, dim=-1).max())
        far = T.clone()
        direction = torch.tensor([1.0, 1.0, 1.0], dtype=T.dtype) / (3 ** 0.5)
        far[:, :3, 3] = centre + direction * (radius * 5.0 + 1.0)
        res = solve_ik(chain, far, q0, lam=LAM, max_iters=MAX_ITERS, trust=0.2)
        for k in range(len(q0)):
            trials.append({
                "metrics": {"residual_reported": bool(
                    not bool(res["converged"][k]) and float(res["residual_norm"][k]) > 1e-5
                )},
                "conditions": {"robot": robot, "dof": chain.n},
                "repro": {"robot": robot, "seed": robot_seed(robot)},
            })
    return trials


def case_nullspace_drift() -> list[dict]:
    """A secondary objective must not move the tool.

    The projector uses the TRUE pseudo-inverse, not the damped one: (I - J_dls J)
    is only approximately a projector and leaks O(lambda^2) into the tool.
    """
    trials = []
    for robot in ROBOTS:
        try:
            chain = arm(robot)
        except Exception as exc:
            print(f"skip {robot}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        g = torch.Generator().manual_seed(robot_seed(robot))
        q = chain.sample(N_PER_ROBOT, g)
        J = body_jacobian(chain, q)
        z = torch.randn(N_PER_ROBOT, chain.n, generator=g)
        drift = torch.linalg.norm((J @ (nullspace_projector(J) @ z[..., None]))[..., 0], dim=-1)
        for k in range(len(q)):
            trials.append({
                "metrics": {"tool_drift": float(drift[k])},
                "conditions": {"robot": robot, "dof": chain.n, "inverse_kind": "pinv",
                               "redundant": chain.n > 6},
                "repro": {"robot": robot, "seed": robot_seed(robot), "inverse_kind": "pinv"},
            })
    return trials


CASES = {
    "poe_fk": case_poe_fk,
    "jacobian_fd": case_jacobian_fd,
    "ik_convergence": case_ik_convergence,
    "ik_near_singular": case_ik_near_singular,
    "ik_step_bound": case_ik_step_bound,
    "ik_unreachable": case_ik_unreachable,
    "nullspace_drift": case_nullspace_drift,
}


def main() -> int:
    global N_PER_ROBOT, BASE_SEED, SIGMA_THRESH, MAX_ITERS
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    out, case = sys.argv[1], sys.argv[2]
    if "--n" in sys.argv:
        N_PER_ROBOT = int(sys.argv[sys.argv.index("--n") + 1])
    if "--seed" in sys.argv:
        BASE_SEED = int(sys.argv[sys.argv.index("--seed") + 1])
    if "--sigma-thresh" in sys.argv:
        SIGMA_THRESH = float(sys.argv[sys.argv.index("--sigma-thresh") + 1])
    if "--iters" in sys.argv:
        MAX_ITERS = int(sys.argv[sys.argv.index("--iters") + 1])
    if case not in CASES:
        print(f"unknown case {case!r}; have {sorted(CASES)}", file=sys.stderr)
        return 2
    trials = CASES[case]()
    Path(out).write_text(json.dumps({"trials": trials}, indent=2))
    print(f"{case}: {len(trials)} trials -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
