#!/usr/bin/env python
"""Emit trials for the component-belief MCP.

Usage: python tools/emit_trials.py $OUT <case>

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
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_default_dtype(torch.float64)

from screwhead.kinematics import body_jacobian, fk, space_jacobian  # noqa: E402
from screwhead.mjcf import from_mjcf  # noqa: E402
from screwhead.se3 import adjoint, inverse, log_se3  # noqa: E402

ASSETS = Path(os.environ.get(
    "ROBOT_ASSETS",
    "/home/edge-host/Documents/GitHub/vla_jepa/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots",
))
ROBOTS = ("panda", "ur5e", "iiwa", "kinova3", "jaco")
N_PER_ROBOT = int(os.environ.get("N_PER_ROBOT", 200))


def geodesic(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Rotation distance without arccos blowing up near identity.

    arccos((tr-1)/2) has a sqrt singularity at 0, so a machine-precision-exact
    rotation reads as 1e-7 rad of "error". This does not.
    """
    d = Ra.T @ Rb
    return float(2 * np.arcsin(min(1.0, np.linalg.norm(d - np.eye(3), "fro") / (2 * np.sqrt(2)))))


def load(robot: str):
    import mujoco
    f = ASSETS / robot / "robot.xml"
    model = mujoco.MjModel.from_xml_path(str(f))
    return model, mujoco.MjData(model), from_mjcf(f, name=robot)


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
        q = wide_sample(chain, N_PER_ROBOT, seed=hash(robot) % 2**31)
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
        q = wide_sample(chain, max(20, N_PER_ROBOT // 10), seed=7)
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
            })
    return trials


CASES = {"poe_fk": case_poe_fk, "jacobian_fd": case_jacobian_fd}


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    out, case = sys.argv[1], sys.argv[2]
    if case not in CASES:
        print(f"unknown case {case!r}; have {sorted(CASES)}", file=sys.stderr)
        return 2
    trials = CASES[case]()
    Path(out).write_text(json.dumps({"trials": trials}, indent=2))
    print(f"{case}: {len(trials)} trials -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
