"""The NumPy kinematics (screwhead/geometry/kin_np.py) must agree with the torch reference.

The servo is the execution interface the teacher and the student share, so swapping its
arithmetic is only allowed if nothing observable changes. These tests hold the fast path
to the reference on the arm LIBERO uses, including the branches that are easy to get
subtly wrong: joint-limit clamping inside the damped solve, the log near a half turn, and
rows of a batched IK that converge at different iterations.

  .venv-libero/bin/python -m pytest tests/test_kin_np.py -q
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from screwhead.geometry import ik, kin_np, se3
from screwhead.geometry.kinematics import body_jacobian, fk
from screwhead.sim.libero import panda_chain

CHAIN = panda_chain()
NP = kin_np.NpChain.of(CHAIN)
RNG = np.random.default_rng(0)
LIM = CHAIN.limits.numpy()


def configs(k: int, margin: float = 0.0) -> np.ndarray:
    lo, hi = LIM[:, 0] + margin, LIM[:, 1] - margin
    return lo + RNG.random((k, len(lo))) * (hi - lo)


def test_fk_and_body_jacobian():
    th = configs(64)
    T_ref = fk(CHAIN, torch.tensor(th)).numpy()
    J_ref = body_jacobian(CHAIN, torch.tensor(th)).numpy()
    T, J = kin_np.fk_jac(NP, th)
    assert np.abs(T - T_ref).max() < 1e-12
    assert np.abs(J - J_ref).max() < 1e-12
    assert np.abs(kin_np.fk(NP, th) - T_ref).max() < 1e-12


@pytest.mark.parametrize("angle", [0.0, 1e-10, 1e-4, 0.3, 2.0, np.pi - 1e-9, np.pi])
def test_exp_and_log(angle):
    axis = RNG.normal(size=(16, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    V = np.concatenate([axis * angle, RNG.normal(size=(16, 3)) * 0.2], 1)
    T_ref = se3.exp_twist(torch.tensor(V)).numpy()
    T = kin_np.exp_twist(V)
    assert np.abs(T - T_ref).max() < 1e-12
    L_ref = se3.log_se3(torch.tensor(T_ref)).numpy()
    assert np.abs(kin_np.log_se3(T_ref) - L_ref).max() < 1e-9


@pytest.mark.parametrize("secondary", [False, True])
def test_decode_twist_with_clamping(secondary):
    th = configs(128)
    th[::3] = np.clip(th[::3], None, LIM[:, 1] - 1e-3)       # a third sit on an upper limit
    th[1::3] = np.clip(th[1::3], LIM[:, 0] + 1e-3, None)
    tw = RNG.normal(size=(128, 6)) * 0.5
    sec = (RNG.normal(size=(128, 7)) * 0.1) if secondary else None
    ref = ik.decode_twist(CHAIN, torch.tensor(th), torch.tensor(tw), dt=1.0, lam=0.01,
                          secondary=None if sec is None else torch.tensor(sec))
    new, delta, clamped = kin_np.decode_twist(NP, th, tw, dt=1.0, lam=0.01, secondary=sec)
    assert clamped.any(), "the test must exercise the clamping loop"
    assert np.array_equal(clamped, ref.clamped.numpy())
    assert np.abs(new - ref.theta.numpy()).max() < 1e-10
    assert np.abs(delta - ref.delta.numpy()).max() < 1e-10


def test_solve_ik_batched():
    goal = configs(48, margin=0.2)
    targets = kin_np.fk(NP, goal)
    q0 = configs(1, margin=0.3)[0]
    ref = ik.solve_ik(CHAIN, torch.tensor(targets), torch.tensor(q0)[None].expand(48, -1),
                      lam=0.02, max_iters=200, trust=0.2)
    out = kin_np.solve_ik(NP, targets, q0[None], lam=0.02, max_iters=200, trust=0.2)
    assert np.array_equal(out["converged"], ref["converged"].numpy())
    assert np.array_equal(out["iters"], ref["iters"].numpy())
    assert np.abs(out["theta"] - ref["theta"].numpy()).max() < 1e-8


def test_sigma_min():
    th = configs(32)
    ref = ik.sigma_min(CHAIN, torch.tensor(th)).numpy()
    assert np.abs(kin_np.sigma_min(NP, th) - ref).max() < 1e-10
