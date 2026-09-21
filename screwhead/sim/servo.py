"""Execute a twist stream as an absolute joint reference, not as per-step deltas.

Measured on LIBERO's JOINT_POSITION controller at kp=4000: one control period
achieves 82% of a commanded joint step (p10 0.79, p90 0.83), in the right
direction (cos 0.998). Commanding each twist as a delta from the MEASURED joints
therefore compounds that shortfall -- the tool trails a replayed demonstration
by 38 mm after 20 steps and 102 mm by the end, outside a ~30 mm grasp basin, and
replaying a demonstration's own twist labels succeeds 0/50 where the same demos
replayed as joint targets succeed 91%.

The joint replay avoids this because its target is absolute: lag at one step is
corrected at the next. So does this. Each twist is integrated onto a REFERENCE
configuration -- decoded at the reference, not at the lagging measurement -- and
the controller is commanded toward the reference. Tracking error then settles
at a bounded offset (about 0.22 of one step) instead of accumulating.

Anti-windup: when the arm is blocked (contact, a joint limit the controller
respects but the reference does not), the reference is held within `max_lag`
of the measurement so it cannot run away and snap the arm when released.

A gain correction (command /= 0.82) would be tuned to this controller, this
stiffness and this load. The reference is correct by construction.
"""
from __future__ import annotations

import numpy as np

from ..geometry import kin_np
from ..geometry.interface import ActionSpec
from ..geometry.poe import Chain


class TwistServo:
    def __init__(self, chain: Chain, spec: ActionSpec, joint_action_scale: float,
                 lam: float = 0.01, iters: int = 3, max_lag: float = 0.05,
                 max_pose_err: float = 0.02, limit_gain: float = 0.5):
        self.chain, self.spec, self.jas = chain, spec, joint_action_scale
        self.lam, self.iters, self.max_lag, self.max_pose_err = lam, iters, max_lag, max_pose_err
        self.ref: np.ndarray | None = None
        self.T_ref: np.ndarray | None = None
        # the arithmetic runs in NumPy (screwhead/geometry/kin_np.py, held to the torch reference
        # by tests/test_kin_np.py): 17x faster per step, the same numbers
        self._np = kin_np.NpChain.of(chain)
        self.reanchors = 0          # diagnostics: how often the pose reference was abandoned
        self.posture: np.ndarray | None = None   # null-space target joints, or None
        self.posture_gain = 0.0
        self.limit_gain = limit_gain             # null-space push away from the joint limits
        lim = chain.limits.detach().cpu().numpy().astype(float)
        self._mid = lim.mean(1)
        self._span = np.maximum(lim[:, 1] - lim[:, 0], 1e-6)
        self.limit_clamps = 0

    def reset(self, theta_measured: np.ndarray) -> None:
        self.ref = np.asarray(theta_measured, np.float64).copy()
        self.T_ref = kin_np.fk(self._np, self.ref)[0]

    def command(self, theta_measured: np.ndarray, twist: np.ndarray) -> np.ndarray:
        """twist: body twist RATE (rad/s, m/s), moment first. Returns normalised joint action.

        Two integrations, each removing one measured loss:
          - the POSE reference integrates the twist exactly on SE(3), so the 4%
            per-step shrink of a damped solve cannot accumulate into the path;
          - the JOINT reference is solved toward that pose and commanded as an
            absolute target, so the controller's 18% per-step lag cannot either.
        """
        meas = np.asarray(theta_measured, np.float64)
        if self.ref is None:
            self.reset(meas)
        V = np.asarray(twist, np.float64)
        self.T_ref = self.T_ref @ kin_np.exp_twist(V * self.spec.dt)
        th = self.ref[None].copy()
        for _ in range(self.iters):
            T, J = kin_np.fk_jac(self._np, th)
            e = kin_np.log_se3(kin_np.inverse(T) @ self.T_ref[None])         # body-frame pose error
            secondary = None
            if self.limit_gain > 0:
                # The 7th joint is not specified by a 6-D twist. Spend it staying off the
                # limits: measured reaching a bowl on the cabinet, the damped solve clamped
                # against a limit on 110 of 200 steps and the tool settled 20 mm off the
                # pose it was asking for, never closing. Pulling the redundancy toward
                # mid-range in the null space leaves the tool pose untouched.
                secondary = self.limit_gain * (self._mid - th) / self._span
            if self.posture is not None and self.posture_gain > 0:
                # an explicit posture target overrides it (used to compare branches)
                secondary = self.posture_gain * (np.asarray(self.posture, float) - th)
            th, _, clamped = kin_np.decode_twist(self._np, th, e, dt=1.0, lam=self.lam,
                                                 secondary=secondary, J=J)
            self.limit_clamps += int(bool(clamped.any()))
        T = kin_np.fk(self._np, th)
        e = kin_np.log_se3(kin_np.inverse(T) @ self.T_ref[None])[0]
        if float(np.linalg.norm(e)) > self.max_pose_err:
            # the pose reference has left what the arm can reach (a limit, a
            # singularity, a collision): re-anchor it rather than let it run away
            self.T_ref = T[0]
            self.reanchors += 1
        self.ref = th[0].copy()
        self.ref = meas + np.clip(self.ref - meas, -self.max_lag, self.max_lag)
        return np.clip((self.ref - meas) / self.jas, -1.0, 1.0)
