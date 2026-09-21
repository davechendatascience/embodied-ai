"""Can the arm actually use a pose? Inverse kinematics, conditioning, and the simulator.

Grasp synthesis (grasp_planner.py) proposes; this disposes. Three things decide, each
learned by the teacher failing without it:
  convergence    the pose is reachable at all
  conditioning   the arm has room to move there (smallest singular value, distance to the
                 joint limits): the highest converging crossing put the arm at sigma 0.021,
                 nearly straight, where the servo cannot track a twist
  the simulator  the ARM, not just the tool, is not inside a fixture: every candidate for a
                 bowl beside the cabinet converged with 0.75 rad of margin and the forearm
                 sat in the cabinet base
"""
from __future__ import annotations

import numpy as np

from . import contacts
from .frames import Z, pose
from .kin_np import NpChain, sigma_min, solve_ik
from .scene import _geom_half

IK = dict(lam=0.02, max_iters=200, trust=0.2)
MIN_SIGMA = 0.05          # a crossing or carry pose must be at least this well conditioned
MIN_MARGIN = 0.15         # rad from every joint limit
SIGMA_WEIGHT = 8.0        # sigma is scaled to compete with joint margin in a grasp's score
TOUCH_COST = 0.3          # score lost for brushing something loose at a candidate pose
CONDITION_BARS = (0.35, 0.20, 0.05)   # take the first candidate clearing the highest bar
COLUMN_PROBES = (0.05, 0.10)          # heights above the pre-grasp checked on the way down
TRANSIT_STEP = 0.05       # crossing heights tried, top down
CARRY_STEP = 0.03
CORRIDOR = 0.08           # obstacles within this of the path's footprint count
CLEARANCE = 0.05          # cross this far above the tallest of them
CAP_ABOVE = 0.30          # never cross more than this above the target
CARRY_FLOOR = 0.05        # a carry at least this above the drop point
CARRY_EXTRA = 0.03        # the carried object's bottom clears obstacles by this more
EPS_LEN2 = 1e-9


class Reach:
    def __init__(self, env, config):
        self.env, self.scene, self.k = env, env.scene, config
        self.last_choice: dict = {}
        self._geom_mask: np.ndarray | None = None
        self._geom_half: np.ndarray | None = None

    # -- primitives -----------------------------------------------------------------------
    def solve(self, Ts: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(joints, converged, smallest singular value, joint margin) per pose, from here."""
        c = NpChain.of(self.env.chain)
        q0 = np.asarray(self.env.raw["robot0_joint_pos"], float)
        res = solve_ik(c, np.stack(Ts), q0[None], **IK)
        th = res["theta"]
        margin = np.minimum(th - c.limits[:, 0], c.limits[:, 1] - th).min(1)
        return th, res["converged"], sigma_min(c, th), margin

    def collides(self, theta: np.ndarray, allow: int = -1, aperture: float | None = None) -> int:
        """0 clear, 1 the arm brushes something loose, 2 it is inside something fixed --
        found by putting the configuration in the simulator, looking, and restoring it.

        The fingers go where the grasp will have them: screened fully open (77 mm), every
        rim grasp on the bowl in the drawer hit the drawer walls and none was feasible.
        """
        sim = self.env.env.sim
        qpos, qvel = sim.data.qpos.copy(), sim.data.qvel.copy()
        sim.data.qpos[self.env.joint_indexes] = np.asarray(theta, float)
        if aperture is not None:
            sim.data.qpos[self.env.gripper_indexes] = [aperture / 2, -aperture / 2]
        sim.forward()
        grade = contacts.contact_grade(self.scene.m, self.scene.d, allow)
        sim.data.qpos[:] = qpos
        sim.data.qvel[:] = qvel
        sim.forward()
        return grade

    # -- choosing a grasp -------------------------------------------------------------------
    def choose(self, cands: list, via: np.ndarray | None, allow: int = -1, strict: bool = False):
        """The candidate grasp the arm can use: reachable at the grasp, down the column
        above it, and (when known) at the place pose the same grasp must reach.

        Shape proposes in list order (narrow faces first), conditioning decides how far
        down that list to look: the first merely-feasible candidate put the arm where the
        damped solve clamps on a limit, the best-conditioned one ignored the shape.
        """
        Ts, meta = [], []
        for i, (R, p_g, _w, app) in enumerate(cands):
            pre = p_g - app * self.k.approach
            column = [pre + Z * h for h in COLUMN_PROBES]
            for pt in [p_g, pre, *column] + ([via] if via is not None else []):
                Ts.append(pose(R, pt))
                meta.append(i)
        th, conv, sig, margin = self.solve(Ts)
        scores = {i: s for i in range(len(cands))
                  if (s := self._score(cands[i], [j for j, mi in enumerate(meta) if mi == i],
                                       th, conv, sig, margin, allow)) is not None}
        if not scores:
            if strict:
                return None
            self.last_choice = dict(n=len(cands), feasible=0, score=None, forced=True)
            return cands[0]
        pick = next((ok[0] for bar in CONDITION_BARS
                     if (ok := [i for i in sorted(scores) if scores[i] >= bar])), None)
        if pick is None:
            pick = max(scores, key=scores.get)
        self.last_choice = dict(n=len(cands), feasible=len(scores), score=float(scores[pick]), forced=False)
        return cands[pick]

    def _score(self, cand, sel, th, conv, sig, margin, allow) -> float | None:
        if not all(conv[j] for j in sel):
            return None
        ap = min(self.k.max_grip, cand[2] + self.k.grip_margin)
        touch = max(self.collides(th[j], allow, ap) for j in sel)
        if touch == 2:                                  # reachable on paper, the arm in the cabinet
            return None
        return min(min(margin[j] for j in sel), SIGMA_WEIGHT * min(sig[j] for j in sel)) - TOUCH_COST * touch

    # -- how high to cross ------------------------------------------------------------------
    def transit_height(self, p_from: np.ndarray, p_to: np.ndarray, exclude: int) -> float:
        """A height that clears everything standing between here and there: every geom
        whose footprint is within 8 cm of the line, minus the robot and the target, at its
        exact top (not its bounding radius -- the table's is a metre)."""
        m, d = self.scene.m, self.scene.d
        if self._geom_mask is None:
            self._geom_mask = np.array([not contacts.is_robot(contacts.body_name(m, int(m.geom_bodyid[g])))
                                        for g in range(m.ngeom)])
            self._geom_half = np.stack([_geom_half(m, g) for g in range(m.ngeom)])
        pos = d.geom_xpos - self.scene.base
        tops = pos[:, 2] + (np.abs(d.geom_xmat.reshape(-1, 3, 3)[:, 2, :]) * self._geom_half).sum(1)
        rad = np.linalg.norm(self._geom_half[:, :2], axis=1)
        a, b = np.asarray(p_from[:2], float), np.asarray(p_to[:2], float)
        ab = b - a
        L2 = float(ab @ ab)
        t = np.clip(((pos[:, :2] - a) @ ab) / L2, 0.0, 1.0) if L2 > EPS_LEN2 else np.zeros(len(pos))
        near = np.linalg.norm(pos[:, :2] - (a + t[:, None] * ab), axis=1) - rad < CORRIDOR
        keep = near & self._geom_mask & (np.asarray(m.geom_bodyid) != exclude)
        floor = float(p_to[2] + self.k.approach)
        if not keep.any():
            return floor
        return float(min(max(np.max(tops[keep]) + CLEARANCE, floor), p_to[2] + CAP_ABOVE))

    def reachable_transit(self, R: np.ndarray, p_start: np.ndarray, body: int) -> float:
        """The highest crossing height the arm can hold with room to move, at or below the
        one the obstacles ask for (clearing the cabinet asks for a pose the Panda cannot
        hold: 565 re-anchors in one episode, touching nothing)."""
        top = self.transit_height(self.env.snapshot()["p_tool"], p_start, body)
        floor = float(p_start[2] + self.k.approach)
        hs = list(np.arange(top, floor - EPS_LEN2, -TRANSIT_STEP)) or [floor]
        if hs[-1] > floor:
            hs.append(floor)
        th, conv, sig, margin = self.solve([pose(R, np.array([p_start[0], p_start[1], h])) for h in hs])
        for i, h in enumerate(hs):
            if conv[i] and sig[i] > MIN_SIGMA and margin[i] > MIN_MARGIN and self.collides(th[i], body) < 2:
                return float(h)
        return floor

    def carry_height(self, box, q, target_q, R, p, body: int) -> float:
        """Object-origin height for a carry: its bottom clears what is between here and the
        drop, and the arm holds the tool there with room to move ALONG THE WHOLE CARRY --
        over the pick point, halfway, and over the drop (a fixed "region top + 12 cm"
        stalled a top-drawer carry 200-400 mm short).

        Checked over the drop alone, a marginal grasp was lifted, over the pick point, into
        a joint limit at the same height every episode: the servo clamped, the wrist jerked
        and the bowl fell (spatial 6, six of six failures, lost at 0.22 m)."""
        hang = float(q[2] - (box.world_centre[2] - np.abs(box.R @ np.diag(box.half)).sum(1)[2]))
        clear = self.transit_height(q, target_q, body) - CLEARANCE + hang + CARRY_EXTRA
        lo = float(target_q[2] + CARRY_FLOOR)
        hi = float(min(max(clear, lo), target_q[2] + CAP_ABOVE))
        hs = list(np.arange(hi, lo - EPS_LEN2, -CARRY_STEP)) or [lo]
        tool_off = p - q
        stops = [q[:2], (q[:2] + target_q[:2]) / 2, target_q[:2]]      # pick, halfway, drop
        _th, conv, sig, margin = self.solve([pose(R, np.array([xy[0], xy[1], h]) + tool_off)
                                             for h in hs for xy in stops])
        ok = (conv & (sig > MIN_SIGMA) & (margin > MIN_MARGIN)).reshape(len(hs), len(stops)).all(1)
        return next((float(h) for h, good in zip(hs, ok, strict=True) if good), lo)
