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

from ..sim import contacts
from ..geometry.frames import Z, pose
from ..geometry.kin_np import NpChain, sigma_min, solve_ik
from ..sim.scene import geom_box

IK = dict(lam=0.02, max_iters=200, trust=0.2)
MIN_SIGMA = 0.05          # a crossing or carry pose must be at least this well conditioned
MIN_MARGIN = 0.15         # rad from every joint limit
SIGMA_WEIGHT = 8.0        # sigma is scaled to compete with joint margin in a grasp's score
CONDITION_BARS = (0.35, 0.20, 0.05)   # take the first candidate clearing the highest bar
COLUMN_PROBES = (0.05, 0.10)          # heights above the pre-grasp checked on the way down
TRANSIT_STEP = 0.05       # crossing heights tried, top down
CARRY_STEP = 0.03
LINE_STEP = 0.01          # m between the points of a line tested against the geoms' boxes
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
        self.last_carry: dict = {}
        # DEF-witness-or-refusal says a query answers with a candidate satisfying its predicate or
        # with nothing, and that is what this should do. It is off until the screen is calibrated,
        # because DEF-reachable-pose calls itself a conservative stand-in and refusing on a
        # conservative screen refuses work the arm can do: measured over 1500 episodes, turning it
        # on cost 56 successes, 50 of them libero_goal 6, which the screen rejects 50 times out of
        # 50 and which the rejected grasp solves 50 times out of 50. Until P(success | the screen
        # found nothing) is measured and the thresholds moved, forcing loses less than refusing.
        self.refuse_when_empty = False
        self._geom_mask: np.ndarray | None = None
        self._geom_half: np.ndarray | None = None
        self._geom_centre: np.ndarray | None = None

    # -- primitives -----------------------------------------------------------------------
    def solve(self, Ts: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(joints, converged, smallest singular value, joint margin) per pose, from here."""
        c = NpChain.of(self.env.chain)
        q0 = np.asarray(self.env.raw["robot0_joint_pos"], float)
        res = solve_ik(c, np.stack(Ts), q0[None], **IK)
        th = res["theta"]
        margin = np.minimum(th - c.limits[:, 0], c.limits[:, 1] - th).min(1)
        return th, res["converged"], sigma_min(c, th), margin

    def all_reachable(self, Ts: list[np.ndarray]) -> bool:
        """Every pose converges with DEF-reachable-pose's conditioning and joint margin."""
        _th, conv, sig, margin = self.solve(Ts)
        return bool((conv & (sig > MIN_SIGMA) & (margin > MIN_MARGIN)).all())

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
    def choose(self, cands: list, via: np.ndarray | None, allow: int = -1, extra=None,
               force: bool = False):
        """The candidate grasp the arm can use, or None when there is no such candidate.

        Reachable at the grasp, down the column above it, and (when known) at the place pose the
        same grasp must reach. Shape proposes in list order (narrow faces first), conditioning
        decides how far down that list to look: the first merely-feasible candidate put the arm
        where the damped solve clamps on a limit, the best-conditioned one ignored the shape.
        `extra(cand)` adds whole poses the candidate must also reach (the end of a drive).

        None means none: this used to return cands[0] when nothing passed, so a caller received a
        grasp that failed the very screen it asked for and could not tell. Measured on libero_goal
        9, the forced grasp was taken on a 58.5 mm bottle with feasible 0, and the jaws then
        oscillated between held and lost at an 11 mm aperture for four hundred steps before tipping
        it over -- 1 success in 50. DEF-witness-or-refusal: a query answers with a candidate
        satisfying the predicate it names, or with nothing.
        """
        Ts, meta, kind = [], [], []
        for i, (R, p_g, _w, app) in enumerate(cands):
            pre = p_g - app * self.k.approach
            column = [pre + Z * h for h in COLUMN_PROBES]
            names = ["grasp", "pre"] + [f"column{h:g}" for h in COLUMN_PROBES] + \
                    (["via"] if via is not None else [])
            for pt, nm in zip([p_g, pre, *column] + ([via] if via is not None else []), names,
                              strict=True):
                Ts.append(pose(R, pt))
                meta.append(i)
                kind.append(nm)
            for T in (extra(cands[i]) if extra is not None else []):
                Ts.append(T)
                meta.append(i)
                kind.append("drive")
        th, conv, sig, margin = self.solve(Ts)
        rated = {i: r for i in range(len(cands))
                 if (r := self._score(cands[i], [j for j, mi in enumerate(meta) if mi == i],
                                      th, conv, sig, margin, allow)) is not None}
        if not rated:
            # Which probe emptied the screen, and on what. _score rejects on exactly two things --
            # IK failing to converge at some probed pose, or the arm penetrating something -- so
            # naming the pose and the reason says what to loosen. DEF-reachable-pose calls itself
            # "a conservative stand-in", and refusing on a conservative screen refuses states the
            # arm can in fact reach: measured, libero_goal 6 is 50/50 on a grasp this screen
            # rejects every time.
            self.last_choice = dict(n=len(cands), feasible=0, score=None, forced=force,
                                    rejected=self._why_empty(meta, kind, th, conv, allow, cands))
            # Forcing is the caller's last resort, never a tier's: asked tier by tier, an empty
            # tier must answer None so the caller moves on to the next one. Forcing here on every
            # call made the first tier force its own first candidate and hid feasible grasps in
            # later tiers -- which changed libero_spatial 4 from 41 to 45 and 6 from 44 to 42 at
            # the same seed while "forcing restored" was claimed and not measured.
            return cands[0] if force else None
        # touch nothing if anything clean is feasible. Brushing a loose object used to cost
        # only 0.3 of score, so a candidate early in the list that pushed the palm into the
        # wine bottle beside the bowl beat clean ones further down -- the tool was shoved off
        # its column at the pre-grasp and never recovered (goal 3)
        clean = {i: sc for i, (sc, touch) in rated.items() if touch == 0}
        scores = clean or {i: sc for i, (sc, _touch) in rated.items()}
        pick = next((ok[0] for bar in CONDITION_BARS
                     if (ok := [i for i in sorted(scores) if scores[i] >= bar])), None)
        if pick is None:
            pick = max(scores, key=scores.get)
        self.last_choice = dict(n=len(cands), feasible=len(rated), clean=len(clean),
                                score=float(scores[pick]))
        return cands[pick]

    def _why_empty(self, meta, kind, th, conv, allow, cands) -> dict:
        """Per probed pose kind, how many candidates it rejected and how -- the calibration datum."""
        out: dict[str, int] = {}
        for i in range(len(cands)):
            sel = [j for j, mi in enumerate(meta) if mi == i]
            ap = min(self.k.max_grip, cands[i][2] + self.k.grip_margin)
            for j in sel:
                if not conv[j]:
                    out[f"{kind[j]}:no_ik"] = out.get(f"{kind[j]}:no_ik", 0) + 1
                elif self.collides(th[j], allow, ap) == 2:
                    out[f"{kind[j]}:collides"] = out.get(f"{kind[j]}:collides", 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1])[:6])

    def _score(self, cand, sel, th, conv, sig, margin, allow) -> tuple[float, int] | None:
        """(conditioning score, contact grade) of a candidate, or None if unusable."""
        if not all(conv[j] for j in sel):
            return None
        ap = min(self.k.max_grip, cand[2] + self.k.grip_margin)
        touch = max(self.collides(th[j], allow, ap) for j in sel)
        if touch == 2:                                  # reachable on paper, the arm in the cabinet
            return None
        return min(min(margin[j] for j in sel), SIGMA_WEIGHT * min(sig[j] for j in sel)), touch

    # -- how high to cross ------------------------------------------------------------------
    def _geoms_now(self):
        """(centres, plan radii, bottoms, tops, keep) of every geom now, base frame; keep masks out
        the robot's geoms, which a test of the scene must not see (a ray from above the drawer's
        bar met the hand over it and called the bar covered)."""
        m, d = self.scene.m, self.scene.d
        if self._geom_mask is None:
            self._geom_mask = np.array([not contacts.is_robot(contacts.body_name(m, int(m.geom_bodyid[g])))
                                        for g in range(m.ngeom)])
            boxes = [geom_box(m, g) for g in range(m.ngeom)]
            self._geom_centre = np.stack([b[0] for b in boxes])
            self._geom_half = np.stack([b[1] for b in boxes])
        xmat = d.geom_xmat.reshape(-1, 3, 3)
        pos = d.geom_xpos - self.scene.base + np.einsum("gij,gj->gi", xmat, self._geom_centre)
        hz = (np.abs(xmat[:, 2, :]) * self._geom_half).sum(1)
        rad = np.linalg.norm(self._geom_half[:, :2], axis=1)
        return pos, rad, pos[:, 2] - hz, pos[:, 2] + hz, self._geom_mask

    def column_clear(self, xy: np.ndarray, radius: float, z_from: float, ignore: set[int]) -> bool:
        """No geom of the scene (the robot's aside, and bodies in `ignore`) lies over a disc of
        `radius` about `xy` anywhere above `z_from`."""
        pos, rad, bottoms, _tops, keep = self._geoms_now()
        near = np.linalg.norm(pos[:, :2] - np.asarray(xy, float)[:2], axis=1) - rad < radius
        body = np.asarray(self.scene.m.geom_bodyid)
        mine = np.isin(body, list(ignore)) if ignore else np.zeros(len(body), bool)
        return not bool((near & keep & ~mine & (bottoms > z_from)).any())

    def transit_height(self, p_from: np.ndarray, p_to: np.ndarray, exclude: int) -> float:
        """A height that clears everything standing between here and there: every geom
        whose footprint is within 8 cm of the line, minus the robot and the target, at its
        exact top (not its bounding radius -- the table's is a metre)."""
        m, d = self.scene.m, self.scene.d
        if self._geom_mask is None:
            self._geom_mask = np.array([not contacts.is_robot(contacts.body_name(m, int(m.geom_bodyid[g])))
                                        for g in range(m.ngeom)])
            boxes = [geom_box(m, g) for g in range(m.ngeom)]
            self._geom_centre = np.stack([b[0] for b in boxes])
            self._geom_half = np.stack([b[1] for b in boxes])
        xmat = d.geom_xmat.reshape(-1, 3, 3)
        pos = d.geom_xpos - self.scene.base + np.einsum("gij,gj->gi", xmat, self._geom_centre)
        tops = pos[:, 2] + (np.abs(xmat[:, 2, :]) * self._geom_half).sum(1)
        a, b = np.asarray(p_from[:2], float), np.asarray(p_to[:2], float)
        # distance in plan from the line to each geom's own box, not to its bounding circle: the
        # room's walls (world geoms 2.1 m tall, circles 2-3 m wide) were "near" every carry and put
        # every carry at the cap, the target + 30 cm, where LIBERO's human demos lift 120-160 mm
        n = max(2, int(np.ceil(float(np.linalg.norm(b - a)) / LINE_STEP)) + 1)
        line = a + np.linspace(0.0, 1.0, n)[:, None] * (b - a)
        rel = np.concatenate([line[None, :, :] - pos[:, None, :2], np.zeros((len(pos), n, 1))], axis=2)
        local = np.einsum("gji,gkj->gki", xmat, rel)
        out = np.linalg.norm(np.maximum(np.abs(local) - self._geom_half[:, None, :], 0.0), axis=2)
        near = out.min(1) < CORRIDOR
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
        hs = list(np.arange(hi, lo - EPS_LEN2, -CARRY_STEP))
        if not hs or hs[-1] > lo + EPS_LEN2:
            hs.append(lo)                       # the floor is always one of the heights solved
        tool_off = p - q
        stops = [q[:2], (q[:2] + target_q[:2]) / 2, target_q[:2]]      # pick, halfway, drop
        th, conv, sig, margin = self.solve([pose(R, np.array([xy[0], xy[1], h]) + tool_off)
                                            for h in hs for xy in stops])
        n = len(stops)
        ok = (conv & (sig > MIN_SIGMA) & (margin > MIN_MARGIN)).reshape(len(hs), n).all(1)
        for i, h in enumerate(hs):              # the highest height reachable at every stop,
            if ok[i] and all(self.collides(th[i * n + j], body) < 2 for j in range(n)):
                self.last_carry = dict(verified=True, h=round(float(h), 3))
                return float(h)                 # with the arm out of the fixtures there
        # none is: the floor, logged as unverified with its worst stop's conditioning (a
        # carry this low is the gentlest; one in five carries on libero_spatial lands here)
        worst = np.where(conv, np.minimum(margin, SIGMA_WEIGHT * sig), -np.inf).reshape(len(hs), n).min(1)
        self.last_carry = dict(verified=False, h=round(lo, 3), worst=round(float(worst[-1]), 3))
        return lo
