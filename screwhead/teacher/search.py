"""The optimization teacher's receding-horizon search (BRN-optimization-teacher, BRN-screw-search).

At each control step: sample plans -- sequences of segments, each one normalized body twist held
for the segment's periods and one gripper snap level -- around the mean plan; roll each out from a
restored copy of the state (exec_state) with a fork of the executed watch (verdicts); rank them
lexicographically by (the rollout recorded a violation; for a violating plan the negated period of
its violation, otherwise the settled period or horizon + 1; in how many of its periods an unnamed
body could leave its rest; the terminal order at the end state);
move the mean toward the better plans with weights that depend only on rank, the step bounded by a
KL budget to the start distribution; execute the first action of the best plan.

Everything in Settings is a compute parameter: it changes which plan a finite search finds, never
the problem (the fewest steps to a settled state with no violation) or a verdict. The terminal
order is a compute setting too; its form comes from LMA-loss-lower-bounds-steps.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry.box_distance import BoxSet, _point_box_sq
from ..sim import exec_state
from ..sim.gripper_servo import target_to_channel

LEVELS = (0.0, 0.026, 0.08)        # the student's snap levels (tools/distill.py)


@dataclass
class Settings:
    horizon: int = 12              # periods a plan covers
    segments: int = 4              # segments per plan (a plan's periods are split evenly)
    samples: int = 16              # plans per iteration
    iters: int = 2                 # distribution updates per control step
    spread: float = 0.35           # initial spread of the twist, in normalized twist coordinates
    shrink: float = 0.6            # spread factor applied when an iteration does not improve the best key
    level_floor: float = 0.1       # least probability of every snap level in the start categorical
    kl: float = 2.0                # KL budget of one update against the start distribution
    seed: int = 20260923           # fixed: the label is then a function of the state
    gradient_step: float = 0.0     # first-order proposal (BRN-screw-search (c)); 0 = off
    key_order: str = "plain"       # where stability sits in the key: plain | stability | tiebreak.
    #                                Measured on libero_goal 8 i0 and libero_object 1 i0: tiebreak is
    #                                identical to plain (exact ties on the terminal order are vanishing),
    #                                stability starves progress (object 1 timed out at terminal 0.30).

    def periods(self) -> list[int]:
        base, extra = divmod(self.horizon, self.segments)
        return [base + (i < extra) for i in range(self.segments)]


@dataclass
class Plan:
    twist: np.ndarray              # (segments, 6), normalized body twist, moment first
    level: np.ndarray              # (segments,) index into LEVELS

    def action(self, period: int, periods: list[int]) -> np.ndarray:
        k, seen = 0, 0
        for k, n in enumerate(periods):                              # noqa: B007 - k is the segment
            if period < seen + n:
                break
            seen += n
        return np.concatenate([np.clip(self.twist[k], -1, 1), [target_to_channel(LEVELS[self.level[k]])]])


@dataclass
class Rollout:
    violated: bool
    period: int                    # the violation's period (negated in the key) or the settled period
    settled: bool
    terminal: float
    violation: str = ""
    unsettled: int = 0             # periods in which an unnamed body could leave its rest


@dataclass
class Report:
    """What one control step's search found, recorded with the demonstration."""
    key: tuple
    best: Plan
    settled_in: int | None
    violations: int
    terminal: float
    unsettled: int = 0             # periods of the chosen plan in which an unnamed body could leave its rest
    spreads: list[float] = field(default_factory=list)


class Search:
    def __init__(self, env, verdicts, loss, settings: Settings | None = None, policy=None):
        self.env, self.v, self.loss = env, verdicts, loss
        self.s = settings or Settings()
        self.policy = policy
        self.m, self.d = env.scene.raw()
        self._boxes: dict[str, BoxSet] = {}
        self._mean: Plan | None = None

    # -- one control step -------------------------------------------------------------------
    def act(self, watch, start_reference: dict, previous_reference: dict) -> tuple[np.ndarray, Report]:
        """The action to execute from the current state, and what the search found.
        `start_reference` is the episode start's supports and faces and `previous_reference` the
        previous executed period's: a body is disturbed when it matches neither, so undoing a
        disturbance is not one while a fresh knock from a disturbed state still is."""
        s, periods = self.s, self.s.periods()
        saved = exec_state.save(self.env)
        rng = np.random.default_rng(s.seed)
        mean, levels = self._start()
        spread, spreads = s.spread, []
        best: tuple[tuple, Plan, Rollout] | None = None
        for _ in range(s.iters):
            plans = self._sample(rng, mean, levels, spread)
            scored = []
            for i, plan in enumerate(plans):
                roll = self._rollout(saved, watch, plan, periods, start_reference, previous_reference)
                scored.append((self._key(roll), i, plan, roll))
            scored.sort(key=lambda r: (r[0], r[1]))                  # ties: the lowest sample index
            if best is None or scored[0][0] < best[0]:
                best = (scored[0][0], scored[0][2], scored[0][3])
            else:
                spread *= s.shrink
            spreads.append(spread)
            mean, levels = self._update(mean, levels, [p for _, _, p, _ in scored], spread)
        exec_state.restore(self.env, saved)
        key, plan, roll = best
        self._mean = plan
        report = Report(key=key, best=plan, settled_in=roll.period if roll.settled else None,
                        violations=sum(r.violated for _, _, _, r in scored), terminal=roll.terminal,
                        unsettled=roll.unsettled, spreads=spreads)
        return plan.action(0, periods), report

    # -- sampling ---------------------------------------------------------------------------
    def _start(self) -> tuple[np.ndarray, np.ndarray]:
        """The start distribution: the policy's plan, or the previous step's shifted by one
        period, or zeros; and a categorical over levels with every level above the floor."""
        n, s = self.s.segments, self.s
        if self.policy is not None:
            mean, probs = self.policy(self.env, self.v)
        else:
            mean = np.zeros((n, 6)) if self._mean is None else np.roll(self._mean.twist, -1, axis=0)
            probs = np.full((n, len(LEVELS)), 1.0 / len(LEVELS))
            if self._mean is not None:
                probs = np.full((n, len(LEVELS)), s.level_floor)
                probs[np.arange(n), np.roll(self._mean.level, -1)] = 1.0 - s.level_floor * (len(LEVELS) - 1)
        if s.gradient_step:
            mean = mean - s.gradient_step * self._gradient()
        return np.clip(mean, -1, 1), probs / probs.sum(axis=1, keepdims=True)

    def _sample(self, rng, mean: np.ndarray, probs: np.ndarray, spread: float) -> list[Plan]:
        n = self.s.segments
        out = [Plan(np.clip(mean, -1, 1), np.array([int(np.argmax(p)) for p in probs]))]   # the mean plan
        for _ in range(self.s.samples - 1):
            twist = np.clip(mean + spread * rng.standard_normal((n, 6)), -1, 1)
            level = np.array([rng.choice(len(LEVELS), p=p) for p in probs])
            out.append(Plan(twist, level))
        return out

    def _update(self, mean, probs, ordered: list[Plan], spread: float) -> tuple[np.ndarray, np.ndarray]:
        """Rank weights (CMA-ES's log rule) over the better half; the mean's step is bounded by
        the KL budget, which for an isotropic Gaussian is |step| <= spread sqrt(2 KL)."""
        mu = max(2, len(ordered) // 2)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        target = np.einsum("i,ijk->jk", w, np.stack([p.twist for p in ordered[:mu]]))
        step = target - mean
        norm = float(np.linalg.norm(step))
        limit = spread * np.sqrt(2 * self.s.kl)
        if norm > limit > 0:
            step *= limit / norm
        counts = np.full_like(probs, self.s.level_floor)
        for weight, plan in zip(w, ordered[:mu], strict=True):
            counts[np.arange(self.s.segments), plan.level] += weight
        return np.clip(mean + step, -1, 1), counts / counts.sum(axis=1, keepdims=True)

    # -- rollouts ---------------------------------------------------------------------------
    def _rollout(self, saved, watch, plan: Plan, periods: list[int], start_reference: dict,
                 previous_reference: dict) -> Rollout:
        exec_state.restore(self.env, saved)
        w = watch.fork()
        previous, unsettled = previous_reference, 0
        for period in range(self.s.horizon):
            self.env.execute(plan.action(period, periods), substep=w.substep)
            success = self.env.success()
            w.period_end(success)
            violation = w.pending() or self.v.disturbed(start_reference, previous) or self.v.lost()
            unsettled += bool(self.v.unsettled())
            if violation:
                return Rollout(True, period, False, self.terminal_order(), violation, unsettled)
            if success and self.v.settled(w):
                return Rollout(False, period, True, 0.0, "", unsettled)
            previous = self.v.reference()
        doomed = w.doomed()          # a loss open at the end has already failed unless contact returns
        return Rollout(bool(doomed), self.s.horizon if doomed else self.s.horizon + 1, False,
                       self.terminal_order(), doomed, unsettled)

    def _key(self, r: Rollout) -> tuple:
        """Violation first, then how soon it settles (how late it violates), then -- where
        key_order puts it -- in how many of its periods an unnamed body could leave its rest,
        and the terminal order."""
        head = (1, -r.period) if r.violated else (0, r.period)
        tail = {"plain": (r.terminal,), "stability": (r.unsettled, r.terminal),
                "tiebreak": (r.terminal, r.unsettled)}[self.s.key_order]
        return head + tail

    # -- the terminal order -----------------------------------------------------------------
    def terminal_order(self) -> float:
        """Sum over the goal's conjuncts of the tool travel the simplest carry needs: the tool
        point's distance to what must move, plus what must move's distance to acceptance (half the
        task loss's distance for a placement, the handle's arc for a joint). 0 where LIBERO
        accepts (LMA-loss-lower-bounds-steps)."""
        total = 0.0
        p_tool = self.env.tool_state()["p_tool"] + self.env.scene.base
        for term, goal in zip(self.loss.terms(), self.loss.goals, strict=True):
            if term.satisfied:
                continue
            if goal in self.loss._joint:
                a = self.env.scene.articulation(goal[1])
                total += self._geom_distance(int(a["handle_geom"]), p_tool) + self._handle_arc(a, term.distance)
            else:
                total += self._object_distance(goal[1], p_tool) + term.distance / 2
        return float(total)

    def _handle_arc(self, articulation: dict, distance: float) -> float:
        """How far the handle itself must travel: an arc for a hinge (its centre's radius about
        the axis), the joint's own distance for a slide."""
        import mujoco
        if articulation["jnt_type"] == mujoco.mjtJoint.mjJNT_SLIDE:
            return abs(distance)
        centre = np.asarray(self.d.geom_xpos[int(articulation["handle_geom"])], float)
        axis = np.asarray(articulation["axis"], float)
        anchor = np.asarray(articulation["anchor"], float) + self.env.scene.base
        r = centre - anchor
        return float(np.linalg.norm(r - (r @ axis) * axis) * abs(distance))

    def _object_distance(self, name: str, p_tool: np.ndarray) -> float:
        boxes = self._boxes.get(name)
        if boxes is None:
            ids = [self.m.geom(g).id for g in self.v.lib.get_object(name).contact_geoms]
            boxes = self._boxes[name] = BoxSet(self.m, ids)
        c, R = boxes.pose(self.d)
        return float(np.sqrt(min(_point_box_sq(p_tool[0], p_tool[1], p_tool[2], c[k], R[k], boxes.h[k])
                                 for k in range(len(c)))))

    def _geom_distance(self, geom: int, p: np.ndarray) -> float:
        c = np.asarray(self.d.geom_xpos[geom], float)
        R = np.asarray(self.d.geom_xmat[geom], float).reshape(3, 3)
        return float(np.sqrt(_point_box_sq(p[0], p[1], p[2], c, R, np.asarray(self.m.geom_size[geom], float))))

    def _gradient(self) -> np.ndarray:
        raise NotImplementedError("BRN-screw-search (c): the first-order proposal is not built yet")
