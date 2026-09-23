"""Whether the grip let an object go where it was not meant to: CTR-teacher-gentle's metrics.

One definition for every tool that reports them (tools/teacher_motion.py, tools/skill_eval.py).
It was written once, inside teacher_motion.py, over four cases; libero_goal 1 and 8 were never
among them, and both read 50/50 in the reliability sweep while the rim-pinched bowl fell from
28 cm on to its target in 5 of 8 probed episodes. Success certified the drop; this counts it.

A drop is a fall, not a reading. held() flickers -- a tilted plate, a bottle between squeezes --
and counting each True -> False with the object off its support read 18-40 "drops" an episode on
libero_goal 5 and 9, none of them a fall. So a loss away from a release is only a candidate, and it
counts once the object has fallen DROP_FALL below where it was let go, not held again, within
DROP_CONFIRM steps (LMA-loss-needs-robust-evidence: never on an instantaneous signal). A real drop
confirms in one or two steps: 10 mm of free fall takes 45 ms.

An observer: it reads the state and the teacher's phase and changes neither.
"""
from __future__ import annotations

import numpy as np

RELEASE_WINDOW = 10      # steps: a let-go this soon after a release phase was meant
DROP_GAP = 0.010         # m: an unmeant let-go below this is a set-down, not a drop
DROP_FALL = 0.010        # m: how far a let-go object must fall to count as dropped
DROP_CONFIRM = 10        # steps it has to fall that far, unheld


def support_gap(scene, planner, obj: str) -> float:
    """Gap from the object's bottom to the surface straight below it (nan if none is found)."""
    box = scene.object_box(obj)
    ext = np.abs(box.R) @ box.half
    c = box.world_centre
    bottom = c[2] - ext[2]
    _g, dist = planner._ray(np.array([c[0], c[1], bottom - 0.001]), np.array([0.0, 0.0, -1.0]),
                            scene.body_id(obj))
    return dist + 0.001 if dist >= 0 else float("nan")


class GripWatch:
    """Call reset() with each episode and step() after the teacher acts, before the env steps."""

    def __init__(self, env, teacher):
        self.env, self.teacher = env, teacher
        self.objs = sorted({st.obj for st in teacher.plan if st.obj})
        self.reset()

    def reset(self) -> None:
        self.held = dict.fromkeys(self.objs, False)
        self.last_release = -10**6
        self.gaps: list[float] = []
        self.drops = 0
        self.pending: dict[str, tuple[int, float]] = {}     # obj -> (step let go, height then)

    def step(self) -> None:
        sk, t = self.teacher.skills, self.env.t
        if self.teacher.phase.endswith("release"):
            self.last_release = t
        for o in self.objs:
            h = sk.held(o)
            z = float(sk.scene.object_box(o).world_centre[2])
            if o in self.pending:
                t0, z0 = self.pending[o]
                if h or t - t0 > DROP_CONFIRM:
                    del self.pending[o]                        # held again, or never fell: a flicker
                elif z0 - z >= DROP_FALL:
                    self.drops += 1
                    del self.pending[o]
            if self.held[o] and not h:
                gap = support_gap(sk.scene, sk.planner, o)
                if t - self.last_release <= RELEASE_WINDOW:
                    self.gaps.append(gap)
                elif gap > DROP_GAP:
                    self.pending[o] = (t, z)
            self.held[o] = h

    def metrics(self) -> dict:
        """release_gap_mm is -1 when nothing was let go on purpose: the ledger's rules cannot
        compare a missing value."""
        return dict(release_gap_mm=round(1000 * max(self.gaps), 1) if self.gaps else -1.0,
                    releases=len(self.gaps), drop_count=self.drops)
