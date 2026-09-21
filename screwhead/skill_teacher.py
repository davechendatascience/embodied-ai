"""A demonstration teacher for any LIBERO task, driven by the task's own goal.

The step it is executing is decided from the CURRENT state -- which goal conjuncts already
hold, whether the object is in the gripper -- and never from stored progress, so it can
label any state a student reaches (DAgger; belief.yaml IFC-teacher__expert). That is the
same property the hand-written spatial programs have, without a program per task.
"""
from __future__ import annotations

import numpy as np

from .gripper_servo import A_OPEN
from .skills import Skills, SkillConfig


class SkillTeacher:
    def __init__(self, env, config: SkillConfig | None = None):
        self.env = env
        self.spec = env.task_spec
        self.skills = Skills(env, config)
        self.plan = self.spec.plan
        self.phase = ""
        self.step_index = 0

    # -- goal predicates, scored by LIBERO itself ------------------------------------
    def satisfied(self, goal: tuple) -> bool:
        """LIBERO's own predicate on LIBERO's own conjunct.

        Reimplementing these drifts from the thing that scores the episode: my box test
        called In true when the can hung 8 mm above the basket rim, so the teacher let go
        of the goal while still holding the can and the episode scored 0. LIBERO's In is
        contact AND containment, and every problem class exposes `_eval_predicate`.
        """
        return bool(self.env.scene.env._eval_predicate(list(goal)))

    def current_step(self):
        """The first plan step whose goal is not yet satisfied; picks stay active until the
        object is actually held, places until the predicate holds."""
        for i, step in enumerate(self.plan):
            if step.skill == "pick":
                nxt = self.plan[i + 1] if i + 1 < len(self.plan) else None
                goal = self._goal_for(nxt) if nxt is not None else None
                if goal is not None and self.satisfied(goal):
                    continue
                if not self.skills.held(step.obj):
                    return i, step
                continue
            goal = self._goal_for(step)
            if goal is None or not self.satisfied(goal):
                return i, step
        return len(self.plan), None

    def _goal_for(self, step):
        return None if step is None else step.goal

    # -- the feedback law -----------------------------------------------------------
    def act(self, s: dict | None = None) -> np.ndarray:
        s = s or self.env.snapshot()
        if s["success"]:
            self.phase = "done"
            return self.skills.action(np.zeros(6), A_OPEN)
        i, step = self.current_step()
        self.step_index = i
        if step is None:
            held = [st.obj for st in self.plan
                    if st.skill in ("place_in", "place_on") and self.skills.held(st.obj)]
            if held:                           # let go where it is: retreating while gripping
                self.phase = "release"         # lifts the object back out of the container
                return self.skills.action(np.zeros(6), A_OPEN)
            self.phase = "settle"
            return self.skills.retreat(s)
        if step.skill == "pick":
            nxt = self.plan[i + 1] if i + 1 < len(self.plan) else None
            via = (self.skills.via_for(nxt.region)                 # the grasp must also reach
                   if nxt is not None and nxt.skill in ("place_in", "place_on") else None)
            a = self.skills.pick(step.obj, s, via=via)
        elif step.skill in ("place_in", "place_on"):
            a = self.skills.place(step.obj, step.region, s, inside=step.skill == "place_in")
        elif step.skill == "articulate":
            a = self.skills.articulate(step.region, step.mode, s)
        elif step.skill == "turn":
            a = self.skills.articulate(step.region, step.mode, s)
        else:
            raise NotImplementedError(f"skill {step.skill!r}")
        self.phase = f"{step.skill}:{self.skills.phase}"
        return a
