"""A demonstration teacher for any LIBERO task, driven by the task's own goal.

The step it is executing is decided from the CURRENT state -- which goal conjuncts already
hold, whether the object is in the gripper -- and never from stored progress, so it can
label any state a student reaches (DAgger; belief.yaml IFC-teacher__expert). That is the
same property the hand-written spatial programs have, without a program per task.
"""
from __future__ import annotations

import numpy as np

from ..sim.gripper_servo import A_OPEN
from .skills import Skills, SkillConfig
from .task_spec import Step
from .refusal import Refusal

HANDLE_AWAY = 0.10        # m: the tool this far from a container's handle is not working it


class SkillTeacher:
    def __init__(self, env, config: SkillConfig | None = None):
        self.env = env
        self.spec = env.task_spec
        self.skills = Skills(env, config)
        self.plan = self.spec.plan
        self.phase = ""
        self.step_index = 0
        self.step = None

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
                # delivered and let go: its container is no longer a precondition. Asked first, a
                # closing drawer read as shut and the done pick had it opened again, against the
                # close step, for 500 steps (libero_10 3)
                if goal is not None and self.satisfied(goal) and not self.skills.held(step.obj):
                    continue
                closed = self._closed_container(nxt)
                if closed is not None:
                    # an object the open container would crowd is moved out of its way
                    # first: with one gripper it cannot be held while the container opens
                    spot = self.skills.clearing_spot(step.obj, closed, self.env.snapshot()["R_tool"])
                    if spot is not None and (self.skills.held(step.obj)
                                             or not self.skills.at_spot(step.obj, spot)):
                        return i, Step("relocate", obj=step.obj, region=spot)
                if closed is not None and not self.skills.held(step.obj):
                    return i, Step("articulate", region=closed, mode="open",
                                   goal=("open", closed))
                if nxt is not None and nxt.skill in ("place_in", "place_on"):
                    q, target = self.skills.place_target(nxt.obj, nxt.region,
                                                         nxt.skill == "place_in")
                    if self.skills.at_place(q, target):   # delivered, waiting to be scored
                        continue
                if not self.skills.held(step.obj):
                    return i, step
                continue
            goal = self._goal_for(step)
            # a place is done when its goal holds with the object let go: LIBERO's In held for a can
            # still in the jaws over libero_10's basket, the next pick lifted it back out, and the
            # teacher lowered and lifted it for 700 steps (tasks 0 and 7, 0 of 100)
            if (goal is None or not self.satisfied(goal)
                    or (step.skill in ("place_in", "place_on") and self.skills.held(step.obj))):
                return i, step
        return len(self.plan), None

    def _closed_container(self, step) -> str | None:
        """The region a place step fills, if it is behind a door or in a drawer that is
        shut.

        "Open the top drawer and put the bowl inside" is scored as In(bowl, top_region)
        alone -- no Open conjunct -- and the drawer starts closed, so a plan read off the
        goal set the bowl down on the drawer's lid, 0/20. Ten LIBERO tasks have this
        shape. Opening is a precondition of filling, and whether it holds is read from
        the scene each step, like everything else.
        """
        if step is None or step.skill != "place_in":
            return None
        try:
            art = self.env.scene.articulation(step.region)
        except ValueError:
            return None                       # not an articulated region
        # open with room to spare: released the moment LIBERO's threshold was crossed, the
        # drawer sat 4.7 mm past it, the arm brushed the cabinet on its way out, and the
        # precondition flickered between "open" and "shut" every few steps
        # from the joint and LIBERO's own threshold for it (scene.ARTICULATION): asked of a region,
        # LIBERO's Open predicate read the microwave's heating region as shut with its door wide
        # open, and the teacher drove the open door for 600 steps (libero_10 9, 0 of 20)
        # Open by LIBERO's threshold and admitting the object is enough while the tool is away from
        # the container's handle: libero_10's bottom drawer starts 3-10 mm past the threshold,
        # LIBERO's humans put the bowl in as it is (50 of 50), and asked for the 12 mm margin the
        # teacher reached for its handle for 800 steps. At the handle -- mid-drag -- the margin
        # stands: stopping the hook at the threshold left libero_goal 3's drawer too little open
        # for the pick's descent (2 of 50)
        past = (art["qpos"] - art["thresholds"]["open"]) * art["sign"]
        away = float(np.linalg.norm(self.env.snapshot()["p_tool"] - art["handle_world"])) > HANDLE_AWAY
        wide = (past >= min(self.skills.k.open_margin, art["open_room"] - 1e-3)
                or (past >= 0 and away and self.skills.admits(step.obj, step.region)))
        return None if wide else step.region

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
        self.step = step                       # what is being executed (may be a precondition)
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
            placing = nxt is not None and nxt.skill in ("place_in", "place_on")
            via = self.skills.via_for(nxt.region) if placing else None   # the grasp must also reach
            place = (nxt.region, nxt.skill == "place_in") if placing else None
            a = self.skills.leave_handle(s)
            if a is None:
                a = self.skills.pick(step.obj, s, via=via, place=place)
        elif step.skill in ("place_in", "place_on"):
            a = self.skills.place(step.obj, step.region, s, inside=step.skill == "place_in")
        elif step.skill == "push":
            a = self.skills.push(step.obj, step.region, s)
        elif step.skill == "relocate":                 # place picks it up first (regrasp)
            a = self.skills.place(step.obj, step.region, s, inside=False)
        elif step.skill == "articulate" or step.skill == "turn":
            a = self.skills.articulate(step.region, step.mode, s)
        else:
            # The plan named a skill this teacher does not have. This teacher walks the plan
            # task_spec builds; it is not the regression planner BRN-regression-planner designs,
            # which is not implemented. Whatever builds the plan, a step no skill here executes is
            # refused rather than attempted with some other skill, and recorded as a refusal.
            raise Refusal("skill", step.skill,
                          "no declared skill produces the motion this goal needs")
        self.phase = f"{step.skill}:{self.skills.phase}"
        return a
