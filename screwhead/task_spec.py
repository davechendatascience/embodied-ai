"""A LIBERO task, read rather than hand-written.

The ten libero_spatial programs were written per task, with their constants tuned on
those tasks (screwhead/scripted_teacher.py). That does not reach 130 tasks, and the
held-out splits showed what it costs: a task whose bowl sits on an unseen fixture
scored 0/20 while table-top tasks transferred (belief.yaml, CTR-held-out-task).

What LIBERO actually gives us is small. Over all 130 bddl files the goal predicates are
  in 63, on 61, close 11, turnon 8, open 7, turnoff 1
in 15 distinct combinations, 110 tasks with one conjunct and 20 with two or three. Each
predicate names an object and a region, and both are readable at runtime: regions are
MuJoCo sites with a pose and half-extents, objects are bodies whose collision geoms give
a bounding box. So a task is a goal to satisfy, and a plan is the skills that satisfy it.

  In(object, region)      -> pick(object), place_in(region)
  On(object, target)      -> pick(object), place_on(target)
  Open(region) / Close    -> articulate(region, open|close)   (drawer, door)
  TurnOn / TurnOff(thing) -> turn(thing, on|off)              (stove knob)

Ordering within a conjunction follows the dependencies: open a container before putting
something in it, close it afterwards. LIBERO scores the final state only, so anything
consistent with those two rules is acceptable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PICK_PLACE = {"in", "on"}
ARTICULATE = {"open", "close"}
SWITCH = {"turnon", "turnoff"}


@dataclass(frozen=True)
class Step:
    """One skill invocation. `obj` is a body name, `region` a site name, both as the
    simulator knows them (the bddl region key IS the site name after robosuite's prefix)."""
    skill: str                  # pick | place_in | place_on | articulate | turn
    obj: str | None = None
    region: str | None = None
    mode: str | None = None     # open/close for articulate, on/off for turn
    goal: tuple | None = None   # the bddl conjunct this step satisfies, for LIBERO to score


@dataclass
class TaskSpec:
    suite: str
    name: str
    instruction: str
    goals: list[tuple]                       # [('in', 'alphabet_soup_1', 'basket_1_contain_region'), ...]
    objects: dict[str, str] = field(default_factory=dict)      # instance -> category
    fixtures: dict[str, str] = field(default_factory=dict)
    regions: dict[str, dict] = field(default_factory=dict)
    obj_of_interest: list[str] = field(default_factory=list)

    @property
    def plan(self) -> list[Step]:
        return plan_for(self.goals)


def parse(path: str | Path, suite: str | None = None) -> TaskSpec:
    from libero.libero.envs.bddl_utils import robosuite_parse_problem
    path = Path(path)
    p = robosuite_parse_problem(str(path))
    lang = p["language_instruction"]
    objects = {inst: cat for cat, insts in p["objects"].items() for inst in insts}
    fixtures = {inst: cat for cat, insts in p["fixtures"].items() for inst in insts}
    return TaskSpec(suite=suite or path.parent.name, name=path.stem,
                    instruction=" ".join(lang) if isinstance(lang, list) else str(lang),
                    goals=[tuple(g) for g in p["goal_state"]],
                    objects=objects, fixtures=fixtures, regions=p["regions"],
                    obj_of_interest=list(p["obj_of_interest"]))


def plan_for(goals: list[tuple]) -> list[Step]:
    """Goal conjunction -> skill sequence. Raises on a predicate we cannot satisfy, so a
    suite we cannot yet do is a loud failure rather than a silently empty plan."""
    opens, places, closes, switches = [], [], [], []
    close_targets = {g[1] for g in goals if g[0] == "close"}
    for g in goals:
        pred = g[0].lower()
        if pred in PICK_PLACE:
            obj, region = g[1], g[2]
            # putting something into a container that the same goal asks to close: it has
            # to be open first, and LIBERO's drawers start closed
            if region in close_targets:
                opens.append(Step("articulate", region=region, mode="open"))
            places.append(Step("pick", obj=obj))
            places.append(Step("place_in" if pred == "in" else "place_on", obj=obj, region=region,
                               goal=tuple(g)))
        elif pred in ARTICULATE:
            (opens if pred == "open" else closes).append(
                Step("articulate", region=g[1], mode=pred, goal=tuple(g)))
        elif pred in SWITCH:
            switches.append(Step("turn", region=g[1], mode="on" if pred == "turnon" else "off",
                                 goal=tuple(g)))
        else:
            raise ValueError(f"no skill for predicate {pred!r} in {goals}")
    # open containers, then fill them, then close, then switch (a knob is easier to reach
    # with an empty gripper, and a closed drawer must stay closed)
    return opens + places + closes + switches


def all_tasks(root: str | Path) -> list[TaskSpec]:
    root = Path(root)
    return [parse(f, suite=f.parent.name) for f in sorted(root.glob("*/*.bddl"))]
