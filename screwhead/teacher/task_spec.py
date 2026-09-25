"""A LIBERO task, read rather than hand-written.

The ten libero_spatial programs were written per task, with their constants tuned on
those tasks (screwhead/scripted/scripted_teacher.py). That does not reach 130 tasks, and the
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
                             push(object, target) for a category declared moved_by push
  Open(region) / Close    -> articulate(region, open|close)   (drawer, door)
  TurnOn / TurnOff(thing) -> turn(thing, on|off)              (stove knob)

Ordering within a conjunction follows the dependencies: open a container before putting
something in it, close it afterwards. LIBERO scores the final state only, so anything
consistent with those two rules is acceptable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

AFFORDANCES = Path(__file__).with_name("affordances.yaml")

PICK_PLACE = {"in", "on"}
ARTICULATE = {"open", "close"}
SWITCH = {"turnon", "turnoff"}


@dataclass(frozen=True)
class Step:
    """One skill invocation. `obj` is a body name, `region` a site name, both as the
    simulator knows them (the bddl region key IS the site name after robosuite's prefix)."""
    skill: str                  # pick | place_in | place_on | push | articulate | turn | relocate
    obj: str | None = None
    region: str | None = None
    mode: str | None = None     # open/close for articulate, on/off for turn
    goal: tuple | None = None   # the bddl conjunct this step satisfies, for LIBERO to score


@dataclass
class TaskSpec:
    suite: str
    name: str
    goals: list[tuple]                       # [('in', 'alphabet_soup_1', 'basket_1_contain_region'), ...]
    objects: dict[str, str] = field(default_factory=dict)      # instance -> category
    fixtures: dict[str, str] = field(default_factory=dict)

    @property
    def plan(self) -> list[Step]:
        return plan_for(self.goals, self.objects)


@lru_cache(maxsize=1)
def affordances() -> dict:
    """screwhead/teacher/affordances.yaml's categories (AXM-category-affordances-declared)."""
    import yaml
    return yaml.safe_load(AFFORDANCES.read_text())["categories"]


def moved_by(category: str | None) -> str:
    """How a category is moved: pick unless the table declares otherwise."""
    return (affordances().get(category or "", {}) or {}).get("moved_by", "pick")


def held_by(category: str | None) -> str | None:
    """Where a category is held, when the table declares it with measured or declared evidence (the
    grasp planner's tier names); an entry backed by nothing is not relied on."""
    entry = affordances().get(category or "", {}) or {}
    evidence = entry.get("evidence") or {}
    return entry.get("held_by") if ("measured" in evidence or "declared" in evidence) else None


def also_held_by(category: str | None) -> str | None:
    """A second place a category is held, offered after its own tiers (only 'handle' is read), under the
    same evidence gate as held_by."""
    entry = affordances().get(category or "", {}) or {}
    evidence = entry.get("evidence") or {}
    return entry.get("also_held_by") if ("measured" in evidence or "declared" in evidence) else None


def parse(path: str | Path, suite: str | None = None) -> TaskSpec:
    from libero.libero.envs.bddl_utils import robosuite_parse_problem
    path = Path(path)
    p = robosuite_parse_problem(str(path))
    objects = {inst: cat for cat, insts in p["objects"].items() for inst in insts}
    fixtures = {inst: cat for cat, insts in p["fixtures"].items() for inst in insts}
    return TaskSpec(suite=suite or path.parent.name, name=path.stem,
                    goals=[tuple(g) for g in p["goal_state"]], objects=objects, fixtures=fixtures)


def _supports_first(goals: list[tuple]) -> list[tuple]:
    """The goals, a goal that moves an object ordered before any goal that sets something on or in
    that object; otherwise in the bddl's order. Stacked first, libero_90 63's lower bowl had the
    upper one nested in it when it had to go into the tray, its rim was covered, and no grasp of it
    closed (0 of 20)."""
    places = [g for g in goals if g[0].lower() in PICK_PLACE]
    rest = [g for g in goals if g[0].lower() not in PICK_PLACE]
    ordered, pending = [], list(places)
    while pending:
        free = [g for g in pending if not any(h[1] == g[2] for h in pending if h is not g)]
        nxt = free[0] if free else pending[0]           # a cycle cannot be ordered: keep the bddl's order
        ordered.append(nxt)
        pending.remove(nxt)
    return ordered + rest


def plan_for(goals: list[tuple], objects: dict[str, str] | None = None) -> list[Step]:
    """Goal conjunction -> skill sequence. Raises on a predicate we cannot satisfy, so a
    suite we cannot yet do is a loud failure rather than a silently empty plan. `objects`
    (instance -> category) lets the affordance table say how each object is moved."""
    opens, places, closes, switches = [], [], [], []
    for g in _supports_first(goals):
        pred = g[0].lower()
        if pred in PICK_PLACE:
            obj, region = g[1], g[2]
            # a container the object goes into is opened by the pick's precondition when it is
            # shut (SkillTeacher.current_step). An open step added here had no goal, so it was
            # never done: with libero_10's bottom drawer already open, the teacher reached for its
            # handle for 600 steps and never picked the bowl (0 of 20)
            if pred == "on" and moved_by((objects or {}).get(obj)) == "push":
                places.append(Step("push", obj=obj, region=region, goal=tuple(g)))
                continue
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
    # a container nothing goes into is closed first, before anything is opened: closed after, the arm
    # pressing libero_90 23's bottom drawer shut pushed the top drawer it had just opened back in
    # (0 of 20). Then open, fill, close what was filled, and switch last (a knob is easier to reach
    # with an empty gripper)
    filled = {g[2] for g in goals if g[0].lower() in PICK_PLACE}

    def fills(c) -> bool:
        # a fixture is filled when a region of its own is: "close the microwave" names microwave_1, the
        # mug goes into microwave_1_heating_region -- read apart, the door was closed first and the mug
        # was then carried to a shut microwave (libero_10 9, 0 of 50)
        return any(f == c.region or f.startswith(f"{c.region}_") for f in filled)
    first = [c for c in closes if not fills(c)]
    last = [c for c in closes if fills(c)]
    return first + opens + places + last + switches

