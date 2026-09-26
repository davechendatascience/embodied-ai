"""A generated task, written in LIBERO's task-file format by LIBERO's own writer.

Every movable object gets a rectangular region of its own on the table, centred where the
generator put it; LIBERO's samplers draw its pose inside that region at reset. Region
coordinates are the table's: on the kitchen table they are world x, y, with the robot's base
at (-0.66, 0) (measured).
"""
from __future__ import annotations

from dataclasses import dataclass

WORKSPACE = "kitchen_table"


@dataclass(frozen=True)
class Placement:
    name: str                       # the instance, as the task file names it (akita_black_bowl_1)
    category: str                   # LIBERO's category (akita_black_bowl)
    xy: tuple[float, float]         # region centre, table coordinates
    half: float                     # region half-side, m
    yaw: float                      # rad, drawn exactly (LIBERO samples yaw in [yaw, yaw])

    @property
    def region(self) -> str:
        return f"{self.name}_init_region"

    def ranges(self) -> tuple[float, float, float, float]:
        x, y = self.xy
        return (x - self.half, y - self.half, x + self.half, y + self.half)


def instance_name(category: str, k: int) -> str:
    """LIBERO's naming: the k-th object of a category (1-based)."""
    return f"{category}_{k}"


def write(language: str, placements: list[Placement], goal: list[tuple], interest: list[str] | None = None,
          fixtures: tuple[Placement, ...] = (), workspace: str = WORKSPACE) -> str:
    """The task file's text. `goal`: LIBERO's goal literals, e.g. [("On", "butter_1", "plate_1")], all to hold.
    `fixtures`: a family's fixtures, each at an exact pose (half 0: the reset draws it where declared, so the scene
    can match the family's reference); `workspace`: the table (kitchen_table, study_table)."""
    from libero.libero.envs.objects import OBJECTS_DICT
    from libero.libero.utils.bddl_generation_utils import (get_affordance_region_kwargs_list_from_fixture_info, get_result,
                                                          get_xy_region_kwargs_list_from_regions_info)
    from libero.libero.utils.object_utils import get_affordance_regions
    from libero.libero.utils.task_generation_utils import get_suite_generator_func
    everything = list(fixtures) + list(placements)
    regions = {p.region: {"target": workspace, "ranges": [p.ranges()], "yaw_rotation": [(p.yaw, p.yaw)]}
               for p in everything}
    affordances = get_affordance_regions(OBJECTS_DICT)
    movable: dict[str, list[str]] = {}
    for p in placements:
        movable.setdefault(p.category, []).append(p.name)
    fixed: dict[str, list[str]] = {workspace: [workspace]}
    for f in fixtures:
        fixed.setdefault(f.category, []).append(f.name)
    text = get_suite_generator_func(workspace)(
        language=language,
        xy_region_kwargs_list=get_xy_region_kwargs_list_from_regions_info(regions),
        # each object's and fixture's own regions (the basket's contain_region, a cabinet's drawers), declared as
        # LIBERO's scene templates declare them
        affordance_region_kwargs_list=get_affordance_region_kwargs_list_from_fixture_info(
            {p.name: affordances[p.category] for p in everything if p.category in affordances}),
        fixture_object_dict=fixed,
        movable_object_dict=movable,
        objects_of_interest=list(interest if interest is not None else [p.name for p in placements]),
        init_states=[("On", p.name, f"{workspace}_{p.region}") for p in everything],
        goal_states=[("And", *goal)] if goal else [])
    return get_result(text)
