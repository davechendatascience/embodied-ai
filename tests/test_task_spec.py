"""task_spec.plan_for reads how an object is moved from the affordance table.

  .venv-libero/bin/python -m pytest tests/test_task_spec.py -q

libero_goal 5's plate is declared moved_by push (137 mm across, beyond the 75 mm opening):
planned as pick and place_on it scored 3 of 50; as a push, 20 of 20.
"""
from __future__ import annotations

from screwhead.teacher.task_spec import moved_by, plan_for


def test_a_push_category_is_pushed_not_picked():
    plan = plan_for([("on", "plate_1", "main_table_stove_front_region")], {"plate_1": "plate"})
    assert [s.skill for s in plan] == ["push"]
    assert plan[0].goal == ("on", "plate_1", "main_table_stove_front_region")


def test_a_plate_as_the_target_does_not_make_the_bowl_pushed():
    plan = plan_for([("on", "akita_black_bowl_1", "plate_1")],
                    {"akita_black_bowl_1": "akita_black_bowl", "plate_1": "plate"})
    assert [s.skill for s in plan] == ["pick", "place_on"]


def test_an_undeclared_or_missing_category_is_picked():
    assert moved_by(None) == "pick" and moved_by("no_such_category") == "pick"
    assert [s.skill for s in plan_for([("on", "x_1", "y_region")])] == ["pick", "place_on"]


def test_in_goals_are_never_pushed():
    plan = plan_for([("in", "plate_1", "basket_1_contain_region")], {"plate_1": "plate"})
    assert [s.skill for s in plan] == ["pick", "place_in"]
