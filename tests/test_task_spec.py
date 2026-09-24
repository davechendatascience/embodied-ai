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


def test_fill_then_close_has_no_goalless_open_step():
    """libero_10 3: In(bowl, drawer) and Close(drawer), the drawer already open. A goal-less open step
    was never done, and the teacher reached for the open drawer's handle for 600 steps."""
    plan = plan_for([("in", "akita_black_bowl_1", "white_cabinet_1_bottom_region"),
                     ("close", "white_cabinet_1_bottom_region")],
                    {"akita_black_bowl_1": "akita_black_bowl"})
    assert [s.skill for s in plan] == ["pick", "place_in", "articulate"]
    assert plan[-1].mode == "close" and plan[-1].goal == ("close", "white_cabinet_1_bottom_region")
    assert all(s.goal is not None for s in plan if s.skill == "articulate")


def test_a_standalone_close_comes_before_the_opens():
    """libero_90 23: closed after the top drawer was opened, pressing the bottom drawer shut pushed the
    top one back in (0 of 20)."""
    plan = plan_for([("close", "white_cabinet_1_bottom_region"), ("open", "white_cabinet_1_top_region")])
    assert [(s.skill, s.mode) for s in plan] == [("articulate", "close"), ("articulate", "open")]


def test_a_support_is_moved_before_anything_is_set_on_it():
    """libero_90 63: stacked first, the lower bowl's rim was covered by the upper one when it had to go
    into the tray (0 of 20)."""
    plan = plan_for([("on", "akita_black_bowl_1", "akita_black_bowl_2"),
                     ("in", "akita_black_bowl_2", "wooden_tray_1_contain_region")])
    assert [(s.skill, s.obj) for s in plan] == [("pick", "akita_black_bowl_2"), ("place_in", "akita_black_bowl_2"),
                                                ("pick", "akita_black_bowl_1"), ("place_on", "akita_black_bowl_1")]


def test_independent_places_keep_the_bddl_order():
    plan = plan_for([("in", "alphabet_soup_1", "basket_1_contain_region"),
                     ("in", "tomato_sauce_1", "basket_1_contain_region")])
    assert [s.obj for s in plan if s.skill == "pick"] == ["alphabet_soup_1", "tomato_sauce_1"]
