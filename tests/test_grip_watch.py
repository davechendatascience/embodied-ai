"""grip_watch.py's drop and release classification, on scripted grip histories.

  .venv-libero/bin/python -m pytest tests/test_grip_watch.py -q

CTR-teacher-gentle reads drop_count == 0. Counted on every held() True -> False with the object
off its support, it read 18-40 an episode on libero_goal 5 and 9 where nothing fell; counted not at
all, libero_goal 1 and 8 read 50/50 while the bowl fell from 28 cm. These pin the definition
between the two: a drop is a fall of DROP_FALL within DROP_CONFIRM steps, unheld.
"""
from __future__ import annotations

import types

import numpy as np

from screwhead.teacher import grip_watch
from screwhead.teacher.grip_watch import DROP_CONFIRM, DROP_FALL, GripWatch


class Script:
    """An env/teacher pair replaying one object's (held, height, phase) per step."""

    def __init__(self, rows, gap=0.05):
        self.rows, self.i, self.t = rows, 0, 0
        box = types.SimpleNamespace(R=np.eye(3), half=np.full(3, 0.01), world_centre=np.zeros(3))
        self.scene = types.SimpleNamespace(object_box=lambda _o: self._box(box))
        self.skills = types.SimpleNamespace(held=lambda _o: self.rows[self.i][0], scene=self.scene,
                                            planner=None)
        self.plan = [types.SimpleNamespace(obj="bowl")]
        self.gap = gap

    def _box(self, box):
        box.world_centre = np.array([0.0, 0.0, self.rows[self.i][1]])
        return box

    @property
    def phase(self):
        return self.rows[self.i][2]


def run(rows, gap=0.05):
    s = Script(rows, gap)
    grip_watch.support_gap = lambda *_args: s.gap
    w = GripWatch(s, s)
    for i in range(len(rows)):
        s.i, s.t = i, i
        w.step()
    return w.metrics()


def carry(n, z=0.28):
    return [(True, z, "place:carry")] * n


def test_a_fall_after_letting_go_is_a_drop():
    fall = [(False, 0.28 - k * 0.012, "pick:up") for k in range(5)]
    assert run(carry(5) + fall)["drop_count"] == 1


def test_a_flicker_that_is_held_again_is_not():
    flick = [(False, 0.28, "place:squeeze"), (True, 0.28, "place:lift")]
    assert run(carry(5) + flick * 4 + carry(3))["drop_count"] == 0


def test_a_let_go_that_never_falls_is_not():
    rest = [(False, 0.28 - 0.5 * DROP_FALL, "pick:up")] * (DROP_CONFIRM + 5)
    assert run(carry(5) + rest)["drop_count"] == 0


def test_a_fall_after_the_window_is_not_counted():
    late = ([(False, 0.28, "pick:up")] * (DROP_CONFIRM + 1)
            + [(False, 0.28 - 2 * DROP_FALL, "pick:up")])
    assert run(carry(5) + late)["drop_count"] == 0


def test_a_release_is_a_release_not_a_drop():
    rel = [(True, 0.03, "place:release"), (False, 0.01, "place:release"), (False, 0.0, "settle")]
    m = run(carry(5, 0.03) + rel, gap=0.004)
    assert m["drop_count"] == 0 and m["releases"] == 1 and m["release_gap_mm"] == 4.0


def test_a_let_go_on_the_support_is_a_set_down():
    down = [(False, 0.0 - k * 0.012, "pick:up") for k in range(3)]
    assert run(carry(5, 0.0) + down, gap=0.002)["drop_count"] == 0


def test_a_release_is_measured_when_decided_even_if_held_went_first():
    # lowered into a basket: the object touches it and held() reads false before the jaws open
    lower = [(True, 0.05, "place:lower"), (False, 0.03, "place:lower"), (False, 0.03, "place:lower")]
    rel = [(False, 0.03, "place:release"), (False, 0.03, "place:release"), (False, 0.03, "settle")]
    m = run(carry(5, 0.05) + lower + rel, gap=0.012)
    assert m["releases"] == 1 and m["release_gap_mm"] == 12.0 and m["drop_count"] == 0
