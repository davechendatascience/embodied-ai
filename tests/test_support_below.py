"""scan.support_below finds the top of what an object rests on, not the far side of it.

  .venv-libero/bin/python -m pytest tests/test_support_below.py -q

The grasp floor is the support plus the fingers' reach. A resting object's bottom is flush with
its support, so a ray started 1 mm under the bottom began inside the table's box, met its
underside 49 mm down, and the floor never held: on libero_goal 6 the fingertips went 4.3 mm into
the table and the squeeze stalled for up to 188 steps.
"""
from __future__ import annotations

import types

import numpy as np

from screwhead.sim.scan import support_below

TABLE_TOP, TABLE_THICK = -0.012, 0.05


def _ray_boxes(boxes):
    """MuJoCo's ray semantics over axis-aligned boxes: the nearest intersection at a positive
    distance, so a ray that starts inside a box meets its far face; -1 on a miss."""
    def ray(p, v, exclude):
        best = -1.0
        for body, lo, hi in boxes:
            if body == exclude:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                t1, t2 = (lo - p) / v, (hi - p) / v
            near = np.nanmax(np.where(v != 0, np.minimum(t1, t2), -np.inf))
            far = np.nanmin(np.where(v != 0, np.maximum(t1, t2), np.inf))
            inside = np.all((v != 0) | ((p >= lo) & (p <= hi)))
            if not inside or far < max(near, 0.0):
                continue
            t = near if near > 0 else far
            if t > 0 and (best < 0 or t < best):
                best = float(t)
        return None, best
    return ray


def _scene(bottom, half_h=0.009):
    box = types.SimpleNamespace(R=np.eye(3), half=np.array([0.04, 0.02, half_h]),
                                world_centre=np.array([0.6, 0.1, bottom + half_h]))
    return types.SimpleNamespace(object_box=lambda _o: box, body_id=lambda _o: 7)


TABLE = (1, np.array([0.0, -0.5, TABLE_TOP - TABLE_THICK]), np.array([1.2, 0.5, TABLE_TOP]))


def test_flush_on_the_table_meets_its_top():
    got = support_below(_scene(TABLE_TOP), _ray_boxes([TABLE]), "cheese")
    assert abs(got - TABLE_TOP) < 1e-9


def test_settled_a_millimetre_into_the_table_meets_its_top():
    got = support_below(_scene(TABLE_TOP - 0.001), _ray_boxes([TABLE]), "cheese")
    assert abs(got - TABLE_TOP) < 1e-9


def test_the_object_itself_is_not_its_support():
    own = (7, np.array([0.5, 0.0, TABLE_TOP]), np.array([0.7, 0.2, TABLE_TOP + 0.018]))
    got = support_below(_scene(TABLE_TOP), _ray_boxes([own, TABLE]), "cheese")
    assert abs(got - TABLE_TOP) < 1e-9


def test_nothing_below_reads_as_the_object_bottom():
    got = support_below(_scene(0.2), _ray_boxes([]), "cheese")
    assert abs(got - 0.2) < 1e-9
