"""screwhead/teacher/task_loss.py: the tolerances probed through LIBERO's predicates equal the
ones in its code, and the loss is zero exactly where LIBERO accepts on a live scene.

  PYTHONPATH=third_party/LIBERO:. .venv-libero/bin/python -m pytest tests/test_task_loss.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("libero.libero")


@pytest.fixture(scope="module")
def goal_env():
    from screwhead.sim.task_env import TaskEnv
    env = TaskEnv("libero_goal", 1, seed=0, render=False)        # on(bowl, stove cook region)
    env.reset(0)
    yield env
    env.close()


def test_on_top_tolerances_equal_libero():
    from screwhead.teacher.task_loss import probe_ontop
    t = probe_ontop()
    assert t["r"] == pytest.approx(0.03, abs=1e-12)
    assert t["dz"] == pytest.approx(0.0, abs=1e-12)


def test_site_band_and_slack_equal_libero(goal_env):
    from screwhead.teacher.task_loss import probe_site
    sites = goal_env.env.env.object_sites_dict
    band = probe_site(sites["flat_stove_1_cook_region"])
    assert band["under_below"] == pytest.approx(0.005, abs=1e-12)
    assert band["under_above"] == pytest.approx(0.10, abs=1e-12)
    box = probe_site(sites["wooden_cabinet_1_top_region"])
    assert np.allclose(box["in_lo_slack"], [0, 0, 0.01], atol=1e-12)
    assert np.allclose(box["in_hi_slack"], 0.0, atol=1e-12)


def test_joint_thresholds_equal_libero(goal_env):
    from screwhead.teacher.task_loss import probe_joint
    lib = goal_env.env.env
    drawer = probe_joint(lib, "open", "wooden_cabinet_1_middle_region")["joints"][0]
    assert drawer["theta"] == pytest.approx(-0.14, abs=1e-12) and drawer["side"] == -1.0
    knob = probe_joint(lib, "turnon", "flat_stove_1")["joints"][0]
    assert knob["theta"] == pytest.approx(0.5, abs=1e-12) and knob["side"] == 1.0


def test_loss_is_zero_exactly_where_libero_accepts(goal_env):
    """Teleport the bowl over and onto the cook region: the loss is 0 iff LIBERO accepts,
    and positive (at least M) wherever it rejects."""
    import mujoco
    from screwhead.teacher.task_loss import TaskLoss
    loss = TaskLoss(goal_env)
    lib = goal_env.env.env
    m, d = lib.sim.model._model, lib.sim.data._data
    jid = m.body("akita_black_bowl_1_main").jntadr[0]
    adr = m.jnt_qposadr[jid]
    c = lib.sim.data.get_site_xpos("flat_stove_1_cook_region").copy()
    saved = d.qpos.copy()
    seen = set()
    for dx in np.linspace(-0.06, 0.06, 7):
        for dz in np.linspace(-0.01, 0.12, 14):
            d.qpos[adr:adr + 3] = c + np.array([dx, 0.0, dz])
            mujoco.mj_forward(m, d)
            accepted = bool(lib._check_success())
            value = loss()
            assert (value == 0.0) == accepted
            assert accepted or value >= loss.m
            seen.add(accepted)
    d.qpos[:] = saved
    mujoco.mj_forward(m, d)
    assert seen == {True, False}
