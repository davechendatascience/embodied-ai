"""screwhead/teacher/settle.py and screwhead/teacher/verdicts.py against cases known
independently: a box on an incline (Coulomb friction, measured slope), and a skill-teacher
placement into a basket, whose basket stays put while the cheese is dropped into it.

  PYTHONPATH=third_party/LIBERO:. .venv-libero/bin/python -m pytest tests/test_verdicts.py -q
"""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from screwhead.teacher import settle

MU = 0.95


def _incline(ratio: float, yaw: float):
    """A box resting on a plane tilted to tan(theta) = ratio * mu, the slope turned by yaw."""
    theta = np.arctan(ratio * MU)
    axis = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    quat = np.r_[np.cos(theta / 2), np.sin(theta / 2) * axis]
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat)
    n = R.reshape(3, 3)[:, 2]
    q = " ".join(map(str, quat))
    xml = f"""<mujoco><option cone="elliptic" impratio="20" timestep="0.002"/><worldbody>
      <geom type="box" size="1 1 0.01" pos="{-0.01 * n[0]} {-0.01 * n[1]} {-0.01 * n[2]}" quat="{q}"
            friction="{MU} 0.005 0.0001"/>
      <body name="obj" pos="{0.02 * n[0]} {0.02 * n[1]} {0.02 * n[2]}" quat="{q}"><freejoint/>
        <geom type="box" size="0.03 0.03 0.02" mass="0.1" friction="{MU} 0.005 0.0001"/></body>
    </worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    for _ in range(50):
        mujoco.mj_step(m, d)
    return m, d, m.body("obj").id


@pytest.mark.parametrize("yaw", np.linspace(0, np.pi / 2, 7))
def test_equilibrium_accepted_below_the_pyramid_and_rejected_past_mu(yaw):
    """Accepted wherever the needed friction is below the pyramid's faces (0.854 mu), and the
    accepted box stays put; rejected once the slope needs more than mu."""
    m, d, root = _incline(0.84, yaw)
    assert settle.in_equilibrium(m, d, root)
    p0 = d.xpos[root].copy()
    for _ in range(1000):
        mujoco.mj_step(m, d)
    assert np.linalg.norm(d.xpos[root] - p0) < 1e-3
    m, d, root = _incline(1.02, yaw)
    assert not settle.in_equilibrium(m, d, root)


def test_basket_placement_verdicts():
    """libero_object 1 (cream cheese into the basket): the basket is not disturbed by the
    placement, the skill teacher's drop from above the basket is a release violation, and the
    placement settles after LIBERO's success."""
    pytest.importorskip("libero.libero")
    from screwhead.sim.gripper_servo import channel_to_target
    from screwhead.sim.sim_arm import Execution
    from screwhead.sim.task_env import TaskEnv
    from screwhead.teacher.skill_teacher import SkillTeacher
    from screwhead.teacher.task_loss import TaskLoss
    from screwhead.teacher.verdicts import GENTLE_GAP, Verdicts

    levels = (0.0, 0.026, 0.08)
    te = TaskEnv("libero_object", 1, seed=0, render=False, execution=Execution(lean=True, anchor=True, scale_lead=True))
    te.reset(0)
    v = Verdicts(te, TaskLoss(te))
    start = v.reference()
    teacher, watch = SkillTeacher(te), v.release_watch()
    success = settled = None
    for t in range(400):
        a = teacher.act(te.snapshot())
        lv = int(np.argmin(np.abs(np.asarray(levels) - float(channel_to_target(a[6])))))
        te.execute(np.r_[np.clip(a[:6], -1, 1), 1.0 - 2.0 * levels[lv] / 0.08], substep=watch.substep)
        assert v.disturbed(start) == "", t
        success = t if success is None and te.success() else success
        if v.settled():
            settled = t
            break
    assert success is not None and settled is not None and settled >= success
    verdict = watch.finish()
    assert verdict.startswith("release cream_cheese_1")
    assert float(verdict.split("gap ")[1].split(" mm")[0]) > GENTLE_GAP * 1000
    assert v.lost() == ""
