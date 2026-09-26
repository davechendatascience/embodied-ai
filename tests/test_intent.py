"""DEF-intended-contact as screwhead.sim.intent computes it: a held object is intended, an arm link in a fixture is
not, and neither two fingers nor bodies one joint apart count as self-contact.

  PYTHONPATH=third_party/LIBERO:. .venv-libero/bin/python -m pytest tests/test_intent.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("libero.libero")


def _env():
    from screwhead.sim.task_env import TaskEnv
    env = TaskEnv("libero_spatial", 0, horizon=600, seed=55500, render=False)
    env.reset(init_index=0)
    return env


def _classify(env, it):
    ts = env.tool_state()
    return it.classify(ts["R_tool"], env.scene.base + ts["p_tool"])


def test_held_bowl_is_intended():
    """The teacher's pick of the bowl: once both fingers hold it, its contacts are intended, and nothing else is
    touched."""
    from screwhead.sim.intent import Intent
    from screwhead.teacher.skill_teacher import SkillTeacher
    env = _env()
    it = Intent(env.scene.m, env.scene.d)
    teacher = SkillTeacher(env)
    rules = set()
    while env.t < 150 and "held" not in rules:
        env.step(teacher.act(env.snapshot()))
        rules = {c.rule for c in _classify(env, it)}
    env.close()
    assert "held" in rules
    assert not rules & {"UNINTENDED", "SELF"}


def test_arm_in_the_table_is_unintended():
    """Written into the table, the arm's contacts are unintended; the two closed fingers touching are not
    self-contact."""
    from screwhead.sim.intent import Intent
    env = _env()
    it = Intent(env.scene.m, env.scene.d)
    sim = env.env.sim
    q = sim.data.qpos[env.joint_indexes].copy()
    q[1] += 0.9                                    # the shoulder pitched down: the hand goes into the table
    sim.data.qpos[env.joint_indexes] = q
    sim.data.qpos[env.gripper_indexes] = [0.0, 0.0]
    sim.forward()
    got = _classify(env, it)
    env.close()
    assert any(c.rule == "UNINTENDED" for c in got)
    assert not any(c.rule == "SELF" and c.robot.startswith("gripper") and c.other.startswith("gripper") for c in got)


def test_self_contact_needs_two_joints():
    """Bodies one joint apart touch by construction; the rule counts joints on the tree path between them."""
    from screwhead.sim.intent import Intent
    env = _env()
    it = Intent(env.scene.m, env.scene.d)
    m = it.m
    links = [b for b in range(m.nbody) if it.names[b].startswith("robot0_link")]
    env.close()
    assert it._joint_count(links[1], links[2]) == 1
    assert it._joint_count(links[1], links[4]) == 3
    assert np.all([it._joint_count(b, b) == 0 for b in links])
