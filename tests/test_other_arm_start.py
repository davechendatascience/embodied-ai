"""Another arm starts in the Panda's scene (BRN-other-arm-starts-at-the-panda-tool-pose): its tool at the pose LIBERO's
Panda was recorded at, every object where the Panda's reset leaves it, and the fixtures drawn from the Panda's stream.

  PYTHONPATH=third_party/LIBERO:. .venv-libero/bin/python -m pytest tests/test_other_arm_start.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("libero.libero")


def _reset(robot: str, suite: str, task: int):
    from screwhead.sim.sim_arm import Execution
    from screwhead.sim.task_env import TaskEnv
    env = TaskEnv(suite, task, horizon=600, seed=555 * 100 + task, render=False, execution=Execution(robot=robot))
    env.reset(init_index=0)
    return env


def _objects(env) -> dict:
    m, q = env.env.sim.model, np.asarray(env.env.sim.data.qpos, float)
    return {m.joint_id2name(j): q[int(m.jnt_qposadr[j]):int(m.jnt_qposadr[j]) + 3].copy()
            for j in range(m.njnt) if int(m.jnt_type[j]) == 0}


def _fixtures(env) -> np.ndarray:
    m = env.env.sim.model
    return np.array([m.body_pos[b] for b in range(1, m.nbody)
                     if int(m.body_parentid[b]) == 0 and not (m.body_id2name(b) or "").startswith(("robot", "mount"))])


def test_kinova3_starts_in_the_pandas_scene():
    """libero_goal 3: in robosuite's own start pose the Kinova3 knocked the wine bottle 3 m off the table."""
    from screwhead.geometry.kin_np import NpChain, fk
    from screwhead.sim.libero_env import panda_tool_pose
    panda = _reset("Panda", "libero_goal", 3)
    ref = _objects(panda)
    recorded = panda_tool_pose(panda.init_states[0])[:3, 3]
    panda.close()
    env = _reset("Kinova3", "libero_goal", 3)
    objs = _objects(env)
    tool = fk(NpChain.of(env.chain), np.asarray(env.raw["robot0_joint_pos"], float)[None])[0][:3, 3]
    env.close()
    assert objs.keys() == ref.keys()
    assert max(float(np.linalg.norm(objs[k] - ref[k])) for k in ref) < 1e-9
    assert float(np.linalg.norm(tool - recorded)) < 1e-3


def test_ur5e_draws_the_pandas_fixtures():
    """The 6-joint UR5e's reset draws one normal fewer; unaligned, its fixtures came from a shifted stream."""
    panda = _reset("Panda", "libero_goal", 0)
    ref = _fixtures(panda)
    panda.close()
    env = _reset("UR5e", "libero_goal", 0)
    got = _fixtures(env)
    env.close()
    assert got.shape == ref.shape and np.array_equal(got, ref)


def test_ur5e_starts_in_its_home_family():
    """BRN-other-arm-starts-in-its-home-family: the UR5e starts with its shoulder panned to the far side of the base
    (the family calibrated on libero_90), its tool still at the recorded pose; an arm without a declared home family
    starts as before."""
    from screwhead.geometry.kin_np import NpChain, fk
    from screwhead.sim.libero_env import panda_tool_pose
    from screwhead.sim.sim_arm import HOME_FAMILY, ik_family
    env = _reset("UR5e", "libero_goal", 0)
    q = np.asarray(env.raw["robot0_joint_pos"], float)
    tool = fk(NpChain.of(env.chain), q[None])[0][:3, 3]
    recorded = panda_tool_pose(env.init_states[0])[:3, 3]
    reached = env.home_family_reached
    env.close()
    assert reached is True
    assert ik_family(q) == HOME_FAMILY[("UR5e", "PandaGripper")]
    assert float(np.linalg.norm(tool - recorded)) < 1e-3
    other = _reset("Kinova3", "libero_goal", 0)
    assert other.home_family_reached is None
    other.close()
