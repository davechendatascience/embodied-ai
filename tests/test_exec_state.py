"""screwhead/sim/exec_state.py: a rollout from a restored execution state repeats the original
bit for bit -- from the reset state (the controller's cache still in use), from mid-episode, and
restored into a second instance of the task (BRN-execution-state-restore).

  PYTHONPATH=third_party/LIBERO:. .venv-libero/bin/python -m pytest tests/test_exec_state.py -q
"""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

pytest.importorskip("libero.libero")

from screwhead.sim import exec_state  # noqa: E402
from screwhead.sim.sim_arm import Execution  # noqa: E402
from screwhead.sim.task_env import TaskEnv  # noqa: E402

LEAN = Execution(lean=True, anchor=True, scale_lead=True)


def _integration(te) -> np.ndarray:
    m, d = te.scene.raw()
    out = np.empty(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(m, d, out, mujoco.mjtState.mjSTATE_INTEGRATION)
    return out


def _actions(rng, n: int) -> list[np.ndarray]:
    return [np.r_[rng.uniform(-1, 1, 6), rng.choice([-1.0, 1.0])] for _ in range(n)]


def _run(te, actions) -> list[np.ndarray]:
    out = []
    for a in actions:
        te.execute(a)
        out.append(_integration(te))
    return out


@pytest.fixture(scope="module")
def env():
    return TaskEnv("libero_goal", 8, seed=0, render=False, execution=LEAN)


@pytest.mark.parametrize("lead", [0, 30])
def test_restore_repeats_the_rollout(env, lead):
    rng = np.random.default_rng(lead)
    env.reset(0)
    _run(env, _actions(rng, lead))
    saved, t0, snap = exec_state.save(env), env.t, dict(env.snapshot())
    actions = _actions(rng, 15)
    first = _run(env, actions)
    exec_state.restore(env, saved)
    assert env.t == t0
    assert all(np.array_equal(snap[k], v) for k, v in env.snapshot().items())
    again = _run(env, actions)
    assert all(np.array_equal(a, b) for a, b in zip(first, again, strict=True))


def test_restore_into_another_instance(env):
    rng = np.random.default_rng(7)
    env.reset(0)
    _run(env, _actions(rng, 20))
    saved = exec_state.save(env)
    actions = _actions(rng, 15)
    first = _run(env, actions)
    other = TaskEnv("libero_goal", 8, seed=1, render=False, execution=LEAN)
    other.reset(1)
    exec_state.restore(other, saved, other_instance=True)
    again = _run(other, actions)
    assert all(np.array_equal(a, b) for a, b in zip(first, again, strict=True))
