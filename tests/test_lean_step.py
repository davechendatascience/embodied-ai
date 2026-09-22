"""SimArm._advance (Execution.lean) leaves the integration state bit-identical to robosuite's
env.step under the same commands, the first period after a reset included
(BRN-lean-step-keeps-controller-cache).

  PYTHONPATH=third_party/LIBERO:. .venv-libero/bin/python -m pytest tests/test_lean_step.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("libero.libero")


def _state(env) -> np.ndarray:
    """Everything the next step integrates from (time, qpos, qvel, act, warmstart, ctrl,
    applied forces, ...). Not qacc: the lean period's closing forward recomputes derived
    quantities at the new state on purpose (BRN-policies-read-one-forwarded-state)."""
    import mujoco
    m, d = env.env.sim.model._model, env.env.sim.data._data
    out = np.empty(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(m, d, out, mujoco.mjtState.mjSTATE_INTEGRATION)
    return out


@pytest.mark.parametrize("suite,task", [("libero_goal", 8), ("libero_object", 1), ("libero_goal", 0)])
def test_lean_period_is_env_step(suite, task):
    from screwhead.sim.task_env import TaskEnv
    rng = np.random.default_rng(0)
    runs = []
    for lean in (False, True):
        env = TaskEnv(suite, task, seed=0, render=False)
        env.lean = lean
        env.reset(0)
        cmds = rng.uniform(-1, 1, size=(40, env.env.env.action_dim)) if not runs else runs[0][1]
        states = []
        for cmd in cmds:
            cmd = cmd.copy()
            cmd[-1] = 1.0 if cmd[-1] > 0 else -1.0          # close / open, so the fingers meet things
            env._gains()
            if lean:
                env._advance(cmd)
            else:
                env.env.step(cmd)
            states.append(_state(env))
        runs.append((np.array(states), cmds))
        env.close()
    assert np.array_equal(runs[0][0], runs[1][0])
