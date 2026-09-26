"""task_env_place.integration_digest: equal for equal integration states, blind to what a step does not read."""
import mujoco
import numpy as np

from screwhead.sim.task_env_place import integration_digest

XML = """<mujoco><worldbody><body><joint name="j" type="hinge"/><geom size="0.1" mass="1"/></body></worldbody>
<actuator><motor joint="j"/></actuator></mujoco>"""


def _placed(qpos: float):
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    d.qpos[0] = qpos
    mujoco.mj_forward(m, d)
    return m, d


def test_equal_states_equal_digests():
    assert integration_digest(*_placed(0.3)) == integration_digest(*_placed(0.3))


def test_a_field_one_step_reads_changes_it():
    base = integration_digest(*_placed(0.3))
    assert integration_digest(*_placed(0.31)) != base
    m, d = _placed(0.3)
    d.ctrl[0] = 1.0                                   # an input left at placement is part of the state
    assert integration_digest(m, d) != base
    m, d = _placed(0.3)
    d.qacc_warmstart[0] = 1.0
    assert integration_digest(m, d) != base


def test_derived_quantities_are_not_in_it():
    m, d = _placed(0.3)
    base = integration_digest(m, d)
    d.xpos[:] = np.nan                                # recomputed by the next step, not read by it
    assert integration_digest(m, d) == base
