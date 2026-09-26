"""gripper_servo.squeeze_command: a target channel as the two-valued command VLA_EXECUTION's gripper takes."""
from screwhead.sim.gripper_servo import A_OPEN, SATURATE, squeeze_command, target_to_channel


def test_only_a_squeeze_closes():
    assert squeeze_command(target_to_channel(0.0)) == 1.0
    assert squeeze_command(target_to_channel(0.5 * SATURATE)) == 1.0
    assert squeeze_command(target_to_channel(0.026)) == -1.0       # a pre-shape cannot be held: open
    assert squeeze_command(target_to_channel(A_OPEN)) == -1.0
