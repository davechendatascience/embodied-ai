"""The gripper's TwistServo: the policy names an aperture, a fixed servo reaches it.

robosuite's gripper action is incremental and read by sign -- close, hold, open --
and the fingers lag the command by ~0.12 s. A demonstration program that pre-shapes
the jaws to 26 mm (the drawer task, so the fingers clear the cabinet) must therefore
switch on aperture + rate * tau inside a 3 mm band. As a policy output that is a
controller, not an intent: measured, a student saw 821 pre-shape frames in 18
episodes, matched the teacher's command on 20-25% of held-out pre-shape frames while
the switching rule itself matched 94%, and after one DAgger round the drawer task
fell from 6/10 to 0/10.

So the gripper is split the way the arm already is. The policy emits WHERE the jaws
should be -- a target aperture, embodiment-free in metres -- and this servo, reading
the gripper's own measured aperture and rate, produces the three-valued command. The
teacher's pre-shape law is reproduced exactly (same tau, same band), so a program
executed through the servo behaves as it did when it emitted commands itself.

Channel convention (the 7th action entry): g = 1 - 2 a / A_OPEN, so closed is +1
(the old "close" sign), fully open is -1, and the drawer's 26 mm is +0.35.
"""
from __future__ import annotations

import numpy as np

A_OPEN = 0.08          # Panda: finger qpos 0..0.04 each, aperture = q0 - q1
SATURATE = 0.002       # targets this close to either end are "fully closed/open": command, don't regulate

OPEN_PHASES = {"done", "reopen", "release"}
CLOSED_PHASES = {"close", "squeeze", "lower", "lift", "carry", "exit"}
APPROACH_PHASES = {"approach", "descend", "preshape", "rise"}


def target_to_channel(aperture):
    return 1.0 - 2.0 * np.asarray(aperture, np.float64) / A_OPEN


def channel_to_target(g):
    return (1.0 - np.clip(np.asarray(g, np.float64), -1.0, 1.0)) / 2.0 * A_OPEN


class GripperServo:
    def __init__(self, tau: float = 0.12, band: float = 0.003):
        self.tau, self.band = tau, band

    def command(self, target: float, aperture: float, rate: float) -> float:
        """Target aperture (m), measured aperture (m) and rate (m/s) -> -1 open, 0 hold, +1 close."""
        if target <= SATURATE:
            return 1.0                      # squeeze: keep closing onto whatever is between the pads
        if target >= A_OPEN - SATURATE:
            return -1.0
        pred = aperture + rate * self.tau   # where the lagging fingers will be
        return 1.0 if pred > target + self.band else (-1.0 if pred < target - self.band else 0.0)


def squeeze_command(g) -> float:
    """The +1/-1 command a LIBERO-style gripper (VLA_EXECUTION's command mode) is given for a target channel: close
    only for a squeeze (a target within SATURATE of shut), open for anything else -- fully open and a pre-shape alike,
    which a two-valued gripper cannot hold."""
    return 1.0 if float(channel_to_target(g)) <= SATURATE else -1.0


def snap_channel(g, levels):
    """Nearest of the target apertures (m) the programs actually use, as a channel value.

    The regressed target averages the modes it was trained on: at the drawer's grasp
    the labels step from 26 mm to 0 and the student said ~9 mm, which the servo then
    HELD instead of squeezing (measured, round-2 target model: 12% of drawer close
    frames within 3 mm raw, 88% exact after snapping). The same reason the command
    channel is decoded to -1/0/+1."""
    t = channel_to_target(g)
    lv = np.asarray(sorted(levels), np.float64)
    return target_to_channel(lv[np.abs(np.asarray(t)[..., None] - lv).argmin(-1)])


def program_target(phase: str, command: float, preshape_aperture: float | None) -> float:
    """The aperture a demonstration program is regulating toward, from its phase.

    Approach-family phases hold the pre-shape when the task has one and are open
    otherwise; grasp and transport are closed; release and recovery are open. An
    unknown phase follows the sign of the command it emitted."""
    if phase in OPEN_PHASES:
        return A_OPEN
    if phase in CLOSED_PHASES:
        return 0.0
    if phase in APPROACH_PHASES:
        return A_OPEN if preshape_aperture is None else float(preshape_aperture)
    return A_OPEN if command < 0 else 0.0
