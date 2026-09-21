"""A linear ramp of the joint goal across each control period, for robosuite's joint controller.

LIBERO steps the joint goal once per 50 ms period, and the stiff PD behind it (kp 4000,
critically damped) makes the tool lunge and coast inside every period: 0.08 -> 0.20 ->
0.09 m/s within one steady carry step, acceleration p95 12 m/s^2 at the 2 ms physics
substep. robosuite 1.4.0's own LinearInterpolator is not linear: it returns
start + (goal - start) / (N - k) without advancing start, so half of each move lands in
the last two of 25 substeps (measured: the joint torque sagged mid-period and spiked to
-178 N m at its end). This ramp moves the goal by equal increments instead.
"""
from __future__ import annotations

import numpy as np


class JointRamp:
    """Duck-types robosuite's interpolator for JointPositionController (set_goal,
    get_interpolated_goal, order)."""
    order = 1

    def __init__(self, substeps: int, start: np.ndarray):
        self.n = max(1, int(substeps))
        self.start = np.array(start, float)
        self.goal = self.start.copy()
        self.k = self.n

    def value(self) -> np.ndarray:
        return self.start + (self.goal - self.start) * (self.k / self.n)

    def set_goal(self, goal) -> None:
        """A new goal ramps from wherever the goal is now, finished or not."""
        self.start = self.value()
        self.goal = np.array(goal, float)
        self.k = 0

    def get_interpolated_goal(self) -> np.ndarray:
        self.k = min(self.k + 1, self.n)
        return self.value()


def install(controller, fraction: float, substeps_per_period: int) -> None:
    """Give the (rebuilt-on-reset) joint controller a ramp over `fraction` of each period."""
    if fraction <= 0 or isinstance(controller.interpolator, JointRamp):
        return
    q = np.asarray(controller.sim.data.qpos[controller.qpos_index], float)
    controller.interpolator = JointRamp(round(fraction * substeps_per_period), q)
