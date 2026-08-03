"""The contract a planner may rely on, and nothing more."""

import abc

import numpy as np


class RobotPoints(abc.ABC):
    """Robot surface points as a function of joint configuration.

    `n` points, fixed identity for the lifetime of the object: point i is the
    same material point in every call. That is not decoration -- `robot_velocity`
    and `robot_acceleration` are finite differences of consecutive frames and
    make up 6 of the 16 robot feature channels, so resampling between calls
    turns them into noise (`NOTES.md` section 4).
    """

    n: int
    joint_names: list

    @abc.abstractmethod
    def at_config(self, q):
        """(points, normals) in the world frame for joint vector `q` (J,)."""

    @abc.abstractmethod
    def current_config(self):
        """The robot's joint vector right now, (J,)."""

    def trajectory(self, qs):
        """(T, n, 3) world points for a sequence of configurations (T, J).

        This is a candidate action in the layout the bridge wants: WORLD
        coordinates, uncentred. Centring happens inside the service, and
        pre-centring double-shifts the gripper about a metre from the scene --
        a mistake that reads exactly like "the model cannot rank actions".
        """
        qs = np.asarray(qs, dtype=np.float64)
        out = np.empty((len(qs), self.n, 3), dtype=np.float32)
        for i, q in enumerate(qs):
            out[i] = self.at_config(q)[0]
        return out

    def integrate(self, q0, deltas):
        """(T+1, J) configurations from a start and per-step joint deltas (T, J).

        Kept here rather than in the planner so both backends agree on what an
        action MEANS: a sequence of joint increments from the current pose.
        """
        q0 = np.asarray(q0, dtype=np.float64)
        deltas = np.asarray(deltas, dtype=np.float64)
        return np.vstack([q0, q0 + np.cumsum(deltas, axis=0)])
