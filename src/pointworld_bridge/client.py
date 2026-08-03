"""Talk to the PointWorld service from the LIBERO venv.

Imports numpy and stdlib only, and that is a hard rule: this runs in `.venv`,
where torch 2.11 and the PointWorld tree do not exist.

**The socket is the entire boundary.** This module does not know how to start
the service, which interpreter it runs under, or what environment it needs --
deliberately. An earlier version had `start_server()` here, shelling out to
`.venv-pw/bin/python` and setting `CUMM_CUDA_*` from this side, which made the
LIBERO venv responsible for configuring the PointWorld venv. That is the same
coupling `HANDOFF.md` warns about, one layer up: the two stacks must not know
about each other at all. Lifecycle belongs to `scripts/pointworld_serve.sh`,
which is shell and therefore belongs to neither.

    from pointworld_bridge.client import PointWorldClient

    with PointWorldClient() as pw:
        pw.observe(episode_arrays)                 # once per observation
        head, out = pw.rollout(candidates, goal_idx, goal_pos)
"""

import socket

import numpy as np

from .protocol import DEFAULT_SOCKET, recv_frame, send_frame


class BridgeError(RuntimeError):
    """The service reported a failure, rather than the socket failing."""


class PointWorldClient:
    def __init__(self, path=DEFAULT_SOCKET, timeout=120.0):
        self.path = path
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        try:
            self.sock.connect(path)
        except OSError as exc:
            raise BridgeError(
                f"no PointWorld service on {path} ({exc}). Start it in its own "
                f"venv, from its own shell:\n"
                f"    scripts/pointworld_serve.sh --socket {path}"
            ) from exc

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.sock.close()

    def _call(self, op, arrays=None, **header):
        send_frame(self.sock, dict(op=op, **header), arrays or {})
        head, out = recv_frame(self.sock)
        if not head.get("ok"):
            raise BridgeError(head.get("error", "unknown error"))
        return head, out

    # ------------------------------------------------------------------ ops
    def ping(self):
        head, _ = self._call("ping")
        return head

    def features(self):
        """(Ns, C) per-point DINOv3 features from the last `observe`.

        Same points, same order, same frame as the cloud the planner masks
        over -- which is the only reason a mask derived from these can be
        handed straight back as `goal_idx`.
        """
        head, out = self._call("features")
        return out["features"]

    def observe(self, arrays):
        """Send one camera observation. Returns the service's timing header.

        `arrays` is the recorder's own layout, so a live observation and a
        recorded episode reach the model by the same path.
        """
        keep = ("scene_flows", "scene_colors", "scene_normals", "robot_flows",
                "robot_normals", "gripper_open", "rgb", "depth", "intrinsic",
                "extrinsic")
        head, out = self._call("observe", {k: arrays[k] for k in keep})
        self.centre = out["centre"]
        return head

    def rollout(self, robot_flows, goal_idx=None, goal_pos=None, return_pred=False,
                avoid_idx=None, avoid_weight=0.0):
        """Score candidate gripper trajectories.

        robot_flows : (K, T, Nr, 3) in WORLD coordinates. Not centred -- the
                      service centres them, and pre-centring double-shifts.
        goal_idx    : (M,) indices of the scene points that must reach a target
        goal_pos    : (M, 3) where those points should end up, world frame
        """
        arrays = {"robot_flows": np.ascontiguousarray(robot_flows, dtype=np.float32)}
        if goal_idx is not None:
            arrays["goal_idx"] = np.asarray(goal_idx, dtype=np.int64)
            arrays["goal_pos"] = np.ascontiguousarray(goal_pos, dtype=np.float32)
        if avoid_idx is not None and len(avoid_idx) and avoid_weight > 0:
            arrays["avoid_idx"] = np.asarray(avoid_idx, dtype=np.int64)
        head, out = self._call("rollout", arrays, return_pred=bool(return_pred),
                               avoid_weight=float(avoid_weight))
        return head, out
