"""Length-prefixed JSON over a UNIX socket. Imported by BOTH venvs.

stdlib only, deliberately: `.venv` must be able to speak to the cuRobo service
without importing anything cuRobo or torch related, and `.venv-curobo` must not
need mujoco. The socket is the entire boundary -- same rule as
`pointworld_bridge`.
"""

import json
import socket
import struct

SOCKET_PATH = "/tmp/curobo_ik.sock"


def send(sock, obj):
    payload = json.dumps(obj).encode()
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def recv(sock):
    head = b""
    while len(head) < 4:
        chunk = sock.recv(4 - len(head))
        if not chunk:
            return None
        head += chunk
    (n,) = struct.unpack("!I", head)
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(min(65536, n - len(buf)))
        if not chunk:
            return None
        buf += chunk
    return json.loads(buf.decode())


def serve(path, handler):
    import os

    if os.path.exists(path):
        os.unlink(path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(4)
    print(f"curobo-ik: listening on {path}", flush=True)
    while True:
        conn, _ = srv.accept()
        try:
            while True:
                req = recv(conn)
                if req is None:
                    break
                send(conn, handler(req))
        finally:
            conn.close()


class Client:
    """The `.venv` side. numpy is not even required."""

    def __init__(self, path=SOCKET_PATH, timeout=30.0):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.sock.connect(path)

    def solve(self, q_start, goal_pose, world, joint_names):
        send(self.sock, {"q_start": list(q_start), "goal": list(goal_pose),
                         "world": world, "joint_names": list(joint_names)})
        return recv(self.sock)

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass
