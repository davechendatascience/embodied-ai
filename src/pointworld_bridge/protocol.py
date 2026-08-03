"""Framing for the cross-venv bridge. Imports numpy and stdlib, nothing else.

This module is imported by BOTH venvs, so it must not touch torch, mujoco, or
anything else that only exists on one side. That constraint is the whole
reason the file exists separately.

Serialization is the `.npy` format inside a zip (`np.savez`), NOT pickle.
numpy 1.26 is on the LIBERO side and numpy 2.5 on the PointWorld side, and
pickled arrays are not reliably portable across a numpy major version, while
the NPY format is versioned and stable by design. `allow_pickle=False` is left
at its default so a stray object array fails loudly here rather than
deserializing into something surprising on the far side.

Frame layout:

    b"PWB1" | uint32 header_len | uint32 blob_len | header JSON | npz blob

Scalars, strings and flags travel in the JSON header; bulk numeric data
travels in the npz. Keeping them apart avoids numpy's unicode dtypes, which
did change between the two versions in use.
"""

import io
import json
import socket
import struct

import numpy as np

MAGIC = b"PWB1"
_PREFIX = struct.Struct(">II")
DEFAULT_SOCKET = "/tmp/pointworld_bridge.sock"


def encode(header, arrays):
    """(dict, dict[str, ndarray]) -> bytes"""
    buf = io.BytesIO()
    np.savez(buf, **{k: np.ascontiguousarray(v) for k, v in arrays.items()})
    blob = buf.getvalue()
    head = json.dumps(header).encode("utf-8")
    return MAGIC + _PREFIX.pack(len(head), len(blob)) + head + blob


def decode(payload):
    """bytes -> (header, dict[str, ndarray])"""
    if payload[:4] != MAGIC:
        raise ValueError(f"bad frame magic: {payload[:4]!r}")
    head_len, blob_len = _PREFIX.unpack(payload[4:12])
    head = json.loads(payload[12:12 + head_len].decode("utf-8"))
    blob = payload[12 + head_len:12 + head_len + blob_len]
    with np.load(io.BytesIO(blob)) as z:
        arrays = {k: z[k] for k in z.files}
    return head, arrays


def _recv_exactly(sock, n):
    chunks, got = [], 0
    while got < n:
        chunk = sock.recv(min(1 << 20, n - got))
        if not chunk:
            raise ConnectionError(f"peer closed after {got}/{n} bytes")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def send_frame(sock, header, arrays):
    payload = encode(header, arrays)
    sock.sendall(struct.pack(">Q", len(payload)) + payload)


def recv_frame(sock):
    (length,) = struct.unpack(">Q", _recv_exactly(sock, 8))
    return decode(_recv_exactly(sock, length))
