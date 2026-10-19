"""Wire between dispatcher and per-family server (K5). stdlib only; no authkey, no pickle.

    listener = wire.listen()                         # Listener(('127.0.0.1', 0)); port = listener_port(listener)
    wire.write_port_file(path, port)                 # atomic: tmp file, then os.replace
    port = wire.wait_port_file(path, 60.0, proc)     # polls; TimeoutError, or RuntimeError if proc already exited
    conn = wire.connect(port)
    wire.send(conn, {"op": "ping"})                  # JSON through Connection.send_bytes
    msg = wire.recv(conn, timeout_s=1.0)             # dict; TimeoutError if nothing arrives in time

Only send_bytes / recv_bytes / poll are used on a connection, so a test double needs just those three.
Messages are validated by contracts.validate_server_msg, not here.
"""
from __future__ import annotations

import json
import os
import time
from multiprocessing.connection import Client, Listener

from exp2.contracts import ContractError

HOST = "127.0.0.1"
POLL_S = 0.02


def listen() -> Listener:
    """Listen on an ephemeral loopback port, without authentication."""
    return Listener((HOST, 0))


def listener_port(listener: Listener) -> int:
    return int(listener.address[1])


def write_port_file(path, port: int) -> None:
    """Publish `port` so that a reader sees either no file or the complete number, never a partial write."""
    path = os.fspath(path)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="ascii", newline="\n") as fh:
        fh.write(f"{int(port)}\n")
        fh.flush()
        os.fsync(fh.fileno())
    for attempt in range(50):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:               # Windows only: a reader has the destination open for a moment
            if attempt == 49:
                raise
            time.sleep(0.01)


def wait_port_file(path, timeout_s: float, proc=None) -> int:
    """Poll for the port file. `proc` (subprocess.Popen) that exits first -> RuntimeError; deadline -> TimeoutError."""
    path = os.fspath(path)
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            with open(path, "r", encoding="ascii") as fh:
                text = fh.read().strip()
            if text:
                return int(text)
        except (FileNotFoundError, ValueError):
            pass
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server process exited with code {proc.returncode} before writing {path}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"no port file {path} after {timeout_s} s")
        time.sleep(POLL_S)


def connect(port: int):
    return Client((HOST, int(port)))


def send(conn, msg: dict) -> None:
    if not isinstance(msg, dict):
        raise TypeError(f"wire messages are dicts, got {type(msg).__name__}")
    conn.send_bytes(json.dumps(msg, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def recv(conn, timeout_s: float | None = None) -> dict:
    """Next message as a dict. timeout_s None waits forever. EOFError / OSError pass through when the peer is gone."""
    if timeout_s is not None and not conn.poll(timeout_s):
        raise TimeoutError(f"no message within {timeout_s} s")
    obj = json.loads(conn.recv_bytes().decode("utf-8"))
    if not isinstance(obj, dict):
        raise ContractError(f"wire message is not an object: {obj!r}")
    return obj
