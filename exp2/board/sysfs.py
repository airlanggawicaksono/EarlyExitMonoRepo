"""Client side of the root helper: start it (sudo or dry), then write/read sysfs files and run board tools.

    RootHelper.start(mode="sudo")                      real board: sudo -S -k -p "" python -B -m exp2.board.root_helper
    RootHelper.start(mode="dry", fake_root=DIR)        no root: helper re-roots every path under DIR

Start-up order (real mode): the password goes down the pipe ONCE, then the client waits for the hello line
(15 s max) before it sends any request, because sudo may read more than one line from the pipe.
The password is read at run time from env JETSON_SUDO_PW or getpass and is never written to a file.

`oplog` lists the calls that change state, in call order: ("write", path, value) and ("run", argv_tuple, None).
Reads are not logged. Every failure (helper refused, died, timed out, replied ok=false) raises ClockError.
"""
import getpass
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from exp2.contracts import SUDO_PW_ENV, ClockError

REPO_ROOT = Path(__file__).resolve().parents[2]
HELLO_TIMEOUT_S = 15.0
REQUEST_TIMEOUT_S = 10.0


def default_sudo_argv():
    """The real start command (K17). -B keeps root from leaving root-owned __pycache__ in the repo."""
    return ["sudo", "-S", "-k", "-p", "", sys.executable, "-B", "-m", "exp2.board.root_helper"]


def read_password():
    """Env JETSON_SUDO_PW if set, else an interactive getpass prompt. Never stored anywhere."""
    pw = os.environ.get(SUDO_PW_ENV)
    if pw is not None:
        return pw
    return getpass.getpass("sudo password for the Jetson root helper: ")


def _pump(stream, out):
    """Reader thread: push every JSON-object line to `out`; push None at EOF. Other lines are ignored."""
    for line in stream:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if isinstance(msg, dict):
            out.put(msg)
    out.put(None)


class RootHelper:
    def __init__(self, proc, replies, reader, hello):
        self._proc = proc
        self._replies = replies
        self._reader = reader
        self._lock = threading.Lock()
        self._closed = False
        self.hello = hello
        self.oplog = []

    @classmethod
    def start(cls, mode="sudo", fake_root=None, sudo_argv=None, hello_timeout_s=HELLO_TIMEOUT_S):
        if mode == "dry":
            if fake_root is None:
                raise ValueError("mode='dry' needs fake_root")
            argv = [sys.executable, "-B", "-m", "exp2.board.root_helper", "--fake-root", str(fake_root)]
        elif mode == "sudo":
            argv = list(sudo_argv) if sudo_argv is not None else default_sudo_argv()
        else:
            raise ValueError(f"unknown mode {mode!r}")
        env = {k: v for k, v in os.environ.items() if k != SUDO_PW_ENV}
        pw = read_password() if mode == "sudo" else None
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                encoding="utf-8", cwd=str(REPO_ROOT), env=env)
        replies = queue.Queue()
        reader = threading.Thread(target=_pump, args=(proc.stdout, replies), daemon=True)
        reader.start()
        try:
            if pw is not None:
                try:
                    proc.stdin.write(pw + "\n")
                    proc.stdin.flush()
                except OSError:
                    pass  # sudo already exited; the hello wait below reports it with the exit code
                pw = None
            hello = cls._wait_hello(proc, replies, hello_timeout_s)
        except BaseException:
            cls._kill(proc)
            raise
        return cls(proc, replies, reader, hello)

    @staticmethod
    def _wait_hello(proc, replies, timeout_s):
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                msg = replies.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                raise ClockError(f"root helper sent no hello within {timeout_s:g} s (sudo refused or stuck?)")
            if msg is None:
                try:
                    rc = proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    rc = None
                raise ClockError(f"root helper exited before hello (rc={rc}); wrong password or sudo refused?")
            if msg.get("op") == "hello":
                return msg

    @staticmethod
    def _kill(proc):
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        for stream in (proc.stdin, proc.stdout):
            try:
                stream.close()
            except OSError:
                pass

    # ---- requests --------------------------------------------------------------------------------
    def _request(self, msg, timeout_s=REQUEST_TIMEOUT_S):
        with self._lock:
            if self._closed:
                raise ClockError("root helper is closed")
            try:
                self._proc.stdin.write(json.dumps(msg) + "\n")
                self._proc.stdin.flush()
            except OSError as exc:
                raise ClockError(f"root helper is gone: {exc}") from exc
            deadline = time.monotonic() + timeout_s
            while True:
                try:
                    reply = self._replies.get(timeout=max(0.0, deadline - time.monotonic()))
                except queue.Empty:
                    self._kill(self._proc)
                    self._closed = True
                    raise ClockError(f"root helper did not answer {msg['op']} within {timeout_s:g} s")
                if reply is None:
                    self._closed = True
                    raise ClockError(f"root helper exited while handling {msg['op']}")
                if "ok" in reply:
                    break
        if not reply["ok"]:
            raise ClockError(f"root helper {msg['op']} failed: {reply.get('err')}")
        return reply

    def write(self, path, value):
        """Write `value` to `path` as root; returns the content read back right after the write."""
        self.oplog.append(("write", str(path), str(value)))
        return self._request({"op": "write", "path": str(path), "value": str(value)})["readback"]

    def read(self, path):
        return self._request({"op": "read", "path": str(path)})["value"]

    def run(self, argv, timeout_s=30.0):
        """Run argv as root. Returns (returncode, stdout+stderr). A non-zero rc is NOT an error here."""
        argv = [str(a) for a in argv]
        self.oplog.append(("run", tuple(argv), None))
        reply = self._request({"op": "run", "argv": argv, "timeout_s": timeout_s}, timeout_s=timeout_s + 10)
        return reply["rc"], reply["out"]

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._proc.stdin.write(json.dumps({"op": "stop"}) + "\n")
                self._proc.stdin.flush()
                self._proc.stdin.close()
                self._proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
            self._kill(self._proc)
        self._reader.join(timeout=2)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
