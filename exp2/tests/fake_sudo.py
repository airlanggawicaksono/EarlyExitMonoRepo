"""Fake `sudo -S` for tests of exp2.board.sysfs.RootHelper (script, not imported by tests).

Usage:  python fake_sudo.py [--log FILE] [--refuse] [--delay-hello S] -- CMD [ARG ...]

Behaviour, like `sudo -S -k -p ""`:
  1. Read ONE line from stdin, byte by byte (the password; only its length is logged).
  2. Env FAKE_SUDO_EXPECT set and different from the line: print "Sorry, try again." and exit 1.
  3. --refuse: print "Sorry, try again." and exit 1 without starting CMD.
  4. Otherwise run CMD and forward the rest of stdin to it line by line, and its stdout back.
     --delay-hello S holds the first line CMD prints (the hello) for S seconds.

Every forwarded line is appended to --log as "<seq> PW|IN|OUT|REFUSE <text>", so a test can check the
order in which the client sent requests and the helper answered.
"""
import argparse
import os
import subprocess
import sys
import threading
import time


class Log:
    def __init__(self, path):
        self._path = path
        self._lock = threading.Lock()
        self._seq = 0

    def add(self, kind, text=""):
        if not self._path:
            return
        with self._lock:
            self._seq += 1
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(f"{self._seq} {kind} {text}\n")


def read_line_bytewise(fd=0):
    buf = b""
    while True:
        ch = os.read(fd, 1)
        if not ch or ch == b"\n":
            return buf
        buf += ch


def forward_stdin(child, log):
    for raw in iter(sys.stdin.buffer.readline, b""):
        log.add("IN", raw.decode("utf-8", "replace").rstrip("\r\n"))
        try:
            child.stdin.write(raw)
            child.stdin.flush()
        except (BrokenPipeError, OSError):
            return
    try:
        child.stdin.close()
    except OSError:
        pass


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None)
    ap.add_argument("--refuse", action="store_true")
    ap.add_argument("--delay-hello", type=float, default=0.0)
    ap.add_argument("cmd", nargs="+")
    args = ap.parse_args(argv)
    log = Log(args.log)

    secret = read_line_bytewise(0).decode("utf-8", "replace").rstrip("\r")
    log.add("PW", f"len={len(secret)}")
    expected = os.environ.get("FAKE_SUDO_EXPECT")
    if args.refuse or (expected is not None and secret != expected):
        log.add("REFUSE")
        sys.stderr.write("Sorry, try again.\n")
        sys.stderr.flush()
        return 1

    child = subprocess.Popen(args.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    threading.Thread(target=forward_stdin, args=(child, log), daemon=True).start()
    first = True
    for raw in iter(child.stdout.readline, b""):
        if first and args.delay_hello > 0:
            time.sleep(args.delay_hello)
        first = False
        log.add("OUT", raw.decode("utf-8", "replace").rstrip("\r\n"))
        sys.stdout.buffer.write(raw)
        sys.stdout.buffer.flush()
    return child.wait()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
