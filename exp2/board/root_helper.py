"""Root helper for exp2.board: a long-lived process that writes sysfs/debugfs files and runs board tools.

Started by exp2.board.sysfs.RootHelper:
    real: sudo -S -k -p "" python -B -m exp2.board.root_helper        (runs as root)
    dry : python -B -m exp2.board.root_helper --fake-root DIR          (no root, paths re-rooted under DIR)

Protocol: one JSON object per stdin line, one JSON object per stdout line. Lines that are not a JSON object
are ignored (no reply). The first stdout line is the hello:
    {"op":"hello","pid":1234,"uid":0,"fake_root":null}
    {"op":"write","path":P,"value":V}          -> {"ok":true,"readback":"<content of P after the write>"}
    {"op":"read","path":P}                      -> {"ok":true,"value":"<content of P>"}
    {"op":"run","argv":[...],"timeout_s":30}    -> {"ok":true,"rc":0,"out":"<stdout+stderr>"}
    {"op":"stop"}                               -> {"ok":true}
    any failure                                 -> {"ok":false,"err":"..."}

Dry mode also emulates what the kernel and the NVIDIA tools do on the real board:
    devfreq min_freq/max_freq: writing min > max (or max < min) fails; cur_freq is clamped into [min, max].
    nvpmodel -q / -m N, jetson_clocks / --store F / --restore F work on files under the fake root.
Stdlib only: this module runs as root, so it must not import anything else from exp2.
"""
import argparse
import json
import os
import re
import subprocess
import sys

DEVFREQ_DIR = "/sys/class/devfreq/17000000.gpu"
NVPMODEL_STATE = "/etc/nvpmodel_state"
NVPMODEL_MODES = {0: "15W", 1: "25W", 2: "MAXN_SUPER"}


def _slurp(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read().strip()


class Helper:
    def __init__(self, fake_root=None):
        self.fake_root = fake_root

    # ---- paths and files -------------------------------------------------------------------------
    def real(self, path):
        if self.fake_root is None:
            return path
        return os.path.join(self.fake_root, path.lstrip("/\\"))

    def read_file(self, path):
        return _slurp(self.real(path))

    def write_file(self, path, value):
        real = self.real(path)
        if self.fake_root is not None:
            os.makedirs(os.path.dirname(real), exist_ok=True)
        with open(real, "w", encoding="utf-8") as fh:
            fh.write(value)

    # ---- ops -------------------------------------------------------------------------------------
    def op_write(self, msg):
        path, value = msg["path"], str(msg["value"])
        if self.fake_root is not None:
            self._dry_check_devfreq(path, value)
        self.write_file(path, value)
        if self.fake_root is not None:
            self._dry_clamp_cur(path)
        return {"ok": True, "readback": self.read_file(path)}

    def op_read(self, msg):
        return {"ok": True, "value": self.read_file(msg["path"])}

    def op_run(self, msg):
        argv = [str(a) for a in msg["argv"]]
        timeout_s = float(msg.get("timeout_s", 30))
        if self.fake_root is not None:
            rc, out = self._dry_run(argv)
        else:
            proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                  timeout=timeout_s, check=False)
            rc, out = proc.returncode, proc.stdout
        return {"ok": True, "rc": rc, "out": out}

    # ---- dry-mode emulation ----------------------------------------------------------------------
    def _devfreq_file(self, name):
        return os.path.join(DEVFREQ_DIR, name)

    def _is_devfreq_range(self, path):
        return os.path.dirname(path.replace("\\", "/")) == DEVFREQ_DIR and \
            os.path.basename(path) in ("min_freq", "max_freq")

    def _dry_check_devfreq(self, path, value):
        """The kernel rejects min_freq > max_freq (EINVAL). Emulate it so a wrong write order fails."""
        if not self._is_devfreq_range(path):
            return
        name = os.path.basename(path)
        other = int(self.read_file(self._devfreq_file("max_freq" if name == "min_freq" else "min_freq")))
        new = int(value)
        if (name == "min_freq" and new > other) or (name == "max_freq" and new < other):
            raise OSError(22, "Invalid argument (min_freq > max_freq)")

    def _dry_clamp_cur(self, path):
        if not self._is_devfreq_range(path):
            return
        lo = int(self.read_file(self._devfreq_file("min_freq")))
        hi = int(self.read_file(self._devfreq_file("max_freq")))
        cur = int(self.read_file(self._devfreq_file("cur_freq")))
        self.write_file(self._devfreq_file("cur_freq"), str(min(max(cur, lo), hi)))

    def _dry_set_range(self, lo, hi):
        """Write min/max in the order that keeps min <= max (what the real tool does)."""
        cur_lo = int(self.read_file(self._devfreq_file("min_freq")))
        order = ("max_freq", "min_freq") if lo >= cur_lo else ("min_freq", "max_freq")
        for name in order:
            self.op_write({"path": self._devfreq_file(name), "value": lo if name == "min_freq" else hi})

    def _dry_mode_id(self):
        text = self.read_file(NVPMODEL_STATE)
        found = re.search(r"-?\d+", text)
        if found:
            return int(found.group())
        for mode_id, name in NVPMODEL_MODES.items():
            if name in text:
                return mode_id
        raise ValueError(f"cannot parse {NVPMODEL_STATE}: {text!r}")

    def _dry_run(self, argv):
        prog = os.path.basename(argv[0])
        if prog == "nvpmodel" and argv[1:] == ["-q"]:
            mode_id = self._dry_mode_id()
            return 0, f"NV Power Mode: {NVPMODEL_MODES[mode_id]}\n{mode_id}\n"
        if prog == "nvpmodel" and len(argv) == 3 and argv[1] == "-m":
            mode_id = int(argv[2])
            if mode_id not in NVPMODEL_MODES:
                return 255, f"NVPM ERROR: Invalid power mode {mode_id}\n"
            self.write_file(NVPMODEL_STATE, f"{mode_id}\n")
            return 0, ""
        if prog == "jetson_clocks" and len(argv) == 1:
            top = max(int(f) for f in self.read_file(self._devfreq_file("available_frequencies")).split())
            self._dry_set_range(top, top)
            return 0, ""
        if prog == "jetson_clocks" and len(argv) == 3 and argv[1] == "--store":
            state = {n: self.read_file(self._devfreq_file(n)) for n in ("min_freq", "max_freq")}
            self.write_file(argv[2], json.dumps(state))
            return 0, ""
        if prog == "jetson_clocks" and len(argv) == 3 and argv[1] == "--restore":
            state = json.loads(self.read_file(argv[2]))
            self._dry_set_range(int(state["min_freq"]), int(state["max_freq"]))
            return 0, ""
        raise ValueError(f"dry mode does not emulate {argv!r}")


def serve(helper, stdin, stdout):
    def emit(obj):
        stdout.write(json.dumps(obj) + "\n")
        stdout.flush()

    uid = os.getuid() if hasattr(os, "getuid") else -1
    emit({"op": "hello", "pid": os.getpid(), "uid": uid, "fake_root": helper.fake_root})
    ops = {"write": helper.op_write, "read": helper.op_read, "run": helper.op_run}
    for line in stdin:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if not isinstance(msg, dict):
            continue
        op = msg.get("op")
        if op == "stop":
            emit({"ok": True})
            return 0
        try:
            if op not in ops:
                raise ValueError(f"unknown op {op!r}")
            reply = ops[op](msg)
        except Exception as exc:  # every failure goes back to the client; the helper keeps serving
            reply = {"ok": False, "err": f"{type(exc).__name__}: {exc}"}
        emit(reply)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--fake-root", default=None)
    args = ap.parse_args(argv)
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    return serve(Helper(args.fake_root), sys.stdin, sys.stdout)


if __name__ == "__main__":
    sys.exit(main())
