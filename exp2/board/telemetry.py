"""Board telemetry sampler (contract K10).

    python -m exp2.board.telemetry --out DIR [--hz 20] [--clocks-hz 1] [--duration S]
                                   [--sysfs-root /] [--no-tegrastats]

Writes DIR/power.csv (VDD_IN rail, --hz), DIR/clocks.csv (GPU/EMC clocks, memory, temperatures,
--clocks-hz) and DIR/meta.json. The VDD_IN channel is found by its label, never by position.
Stops on stdin EOF, SIGTERM (POSIX) or when --duration expires; rows are flushed every 5 s and on exit.
Exit codes: 0 done, 2 usage error or VDD_IN not found.
"""
import argparse
import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from exp2.contracts import CLOCKS_CSV_FIXED_COLS, POWER_CSV_COLS, SCHEMAS

SCHEMA = SCHEMAS["telemetry_meta"]
RAIL_LABEL = "VDD_IN"
FLUSH_EVERY_NS = 5_000_000_000
BUFFER_BYTES = 1 << 16          # rows reach the disk only through the 5 s flush (or exit)
_GPU_DEVFREQ = re.compile(r"gpu|ga10b|gv11b|gp10b")
_EMC_FREQ = re.compile(r"EMC_FREQ\s+[\d.]+%@(\d+(?:\.\d+)?)")


class TelemetryError(RuntimeError):
    """A required source is missing (VDD_IN); the CLI turns this into exit code 2."""


def parse_emc_mhz(line):
    """EMC frequency in MHz from a tegrastats line (`EMC_FREQ 12%@3199`), or None."""
    m = _EMC_FREQ.search(line)
    return float(m.group(1)) if m else None


def _warn(msg):
    print(f"telemetry: {msg}", file=sys.stderr, flush=True)


def _read_text(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, ValueError):
        return None


def _read_int(path):
    text = _read_text(path)
    try:
        return int(text) if text is not None else None
    except ValueError:
        return None


def _natural(p):
    return (len(p.name), p.name)


def _board_path(root, path):
    """Path as the board sees it: POSIX, leading slash, relative to the sysfs root."""
    return "/" + Path(path).relative_to(root).as_posix()


def _num(x):
    """20.0 -> 20 so the JSON reads like the contract example."""
    return int(x) if float(x).is_integer() else x


class Rail:
    def __init__(self, root, hwmon_dir, channel):
        self.dir = Path(hwmon_dir)
        self.channel = channel
        self.mv_path = self.dir / f"in{channel}_input"
        self.ma_path = self.dir / f"curr{channel}_input"
        self.board_path = _board_path(root, self.dir)
        self.update_interval_ms = _read_int(self.dir / "update_interval")
        self.samples = _read_int(self.dir / "samples")

    def meta(self):
        return {"hwmon": self.board_path, "channel": self.channel, "label": RAIL_LABEL,
                "update_interval_ms": self.update_interval_ms, "samples": self.samples}


def find_vdd_in(root):
    """Locate the INA3221 channel labelled VDD_IN. Raises TelemetryError when there is none."""
    root = Path(root)
    candidates = []
    for d in sorted((root / "sys/class/hwmon").glob("hwmon*"), key=_natural):
        if _read_text(d / "name") == "ina3221":
            candidates.append(d)
    candidates += sorted((root / "sys/bus/i2c/drivers/ina3221").glob("*/hwmon/*"), key=_natural)
    for d in candidates:
        for label_file in sorted(d.glob("in*_label"), key=_natural):
            m = re.fullmatch(r"in(\d+)_label", label_file.name)
            if m and _read_text(label_file) == RAIL_LABEL:
                rail = Rail(root, d, int(m.group(1)))
                if rail.mv_path.is_file() and rail.ma_path.is_file():
                    return rail
    raise TelemetryError(
        f"no {RAIL_LABEL} rail found: looked for an in*_label equal to {RAIL_LABEL} on every hwmon named "
        f"ina3221 under {root / 'sys/class/hwmon'} and on {root / 'sys/bus/i2c/drivers/ina3221'}/*/hwmon/*")


def _find_zones(root):
    """[(column, zone_type, temp_path)] sorted by column name; duplicate types get the zone index."""
    zones = []
    for d in sorted((Path(root) / "sys/class/thermal").glob("thermal_zone*"), key=_natural):
        ztype = _read_text(d / "type")
        if ztype:
            stem = re.sub(r"[^a-z0-9_]", "_", ztype.lower())
            zones.append((stem, ztype, d / "temp", d.name[len("thermal_zone"):]))
    counts = {}
    for stem, *_ in zones:
        counts[stem] = counts.get(stem, 0) + 1
    out = []
    for stem, ztype, temp, idx in zones:
        name = stem if counts[stem] == 1 else f"{stem}_{idx}"
        out.append((f"temp_{name}_mc", ztype, temp))
    return sorted(out, key=lambda z: z[0])


def _find_gpu(root):
    base = Path(root) / "sys/class/devfreq"
    for d in sorted(base.glob("*"), key=_natural):
        if _GPU_DEVFREQ.search(d.name):
            load = next((p for p in (d / "device/load", d / "load") if p.is_file()), None)
            return d, load
    return None, None


class RealClock:
    def mono_ns(self):
        return time.monotonic_ns()

    def wall_ns(self):
        return time.time_ns()

    def wait(self, event, seconds):
        event.wait(seconds)


class Telemetry:
    """clock needs mono_ns(), wall_ns() and wait(event, seconds); tests inject a virtual one."""

    def __init__(self, out_dir, *, root="/", hz=20.0, clocks_hz=1.0, duration_s=None,
                 tegrastats_argv=None, clock=None):
        self.out = Path(out_dir)
        self.root = Path(root)
        self.hz = hz
        self.clocks_hz = clocks_hz
        self.duration_s = duration_s
        self.tegrastats_argv = tegrastats_argv
        self.clock = clock or RealClock()
        self.stop_reason = None
        self._stop = threading.Event()
        self._warned = set()
        self._emc = None
        self._emc_seen = False
        self._tegra = None
        self._tegra_thread = None
        self._closing = False
        self._n = 0
        self._t_first = None
        self._t_last = None

    # ---- control
    def request_stop(self, reason):
        if not self._stop.is_set():
            self.stop_reason = reason
            self._stop.set()

    def _warn_once(self, key, msg):
        if key not in self._warned:
            self._warned.add(key)
            _warn(msg)

    # ---- tegrastats (EMC)
    def _start_tegrastats(self):
        if not self.tegrastats_argv:
            return
        try:
            self._tegra = subprocess.Popen(self.tegrastats_argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                           stdin=subprocess.DEVNULL, text=True, errors="replace", bufsize=1)
        except OSError as e:
            _warn(f"tegrastats could not start ({e}); emc_mhz stays empty")
            return
        self._tegra_thread = threading.Thread(target=self._pump_tegrastats, args=(self._tegra,), daemon=True)
        self._tegra_thread.start()

    def _pump_tegrastats(self, proc):
        try:
            for line in proc.stdout:
                emc = parse_emc_mhz(line)
                if emc is not None:
                    self._emc = emc
                    self._emc_seen = True
        except (OSError, ValueError):
            pass
        if not self._closing:                               # tegrastats ended by itself: do not keep a stale value
            self._emc = None
            self._warn_once("tegra_died", "tegrastats exited early; emc_mhz is empty from now on")

    def _stop_tegrastats(self):
        proc = self._tegra
        if proc is None:
            return
        self._closing = True
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        if self._tegra_thread is not None:
            self._tegra_thread.join(timeout=1)
        proc.stdout.close()
        if not self._emc_seen:
            _warn("tegrastats produced no EMC_FREQ value; emc_mhz stays empty")

    # ---- sampling
    def _power_row(self, t, wall):
        mv = _read_int(self.rail.mv_path)
        ma = _read_int(self.rail.ma_path)
        if mv is None or ma is None:
            bad = self.rail.mv_path if mv is None else self.rail.ma_path
            self._warn_once("rail", f"cannot read {bad}; power rows are skipped until it can be read")
            return None
        return f"{t},{wall},{mv},{ma},{mv * ma / 1000:.3f}"

    def _mem_available_kb(self):
        text = _read_text(self.root / "proc/meminfo") or ""
        for line in text.splitlines():
            if line.startswith("MemAvailable:"):
                return line.split()[1]
        return None

    def _clocks_row(self, t, wall):
        dev = self.gpu_dir
        cells = [
            _read_int(dev / "cur_freq") if dev else None,
            _read_int(dev / "min_freq") if dev else None,
            _read_int(dev / "max_freq") if dev else None,
            _read_int(self.gpu_load) if self.gpu_load else None,
            None if self._emc is None else f"{self._emc:g}",
            self._mem_available_kb(),
            _read_int(self.root / "sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"),
        ]
        cells += [_read_int(path) for _, _, path in self.zones]
        return ",".join([str(t), str(wall)] + ["" if c is None else str(c) for c in cells])

    # ---- meta
    def _write_meta(self):
        """Rewritten atomically at start and at exit, so meta.json on disk always satisfies the contract."""
        achieved = 0.0
        if self._n >= 2 and self._t_last > self._t_first:
            achieved = round((self._n - 1) / ((self._t_last - self._t_first) / 1e9), 3)
        meta = {
            "schema": SCHEMA,
            "vdd_in": self.rail.meta(),
            "target_hz": _num(self.hz),
            "achieved_hz": achieved,
            "clocks_target_hz": _num(self.clocks_hz),
            "devfreq": _board_path(self.root, self.gpu_dir) if self.gpu_dir else None,
            "emc_source": "tegrastats" if self._emc_seen else None,
            "thermal_zones": [ztype for _, ztype, _ in self.zones],
            "boot_id": _read_text(self.root / "proc/sys/kernel/random/boot_id"),
            "t_mono_start_ns": self.t_mono_start,
            "t_wall_start_ns": self.t_wall_start,
            "n_rows": self._n,
        }
        tmp = self.out / "meta.json.tmp"
        tmp.write_text(json.dumps(meta, indent=1) + "\n")
        os.replace(tmp, self.out / "meta.json")

    # ---- main loop
    def run(self):
        """Sample until stopped; returns the number of power rows written."""
        self.rail = find_vdd_in(self.root)                  # before any file is created
        self.gpu_dir, self.gpu_load = _find_gpu(self.root)
        if self.gpu_dir is None:
            _warn("no GPU devfreq node found; gpu_* columns stay empty")
        self.zones = _find_zones(self.root)
        self.out.mkdir(parents=True, exist_ok=True)
        clock = self.clock
        self.t_mono_start = clock.mono_ns()
        self.t_wall_start = clock.wall_ns()
        self._write_meta()
        _warn(f"started: {self.rail.board_path} channel {self.rail.channel} ({RAIL_LABEL}) at {self.hz:g} Hz")
        self._start_tegrastats()
        try:
            with contextlib.ExitStack() as stack:
                pf = stack.enter_context(open(self.out / "power.csv", "w", buffering=BUFFER_BYTES, newline="\n"))
                cf = stack.enter_context(open(self.out / "clocks.csv", "w", buffering=BUFFER_BYTES, newline="\n"))
                pf.write(",".join(POWER_CSV_COLS) + "\n")
                cf.write(",".join(list(CLOCKS_CSV_FIXED_COLS) + [c for c, _, _ in self.zones]) + "\n")
                pf.flush()
                cf.flush()
                self._loop(pf, cf)
        finally:
            self._stop_tegrastats()
            self._write_meta()
        _warn(f"stopped ({self.stop_reason or 'duration reached'}) after {self._n} rows")
        return self._n

    def _loop(self, pf, cf):
        clock = self.clock
        period = round(1e9 / self.hz)
        clocks_period = round(1e9 / self.clocks_hz)
        t0 = self.t_mono_start
        deadline = None if self.duration_s is None else t0 + round(self.duration_s * 1e9)
        next_power = next_clocks = last_flush = t0
        while not self._stop.is_set():
            now = clock.mono_ns()
            if deadline is not None and now >= deadline:
                break
            wall = clock.wall_ns()
            row = self._power_row(now, wall)
            if row is not None:
                pf.write(row + "\n")
                self._n += 1
                self._t_last = now
                if self._t_first is None:
                    self._t_first = now
            if now >= next_clocks:
                cf.write(self._clocks_row(now, wall) + "\n")
                next_clocks = max(next_clocks + clocks_period, now + 1)
            if now - last_flush >= FLUSH_EVERY_NS:
                pf.flush()
                cf.flush()
                last_flush = now
            now2 = clock.mono_ns()
            next_power += period
            if next_power <= now2:                          # stalled: skip the missed slots, never burst
                next_power += ((now2 - next_power) // period + 1) * period
            wait_ns = next_power - now2
            if deadline is not None:
                wait_ns = min(wait_ns, deadline - now2)
            clock.wait(self._stop, max(wait_ns, 0) / 1e9)


def _positive(text):
    value = float(text)
    if not value > 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {text}")
    return value


def _watch_stdin(fd, tele):
    """Stop the sampler when the parent closes our stdin. Raw os.read: no buffered-reader lock at exit."""
    while True:
        try:
            data = os.read(fd, 4096)
        except OSError:
            return                                          # stdin unusable: do not stop on it
        if not data:
            tele.request_stop("stdin EOF")
            return


def main(argv=None, *, stdin_fd=None):
    p = argparse.ArgumentParser(prog="python -m exp2.board.telemetry", description="Board power and clock sampler")
    p.add_argument("--out", required=True, help="directory for power.csv, clocks.csv, meta.json")
    p.add_argument("--hz", type=_positive, default=20.0)
    p.add_argument("--clocks-hz", type=_positive, default=1.0)
    p.add_argument("--duration", type=_positive, default=None, help="seconds; default run until stdin EOF or SIGTERM")
    p.add_argument("--sysfs-root", default="/")
    p.add_argument("--no-tegrastats", action="store_true")
    a = p.parse_args(argv)

    tegra = None
    if not a.no_tegrastats:
        exe = shutil.which("tegrastats")
        if exe:
            tegra = [exe, "--interval", str(max(100, round(1000 / a.clocks_hz)))]
    tele = Telemetry(a.out, root=a.sysfs_root, hz=a.hz, clocks_hz=a.clocks_hz, duration_s=a.duration,
                     tegrastats_argv=tegra)

    if stdin_fd is None:
        try:
            stdin_fd = sys.stdin.fileno()
        except (AttributeError, ValueError, OSError):
            stdin_fd = None
    if stdin_fd is not None:
        threading.Thread(target=_watch_stdin, args=(stdin_fd, tele), daemon=True).start()
    previous = None
    try:
        previous = signal.signal(signal.SIGTERM, lambda signum, frame: tele.request_stop("SIGTERM"))
    except ValueError:                                      # not the main thread
        pass
    try:
        tele.run()
    except TelemetryError as e:
        _warn(f"error: {e}")
        return 2
    except KeyboardInterrupt:
        _warn("interrupted")
        return 0
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
    return 0


if __name__ == "__main__":
    sys.exit(main())
