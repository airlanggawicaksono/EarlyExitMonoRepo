"""Board control (K17): power mode, GPU frequency, EMC lock, jetson_clocks, all through the root helper.

Every state change is read back. ValueError means the request itself is wrong (raised BEFORE any write);
ClockError means the board did not do what was asked (never swallowed here).

    python -m exp2.board.clocks [--dry-run --fake-root DIR] snapshot
    python -m exp2.board.clocks [--dry-run --fake-root DIR] set-gpu MHZ
    python -m exp2.board.clocks [--dry-run --fake-root DIR] lock-emc MHZ
    python -m exp2.board.clocks [--dry-run --fake-root DIR] apply-arm ARM [--gpu MHZ] [--emc MHZ]
Exit codes: 0 ok, 2 ValueError or usage, 4 ClockError.
"""
import argparse
import json
import sys
import time

from exp2.board.sysfs import RootHelper
from exp2.contracts import ARMS, EMC_LOCKABLE_MHZ, POWER_MODE_ID, ClockError

DEVFREQ = "/sys/class/devfreq/17000000.gpu/"
EMC = "/sys/kernel/debug/bpmp/debug/clk/emc/"
BWMGR_HALT = "/sys/kernel/debug/bpmp/debug/bwmgr/bwmgr_halt"
BOOT_ID = "/proc/sys/kernel/random/boot_id"
HZ_PER_MHZ = 1_000_000
GPU_SNAP_TOL_HZ = 1 * HZ_PER_MHZ        # a requested level snaps to a table level within 1 MHz
EMC_READBACK_TOL_HZ = 1 * HZ_PER_MHZ
CUR_FREQ_DELAY_S = 0.05                  # cur_freq is read this long after the min/max writes


def parse_power_mode(text):
    """Parse `nvpmodel -q` ("NV Power Mode: NAME" then the id on the next line), as bench_jetson.py does."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        if "NV Power Mode" not in ln:
            continue
        rest = lines[i + 1:]
        name = ln.split(":")[-1].strip()
        if not name and rest:
            name, rest = rest[0], rest[1:]
        if name and rest and rest[0].isdigit():
            return name, int(rest[0])
    raise ClockError(f"cannot parse `nvpmodel -q` output: {text!r}")


class Board:
    """exp2.contracts.BoardControl on top of a RootHelper (or any object with write/read/run/close)."""

    def __init__(self, helper):
        self.helper = helper

    # ---- helpers ---------------------------------------------------------------------------------
    def _write_checked(self, path, value):
        """Write, then compare the content the helper read back right after the write."""
        got = self.helper.write(path, value)
        if got.strip() != str(value):
            raise ClockError(f"{path}: wrote {value!r}, read back {got!r}")

    def _read_int(self, path):
        return int(self.helper.read(path).split()[0])

    def _run_ok(self, argv, timeout_s=30.0):
        rc, out = self.helper.run(argv, timeout_s=timeout_s)
        if rc != 0:
            raise ClockError(f"`{' '.join(argv)}` failed with rc={rc}: {out.strip()}")
        return out

    # ---- power mode ------------------------------------------------------------------------------
    def query_power_mode(self):
        return parse_power_mode(self._run_ok(["nvpmodel", "-q"]))

    def ensure_power_mode(self, mode_id=POWER_MODE_ID):
        name, cur = self.query_power_mode()
        if cur == mode_id:
            return
        self._run_ok(["nvpmodel", "-m", str(mode_id)], timeout_s=60.0)
        name, cur = self.query_power_mode()
        if cur != mode_id:
            raise ClockError(f"power mode is {name} ({cur}) after `nvpmodel -m {mode_id}`")

    # ---- jetson_clocks ---------------------------------------------------------------------------
    def run_jetson_clocks(self):
        self._run_ok(["jetson_clocks"])

    def store_jetson_clocks(self, path):
        """Save the clock state to `path` and record this boot's id next to it (path + '.boot_id')."""
        boot_id = self.helper.read(BOOT_ID)
        self._run_ok(["jetson_clocks", "--store", str(path)])
        self._write_checked(str(path) + ".boot_id", boot_id)

    def restore_jetson_clocks(self, path):
        stored = self.helper.read(str(path) + ".boot_id")
        now = self.helper.read(BOOT_ID)
        if stored != now:
            raise ClockError(f"refusing to restore {path}: stored in boot {stored!r}, this boot is {now!r}")
        self._run_ok(["jetson_clocks", "--restore", str(path)])

    # ---- GPU -------------------------------------------------------------------------------------
    def _gpu_levels_hz(self):
        return sorted(int(x) for x in self.helper.read(DEVFREQ + "available_frequencies").split())

    def gpu_levels(self):
        return [round(hz / HZ_PER_MHZ) for hz in self._gpu_levels_hz()]

    def _snap_gpu(self, mhz):
        levels = self._gpu_levels_hz()
        want = float(mhz) * HZ_PER_MHZ
        best = min(levels, key=lambda hz: abs(hz - want))
        if abs(best - want) > GPU_SNAP_TOL_HZ:
            raise ValueError(f"GPU level {mhz} MHz is not in the table {[hz // HZ_PER_MHZ for hz in levels]}")
        return best

    def _set_gpu_range(self, lo_hz, hi_hz):
        """Write min_freq/max_freq so that min <= max holds after every write, then read both back."""
        cur_min = self._read_int(DEVFREQ + "min_freq")
        order = ("max_freq", "min_freq") if lo_hz >= cur_min else ("min_freq", "max_freq")
        for name in order:
            self._write_checked(DEVFREQ + name, lo_hz if name == "min_freq" else hi_hz)
        got_min, got_max = self._read_int(DEVFREQ + "min_freq"), self._read_int(DEVFREQ + "max_freq")
        if (got_min, got_max) != (lo_hz, hi_hz):
            raise ClockError(f"GPU range read back {got_min}..{got_max} Hz, wanted {lo_hz}..{hi_hz} Hz")

    def set_gpu_level(self, mhz):
        """Pin the GPU at one table level: going up writes max then min, going down writes min then max.
        Returns the cur_freq read back (MHz)."""
        hz = self._snap_gpu(mhz)
        self._set_gpu_range(hz, hz)
        time.sleep(CUR_FREQ_DELAY_S)
        cur = self._read_int(DEVFREQ + "cur_freq")
        if cur != hz:
            raise ClockError(f"GPU cur_freq is {cur} Hz {CUR_FREQ_DELAY_S * 1000:g} ms after pinning {hz} Hz")
        return round(cur / HZ_PER_MHZ)

    def set_gpu_governor_free(self):
        """Open the DVFS range to the whole table so the governor chooses the frequency again."""
        levels = self._gpu_levels_hz()
        self._set_gpu_range(levels[0], levels[-1])

    # ---- EMC -------------------------------------------------------------------------------------
    def _check_emc(self, mhz):
        if not any(abs(float(mhz) - ok) < 1e-6 for ok in EMC_LOCKABLE_MHZ):
            raise ValueError(f"EMC {mhz} MHz is not lockable; use one of {EMC_LOCKABLE_MHZ}")
        hz = round(float(mhz) * HZ_PER_MHZ)
        lo, hi = self._read_int(EMC + "min_rate"), self._read_int(EMC + "max_rate")
        if not lo <= hz <= hi:
            raise ValueError(f"EMC {mhz} MHz is outside this board's range {lo / 1e6:g}..{hi / 1e6:g} MHz")
        return hz

    def lock_emc(self, mhz):
        """Lock the EMC clock at one lockable point. Returns the rate read back (MHz)."""
        hz = self._check_emc(mhz)
        self._write_checked(EMC + "mrq_rate_locked", 1)
        self._write_checked(EMC + "state", 1)
        self._write_checked(BWMGR_HALT, 1)
        self._write_checked(EMC + "rate", hz)
        got = self._read_int(EMC + "rate")
        if abs(got - hz) > EMC_READBACK_TOL_HZ:
            raise ClockError(f"EMC rate reads {got} Hz after locking {hz} Hz")
        return got / HZ_PER_MHZ

    def unlock_emc(self):
        self._write_checked(BWMGR_HALT, 0)
        self._write_checked(EMC + "mrq_rate_locked", 0)

    # ---- arms and state --------------------------------------------------------------------------
    def apply_arm(self, arm, gpu_mhz, emc_mhz):
        """pinned: GPU level + EMC lock. governor: free GPU + EMC lock. stock: free GPU + EMC unlocked.
        Returns the read-back summary {arm, gpu_mhz (None unless pinned), emc_mhz (None if unlocked), governor_free}."""
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; use one of {ARMS}")
        if arm == "pinned" and gpu_mhz is None:
            raise ValueError("arm 'pinned' needs gpu_mhz")
        if arm != "stock" and emc_mhz is None:
            raise ValueError(f"arm {arm!r} needs emc_mhz")
        if arm == "pinned":
            self._snap_gpu(gpu_mhz)        # validate everything before the first write
        if arm != "stock":
            self._check_emc(emc_mhz)
        gpu_back = emc_back = None
        if arm == "pinned":
            gpu_back = self.set_gpu_level(gpu_mhz)
        else:
            self.set_gpu_governor_free()
        if arm == "stock":
            self.unlock_emc()
        else:
            emc_back = self.lock_emc(emc_mhz)
        return {"arm": arm, "gpu_mhz": gpu_back, "emc_mhz": emc_back, "governor_free": arm != "pinned"}

    def snapshot(self):
        name, mode_id = self.query_power_mode()
        return {
            "power_mode_name": name, "power_mode_id": mode_id, "boot_id": self.helper.read(BOOT_ID),
            "gpu_min_hz": self._read_int(DEVFREQ + "min_freq"), "gpu_max_hz": self._read_int(DEVFREQ + "max_freq"),
            "gpu_cur_hz": self._read_int(DEVFREQ + "cur_freq"),
            "emc_rate_hz": self._read_int(EMC + "rate"), "emc_min_rate_hz": self._read_int(EMC + "min_rate"),
            "emc_max_rate_hz": self._read_int(EMC + "max_rate"),
            "emc_mrq_rate_locked": self._read_int(EMC + "mrq_rate_locked"),
            "emc_state": self._read_int(EMC + "state"), "bwmgr_halt": self._read_int(BWMGR_HALT),
        }

    def close(self):
        self.helper.close()


# ---- CLI ---------------------------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(prog="python -m exp2.board.clocks", description=__doc__.split("\n", 1)[0])
    ap.add_argument("--dry-run", action="store_true", help="use a fake sysfs tree (needs --fake-root)")
    ap.add_argument("--fake-root", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("snapshot")
    sub.add_parser("set-gpu").add_argument("mhz", type=float)
    sub.add_parser("lock-emc").add_argument("mhz", type=float)
    arm = sub.add_parser("apply-arm")
    arm.add_argument("arm")
    arm.add_argument("--gpu", type=float, default=None)
    arm.add_argument("--emc", type=float, default=None)
    return ap


def main(argv=None):
    ap = build_parser()
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    if args.dry_run != (args.fake_root is not None):
        print("error: --dry-run and --fake-root go together", file=sys.stderr)
        return 2
    try:
        if args.dry_run:
            helper = RootHelper.start(mode="dry", fake_root=args.fake_root)
        else:
            helper = RootHelper.start(mode="sudo")
    except ClockError as exc:
        print(f"ClockError: {exc}", file=sys.stderr)
        return 4
    board = Board(helper)
    try:
        if args.cmd == "set-gpu":
            board.set_gpu_level(args.mhz)
        elif args.cmd == "lock-emc":
            board.lock_emc(args.mhz)
        elif args.cmd == "apply-arm":
            board.apply_arm(args.arm, args.gpu, args.emc)
        print(json.dumps(board.snapshot(), indent=2, sort_keys=True))
        return 0
    except ClockError as exc:
        print(f"ClockError: {exc}", file=sys.stderr)
        return 4
    except ValueError as exc:
        print(f"ValueError: {exc}", file=sys.stderr)
        return 2
    finally:
        board.close()


if __name__ == "__main__":
    sys.exit(main())
