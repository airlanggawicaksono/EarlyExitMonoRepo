"""Tests for exp2.board.clocks.Board and its CLI.

Two kinds of helper back the Board:
  * the real RootHelper in dry mode on a make_dry_root tree (integration, checks files on disk);
  * FakeHelper below, an in-memory stand-in with fault injection (stale read-back, ignored writes, ...).
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from exp2.board import clocks
from exp2.board.clocks import Board
from exp2.board.dryroot import make_dry_root
from exp2.board.sysfs import RootHelper
from exp2.contracts import ARMS, BoardControl, ClockError, RootHelperClient

GPU = "/sys/class/devfreq/17000000.gpu/"
EMC = "/sys/kernel/debug/bpmp/debug/clk/emc/"
HALT = "/sys/kernel/debug/bpmp/debug/bwmgr/bwmgr_halt"
BOOT = "/proc/sys/kernel/random/boot_id"
MIN_F, MAX_F, CUR_F = GPU + "min_freq", GPU + "max_freq", GPU + "cur_freq"
RATE, MRQ, STATE = EMC + "rate", EMC + "mrq_rate_locked", EMC + "state"
REPO = Path(clocks.__file__).resolve().parents[2]


# ---- fixtures ------------------------------------------------------------------------------------
@pytest.fixture
def make_board(tmp_path):
    started = []

    def make(**root_kw):
        root = tmp_path / f"root{len(started)}"
        make_dry_root(root, **root_kw)
        helper = RootHelper.start(mode="dry", fake_root=root)
        started.append(helper)
        return Board(helper), root

    yield make
    for helper in started:
        helper.close()


@pytest.fixture
def board_root(make_board):
    return make_board()


def fread(root, path):
    return (root / path.lstrip("/")).read_text().strip()


def fwrite(root, path, text):
    (root / path.lstrip("/")).write_text(text)


def gpu_writes(oplog):
    """('min'|'max', value) for every devfreq range write in the oplog, in order."""
    return [(p.rsplit("/", 1)[1][:3], v) for op, p, v in oplog if op == "write" and p in (MIN_F, MAX_F)]


class FakeHelper:
    """In-memory RootHelper stand-in. Same surface as RootHelper (write/read/run/close/oplog)."""

    def __init__(self, emc_min=204.0, emc_max=3199.0, mode_id=1):
        levels = [m * 1_000_000 for m in (306, 408, 510, 612, 714, 816, 918, 1020)]
        self.files = {
            GPU + "available_frequencies": " ".join(map(str, levels)), MIN_F: str(levels[0]),
            MAX_F: str(levels[-1]), CUR_F: str(levels[-1]), RATE: str(int(emc_max * 1e6)),
            EMC + "min_rate": str(int(emc_min * 1e6)), EMC + "max_rate": str(int(emc_max * 1e6)),
            MRQ: "0", STATE: "1", HALT: "0", BOOT: "boot-A",
        }
        self.modes = {0: "15W", 1: "25W", 2: "MAXN_SUPER"}
        self.mode_id = mode_id
        self.oplog = []
        self.ignore_writes = set()    # write accepted, file keeps its old value, read-back shows the old value
        self.stale_reads = {}         # read(path) returns this whatever the file holds
        self.cur_stuck = False        # cur_freq does not follow min/max (a governor that disagrees)
        self.nvpmodel_set_works = True
        self.nvpmodel_set_rc = 0
        self.closed = False

    def read(self, path):
        if path in self.stale_reads:
            return self.stale_reads[path]
        if path not in self.files:
            raise ClockError(f"no such file {path}")
        return self.files[path]

    def write(self, path, value):
        self.oplog.append(("write", path, str(value)))
        if path not in self.ignore_writes:
            self.files[path] = str(value)
            if path in (MIN_F, MAX_F) and not self.cur_stuck:
                lo, hi, cur = (int(self.files[p]) for p in (MIN_F, MAX_F, CUR_F))
                self.files[CUR_F] = str(min(max(cur, lo), hi))
        return self.files.get(path, "")

    def run(self, argv, timeout_s=30.0):
        self.oplog.append(("run", tuple(argv), None))
        if argv[:2] == ["nvpmodel", "-q"]:
            return 0, f"NV Power Mode: {self.modes[self.mode_id]}\n{self.mode_id}\n"
        if argv[:2] == ["nvpmodel", "-m"]:
            if self.nvpmodel_set_works and self.nvpmodel_set_rc == 0:
                self.mode_id = int(argv[2])
            return self.nvpmodel_set_rc, "" if self.nvpmodel_set_rc == 0 else "NVPM ERROR: nope\n"
        if argv[0] == "jetson_clocks":
            return 0, ""
        raise AssertionError(f"unexpected run {argv}")

    def close(self):
        self.closed = True


# ---- power mode ----------------------------------------------------------------------------------
@pytest.mark.parametrize("text, expected", [
    ("NV Power Mode: MAXN_SUPER\n2\n", ("MAXN_SUPER", 2)),
    ("NV Power Mode: 15W\n0\n", ("15W", 0)),
    ("\nNV Power Mode:\n25W\n1\n", ("25W", 1)),
])
def test_parse_power_mode(text, expected):
    assert clocks.parse_power_mode(text) == expected


@pytest.mark.parametrize("text", ["", "garbage\n", "NV Power Mode: MAXN_SUPER\n", "NV Power Mode: X\nnot-a-number\n"])
def test_parse_power_mode_rejects_unparseable_output(text):
    with pytest.raises(ClockError, match="cannot parse"):
        clocks.parse_power_mode(text)


def test_query_power_mode_on_dry_root(board_root):
    board, root = board_root
    fwrite(root, "/etc/nvpmodel_state", "2\n")
    assert board.query_power_mode() == ("MAXN_SUPER", 2)


def test_ensure_power_mode_already_set_does_not_switch():
    fake = FakeHelper(mode_id=2)
    assert Board(fake).ensure_power_mode(2) is None
    assert fake.oplog == [("run", ("nvpmodel", "-q"), None)]


def test_ensure_power_mode_switches_then_verifies():
    fake = FakeHelper(mode_id=0)
    assert Board(fake).ensure_power_mode(2) is None
    assert fake.oplog == [("run", ("nvpmodel", "-q"), None), ("run", ("nvpmodel", "-m", "2"), None),
                          ("run", ("nvpmodel", "-q"), None)]


def test_ensure_power_mode_default_is_the_contract_mode():
    fake = FakeHelper(mode_id=0)
    Board(fake).ensure_power_mode()
    assert fake.mode_id == 2


def test_ensure_power_mode_mismatch_after_switch_is_clockerror():
    fake = FakeHelper(mode_id=0)
    fake.nvpmodel_set_works = False       # rc 0 but the mode did not change
    with pytest.raises(ClockError, match=r"power mode is 15W \(0\)"):
        Board(fake).ensure_power_mode(2)


def test_ensure_power_mode_switch_command_failure_is_clockerror():
    fake = FakeHelper(mode_id=0)
    fake.nvpmodel_set_rc = 255
    with pytest.raises(ClockError, match="rc=255"):
        Board(fake).ensure_power_mode(2)


def test_ensure_power_mode_on_dry_root_end_to_end(board_root):
    board, root = board_root
    fwrite(root, "/etc/nvpmodel_state", "0\n")
    board.ensure_power_mode(2)
    assert board.query_power_mode() == ("MAXN_SUPER", 2)


# ---- GPU level -----------------------------------------------------------------------------------
def test_gpu_levels_in_mhz_ascending(make_board):
    board, _ = make_board(gpu_levels_mhz=(612, 306, 1020, 408))
    assert board.gpu_levels() == [306, 408, 612, 1020]


def test_set_gpu_level_pins_min_max_and_cur_on_disk(board_root):
    board, root = board_root
    board.set_gpu_level(612)
    assert fread(root, MIN_F) == fread(root, MAX_F) == fread(root, CUR_F) == "612000000"


def test_set_gpu_level_returns_the_read_back_cur_freq_in_mhz(board_root):
    board, _ = board_root
    got = board.set_gpu_level(612)
    assert got == 612 and isinstance(got, int)


def test_set_gpu_level_snaps_within_one_mhz(board_root):
    board, root = board_root
    board.set_gpu_level(612.6)
    assert fread(root, MAX_F) == "612000000"


def test_set_gpu_level_write_order_up_is_max_then_min_down_is_min_then_max(board_root):
    board, _ = board_root
    helper = board.helper
    board.set_gpu_level(612)          # from the open range 306..1020
    assert gpu_writes(helper.oplog) == [("max", "612000000"), ("min", "612000000")]
    del helper.oplog[:]
    board.set_gpu_level(1020)         # up
    assert gpu_writes(helper.oplog) == [("max", "1020000000"), ("min", "1020000000")]
    del helper.oplog[:]
    board.set_gpu_level(408)          # down
    assert gpu_writes(helper.oplog) == [("min", "408000000"), ("max", "408000000")]
    del helper.oplog[:]
    board.set_gpu_level(510)          # up again, from a pin
    assert gpu_writes(helper.oplog) == [("max", "510000000"), ("min", "510000000")]


@pytest.mark.parametrize("bad", [0, 304, 700, 1022, 1100, -612])
def test_set_gpu_level_outside_table_is_valueerror_with_empty_oplog(board_root, bad):
    board, _ = board_root
    with pytest.raises(ValueError, match="not in the table"):
        board.set_gpu_level(bad)
    assert board.helper.oplog == []


def test_set_gpu_level_stale_min_readback_is_clockerror():
    fake = FakeHelper()
    fake.stale_reads[MIN_F] = str(306_000_000)       # the min write "did not take"
    with pytest.raises(ClockError, match="GPU range read back"):
        Board(fake).set_gpu_level(612)


def test_set_gpu_level_stale_max_readback_is_clockerror():
    fake = FakeHelper()
    fake.stale_reads[MAX_F] = str(1_020_000_000)
    with pytest.raises(ClockError, match="GPU range read back"):
        Board(fake).set_gpu_level(612)


def test_set_gpu_level_ignored_write_is_clockerror_at_the_write_readback():
    fake = FakeHelper()
    fake.ignore_writes.add(MAX_F)
    with pytest.raises(ClockError, match="read back"):
        Board(fake).set_gpu_level(612)


def test_set_gpu_level_cur_freq_not_following_is_clockerror():
    fake = FakeHelper()
    fake.cur_stuck = True                            # cur_freq stays at 1020 MHz
    with pytest.raises(ClockError, match="cur_freq is 1020000000"):
        Board(fake).set_gpu_level(612)


def test_set_gpu_level_reads_cur_freq_about_50_ms_after_the_writes(monkeypatch):
    fake = FakeHelper()
    sleeps = []
    monkeypatch.setattr(clocks.time, "sleep", lambda s: sleeps.append((s, len(fake.oplog))))
    Board(fake).set_gpu_level(612)
    assert sleeps == [(0.05, 2)]


def test_set_gpu_governor_free_opens_the_whole_table(board_root):
    board, root = board_root
    board.set_gpu_level(612)
    del board.helper.oplog[:]
    board.set_gpu_governor_free()
    assert (fread(root, MIN_F), fread(root, MAX_F)) == ("306000000", "1020000000")
    assert gpu_writes(board.helper.oplog) == [("min", "306000000"), ("max", "1020000000")]


def test_set_gpu_governor_free_from_the_bottom_pin_writes_max_first(board_root):
    board, root = board_root
    board.set_gpu_level(306)
    del board.helper.oplog[:]
    board.set_gpu_governor_free()
    assert gpu_writes(board.helper.oplog) == [("max", "1020000000"), ("min", "306000000")]
    assert (fread(root, MIN_F), fread(root, MAX_F)) == ("306000000", "1020000000")


def test_set_gpu_governor_free_stale_readback_is_clockerror():
    fake = FakeHelper()
    fake.stale_reads[MAX_F] = str(612_000_000)
    with pytest.raises(ClockError, match="GPU range read back"):
        Board(fake).set_gpu_governor_free()


# ---- EMC -----------------------------------------------------------------------------------------
def test_lock_emc_write_order_is_mrq_state_bwmgr_rate(board_root):
    board, root = board_root
    board.lock_emc(2133.0)
    assert board.helper.oplog == [("write", MRQ, "1"), ("write", STATE, "1"), ("write", HALT, "1"),
                                  ("write", RATE, "2133000000")]
    assert (fread(root, MRQ), fread(root, STATE), fread(root, HALT), fread(root, RATE)) == \
        ("1", "1", "1", "2133000000")


@pytest.mark.parametrize("mhz, rate", [(204.0, "204000000"), (665.6, "665600000"), (2133, "2133000000"),
                                       (3199.0, "3199000000")])
def test_lock_emc_accepts_every_lockable_point(board_root, mhz, rate):
    board, root = board_root
    board.lock_emc(mhz)
    assert fread(root, RATE) == rate


def test_lock_emc_returns_the_read_back_rate_in_mhz(board_root):
    board, _ = board_root
    assert board.lock_emc(665.6) == pytest.approx(665.6)
    assert board.lock_emc(3199) == pytest.approx(3199.0)


def test_lock_emc_below_this_boards_min_rate_is_valueerror_before_any_write(make_board):
    board, _ = make_board(emc_min=800.0)
    with pytest.raises(ValueError, match="outside this board's range"):
        board.lock_emc(665.6)
    assert board.helper.oplog == []


def test_lock_emc_above_this_boards_max_rate_is_valueerror_before_any_write(make_board):
    board, _ = make_board(emc_max=2133.0)
    with pytest.raises(ValueError, match="outside this board's range"):
        board.lock_emc(3199.0)
    assert board.helper.oplog == []


@pytest.mark.parametrize("bad", [1000, 0, 3200.0, 665.0, 2132.9])
def test_lock_emc_non_lockable_point_is_valueerror_before_any_write(board_root, bad):
    board, _ = board_root
    with pytest.raises(ValueError, match="not lockable"):
        board.lock_emc(bad)
    assert board.helper.oplog == []


def test_lock_emc_stale_rate_readback_is_clockerror():
    fake = FakeHelper()
    fake.stale_reads[RATE] = str(3_199_000_000)      # write read-back is honest, the later read is stale
    with pytest.raises(ClockError, match="EMC rate reads 3199000000"):
        Board(fake).lock_emc(665.6)


def test_lock_emc_rate_write_that_does_not_stick_is_clockerror():
    fake = FakeHelper()
    fake.ignore_writes.add(RATE)
    with pytest.raises(ClockError, match="read back"):
        Board(fake).lock_emc(2133.0)


def test_lock_emc_flag_that_does_not_stick_is_clockerror():
    fake = FakeHelper()
    fake.ignore_writes.add(HALT)
    with pytest.raises(ClockError, match="bwmgr_halt"):
        Board(fake).lock_emc(2133.0)


def test_unlock_emc_clears_the_flags(board_root):
    board, root = board_root
    board.lock_emc(2133.0)
    del board.helper.oplog[:]
    board.unlock_emc()
    assert (fread(root, MRQ), fread(root, HALT)) == ("0", "0")
    assert board.helper.oplog == [("write", HALT, "0"), ("write", MRQ, "0")]


# ---- jetson_clocks -------------------------------------------------------------------------------
def test_run_jetson_clocks_pins_gpu_to_the_top_level(board_root):
    board, root = board_root
    board.set_gpu_level(408)
    board.run_jetson_clocks()
    assert fread(root, MIN_F) == fread(root, MAX_F) == "1020000000"


def test_run_jetson_clocks_failure_is_clockerror():
    fake = FakeHelper()
    fake.run = lambda argv, timeout_s=30.0: (1, "jetson_clocks: boom\n")
    with pytest.raises(ClockError, match="rc=1"):
        Board(fake).run_jetson_clocks()


def test_store_records_boot_id_and_restore_puts_the_state_back(board_root):
    board, root = board_root
    board.set_gpu_level(714)
    board.store_jetson_clocks("/tmp/exp2_jc_store")
    assert fread(root, "/tmp/exp2_jc_store.boot_id") == "dry-boot"
    board.run_jetson_clocks()
    assert fread(root, MIN_F) == "1020000000"
    board.restore_jetson_clocks("/tmp/exp2_jc_store")
    assert (fread(root, MIN_F), fread(root, MAX_F)) == ("714000000", "714000000")


def test_restore_with_a_different_boot_id_is_refused_and_changes_nothing(board_root):
    board, root = board_root
    board.set_gpu_level(714)
    board.store_jetson_clocks("/tmp/exp2_jc_store")
    board.run_jetson_clocks()
    fwrite(root, BOOT, "another-boot\n")             # the board rebooted since the store
    del board.helper.oplog[:]
    with pytest.raises(ClockError, match="refusing to restore"):
        board.restore_jetson_clocks("/tmp/exp2_jc_store")
    assert board.helper.oplog == []
    assert fread(root, MIN_F) == "1020000000"


def test_restore_without_a_stored_boot_id_is_clockerror(board_root):
    board, _ = board_root
    with pytest.raises(ClockError):
        board.restore_jetson_clocks("/tmp/never_stored")
    assert board.helper.oplog == []


# ---- arms ----------------------------------------------------------------------------------------
def test_apply_arm_pinned_sets_gpu_level_and_locks_emc(board_root):
    board, root = board_root
    board.apply_arm("pinned", 612, 2133.0)
    assert fread(root, MIN_F) == fread(root, MAX_F) == fread(root, CUR_F) == "612000000"
    assert (fread(root, RATE), fread(root, MRQ), fread(root, HALT)) == ("2133000000", "1", "1")


def test_apply_arm_governor_frees_gpu_and_locks_emc(board_root):
    board, root = board_root
    board.apply_arm("pinned", 612, 2133.0)
    board.apply_arm("governor", None, 3199.0)
    assert (fread(root, MIN_F), fread(root, MAX_F)) == ("306000000", "1020000000")
    assert (fread(root, RATE), fread(root, MRQ), fread(root, HALT)) == ("3199000000", "1", "1")


def test_apply_arm_stock_frees_gpu_and_unlocks_emc(board_root):
    board, root = board_root
    board.apply_arm("pinned", 612, 2133.0)
    board.apply_arm("stock", None, None)
    assert (fread(root, MIN_F), fread(root, MAX_F)) == ("306000000", "1020000000")
    assert (fread(root, MRQ), fread(root, HALT)) == ("0", "0")


def test_apply_arm_stock_ignores_unneeded_arguments(board_root):
    board, root = board_root
    board.apply_arm("stock", 612, 1000)
    assert (fread(root, MRQ), fread(root, HALT)) == ("0", "0")


def test_apply_arm_returns_the_read_back_summary(board_root):
    board, _ = board_root
    assert board.apply_arm("pinned", 816, 3199.0) == {"arm": "pinned", "gpu_mhz": 816, "emc_mhz": 3199.0,
                                                      "governor_free": False}
    assert board.apply_arm("governor", None, 2133.0) == {"arm": "governor", "gpu_mhz": None, "emc_mhz": 2133.0,
                                                         "governor_free": True}
    assert board.apply_arm("stock", None, None) == {"arm": "stock", "gpu_mhz": None, "emc_mhz": None,
                                                    "governor_free": True}


@pytest.mark.parametrize("arm, gpu, emc", [
    ("pinned", 700, 3199.0),         # GPU level not in the table
    ("pinned", 612, 1000),           # EMC point not lockable: nothing may be written, not even the GPU pin
    ("pinned", None, 3199.0),
    ("pinned", 612, None),
    ("governor", None, 1000),
    ("governor", None, None),
    ("overclock", 612, 3199.0),
])
def test_apply_arm_invalid_request_is_valueerror_with_empty_oplog(board_root, arm, gpu, emc):
    board, _ = board_root
    with pytest.raises(ValueError):
        board.apply_arm(arm, gpu, emc)
    assert board.helper.oplog == []


def test_apply_arm_knows_exactly_the_contract_arms():
    assert ARMS == ("pinned", "governor", "stock")


def test_apply_arm_clockerror_propagates(board_root):
    board, root = board_root
    (root / RATE.lstrip("/")).unlink()
    (root / RATE.lstrip("/")).mkdir()                # writing to a directory fails in the helper
    with pytest.raises(ClockError):
        board.apply_arm("governor", None, 3199.0)


def test_board_and_root_helper_satisfy_the_contract_protocols(board_root):
    board, _ = board_root
    assert isinstance(board, BoardControl)
    assert isinstance(board.helper, RootHelperClient)


# ---- snapshot / close ----------------------------------------------------------------------------
def test_snapshot_reports_power_mode_boot_gpu_and_emc(make_board):
    board, root = make_board(boot_id="boot-xyz")
    fwrite(root, "/etc/nvpmodel_state", "2\n")
    snap = board.snapshot()
    assert snap["power_mode_name"] == "MAXN_SUPER" and snap["power_mode_id"] == 2
    assert snap["boot_id"] == "boot-xyz"
    assert (snap["gpu_min_hz"], snap["gpu_max_hz"], snap["gpu_cur_hz"]) == (306_000_000, 1_020_000_000, 1_020_000_000)
    assert (snap["emc_min_rate_hz"], snap["emc_max_rate_hz"]) == (204_000_000, 3_199_000_000)
    assert snap["emc_mrq_rate_locked"] == 0 and snap["bwmgr_halt"] == 0 and snap["emc_state"] == 1
    json.dumps(snap)


def test_snapshot_follows_the_board_state(board_root):
    board, _ = board_root
    board.lock_emc(665.6)
    snap = board.snapshot()
    assert snap["emc_rate_hz"] == 665_600_000 and snap["emc_mrq_rate_locked"] == 1 and snap["bwmgr_halt"] == 1


def test_close_closes_the_helper():
    fake = FakeHelper()
    Board(fake).close()
    assert fake.closed


# ---- CLI -----------------------------------------------------------------------------------------
def cli(root, *args):
    return clocks.main(["--dry-run", "--fake-root", str(root), *args])


def test_cli_snapshot_prints_json_and_exits_0(make_board, capsys):
    _, root = make_board()
    assert cli(root, "snapshot") == 0
    snap = json.loads(capsys.readouterr().out)
    assert snap["gpu_max_hz"] == 1_020_000_000 and snap["boot_id"] == "dry-boot"


def test_cli_set_gpu_changes_the_board(make_board, capsys):
    _, root = make_board()
    assert cli(root, "set-gpu", "612") == 0
    assert fread(root, CUR_F) == "612000000"
    assert json.loads(capsys.readouterr().out)["gpu_cur_hz"] == 612_000_000


def test_cli_lock_emc_changes_the_board(make_board):
    _, root = make_board()
    assert cli(root, "lock-emc", "2133") == 0
    assert fread(root, RATE) == "2133000000"


def test_cli_apply_arm_changes_the_board(make_board):
    _, root = make_board()
    assert cli(root, "apply-arm", "pinned", "--gpu", "816", "--emc", "3199") == 0
    assert (fread(root, MIN_F), fread(root, RATE)) == ("816000000", "3199000000")
    assert cli(root, "apply-arm", "stock") == 0
    assert fread(root, MRQ) == "0"


@pytest.mark.parametrize("args", [
    ("set-gpu", "700"),
    ("lock-emc", "1000"),
    ("apply-arm", "overclock"),
    ("apply-arm", "pinned", "--gpu", "612"),
])
def test_cli_exits_2_on_valueerror(make_board, capsys, args):
    _, root = make_board()
    assert cli(root, *args) == 2
    assert "ValueError" in capsys.readouterr().err


def test_cli_exits_2_when_lock_is_outside_the_boards_range(make_board):
    _, root = make_board(emc_min=800.0)
    assert cli(root, "lock-emc", "665.6") == 2


def test_cli_exits_4_on_clockerror(make_board, capsys):
    _, root = make_board()
    (root / RATE.lstrip("/")).unlink()
    (root / RATE.lstrip("/")).mkdir()
    assert cli(root, "lock-emc", "3199") == 4
    assert "ClockError" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [
    [], ["bogus"], ["set-gpu"], ["--dry-run", "snapshot"], ["--fake-root", "x", "snapshot"],
])
def test_cli_usage_errors_exit_2(argv, capsys):
    assert clocks.main(argv) == 2


def test_cli_as_a_module_returns_the_exit_codes(make_board):
    _, root = make_board()
    base = [sys.executable, "-B", "-m", "exp2.board.clocks", "--dry-run", "--fake-root", str(root)]
    ok = subprocess.run(base + ["set-gpu", "408"], cwd=REPO, capture_output=True, text=True, timeout=60)
    assert ok.returncode == 0 and fread(root, CUR_F) == "408000000"
    bad = subprocess.run(base + ["set-gpu", "700"], cwd=REPO, capture_output=True, text=True, timeout=60)
    assert bad.returncode == 2
