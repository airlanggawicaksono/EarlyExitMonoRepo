"""Tests for exp2.board.telemetry (slice B2, DoD D11, contract K10).

The sysfs tree comes from exp2.board.dryroot.make_dry_root: VDD_IN sits at hwmon3 / channel 2, and the
tests overwrite every channel with distinct values so a hard-coded hwmon1 / in1 reader cannot pass.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from exp2 import contracts
from exp2.board import telemetry
from exp2.board.dryroot import make_dry_root

REPO = Path(__file__).resolve().parents[2]
POWER_HEADER = ["t_mono_ns", "t_wall_ns", "vdd_in_mv", "vdd_in_ma", "vdd_in_mw"]
CLOCKS_FIXED = [
    "t_mono_ns", "t_wall_ns", "gpu_cur_hz", "gpu_min_hz", "gpu_max_hz", "gpu_load_permille",
    "emc_mhz", "mem_available_kb", "cpu0_cur_khz",
]
META_KEYS = {
    "schema", "vdd_in", "target_hz", "achieved_hz", "clocks_target_hz", "devfreq", "emc_source",
    "thermal_zones", "boot_id", "t_mono_start_ns", "t_wall_start_ns", "n_rows",
}
VDD_MV, VDD_MA = 5072, 1203          # the K10 example: 5072 * 1203 / 1000 = 6101.616
TEGRASTATS_LINE = (
    "RAM 2106/7620MB (lfb 4x4MB) SWAP 0/3810MB CPU [1%@1510,0%@1510,off,off] "
    "EMC_FREQ 12%@3199 GR3D_FREQ 0%@[306] VDD_IN 5072mW/5072mW"
)
FAKE_TEGRASTATS = (
    "import sys, time\n"
    "hb, line = sys.argv[1], sys.argv[2]\n"
    "n = 0\n"
    "while True:\n"
    "    n += 1\n"
    "    print(line, flush=True)\n"
    "    with open(hb, 'w') as f:\n"
    "        f.write(str(n))\n"
    "    time.sleep(0.05)\n"
)


# ---------------------------------------------------------------- helpers
def _make_root(path):
    root = make_dry_root(path)
    hw = root / "sys/class/hwmon/hwmon3"
    for name, value in (("in1_input", 1111), ("curr1_input", 2222), ("in3_input", 3333), ("curr3_input", 4444),
                        ("in2_input", VDD_MV), ("curr2_input", VDD_MA)):
        (hw / name).write_text(f"{value}\n")
    return root


@pytest.fixture
def root(tmp_path):
    return _make_root(tmp_path / "root")


def _read_csv(path):
    lines = Path(path).read_text().splitlines()
    header = lines[0].split(",")
    return header, [dict(zip(header, line.split(","))) for line in lines[1:]]


def _meta(out):
    return json.loads((Path(out) / "meta.json").read_text())


def _board_path(root, path):
    return "/" + Path(path).relative_to(root).as_posix()


class FakeClock:
    """Virtual clock: wait() advances time instantly and optionally reports to a probe."""

    def __init__(self, mono0=5_000_000, wall0=1_760_951_700_000_000_000):
        self.now = mono0
        self.start = mono0
        self.wall_offset = wall0 - mono0
        self.probe = None

    def mono_ns(self):
        return self.now

    def wall_ns(self):
        return self.now + self.wall_offset

    def wait(self, event, seconds):
        if self.probe is not None:
            self.probe((self.now - self.start) / 1e9)
        self.now += int(round(seconds * 1e9))


def _run_fake(root, out, duration_s, *, hz=20.0, clocks_hz=1.0, clock=None, tegrastats_argv=None):
    clock = clock or FakeClock()
    tele = telemetry.Telemetry(out, root=root, hz=hz, clocks_hz=clocks_hz, duration_s=duration_s,
                               tegrastats_argv=tegrastats_argv, clock=clock)
    return tele.run()


class _IdleStdin:
    """A pipe whose write end stays open, so the stdin-EOF watcher never fires."""

    def __enter__(self):
        self.r, self.w = os.pipe()
        return self.r

    def __exit__(self, *exc):
        os.close(self.w)
        os.close(self.r)


def _main(argv, **kw):
    with _IdleStdin() as fd:
        return telemetry.main(argv, stdin_fd=fd, **kw)


# ---------------------------------------------------------------- real-time run, shared (read-only)
@pytest.fixture(scope="module")
def run_2s(tmp_path_factory):
    base = tmp_path_factory.mktemp("rt")
    root = _make_root(base / "root")
    out = base / "out"
    rc = _main(["--out", str(out), "--hz", "20", "--clocks-hz", "1", "--duration", "2.0",
                "--sysfs-root", str(root), "--no-tegrastats"])
    power_header, power = _read_csv(out / "power.csv")
    clocks_header, clocks = _read_csv(out / "clocks.csv")
    return {"rc": rc, "root": root, "out": out, "power_header": power_header, "power": power,
            "clocks_header": clocks_header, "clocks": clocks, "meta": _meta(out)}


def test_20hz_for_2s_gives_at_least_32_rows(run_2s):
    assert run_2s["rc"] == 0
    assert len(run_2s["power"]) >= 32          # 80 % of the 40 ideal rows
    assert len(run_2s["power"]) <= 41          # and not a burst of extra rows


def test_power_header_is_exact(run_2s):
    assert run_2s["power_header"] == POWER_HEADER


def test_t_mono_strictly_increasing(run_2s):
    for name in ("power", "clocks"):
        t = [int(r["t_mono_ns"]) for r in run_2s[name]]
        assert all(b > a for a, b in zip(t, t[1:])), name


def test_t_wall_tracks_wall_clock(run_2s):
    now_ns = time.time_ns()
    walls = [int(r["t_wall_ns"]) for r in run_2s["power"]]
    assert all(abs(now_ns - w) < 600 * 10**9 for w in walls)
    assert all(b >= a for a, b in zip(walls, walls[1:]))


def test_power_values_come_from_the_vdd_in_channel_not_hwmon1(run_2s):
    for r in run_2s["power"]:
        assert int(r["vdd_in_mv"]) == VDD_MV            # in2_input, not in1_input (1111)
        assert int(r["vdd_in_ma"]) == VDD_MA            # curr2_input, not curr1_input (2222)


def test_mw_equals_mv_times_ma_over_1000_on_every_row(run_2s):
    for r in run_2s["power"]:
        assert float(r["vdd_in_mw"]) == pytest.approx(int(r["vdd_in_mv"]) * int(r["vdd_in_ma"]) / 1000, abs=1e-9)
    assert run_2s["power"][0]["vdd_in_mw"] == "6101.616"


def test_meta_contents(run_2s):
    meta, root = run_2s["meta"], run_2s["root"]
    hw = root / "sys/class/hwmon/hwmon3"
    assert set(meta) == META_KEYS
    assert meta["schema"] == "exp2.telemetry_meta/1"
    assert meta["vdd_in"] == {
        "hwmon": "/sys/class/hwmon/hwmon3", "channel": 2, "label": "VDD_IN",
        "update_interval_ms": int((hw / "update_interval").read_text()),
        "samples": int((hw / "samples").read_text()),
    }
    assert meta["target_hz"] == 20
    assert meta["clocks_target_hz"] == 1
    assert 10.0 < meta["achieved_hz"] <= 21.0
    assert meta["n_rows"] == len(run_2s["power"])
    assert meta["emc_source"] is None
    assert meta["boot_id"] == "dry-boot"
    assert meta["devfreq"].endswith("17000000.gpu")
    assert isinstance(meta["t_mono_start_ns"], int) and isinstance(meta["t_wall_start_ns"], int)
    assert int(run_2s["power"][0]["t_mono_ns"]) >= meta["t_mono_start_ns"]


def test_outputs_satisfy_the_contract_validators(run_2s):
    contracts.validate_telemetry_meta(run_2s["meta"])
    contracts.validate_csv_header("power", run_2s["power_header"])
    contracts.validate_csv_header("clocks", run_2s["clocks_header"])


def test_clocks_csv_fixed_columns_and_values(run_2s):
    header = run_2s["clocks_header"]
    assert header[:len(CLOCKS_FIXED)] == CLOCKS_FIXED
    assert len(run_2s["clocks"]) >= 1                      # 1 Hz for 2 s gives 2 rows
    row = run_2s["clocks"][0]
    dev = run_2s["root"] / "sys/class/devfreq/17000000.gpu"
    assert row["gpu_cur_hz"] == (dev / "cur_freq").read_text().strip()
    assert row["gpu_min_hz"] == (dev / "min_freq").read_text().strip()
    assert row["gpu_max_hz"] == (dev / "max_freq").read_text().strip()
    assert row["emc_mhz"] == ""
    mem_line = next(x for x in (run_2s["root"] / "proc/meminfo").read_text().splitlines()
                    if x.startswith("MemAvailable:"))
    assert row["mem_available_kb"] == mem_line.split()[1]
    cpu0 = run_2s["root"] / "sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
    assert row["cpu0_cur_khz"] == cpu0.read_text().strip()
    assert row["gpu_load_permille"] != ""


def test_rows_are_complete_and_rectangular(run_2s):
    for name, header in (("power", run_2s["power_header"]), ("clocks", run_2s["clocks_header"])):
        for line in (run_2s["out"] / f"{name}.csv").read_text().splitlines():
            assert len(line.split(",")) == len(header)


# ---------------------------------------------------------------- VDD_IN discovery
def test_vdd_in_channel_is_found_by_label_not_by_position(tmp_path):
    root = _make_root(tmp_path / "r")
    hw = root / "sys/class/hwmon/hwmon3"
    # swap the labels: VDD_IN now lives on channel 3, channel 2 is VDD_SOC
    (hw / "in2_label").write_text("VDD_SOC\n")
    (hw / "in3_label").write_text("VDD_IN\n")
    n = _run_fake(root, tmp_path / "o", 0.5)
    _, rows = _read_csv(tmp_path / "o" / "power.csv")
    assert n == len(rows) > 0
    assert {r["vdd_in_mv"] for r in rows} == {"3333"}
    assert {r["vdd_in_ma"] for r in rows} == {"4444"}
    assert _meta(tmp_path / "o")["vdd_in"]["channel"] == 3


def test_decoy_hwmon_with_other_name_is_ignored(tmp_path):
    root = _make_root(tmp_path / "r")
    decoy = root / "sys/class/hwmon/hwmon1"
    decoy.mkdir(parents=True, exist_ok=True)
    for name, value in (("name", "nvme"), ("in1_label", "VDD_IN"), ("in1_input", "7"), ("curr1_input", "9")):
        (decoy / name).write_text(f"{value}\n")
    _run_fake(root, tmp_path / "o", 0.2)
    _, rows = _read_csv(tmp_path / "o" / "power.csv")
    assert {r["vdd_in_mv"] for r in rows} == {str(VDD_MV)}
    assert _meta(tmp_path / "o")["vdd_in"]["hwmon"] == "/sys/class/hwmon/hwmon3"


def test_i2c_driver_fallback_is_found(tmp_path):
    root = _make_root(tmp_path / "r")
    target = root / "sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3"
    target.parent.mkdir(parents=True)
    shutil.move(str(root / "sys/class/hwmon/hwmon3"), str(target))
    assert not (root / "sys/class/hwmon/hwmon3").exists()
    _run_fake(root, tmp_path / "o", 0.2)
    _, rows = _read_csv(tmp_path / "o" / "power.csv")
    assert len(rows) > 0
    assert {r["vdd_in_mv"] for r in rows} == {str(VDD_MV)}
    assert _meta(tmp_path / "o")["vdd_in"]["hwmon"] == _board_path(root, target)


def test_missing_vdd_in_label_exits_nonzero_with_clear_message(tmp_path, capsys):
    root = _make_root(tmp_path / "r")
    (root / "sys/class/hwmon/hwmon3/in2_label").write_text("VDD_SOC\n")
    rc = _main(["--out", str(tmp_path / "o"), "--duration", "0.2", "--sysfs-root", str(root), "--no-tegrastats"])
    assert rc != 0
    err = capsys.readouterr().err
    assert "VDD_IN" in err
    assert "ina3221" in err


def test_no_ina3221_at_all_exits_nonzero(tmp_path, capsys):
    root = _make_root(tmp_path / "r")
    shutil.rmtree(root / "sys/class/hwmon/hwmon3")
    rc = _main(["--out", str(tmp_path / "o"), "--duration", "0.2", "--sysfs-root", str(root), "--no-tegrastats"])
    assert rc != 0
    assert "VDD_IN" in capsys.readouterr().err


def test_vdd_in_failure_does_not_leave_partial_power_csv(tmp_path):
    root = _make_root(tmp_path / "r")
    (root / "sys/class/hwmon/hwmon3/in2_label").write_text("VDD_SOC\n")
    _main(["--out", str(tmp_path / "o"), "--duration", "0.2", "--sysfs-root", str(root), "--no-tegrastats"])
    assert not (tmp_path / "o" / "power.csv").exists()


# ---------------------------------------------------------------- thermal column names
def _set_zones(root, types):
    base = root / "sys/class/thermal"
    for p in base.glob("thermal_zone*"):
        shutil.rmtree(p)
    base.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(types):
        z = base / f"thermal_zone{i}"
        z.mkdir()
        (z / "type").write_text(f"{t}\n")
        (z / "temp").write_text(f"{40000 + i * 1000}\n")


def test_temp_columns_are_sanitized_and_sorted(tmp_path):
    root = _make_root(tmp_path / "r")
    _set_zones(root, ["tj-thermal", "CPU-therm", "soc0.thermal", "A--B x"])
    _run_fake(root, tmp_path / "o", 0.2)
    header, rows = _read_csv(tmp_path / "o" / "clocks.csv")
    temps = header[len(CLOCKS_FIXED):]
    assert temps == ["temp_a__b_x_mc", "temp_cpu_therm_mc", "temp_soc0_thermal_mc", "temp_tj_thermal_mc"]
    # values follow their own zone, not the column position
    assert rows[0]["temp_tj_thermal_mc"] == "40000"
    assert rows[0]["temp_cpu_therm_mc"] == "41000"
    assert rows[0]["temp_soc0_thermal_mc"] == "42000"
    assert rows[0]["temp_a__b_x_mc"] == "43000"
    assert _meta(tmp_path / "o")["thermal_zones"] == ["A--B x", "CPU-therm", "soc0.thermal", "tj-thermal"]


def test_every_temp_column_matches_the_clean_pattern(run_2s):
    import re
    for col in run_2s["clocks_header"][len(CLOCKS_FIXED):]:
        assert re.fullmatch(r"temp_[a-z0-9_]+_mc", col), col
    assert run_2s["clocks_header"][len(CLOCKS_FIXED):] == sorted(run_2s["clocks_header"][len(CLOCKS_FIXED):])


def test_duplicate_zone_types_get_distinct_columns(tmp_path):
    root = _make_root(tmp_path / "r")
    _set_zones(root, ["cpu-thermal", "cpu-thermal"])
    _run_fake(root, tmp_path / "o", 0.2)
    header, rows = _read_csv(tmp_path / "o" / "clocks.csv")
    temps = header[len(CLOCKS_FIXED):]
    assert len(temps) == len(set(temps)) == 2
    assert {rows[0][c] for c in temps} == {"40000", "41000"}


def test_unreadable_temp_becomes_an_empty_cell(tmp_path):
    root = _make_root(tmp_path / "r")
    _set_zones(root, ["a-thermal", "b-thermal"])
    (root / "sys/class/thermal/thermal_zone0/temp").unlink()
    _run_fake(root, tmp_path / "o", 0.2)
    _, rows = _read_csv(tmp_path / "o" / "clocks.csv")
    assert rows[0]["temp_a_thermal_mc"] == ""
    assert rows[0]["temp_b_thermal_mc"] == "41000"


# ---------------------------------------------------------------- flush, stop, durability
def test_flush_happens_every_5_seconds_with_injected_clock(tmp_path):
    root = _make_root(tmp_path / "r")
    out = tmp_path / "o"
    clock = FakeClock()
    seen = []

    def probe(elapsed_s):
        seen.append((elapsed_s, len((out / "power.csv").read_text().splitlines())))

    clock.probe = probe
    n = _run_fake(root, out, 12.0, clock=clock)
    assert n == 240
    before = [lines for t, lines in seen if t < 5.0]
    assert before and max(before) <= 1                      # header at most: nothing flushed early
    at_5s = next(lines for t, lines in seen if abs(t - 5.0) < 1e-6)
    assert at_5s >= 101                                     # header + rows t=0..5.0 s
    at_995 = next(lines for t, lines in seen if abs(t - 9.95) < 1e-6)
    assert at_995 == at_5s                                  # no flush between 5 s and 10 s
    at_10s = next(lines for t, lines in seen if abs(t - 10.0) < 1e-6)
    assert at_10s >= at_5s + 100                            # second flush at 10 s
    assert len((out / "power.csv").read_text().splitlines()) == 1 + 240


def test_a_stall_skips_missed_slots_instead_of_bursting(tmp_path):
    root = _make_root(tmp_path / "r")
    clock = FakeClock()
    state = {"done": False}

    def stall(elapsed_s):
        if elapsed_s >= 1.0 and not state["done"]:
            clock.now += 500_000_000                        # the sampler was descheduled for 0.5 s
            state["done"] = True

    clock.probe = stall
    n = _run_fake(root, tmp_path / "o", 3.0, clock=clock)
    _, rows = _read_csv(tmp_path / "o" / "power.csv")
    t = [int(r["t_mono_ns"]) for r in rows]
    assert n == len(rows) == 50                             # 21 rows to t=1.0 s, then the 1.55 .. 2.95 s grid
    assert all(b - a >= 50_000_000 for a, b in zip(t, t[1:]))   # strictly increasing, never closer than a period
    assert max(b - a for a, b in zip(t, t[1:])) == 550_000_000  # the stall shows as one gap, not as extra rows


def test_stop_before_first_flush_loses_no_rows(tmp_path):
    root = _make_root(tmp_path / "r")
    n = _run_fake(root, tmp_path / "o", 3.0)
    lines = (tmp_path / "o" / "power.csv").read_text().splitlines()
    assert n == 60 and len(lines) == 1 + 60
    assert lines[-1].count(",") == 4                        # last line is complete
    meta = _meta(tmp_path / "o")
    assert meta["n_rows"] == 60
    assert meta["achieved_hz"] == pytest.approx(20.0, rel=0.02)
    assert len((tmp_path / "o" / "clocks.csv").read_text().splitlines()) == 1 + 3


def test_crash_in_the_loop_still_flushes_rows_and_writes_meta(tmp_path):
    root = _make_root(tmp_path / "r")
    out = tmp_path / "o"
    clock = FakeClock()

    def boom(elapsed_s):
        if elapsed_s >= 1.0:
            raise RuntimeError("simulated crash")

    clock.probe = boom
    with pytest.raises(RuntimeError, match="simulated crash"):
        _run_fake(root, out, 5.0, clock=clock)
    meta = _meta(out)
    contracts.validate_telemetry_meta(meta)
    assert meta["vdd_in"]["channel"] == 2
    rows = (out / "power.csv").read_text().splitlines()
    assert len(rows) - 1 == meta["n_rows"] >= 20            # rows written before the crash were flushed


def test_hwmon_read_failure_skips_the_row_and_warns(tmp_path, capsys):
    root = _make_root(tmp_path / "r")
    clock = FakeClock()
    state = {"done": False}

    def drop_file(elapsed_s):
        if elapsed_s >= 0.5 and not state["done"]:
            (root / "sys/class/hwmon/hwmon3/curr2_input").unlink()
            state["done"] = True

    clock.probe = drop_file
    n = _run_fake(root, tmp_path / "o", 2.0, clock=clock)
    _, rows = _read_csv(tmp_path / "o" / "power.csv")
    assert 10 <= n == len(rows) < 40                        # rows stop, they are never faked
    assert "curr2_input" in capsys.readouterr().err


def test_out_dir_is_created(tmp_path):
    root = _make_root(tmp_path / "r")
    out = tmp_path / "a" / "b" / "telemetry"
    _run_fake(root, out, 0.2)
    assert (out / "power.csv").is_file() and (out / "clocks.csv").is_file() and (out / "meta.json").is_file()


def _spawn(root, out, *extra):
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.Popen(
        [sys.executable, "-m", "exp2.board.telemetry", "--out", str(out), "--sysfs-root", str(root),
         "--no-tegrastats", *extra],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(REPO), env=env, text=True)


def _wait_started(proc, out):
    """Block until the child has written meta.json (written before the first sample)."""
    deadline = time.monotonic() + 20
    while not (Path(out) / "meta.json").exists():
        assert proc.poll() is None, proc.stderr.read()
        assert time.monotonic() < deadline, "telemetry did not start"
        time.sleep(0.05)
    time.sleep(0.5)                                         # let it sample some rows


def test_stdin_eof_stops_with_rc0_in_3s_and_flushed_files(tmp_path):
    root = _make_root(tmp_path / "r")
    out = tmp_path / "o"
    proc = _spawn(root, out, "--hz", "50", "--duration", "60")
    try:
        _wait_started(proc, out)
        t0 = time.monotonic()
        proc.stdin.close()
        rc = proc.wait(timeout=3)
        elapsed = time.monotonic() - t0
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close()
        proc.stderr.close()
    assert rc == 0
    assert elapsed < 3.0
    lines = (out / "power.csv").read_text().splitlines()
    assert len(lines) - 1 >= 5                              # rows sampled before EOF are on disk
    assert all(len(x.split(",")) == 5 for x in lines)       # no torn last line
    assert _meta(out)["n_rows"] == len(lines) - 1


def test_meta_and_headers_written_at_start_survive_a_hard_kill(tmp_path):
    root = _make_root(tmp_path / "r")
    out = tmp_path / "o"
    proc = _spawn(root, out, "--hz", "20", "--duration", "60")
    try:
        _wait_started(proc, out)
        proc.kill()
        proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdin.close()
        proc.stdout.close()
        proc.stderr.close()
    meta = contracts.validate_telemetry_meta(_meta(out))
    assert meta["vdd_in"]["hwmon"] == "/sys/class/hwmon/hwmon3" and meta["n_rows"] == 0
    assert (out / "power.csv").read_text().splitlines()[0].split(",") == POWER_HEADER
    contracts.validate_csv_header("clocks", (out / "clocks.csv").read_text().splitlines()[0].split(","))


@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM delivery is POSIX only")
def test_sigterm_stops_with_rc0_and_flushed_files(tmp_path):
    root = _make_root(tmp_path / "r")
    out = tmp_path / "o"
    proc = _spawn(root, out, "--hz", "50", "--duration", "60")
    try:
        _wait_started(proc, out)
        proc.send_signal(signal.SIGTERM)
        rc = proc.wait(timeout=3)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdin.close()
        proc.stdout.close()
        proc.stderr.close()
    assert rc == 0
    lines = (out / "power.csv").read_text().splitlines()
    assert len(lines) - 1 >= 5
    assert _meta(out)["n_rows"] == len(lines) - 1


def test_duration_expiry_exits_rc0_and_main_returns_zero(tmp_path):
    root = _make_root(tmp_path / "r")
    t0 = time.monotonic()
    rc = _main(["--out", str(tmp_path / "o"), "--duration", "0.6", "--sysfs-root", str(root), "--no-tegrastats"])
    assert rc == 0
    assert 0.5 < time.monotonic() - t0 < 5.0


def test_stdin_eof_in_process_stops_the_run_early(tmp_path):
    root = _make_root(tmp_path / "r")
    r, w = os.pipe()
    os.close(w)                                             # immediate EOF
    try:
        t0 = time.monotonic()
        rc = telemetry.main(["--out", str(tmp_path / "o"), "--duration", "30", "--sysfs-root", str(root),
                             "--no-tegrastats"], stdin_fd=r)
    finally:
        os.close(r)
    assert rc == 0
    assert time.monotonic() - t0 < 3.0
    assert _meta(tmp_path / "o")["n_rows"] == len((tmp_path / "o" / "power.csv").read_text().splitlines()) - 1


@pytest.mark.parametrize("flag,value", [("--hz", "0"), ("--hz", "-5"), ("--clocks-hz", "0"), ("--duration", "0")])
def test_non_positive_rates_are_usage_errors(tmp_path, flag, value):
    root = _make_root(tmp_path / "r")
    with pytest.raises(SystemExit) as e:
        _main(["--out", str(tmp_path / "o"), flag, value, "--sysfs-root", str(root), "--no-tegrastats"])
    assert e.value.code == 2


# ---------------------------------------------------------------- tegrastats / EMC
@pytest.mark.parametrize("line,expected", [
    (TEGRASTATS_LINE, 3199.0),
    ("EMC_FREQ 12%@3199", 3199.0),
    ("x EMC_FREQ 0%@2133 y", 2133.0),
    ("EMC_FREQ 99%@665.6", 665.6),
    ("EMC_FREQ 12%@204 GR3D_FREQ 0%@[306]", 204.0),
])
def test_parse_emc_mhz(line, expected):
    assert telemetry.parse_emc_mhz(line) == expected


@pytest.mark.parametrize("line", [
    "", "RAM 2106/7620MB GR3D_FREQ 0%@[306]", "EMC_FREQ 12%", "EMC_FREQ off", "GR3D_FREQ 0%@[3199]",
])
def test_parse_emc_mhz_returns_none_without_an_emc_frequency(line):
    assert telemetry.parse_emc_mhz(line) is None


def test_tegrastats_stream_feeds_emc_column_and_meta(tmp_path, capsys):
    root = _make_root(tmp_path / "r")
    out, hb = tmp_path / "o", tmp_path / "hb.txt"
    argv = [sys.executable, "-c", FAKE_TEGRASTATS, str(hb), TEGRASTATS_LINE]
    tele = telemetry.Telemetry(out, root=root, hz=20, clocks_hz=4, duration_s=2.2, tegrastats_argv=argv)
    tele.run()
    _, rows = _read_csv(out / "clocks.csv")
    assert rows[-1]["emc_mhz"] == "3199"
    assert _meta(out)["emc_source"] == "tegrastats"
    assert "exited early" not in capsys.readouterr().err   # a normal stop is not a tegrastats failure
    # the child must be gone: its heartbeat stops advancing
    first = hb.read_text()
    time.sleep(0.4)
    assert hb.read_text() == first


def test_emc_changes_are_followed(tmp_path):
    root = _make_root(tmp_path / "r")
    out = tmp_path / "o"
    script = (
        "import sys, time\n"
        "for mhz in (2133, 3199):\n"
        "    for _ in range(12):\n"
        "        print('EMC_FREQ 5%@' + str(mhz), flush=True)\n"
        "        time.sleep(0.05)\n"
        "while True:\n"
        "    time.sleep(0.05)\n"
    )
    telemetry.Telemetry(out, root=root, hz=20, clocks_hz=8, duration_s=2.0,
                        tegrastats_argv=[sys.executable, "-c", script]).run()
    _, rows = _read_csv(out / "clocks.csv")
    values = [r["emc_mhz"] for r in rows if r["emc_mhz"]]
    assert "2133" in values and values[-1] == "3199"


def test_tegrastats_dying_midrun_does_not_leave_a_stale_emc_value(tmp_path, capsys):
    root = _make_root(tmp_path / "r")
    out = tmp_path / "o"
    script = (
        "import time\n"
        "for _ in range(8):\n"
        "    print('EMC_FREQ 5%@3199', flush=True)\n"
        "    time.sleep(0.05)\n"
    )
    telemetry.Telemetry(out, root=root, hz=20, clocks_hz=8, duration_s=2.5,
                        tegrastats_argv=[sys.executable, "-c", script]).run()
    _, rows = _read_csv(out / "clocks.csv")
    values = [r["emc_mhz"] for r in rows]
    assert "3199" in values                                 # it was seen while tegrastats lived
    assert values[-1] == ""                                 # and cleared once tegrastats was gone
    assert "exited early" in capsys.readouterr().err


def test_without_tegrastats_flag_emc_is_empty_and_source_null(run_2s):
    assert {r["emc_mhz"] for r in run_2s["clocks"]} == {""}
    assert run_2s["meta"]["emc_source"] is None


def test_tegrastats_binary_absent_means_empty_emc_and_null_source(tmp_path, monkeypatch):
    root = _make_root(tmp_path / "r")
    monkeypatch.setattr(telemetry.shutil, "which", lambda name, *a, **k: None)
    rc = _main(["--out", str(tmp_path / "o"), "--duration", "0.6", "--sysfs-root", str(root)])
    assert rc == 0
    _, rows = _read_csv(tmp_path / "o" / "clocks.csv")
    assert {r["emc_mhz"] for r in rows} == {""}
    assert _meta(tmp_path / "o")["emc_source"] is None


def test_tegrastats_that_cannot_start_does_not_kill_the_run(tmp_path, capsys):
    root = _make_root(tmp_path / "r")
    n = _run_fake(root, tmp_path / "o", 0.5, tegrastats_argv=[str(tmp_path / "no_such_tegrastats")])
    assert n > 0
    assert _meta(tmp_path / "o")["emc_source"] is None
    assert "tegrastats" in capsys.readouterr().err


def test_tegrastats_without_emc_lines_leaves_source_null(tmp_path, capsys):
    root = _make_root(tmp_path / "r")
    out = tmp_path / "o"
    argv = [sys.executable, "-c", "import time\nprint('RAM 1/2MB', flush=True)\ntime.sleep(30)"]
    telemetry.Telemetry(out, root=root, hz=20, clocks_hz=4, duration_s=1.0, tegrastats_argv=argv).run()
    assert _meta(out)["emc_source"] is None
    assert "EMC" in capsys.readouterr().err
