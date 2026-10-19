"""S0a gate: exp2.board.dryroot (K17 fake sysfs tree)."""
from __future__ import annotations

import pytest

from exp2.board.dryroot import make_dry_root


def rd(root, rel):
    return (root / rel).read_text().strip()


@pytest.fixture
def root(tmp_path):
    return make_dry_root(tmp_path / "r")


class TestReturnValue:
    def test_returns_the_root_path_and_creates_missing_dirs(self, tmp_path):
        target = tmp_path / "deep" / "nested"
        got = make_dry_root(target)
        assert got == target and (got / "sys").is_dir() and (got / "proc/meminfo").is_file()

    def test_accepts_str_and_is_idempotent(self, tmp_path):
        make_dry_root(str(tmp_path))
        make_dry_root(str(tmp_path))
        assert rd(tmp_path, "sys/class/hwmon/hwmon3/name") == "ina3221"

    def test_writes_only_under_the_given_dir(self, tmp_path):
        make_dry_root(tmp_path / "r")
        assert [p.name for p in tmp_path.iterdir()] == ["r"]


class TestIna3221:
    def test_vdd_in_is_labelled_on_hwmon3_in2(self, root):
        hw = "sys/class/hwmon/hwmon3"
        assert rd(root, f"{hw}/name") == "ina3221"
        assert rd(root, f"{hw}/in2_label") == "VDD_IN"
        assert rd(root, f"{hw}/in2_input") == "5072" and rd(root, f"{hw}/curr2_input") == "1203"
        assert int(rd(root, f"{hw}/in2_input")) * int(rd(root, f"{hw}/curr2_input")) / 1000 == pytest.approx(6101.616)

    def test_other_channels_carry_different_labels_and_values(self, root):
        hw = "sys/class/hwmon/hwmon3"
        labels = {n: rd(root, f"{hw}/in{n}_label") for n in (1, 2, 3)}
        assert sorted(labels.values()).count("VDD_IN") == 1 and len(set(labels.values())) == 3
        volts = {n: rd(root, f"{hw}/in{n}_input") for n in (1, 2, 3)}
        amps = {n: rd(root, f"{hw}/curr{n}_input") for n in (1, 2, 3)}
        assert len(set(volts.values())) == 3 and len(set(amps.values())) == 3
        assert (volts[1], amps[1]) != (volts[2], amps[2]) != (volts[3], amps[3])

    def test_update_interval_and_samples_exist(self, root):
        assert int(rd(root, "sys/class/hwmon/hwmon3/update_interval")) >= 1
        assert int(rd(root, "sys/class/hwmon/hwmon3/samples")) >= 1

    def test_position_arguments_move_the_label(self, tmp_path):
        r = make_dry_root(tmp_path, vdd_in_hwmon="hwmon5", vdd_in_channel=3)
        hw = "sys/class/hwmon/hwmon5"
        assert rd(r, f"{hw}/name") == "ina3221" and rd(r, f"{hw}/in3_label") == "VDD_IN"
        assert rd(r, f"{hw}/in3_input") == "5072" and rd(r, f"{hw}/curr3_input") == "1203"
        assert rd(r, f"{hw}/in2_label") != "VDD_IN" and rd(r, f"{hw}/in1_label") != "VDD_IN"
        assert not (r / "sys/class/hwmon/hwmon3").exists()

    def test_hard_coded_hwmon1_in1_would_read_a_wrong_number(self, root):
        """A reader that hard-codes hwmon1/in1 gets the decoy (different name, VDD_IN-looking label, value 7)."""
        assert rd(root, "sys/class/hwmon/hwmon1/name") == "nvme"
        assert rd(root, "sys/class/hwmon/hwmon1/in1_input") == "7"
        assert rd(root, "sys/class/hwmon/hwmon3/in1_input") != rd(root, "sys/class/hwmon/hwmon1/in1_input")
        assert rd(root, "sys/class/hwmon/hwmon3/in1_label") != "VDD_IN"

    def test_only_one_ina3221_exists(self, root):
        names = [rd(root, f"sys/class/hwmon/{d.name}/name") for d in (root / "sys/class/hwmon").iterdir()]
        assert names.count("ina3221") == 1 and len(names) >= 3

    def test_decoy_does_not_overwrite_a_requested_hwmon1(self, tmp_path):
        r = make_dry_root(tmp_path, vdd_in_hwmon="hwmon1", vdd_in_channel=1)
        hw = "sys/class/hwmon/hwmon1"
        assert rd(r, f"{hw}/name") == "ina3221" and rd(r, f"{hw}/in1_label") == "VDD_IN"
        assert rd(r, "sys/class/hwmon/hwmon1/in1_input") == "5072"


class TestGpuDevfreq:
    DEV = "sys/class/devfreq/17000000.gpu"

    def test_default_levels_in_hz_and_stock_start_state(self, root):
        levels = [int(x) for x in rd(root, f"{self.DEV}/available_frequencies").split()]
        assert levels == [m * 1_000_000 for m in (306, 408, 510, 612, 714, 816, 918, 1020)]
        assert rd(root, f"{self.DEV}/min_freq") == "306000000"
        assert rd(root, f"{self.DEV}/max_freq") == "1020000000" == rd(root, f"{self.DEV}/cur_freq")

    def test_custom_levels_are_sorted(self, tmp_path):
        r = make_dry_root(tmp_path, gpu_levels_mhz=(612, 306, 1020))
        assert rd(r, f"{self.DEV}/available_frequencies") == "306000000 612000000 1020000000"
        assert rd(r, f"{self.DEV}/min_freq") == "306000000" and rd(r, f"{self.DEV}/max_freq") == "1020000000"

    def test_load_is_readable_where_telemetry_looks(self, root):
        for rel in (f"{self.DEV}/load", f"{self.DEV}/device/load", "sys/devices/platform/17000000.gpu/load"):
            assert 0 <= int(rd(root, rel)) <= 1000


class TestEmcAndBandwidthManager:
    EMC = "sys/kernel/debug/bpmp/debug/clk/emc"

    def test_rates_in_hz_and_start_state(self, root):
        assert rd(root, f"{self.EMC}/rate") == "3199000000" == rd(root, f"{self.EMC}/max_rate")
        assert rd(root, f"{self.EMC}/min_rate") == "204000000"
        assert rd(root, f"{self.EMC}/mrq_rate_locked") == "0" and rd(root, f"{self.EMC}/state") == "1"
        assert rd(root, "sys/kernel/debug/bpmp/debug/bwmgr/bwmgr_halt") == "0"

    def test_custom_range(self, tmp_path):
        r = make_dry_root(tmp_path, emc_min=665.6, emc_max=2133.0)
        assert rd(r, f"{self.EMC}/min_rate") == "665600000" and rd(r, f"{self.EMC}/max_rate") == "2133000000"
        assert rd(r, f"{self.EMC}/rate") == "2133000000"


class TestMiscFiles:
    def test_boot_id_default_and_custom(self, tmp_path):
        assert rd(make_dry_root(tmp_path / "a"), "proc/sys/kernel/random/boot_id") == "dry-boot"
        assert rd(make_dry_root(tmp_path / "b", boot_id="boot-B"), "proc/sys/kernel/random/boot_id") == "boot-B"

    def test_meminfo_has_memavailable_in_kb(self, root):
        line = next(x for x in (root / "proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:"))
        assert line.split()[2] == "kB" and int(line.split()[1]) > 0

    def test_cpu0_frequency_files(self, root):
        base = "sys/devices/system/cpu/cpu0/cpufreq"
        assert int(rd(root, f"{base}/scaling_cur_freq")) > 0 and int(rd(root, f"{base}/cpuinfo_cur_freq")) > 0

    def test_thermal_zones_exist_and_are_not_in_name_order(self, root):
        base = "sys/class/thermal/thermal_zone"
        zones = [(rd(root, f"{base}{i}/type"), int(rd(root, f"{base}{i}/temp"))) for i in range(3)]
        names = [z[0] for z in zones]
        assert len(set(names)) == 3 and names != sorted(names) and all(t > 1000 for _, t in zones)

    def test_release_and_power_mode_files(self, root):
        assert rd(root, "etc/nv_tegra_release").startswith("# R36")
        assert rd(root, "etc/nvpmodel_state") == "1"

    def test_files_end_with_a_newline_like_sysfs(self, root):
        assert (root / "sys/class/devfreq/17000000.gpu/cur_freq").read_bytes().endswith(b"\n")
        assert b"\r" not in (root / "proc/meminfo").read_bytes()


class TestArgumentChecks:
    @pytest.mark.parametrize("kw", [{"vdd_in_channel": 0}, {"vdd_in_channel": 4}, {"gpu_levels_mhz": ()},
                                    {"emc_min": 3199.0, "emc_max": 204.0}, {"emc_min": 500.0, "emc_max": 500.0}])
    def test_bad_arguments(self, tmp_path, kw):
        with pytest.raises(ValueError):
            make_dry_root(tmp_path, **kw)
