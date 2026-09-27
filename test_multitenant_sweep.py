"""Unit tests for multitenant_sweep.py.

Everything runs offline: filesystem reads of /etc/nvpmodel.conf, the sudo
check, nvpmodel invocations, and multitenant_run.run_scenario are all mocked.
No Jetson hardware required.

Run:  python -m unittest test_multitenant_sweep -v
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

import multitenant_sweep as ms


# A realistic nvpmodel.conf sample for use across tests.
SAMPLE_CONF = """\
# Jetson Orin Nano nvpmodel configuration
< POWER_MODEL ID=0 NAME=MAXN >
CPU_ONLINE CORE_0 1
GPU MAX_FREQ -1

< POWER_MODEL ID=2 NAME=25W >
CPU_ONLINE CORE_0 1
GPU MAX_FREQ 624000

< POWER_MODEL ID=3 NAME=25W_2CORE >
CPU_ONLINE CORE_0 1
CPU_ONLINE CORE_3 0
GPU MAX_FREQ 624000

< POWER_MODEL ID=4 NAME=15W >
CPU_ONLINE CORE_0 1
GPU MAX_FREQ 408000

< PM_CONFIG DEFAULT=0 >
"""


class TestParseNvpmodelConf(unittest.TestCase):
    def test_parses_realistic_sample(self):
        table = ms.parse_nvpmodel_conf(SAMPLE_CONF)
        self.assertEqual(table, {0: "MAXN", 2: "25W", 3: "25W_2CORE", 4: "15W"})

    def test_empty_input_returns_empty_dict(self):
        self.assertEqual(ms.parse_nvpmodel_conf(""), {})
        self.assertEqual(ms.parse_nvpmodel_conf("# only a comment\n"), {})

    def test_single_entry(self):
        conf = "< POWER_MODEL ID=7 NAME=TURBO >\n"
        self.assertEqual(ms.parse_nvpmodel_conf(conf), {7: "TURBO"})

    def test_whitespace_variants(self):
        # Extra spaces inside the angle brackets should still match.
        conf = "<  POWER_MODEL  ID=1  NAME=TEST  >\n"
        self.assertEqual(ms.parse_nvpmodel_conf(conf), {1: "TEST"})


class TestLoadModeTable(unittest.TestCase):
    def test_missing_file_raises_runtime_error(self):
        p = Path("/nonexistent/path/nvpmodel.conf")
        with self.assertRaises(RuntimeError) as ctx:
            ms.load_mode_table(p)
        self.assertIn("not found", str(ctx.exception))

    def test_file_with_no_entries_raises_runtime_error(self):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".conf",
                                         delete=False, encoding="utf-8") as f:
            f.write("# no POWER_MODEL entries\n")
            fname = f.name
        try:
            with self.assertRaises(RuntimeError) as ctx:
                ms.load_mode_table(Path(fname))
            self.assertIn("No POWER_MODEL", str(ctx.exception))
        finally:
            Path(fname).unlink(missing_ok=True)

    def test_valid_file_returns_table(self):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".conf",
                                         delete=False, encoding="utf-8") as f:
            f.write(SAMPLE_CONF)
            fname = f.name
        try:
            table = ms.load_mode_table(Path(fname))
            self.assertEqual(table[0], "MAXN")
            self.assertIn(4, table)
        finally:
            Path(fname).unlink(missing_ok=True)


class TestResolveModeName(unittest.TestCase):
    def setUp(self):
        self.table = {0: "MAXN", 2: "25W", 3: "25W_2CORE", 4: "15W"}

    def test_exact_match(self):
        result = ms.resolve_mode_names(["MAXN"], self.table)
        self.assertEqual(result, [(0, "MAXN")])

    def test_case_insensitive(self):
        result = ms.resolve_mode_names(["maxn"], self.table)
        self.assertEqual(result, [(0, "MAXN")])

    def test_25w_exact_does_not_match_25w_2core(self):
        # "25W" exactly matches ID=2 NAME=25W, not ID=3 NAME=25W_2CORE.
        result = ms.resolve_mode_names(["25W"], self.table)
        self.assertEqual(result, [(2, "25W")])

    def test_prefix_match_when_no_exact(self):
        # A table with only MAXN_SUPER: "MAXN" should match via prefix.
        table = {0: "MAXN_SUPER", 4: "15W"}
        result = ms.resolve_mode_names(["MAXN"], table)
        self.assertEqual(result, [(0, "MAXN_SUPER")])

    def test_unknown_name_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            ms.resolve_mode_names(["99W"], self.table)
        self.assertIn("matches nothing", str(ctx.exception))
        # The error should print the discovered table.
        self.assertIn("MAXN", str(ctx.exception))

    def test_ambiguous_name_raises_value_error(self):
        # No exact match for "MAXN": two prefix candidates MAXN_SUPER and MAXN_PLUS.
        # The request is ambiguous and must be rejected.
        ambig = {0: "MAXN_SUPER", 1: "MAXN_PLUS"}
        with self.assertRaises(ValueError) as ctx:
            ms.resolve_mode_names(["MAXN"], ambig)
        self.assertIn("ambiguous", str(ctx.exception).lower())

    def test_multiple_modes_resolved_in_order(self):
        result = ms.resolve_mode_names(["MAXN", "15W"], self.table)
        self.assertEqual(result, [(0, "MAXN"), (4, "15W")])

    def test_fails_before_running_any_scenario(self):
        # When one name is bad, the whole call fails and nothing is returned.
        with self.assertRaises(ValueError):
            ms.resolve_mode_names(["MAXN", "BADMODE", "15W"], self.table)


class TestSudoCheck(unittest.TestCase):
    def test_failed_sudo_returns_false(self):
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=1)
            self.assertFalse(ms.check_sudo_noninteractive())

    def test_passing_sudo_returns_true(self):
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0)
            self.assertTrue(ms.check_sudo_noninteractive())

    def test_timeout_returns_false(self):
        import subprocess
        with mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired("sudo", 10)):
            self.assertFalse(ms.check_sudo_noninteractive())


class TestReadCurrentModeName(unittest.TestCase):
    def test_parses_nv_power_mode_line(self):
        output = "NV Power Mode: 25W\n2\n"
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(
                returncode=0, stdout=output, stderr=""
            )
            self.assertEqual(ms.read_current_mode_name(), "25W")

    def test_raises_when_line_absent(self):
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(
                returncode=0, stdout="no relevant line here\n", stderr=""
            )
            with self.assertRaises(RuntimeError):
                ms.read_current_mode_name()


class TestVerifyMode(unittest.TestCase):
    def test_matching_mode_does_not_raise(self):
        with mock.patch.object(ms, "read_current_mode_name", return_value="25W"):
            ms.verify_mode("25W")  # should not raise

    def test_mismatch_raises_runtime_error(self):
        with mock.patch.object(ms, "read_current_mode_name", return_value="15W"):
            with self.assertRaises(RuntimeError) as ctx:
                ms.verify_mode("25W")
            self.assertIn("25W", str(ctx.exception))
            self.assertIn("15W", str(ctx.exception))


def _make_sweep_mocks(initial_mode="MAXN", table=None, sudo_ok=True):
    """Return a dict of mock patches for run_sweep tests."""
    if table is None:
        table = {0: "MAXN", 2: "25W", 4: "15W"}
    return {
        "load_table": mock.patch.object(ms, "load_mode_table", return_value=table),
        "read_mode": mock.patch.object(ms, "read_current_mode_name",
                                       return_value=initial_mode),
        "sudo": mock.patch.object(ms, "check_sudo_noninteractive",
                                  return_value=sudo_ok),
        "switch": mock.patch.object(ms, "switch_mode"),
        "verify": mock.patch.object(ms, "verify_mode"),
        "run_scenario": mock.patch.object(ms.mr, "run_scenario",
                                          return_value=[{"tag": "x"}]),
    }


class TestRunSweep(unittest.TestCase):
    def _run_with_mocks(self, **kwargs):
        patches = _make_sweep_mocks(**kwargs)
        with patches["load_table"], patches["read_mode"], \
             patches["sudo"] as mock_sudo, \
             patches["switch"] as mock_switch, \
             patches["verify"], \
             patches["run_scenario"] as mock_run:
            result = ms.run_sweep(
                scenario="llama_yolo",
                modes=["MAXN", "25W"],
                tag="test",
                duration=30.0,
                k=6,
                repeats=1,
                holdout_n=0,
                keep_suspect=False,
                settle_sec=0.0,
            )
        return result, mock_switch, mock_run, mock_sudo

    def test_each_mode_receives_distinct_tag_with_mode_name(self):
        _, _, mock_run, _ = self._run_with_mocks()
        self.assertEqual(mock_run.call_count, 2)
        tags_used = [call.kwargs.get("tag") or call.args[1]
                     for call in mock_run.call_args_list]
        # First call tag must contain "MAXN", second must contain "25W".
        self.assertIn("MAXN", tags_used[0])
        self.assertIn("25W", tags_used[1])
        # The two tags must be different from each other.
        self.assertNotEqual(tags_used[0], tags_used[1])

    def test_failed_sudo_aborts_before_any_measurement(self):
        patches = _make_sweep_mocks(sudo_ok=False)
        with patches["load_table"], patches["read_mode"], \
             patches["sudo"], patches["switch"], \
             patches["verify"], patches["run_scenario"] as mock_run:
            with self.assertRaises(SystemExit):
                ms.run_sweep(
                    scenario="llama_yolo",
                    modes=["MAXN"],
                    tag="test",
                    duration=30.0,
                    k=6,
                    repeats=1,
                    holdout_n=0,
                    keep_suspect=False,
                    settle_sec=0.0,
                )
        mock_run.assert_not_called()

    def test_mode_verification_failure_aborts_sweep(self):
        patches = _make_sweep_mocks()
        with patches["load_table"], patches["read_mode"], \
             patches["sudo"], patches["switch"], \
             mock.patch.object(ms, "verify_mode",
                               side_effect=RuntimeError("mismatch")), \
             patches["run_scenario"] as mock_run:
            with self.assertRaises(RuntimeError):
                ms.run_sweep(
                    scenario="llama_yolo",
                    modes=["MAXN"],
                    tag="test",
                    duration=30.0,
                    k=6,
                    repeats=1,
                    holdout_n=0,
                    keep_suspect=False,
                    settle_sec=0.0,
                )
        mock_run.assert_not_called()

    def test_original_mode_restored_when_scenario_raises(self):
        table = {0: "MAXN", 2: "25W", 4: "15W"}
        switch_calls = []

        def fake_switch(mode_id):
            switch_calls.append(mode_id)

        with mock.patch.object(ms, "load_mode_table", return_value=table), \
             mock.patch.object(ms, "read_current_mode_name", return_value="MAXN"), \
             mock.patch.object(ms, "check_sudo_noninteractive", return_value=True), \
             mock.patch.object(ms, "switch_mode", side_effect=fake_switch), \
             mock.patch.object(ms, "verify_mode"), \
             mock.patch.object(ms.mr, "run_scenario",
                               side_effect=RuntimeError("scenario boom")):
            with self.assertRaises(RuntimeError):
                ms.run_sweep(
                    scenario="llama_yolo",
                    modes=["25W"],
                    tag="test",
                    duration=30.0,
                    k=6,
                    repeats=1,
                    holdout_n=0,
                    keep_suspect=False,
                    settle_sec=0.0,
                )

        # switch_calls: first call switches to 25W (ID=2), second restores MAXN (ID=0).
        self.assertGreaterEqual(len(switch_calls), 2)
        self.assertEqual(switch_calls[0], 2)   # switched to 25W
        self.assertEqual(switch_calls[-1], 0)  # restored to MAXN

    def test_original_mode_restored_on_clean_exit(self):
        table = {0: "MAXN", 2: "25W", 4: "15W"}
        switch_calls = []

        def fake_switch(mode_id):
            switch_calls.append(mode_id)

        with mock.patch.object(ms, "load_mode_table", return_value=table), \
             mock.patch.object(ms, "read_current_mode_name", return_value="MAXN"), \
             mock.patch.object(ms, "check_sudo_noninteractive", return_value=True), \
             mock.patch.object(ms, "switch_mode", side_effect=fake_switch), \
             mock.patch.object(ms, "verify_mode"), \
             mock.patch.object(ms.mr, "run_scenario", return_value=[]):
            ms.run_sweep(
                scenario="llama_yolo",
                modes=["25W"],
                tag="test",
                duration=30.0,
                k=6,
                repeats=1,
                holdout_n=0,
                keep_suspect=False,
                settle_sec=0.0,
            )

        self.assertEqual(switch_calls[0], 2)   # switched to 25W
        self.assertEqual(switch_calls[-1], 0)  # restored to MAXN

    def test_tags_contain_mode_name_for_csv_attribution(self):
        """Each mode's run_scenario call must receive a tag carrying the mode name."""
        table = {0: "MAXN", 4: "15W"}
        call_tags = []

        def capture_run(scenario, tag=None, **kwargs):
            call_tags.append(tag)
            return []

        with mock.patch.object(ms, "load_mode_table", return_value=table), \
             mock.patch.object(ms, "read_current_mode_name", return_value="MAXN"), \
             mock.patch.object(ms, "check_sudo_noninteractive", return_value=True), \
             mock.patch.object(ms, "switch_mode"), \
             mock.patch.object(ms, "verify_mode"), \
             mock.patch.object(ms.mr, "run_scenario", side_effect=capture_run):
            ms.run_sweep(
                scenario="llama_yolo",
                modes=["MAXN", "15W"],
                tag="mytag",
                duration=30.0,
                k=6,
                repeats=1,
                holdout_n=0,
                keep_suspect=False,
                settle_sec=0.0,
            )

        self.assertEqual(len(call_tags), 2)
        self.assertIn("MAXN", call_tags[0])
        self.assertIn("15W", call_tags[1])
        # Tags must differ.
        self.assertNotEqual(call_tags[0], call_tags[1])


if __name__ == "__main__":
    unittest.main()
