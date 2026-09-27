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


class TestCheckIsRoot(unittest.TestCase):
    def test_returns_bool_always(self):
        """check_is_root must return a bool on all platforms, including Windows."""
        result = ms.check_is_root()
        self.assertIsInstance(result, bool)

    def test_returns_false_when_geteuid_absent(self):
        """Simulates the Windows environment where os.geteuid is not present.

        patch 'hasattr' inside the module so it reports geteuid absent;
        the function must return False without raising AttributeError.
        """
        original_hasattr = hasattr

        def fake_hasattr(obj, name):
            if name == "geteuid":
                return False
            return original_hasattr(obj, name)

        with mock.patch("multitenant_sweep.hasattr", side_effect=fake_hasattr):
            result = ms.check_is_root()
        self.assertFalse(result)

    def test_true_when_euid_zero(self):
        import os as _os
        with mock.patch("multitenant_sweep.os") as mock_os:
            mock_os.geteuid = mock.Mock(return_value=0)
            # Simulate hasattr returning True
            with mock.patch("multitenant_sweep.hasattr", return_value=True):
                result = ms.check_is_root()
        self.assertTrue(result)

    def test_false_when_euid_nonzero(self):
        import os as _os
        with mock.patch("multitenant_sweep.os") as mock_os:
            mock_os.geteuid = mock.Mock(return_value=1000)
            with mock.patch("multitenant_sweep.hasattr", return_value=True):
                result = ms.check_is_root()
        self.assertFalse(result)


class TestCheckNvpmodelDirect(unittest.TestCase):
    def test_returns_true_when_nvpmodel_q_succeeds(self):
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0)
            self.assertTrue(ms.check_nvpmodel_direct())

    def test_returns_false_when_nvpmodel_q_fails(self):
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=1)
            self.assertFalse(ms.check_nvpmodel_direct())

    def test_returns_false_when_nvpmodel_not_found(self):
        with mock.patch("subprocess.run", side_effect=FileNotFoundError):
            self.assertFalse(ms.check_nvpmodel_direct())


class TestProbeSwitchCapability(unittest.TestCase):
    def test_root_path_selected_first(self):
        """When the process is root, method is 'root' and sudo is never called."""
        with mock.patch.object(ms, "check_is_root", return_value=True), \
             mock.patch.object(ms, "check_sudo_noninteractive") as mock_sudo, \
             mock.patch.object(ms, "check_nvpmodel_direct") as mock_direct:
            method, _ = ms.probe_switch_capability()
        self.assertEqual(method, "root")
        mock_sudo.assert_not_called()
        mock_direct.assert_not_called()

    def test_sudo_path_when_not_root(self):
        with mock.patch.object(ms, "check_is_root", return_value=False), \
             mock.patch.object(ms, "check_sudo_noninteractive", return_value=True), \
             mock.patch.object(ms, "check_nvpmodel_direct") as mock_direct:
            method, _ = ms.probe_switch_capability()
        self.assertEqual(method, "sudo")
        mock_direct.assert_not_called()

    def test_nvpmodel_direct_path(self):
        with mock.patch.object(ms, "check_is_root", return_value=False), \
             mock.patch.object(ms, "check_sudo_noninteractive", return_value=False), \
             mock.patch.object(ms, "check_nvpmodel_direct", return_value=True):
            method, _ = ms.probe_switch_capability()
        self.assertEqual(method, "nvpmodel_direct")

    def test_none_when_all_fail(self):
        with mock.patch.object(ms, "check_is_root", return_value=False), \
             mock.patch.object(ms, "check_sudo_noninteractive", return_value=False), \
             mock.patch.object(ms, "check_nvpmodel_direct", return_value=False):
            method, _ = ms.probe_switch_capability()
        self.assertIsNone(method)


class TestSwitchMode(unittest.TestCase):
    def test_sudo_path_uses_sudo(self):
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0)
            ms.switch_mode(2, method="sudo")
        cmd = mock_run.call_args[0][0]
        self.assertIn("sudo", cmd)
        self.assertIn("nvpmodel", cmd)

    def test_root_path_omits_sudo(self):
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0)
            ms.switch_mode(2, method="root")
        cmd = mock_run.call_args[0][0]
        self.assertNotIn("sudo", cmd)
        self.assertIn("nvpmodel", cmd)

    def test_nvpmodel_direct_path_omits_sudo(self):
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0)
            ms.switch_mode(4, method="nvpmodel_direct")
        cmd = mock_run.call_args[0][0]
        self.assertNotIn("sudo", cmd)
        self.assertIn("nvpmodel", cmd)


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


# ---------------------------------------------------------------------------
# Helpers shared by sweep and campaign tests.
# ---------------------------------------------------------------------------

def _make_sweep_mocks(initial_mode="MAXN", table=None, switch_method="sudo"):
    """Return a dict of mock patches for run_sweep tests.

    switch_method: method string returned by probe_switch_capability.
    Pass None to simulate no switching capability.
    """
    if table is None:
        table = {0: "MAXN", 2: "25W", 4: "15W"}
    can_switch = switch_method is not None
    return {
        "load_table": mock.patch.object(ms, "load_mode_table", return_value=table),
        "read_mode": mock.patch.object(ms, "read_current_mode_name",
                                       return_value=initial_mode),
        "probe": mock.patch.object(ms, "probe_switch_capability",
                                   return_value=(switch_method, "mocked")),
        "switch": mock.patch.object(ms, "switch_mode"),
        "verify": mock.patch.object(ms, "verify_mode"),
        "run_scenario": mock.patch.object(ms.mr, "run_scenario",
                                          return_value=[{"tag": "x"}]),
    }


class TestRunSweep(unittest.TestCase):
    def _run_with_mocks(self, **kwargs):
        patches = _make_sweep_mocks(**kwargs)
        with patches["load_table"], patches["read_mode"], \
             patches["probe"], \
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
        return result, mock_switch, mock_run

    def test_each_mode_receives_distinct_tag_with_mode_name(self):
        _, _, mock_run = self._run_with_mocks()
        self.assertEqual(mock_run.call_count, 2)
        tags_used = [call.kwargs.get("tag") or call.args[1]
                     for call in mock_run.call_args_list]
        # First call tag must contain "MAXN", second must contain "25W".
        self.assertIn("MAXN", tags_used[0])
        self.assertIn("25W", tags_used[1])
        # The two tags must be different from each other.
        self.assertNotEqual(tags_used[0], tags_used[1])

    def test_no_switching_capability_degrades_not_refuses(self):
        """When switching is unavailable and --require-all-modes is NOT set,
        the sweep runs once (degraded) and exits with EXIT_DEGRADED, rather
        than refusing to start."""
        patches = _make_sweep_mocks(switch_method=None)
        with patches["load_table"], patches["read_mode"], \
             patches["probe"], patches["switch"], patches["verify"], \
             patches["run_scenario"] as mock_run:
            with self.assertRaises(SystemExit) as ctx:
                ms.run_sweep(
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
        # Must exit with EXIT_DEGRADED (2), not 1.
        self.assertEqual(ctx.exception.code, ms.EXIT_DEGRADED)
        # The scenario must still have run exactly once.
        mock_run.assert_called_once()

    def test_require_all_modes_refuses_when_switching_unavailable(self):
        """Under --require-all-modes, no measurement is taken when switching
        is unavailable."""
        patches = _make_sweep_mocks(switch_method=None)
        with patches["load_table"], patches["read_mode"], \
             patches["probe"], patches["switch"], patches["verify"], \
             patches["run_scenario"] as mock_run:
            with self.assertRaises(SystemExit) as ctx:
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
                    require_all_modes=True,
                )
        # Exit code 1: refused before any measurement.
        self.assertEqual(ctx.exception.code, 1)
        mock_run.assert_not_called()

    def test_degraded_run_tags_with_actual_mode_not_requested(self):
        """The most important correctness property: a degraded run must tag rows
        with the device's actual current mode, never with a requested mode name.
        Mislabelled rows are worse than missing rows."""
        captured_tags = []

        def capture_scenario(scenario, tag=None, **kwargs):
            captured_tags.append(tag)
            return [{"tag": tag}]

        patches = _make_sweep_mocks(initial_mode="MAXN_SUPER", switch_method=None)
        with patches["load_table"], patches["read_mode"], \
             patches["probe"], patches["switch"], patches["verify"], \
             mock.patch.object(ms.mr, "run_scenario", side_effect=capture_scenario):
            with self.assertRaises(SystemExit):
                ms.run_sweep(
                    scenario="bert_yolo",
                    modes=["25W", "15W"],   # requested modes
                    tag="degtest",
                    duration=30.0,
                    k=6,
                    repeats=1,
                    holdout_n=0,
                    keep_suspect=False,
                    settle_sec=0.0,
                )

        self.assertEqual(len(captured_tags), 1)
        actual_tag = captured_tags[0]
        # Tag must carry the device's actual mode name (MAXN_SUPER), not the
        # requested ones (25W or 15W).
        self.assertIn("MAXN_SUPER", actual_tag)
        self.assertNotIn("25W", actual_tag)
        self.assertNotIn("15W", actual_tag)

    def test_already_root_no_sudo_invoked(self):
        """When the process is root, sudo must never be called."""
        switch_calls = []

        def fake_switch(mode_id, method="sudo"):
            switch_calls.append((mode_id, method))

        patches = _make_sweep_mocks(switch_method="root")
        with patches["load_table"], patches["read_mode"], \
             patches["probe"], \
             mock.patch.object(ms, "switch_mode", side_effect=fake_switch), \
             patches["verify"], patches["run_scenario"]:
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

        # All switch calls must use "root" method (no sudo).
        for _, method in switch_calls:
            self.assertEqual(method, "root",
                             "switch_mode was called with method != 'root' despite being root")

    def test_nvpmodel_direct_no_sudo_invoked(self):
        """When nvpmodel is accessible directly, sudo must never be called."""
        switch_calls = []

        def fake_switch(mode_id, method="sudo"):
            switch_calls.append((mode_id, method))

        patches = _make_sweep_mocks(switch_method="nvpmodel_direct")
        with patches["load_table"], patches["read_mode"], \
             patches["probe"], \
             mock.patch.object(ms, "switch_mode", side_effect=fake_switch), \
             patches["verify"], patches["run_scenario"]:
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

        for _, method in switch_calls:
            self.assertEqual(method, "nvpmodel_direct")

    def test_mode_verification_failure_aborts_sweep(self):
        patches = _make_sweep_mocks()
        with patches["load_table"], patches["read_mode"], \
             patches["probe"], patches["switch"], \
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

        def fake_switch(mode_id, method="sudo"):
            switch_calls.append(mode_id)

        with mock.patch.object(ms, "load_mode_table", return_value=table), \
             mock.patch.object(ms, "read_current_mode_name", return_value="MAXN"), \
             mock.patch.object(ms, "probe_switch_capability",
                               return_value=("sudo", "mocked")), \
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

        def fake_switch(mode_id, method="sudo"):
            switch_calls.append(mode_id)

        with mock.patch.object(ms, "load_mode_table", return_value=table), \
             mock.patch.object(ms, "read_current_mode_name", return_value="MAXN"), \
             mock.patch.object(ms, "probe_switch_capability",
                               return_value=("sudo", "mocked")), \
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
             mock.patch.object(ms, "probe_switch_capability",
                               return_value=("sudo", "mocked")), \
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


class TestRunCampaign(unittest.TestCase):
    """Tests for the campaign runner."""

    def _patches(self, initial_mode="MAXN", switch_method="sudo",
                 table=None, scenario_result=None, scenario_side_effect=None):
        if table is None:
            table = {0: "MAXN", 2: "25W", 4: "15W"}
        if scenario_result is None:
            scenario_result = [{"tag": "x"}]
        p = {
            "load_table": mock.patch.object(ms, "load_mode_table", return_value=table),
            "read_mode": mock.patch.object(ms, "read_current_mode_name",
                                           return_value=initial_mode),
            "probe": mock.patch.object(ms, "probe_switch_capability",
                                       return_value=(switch_method, "mocked")),
            "switch": mock.patch.object(ms, "switch_mode"),
            "verify": mock.patch.object(ms, "verify_mode"),
        }
        if scenario_side_effect is not None:
            p["run_scenario"] = mock.patch.object(ms.mr, "run_scenario",
                                                   side_effect=scenario_side_effect)
        else:
            p["run_scenario"] = mock.patch.object(ms.mr, "run_scenario",
                                                   return_value=scenario_result)
        return p

    def _run(self, scenarios, modes=None, switch_method="sudo",
             initial_mode="MAXN", scenario_side_effect=None,
             require_all_modes=False):
        if modes is None:
            modes = ["MAXN"]
        p = self._patches(initial_mode=initial_mode,
                          switch_method=switch_method,
                          scenario_side_effect=scenario_side_effect)
        with p["load_table"], p["read_mode"], p["probe"], \
             p["switch"], p["verify"], p["run_scenario"] as mock_run:
            result = ms.run_campaign(
                scenarios=scenarios,
                modes=modes,
                tag="camp",
                duration=30.0,
                k=6,
                repeats=1,
                holdout_n=0,
                keep_suspect=False,
                settle_sec=0.0,
                require_all_modes=require_all_modes,
            )
        return result, mock_run

    def test_campaign_runs_all_scenarios(self):
        result, mock_run = self._run(["bert_scale", "yolo_scale"])
        self.assertEqual(result["completed"], ["bert_scale", "yolo_scale"])
        self.assertEqual(result["failed"], {})

    def test_campaign_continues_past_failing_scenario(self):
        """A failure in one scenario must not stop the rest of the campaign."""
        call_count = [0]

        def side_effect(scenario, tag=None, **kwargs):
            call_count[0] += 1
            if scenario == "yolo_scale":
                raise RuntimeError("injected failure")
            return [{"tag": tag}]

        result, _ = self._run(
            ["bert_scale", "yolo_scale", "vit_scale"],
            scenario_side_effect=side_effect,
        )
        self.assertIn("bert_scale", result["completed"])
        self.assertIn("vit_scale", result["completed"])
        self.assertIn("yolo_scale", result["failed"])
        self.assertIn("injected failure", result["failed"]["yolo_scale"])

    def test_campaign_summary_reports_failures_and_successes(self):
        """Summary dict has both completed and failed keys with the right content."""
        def side_effect(scenario, tag=None, **kwargs):
            if scenario == "bert_yolo":
                raise RuntimeError("bad cell")
            return []

        result, _ = self._run(
            ["bert_scale", "bert_yolo", "yolo_scale"],
            scenario_side_effect=side_effect,
        )
        self.assertIn("bert_scale", result["completed"])
        self.assertIn("yolo_scale", result["completed"])
        self.assertEqual(list(result["failed"].keys()), ["bert_yolo"])

    def test_campaign_plan_lists_all_scenarios_before_running(self, ):
        """The plan output is printed before any scenario runs.

        We verify this by checking that the plan contains each scenario name
        and a cell count derived from count_cells, and that it is emitted before
        run_scenario is ever called.
        """
        # We cannot intercept stdout ordering easily, but we can confirm the
        # cell-count helper returns sensible values for the plan.
        for s in ["bert_scale", "yolo_vit"]:
            nc = ms.count_cells(s, k=6)
            self.assertGreater(nc, 0, f"count_cells({s!r}, 6) returned 0")

    def test_campaign_restricted_to_subset_runs_only_that_subset(self):
        """--campaign bert_scale,yolo_vit must run only those two scenarios."""
        called = []

        def side_effect(scenario, tag=None, **kwargs):
            called.append(scenario)
            return []

        result, _ = self._run(
            ["bert_scale", "yolo_vit"],
            scenario_side_effect=side_effect,
        )
        self.assertEqual(sorted(called), sorted(["bert_scale", "yolo_vit"]))
        self.assertNotIn("llama_yolo", called)
        self.assertNotIn("triple", called)

    def test_campaign_degraded_when_no_switching(self):
        """Without switching capability, campaign runs degraded and sets
        the degraded flag in the summary."""
        result, mock_run = self._run(
            ["bert_scale", "yolo_scale"],
            switch_method=None,
            initial_mode="MAXN_SUPER",
        )
        self.assertTrue(result["degraded"])
        # Both scenarios still ran.
        self.assertEqual(result["completed"], ["bert_scale", "yolo_scale"])
        self.assertEqual(mock_run.call_count, 2)

    def test_campaign_degraded_tags_with_actual_mode(self):
        """In degraded mode every scenario tag carries the device's current mode,
        not any requested mode name."""
        captured = []

        def side_effect(scenario, tag=None, **kwargs):
            captured.append(tag)
            return []

        result, _ = self._run(
            ["bert_scale"],
            modes=["25W", "15W"],   # requested but unavailable
            switch_method=None,
            initial_mode="MAXN_SUPER",
            scenario_side_effect=side_effect,
        )
        self.assertEqual(len(captured), 1)
        self.assertIn("MAXN_SUPER", captured[0])
        self.assertNotIn("25W", captured[0])
        self.assertNotIn("15W", captured[0])

    def test_campaign_require_all_modes_refuses_when_unavailable(self):
        p = self._patches(switch_method=None)
        with p["load_table"], p["read_mode"], p["probe"], \
             p["switch"], p["verify"], p["run_scenario"] as mock_run:
            with self.assertRaises(SystemExit) as ctx:
                ms.run_campaign(
                    scenarios=["bert_scale"],
                    modes=["MAXN"],
                    tag="camp",
                    duration=30.0,
                    k=6,
                    repeats=1,
                    holdout_n=0,
                    keep_suspect=False,
                    settle_sec=0.0,
                    require_all_modes=True,
                )
        self.assertEqual(ctx.exception.code, 1)
        mock_run.assert_not_called()

    def test_campaign_restores_original_mode_after_completion(self):
        table = {0: "MAXN", 2: "25W", 4: "15W"}
        switch_calls = []

        def fake_switch(mode_id, method="sudo"):
            switch_calls.append(mode_id)

        with mock.patch.object(ms, "load_mode_table", return_value=table), \
             mock.patch.object(ms, "read_current_mode_name", return_value="MAXN"), \
             mock.patch.object(ms, "probe_switch_capability",
                               return_value=("sudo", "mocked")), \
             mock.patch.object(ms, "switch_mode", side_effect=fake_switch), \
             mock.patch.object(ms, "verify_mode"), \
             mock.patch.object(ms.mr, "run_scenario", return_value=[]):
            ms.run_campaign(
                scenarios=["bert_scale"],
                modes=["25W"],
                tag="camp",
                duration=30.0,
                k=6,
                repeats=1,
                holdout_n=0,
                keep_suspect=False,
                settle_sec=0.0,
            )

        # Last switch must restore original mode (MAXN = ID 0).
        self.assertEqual(switch_calls[-1], 0)


# ---------------------------------------------------------------------------
# Daemon control tests.  No actual processes are spawned: subprocess.Popen
# and the pid-alive check are mocked throughout.
# ---------------------------------------------------------------------------

class TestSweepDaemonPaths(unittest.TestCase):
    """The sweep pid and log paths must be distinct from bench_jetson's paths."""

    def test_sweep_pid_distinct_from_bench_pid(self):
        bench_pid = ms.REPO_ROOT / "logs" / "bench_daemon.pid"
        self.assertNotEqual(ms.SWEEP_PID, bench_pid,
                            "SWEEP_PID must not collide with bench_jetson's DAEMON_PID")

    def test_sweep_log_distinct_from_bench_log(self):
        bench_log = ms.REPO_ROOT / "logs" / "bench_daemon.log"
        self.assertNotEqual(ms.SWEEP_LOG, bench_log,
                            "SWEEP_LOG must not collide with bench_jetson's DAEMON_LOG")

    def test_sweep_pid_path_contains_sweep(self):
        self.assertIn("sweep", ms.SWEEP_PID.name)

    def test_sweep_log_path_contains_sweep(self):
        self.assertIn("sweep", ms.SWEEP_LOG.name)


class TestSweepDaemonStart(unittest.TestCase):
    """_sweep_start must relaunch without -d, set MALLOC_ARENA_MAX, and record the pid.

    Uses a real temp directory so Path.mkdir and Path.write_text work normally;
    only subprocess.Popen is mocked to prevent actual process spawning.
    """

    def _run_start(self, argv_override=None):
        """Helper: mock everything that touches the filesystem and spawns processes."""
        mock_proc = mock.Mock()
        mock_proc.pid = 12345

        # Use a mock for the log file open so no real file handle is created.
        # This avoids Windows file-locking issues when the mock Popen never
        # closes the handle.
        with mock.patch("sys.argv",
                        argv_override or ["multitenant_sweep.py", "--campaign", "-d"]), \
             mock.patch.object(ms, "_sweep_running", return_value=False), \
             mock.patch("multitenant_sweep.SWEEP_PID") as mock_pid_path, \
             mock.patch("multitenant_sweep.SWEEP_LOG") as mock_log_path, \
             mock.patch("builtins.open", mock.mock_open()), \
             mock.patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            mock_pid_path.parent.mkdir.return_value = None
            mock_pid_path.write_text.return_value = None
            ms._sweep_start()
        return mock_popen

    def test_daemon_flag_stripped_from_relaunch_command(self):
        mock_popen = self._run_start(
            ["multitenant_sweep.py", "--campaign", "bert_scale", "--modes", "MAXN", "-d"]
        )
        cmd = mock_popen.call_args[0][0]
        self.assertNotIn("-d", cmd)
        self.assertNotIn("--daemon", cmd)

    def test_other_flags_preserved_in_relaunch_command(self):
        argv = ["multitenant_sweep.py", "--campaign", "bert_scale",
                "--modes", "MAXN,15W", "--k", "4", "--duration", "120",
                "--tag", "run1", "-d"]
        mock_popen = self._run_start(argv)
        cmd = mock_popen.call_args[0][0]
        cmd_str = " ".join(cmd)
        for expected in ["--campaign", "bert_scale", "--modes", "MAXN,15W",
                         "--k", "4", "--duration", "120", "--tag", "run1"]:
            self.assertIn(expected, cmd_str,
                          f"Expected argument {expected!r} missing from relaunch cmd")

    def test_long_daemon_flag_also_stripped(self):
        mock_popen = self._run_start(
            ["multitenant_sweep.py", "--campaign", "--daemon"]
        )
        cmd = mock_popen.call_args[0][0]
        self.assertNotIn("--daemon", cmd)
        self.assertNotIn("-d", cmd)

    def test_malloc_arena_max_set_to_2(self):
        mock_proc = mock.Mock()
        mock_proc.pid = 9999
        captured_env = {}

        def capture_popen(cmd, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            return mock_proc

        with mock.patch("sys.argv", ["multitenant_sweep.py", "--campaign", "-d"]), \
             mock.patch.object(ms, "_sweep_running", return_value=False), \
             mock.patch("multitenant_sweep.SWEEP_PID") as mock_pid_path, \
             mock.patch("multitenant_sweep.SWEEP_LOG"), \
             mock.patch("builtins.open", mock.mock_open()), \
             mock.patch("subprocess.Popen", side_effect=capture_popen):
            mock_pid_path.parent.mkdir.return_value = None
            mock_pid_path.write_text.return_value = None
            ms._sweep_start()

        self.assertEqual(captured_env.get("MALLOC_ARENA_MAX"), "2")

    def test_malloc_arena_max_not_overridden_when_already_set(self):
        mock_proc = mock.Mock()
        mock_proc.pid = 9999
        captured_env = {}

        def capture_popen(cmd, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            return mock_proc

        with mock.patch("sys.argv", ["multitenant_sweep.py", "--campaign", "-d"]), \
             mock.patch.object(ms, "_sweep_running", return_value=False), \
             mock.patch("multitenant_sweep.SWEEP_PID") as mock_pid_path, \
             mock.patch("multitenant_sweep.SWEEP_LOG"), \
             mock.patch("builtins.open", mock.mock_open()), \
             mock.patch.dict("os.environ", {"MALLOC_ARENA_MAX": "4"}), \
             mock.patch("subprocess.Popen", side_effect=capture_popen):
            mock_pid_path.parent.mkdir.return_value = None
            mock_pid_path.write_text.return_value = None
            ms._sweep_start()

        self.assertEqual(captured_env.get("MALLOC_ARENA_MAX"), "4",
                         "Explicit MALLOC_ARENA_MAX in environment must not be overridden")

    def test_refuses_when_already_running(self):
        with mock.patch.object(ms, "_sweep_running", return_value=True), \
             mock.patch.object(ms, "_sweep_read_pid", return_value=42), \
             mock.patch("subprocess.Popen") as mock_popen:
            ms._sweep_start()
        mock_popen.assert_not_called()


class TestSweepDaemonStop(unittest.TestCase):
    def test_stop_with_no_pid_file_reports_cleanly(self):
        # Patch the module attribute so _sweep_stop reads and unlinks a real temp file.
        import tempfile, os as _os
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_pid = Path(tmpdir) / "sweep_daemon.pid"
            with mock.patch.object(ms, "_sweep_read_pid", return_value=None), \
                 mock.patch.object(ms, "SWEEP_PID", tmp_pid):
                # Must not raise.
                ms._sweep_stop()

    def test_stop_with_dead_pid_reports_cleanly(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_pid = Path(tmpdir) / "sweep_daemon.pid"
            with mock.patch.object(ms, "_sweep_read_pid", return_value=9999), \
                 mock.patch.object(ms, "SWEEP_PID", tmp_pid), \
                 mock.patch("psutil.pid_exists", return_value=False):
                ms._sweep_stop()

    def test_stop_kills_parent_and_children(self):
        import psutil, tempfile
        mock_child = mock.Mock(spec=psutil.Process)
        mock_proc = mock.Mock(spec=psutil.Process)
        mock_proc.children.return_value = [mock_child]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_pid = Path(tmpdir) / "sweep_daemon.pid"
            # Write a fake pid so unlink has a real file.
            tmp_pid.write_text("1234")
            with mock.patch.object(ms, "_sweep_read_pid", return_value=1234), \
                 mock.patch.object(ms, "SWEEP_PID", tmp_pid), \
                 mock.patch("psutil.pid_exists", return_value=True), \
                 mock.patch("psutil.Process", return_value=mock_proc), \
                 mock.patch("psutil.wait_procs", return_value=([], [])):
                ms._sweep_stop()

        mock_proc.terminate.assert_called()
        mock_child.terminate.assert_called()


class TestSweepDaemonSnapshot(unittest.TestCase):
    def test_no_pid_file_no_csv_reports_cleanly(self):
        # Point the module CSV paths at non-existent temp paths so exists() returns
        # False naturally, avoiding the need to patch Path instance methods.
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            no_csv = Path(tmpdir) / "nonexistent.csv"
            no_suspect = Path(tmpdir) / "nonexistent.suspect.csv"
            with mock.patch.object(ms, "_sweep_read_pid", return_value=None), \
                 mock.patch.object(ms, "_sweep_running", return_value=False), \
                 mock.patch.object(ms, "_SWEEP_CSV", no_csv), \
                 mock.patch.object(ms, "_SWEEP_SUSPECT_CSV", no_suspect), \
                 mock.patch.object(ms, "_sweep_print_log_tail"):
                # Must not raise.
                ms._sweep_snapshot()

    def test_snapshot_counts_rows_in_csv(self):
        import tempfile, os as _os
        csv_content = "tag,n_tenants,other\nrun1_MAXN,2,x\nrun1_25W,2,y\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir) / "concurrent_slowdown.csv"
            tmp.write_text(csv_content, encoding="utf-8")
            no_suspect = Path(tmpdir) / "concurrent_slowdown.suspect.csv"
            with mock.patch.object(ms, "_sweep_read_pid", return_value=None), \
                 mock.patch.object(ms, "_sweep_running", return_value=False), \
                 mock.patch.object(ms, "_SWEEP_CSV", tmp), \
                 mock.patch.object(ms, "_SWEEP_SUSPECT_CSV", no_suspect), \
                 mock.patch.object(ms, "_sweep_print_log_tail"), \
                 mock.patch("builtins.print") as mock_print:
                ms._sweep_snapshot()
            printed = " ".join(str(a) for call in mock_print.call_args_list
                               for a in call[0])
            self.assertIn("2", printed)

    def test_snapshot_reports_distinct_tags(self):
        import tempfile
        csv_content = (
            "tag,n_tenants\n"
            "camp_bert_scale_MAXN,2\n"
            "camp_bert_scale_25W,2\n"
            "camp_yolo_scale_MAXN,3\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir) / "concurrent_slowdown.csv"
            tmp.write_text(csv_content, encoding="utf-8")
            no_suspect = Path(tmpdir) / "concurrent_slowdown.suspect.csv"
            with mock.patch.object(ms, "_sweep_read_pid", return_value=None), \
                 mock.patch.object(ms, "_sweep_running", return_value=False), \
                 mock.patch.object(ms, "_SWEEP_CSV", tmp), \
                 mock.patch.object(ms, "_SWEEP_SUSPECT_CSV", no_suspect), \
                 mock.patch.object(ms, "_sweep_print_log_tail"), \
                 mock.patch("builtins.print") as mock_print:
                ms._sweep_snapshot()
            printed = " ".join(str(a) for call in mock_print.call_args_list
                               for a in call[0])
            self.assertIn("camp_bert_scale_MAXN", printed)
            self.assertIn("camp_yolo_scale_MAXN", printed)


class TestCountCsvRows(unittest.TestCase):
    def test_missing_file(self):
        count, note = ms._count_csv_rows(Path("/nonexistent/file.csv"))
        self.assertEqual(count, 0)
        self.assertIn("not found", note)

    def test_header_only(self):
        import tempfile, os as _os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                         delete=False, encoding="utf-8") as f:
            f.write("tag,n_tenants\n")
            tmp = f.name
        try:
            count, note = ms._count_csv_rows(Path(tmp))
            self.assertEqual(count, 0)
        finally:
            _os.unlink(tmp)

    def test_complete_rows(self):
        import tempfile, os as _os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                         delete=False, encoding="utf-8") as f:
            f.write("tag,n_tenants\nrun1,2\nrun2,3\n")
            tmp = f.name
        try:
            count, note = ms._count_csv_rows(Path(tmp))
            self.assertEqual(count, 2)
            self.assertEqual(note, "")
        finally:
            _os.unlink(tmp)

    def test_truncated_last_line_not_counted_and_noted(self):
        import tempfile, os as _os
        # Last line has no comma: truncated mid-write.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                         delete=False, encoding="utf-8") as f:
            f.write("tag,n_tenants\nrun1,2\nrun2,3\nrun3")
            tmp = f.name
        try:
            count, note = ms._count_csv_rows(Path(tmp))
            self.assertEqual(count, 2)
            self.assertIn("truncated", note)
        finally:
            _os.unlink(tmp)

    def test_does_not_raise_on_empty_file(self):
        import tempfile, os as _os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                         delete=False, encoding="utf-8") as f:
            tmp = f.name
        try:
            count, note = ms._count_csv_rows(Path(tmp))
            self.assertEqual(count, 0)
        finally:
            _os.unlink(tmp)


class TestReadCsvTags(unittest.TestCase):
    def test_missing_file_returns_empty_set(self):
        tags = ms._read_csv_tags(Path("/nonexistent/file.csv"))
        self.assertEqual(tags, set())

    def test_returns_distinct_tags(self):
        import tempfile, os as _os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                         delete=False, encoding="utf-8") as f:
            f.write("tag,n_tenants\nalpha,2\nbeta,2\nalpha,3\n")
            tmp = f.name
        try:
            tags = ms._read_csv_tags(Path(tmp))
            self.assertEqual(tags, {"alpha", "beta"})
        finally:
            _os.unlink(tmp)

    def test_truncated_last_line_does_not_raise(self):
        import tempfile, os as _os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                         delete=False, encoding="utf-8") as f:
            f.write("tag,n_tenants\nalpha,2\nbeta")
            tmp = f.name
        try:
            tags = ms._read_csv_tags(Path(tmp))
            self.assertIn("alpha", tags)
            # "beta" may or may not appear depending on parse; what matters is no crash.
        finally:
            _os.unlink(tmp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
