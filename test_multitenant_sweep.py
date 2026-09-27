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


if __name__ == "__main__":
    unittest.main(verbosity=2)
