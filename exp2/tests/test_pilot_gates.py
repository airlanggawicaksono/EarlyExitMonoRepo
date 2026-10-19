"""Pilot gates: one passing and one failing case per gate, summary keys, RUNBOOK coverage, dry run."""
import json
import re
from pathlib import Path

import pytest

from exp2.pilot import pilot as P

RUNBOOK = Path(P.__file__).with_name("RUNBOOK.md")


def _cells(e612, e1020):
    return {"612": {"window_energy_j": e612, "n_done": 10}, "1020": {"window_energy_j": e1020, "n_done": 10}}


def _mem(load):
    return {"all_ready": True, "min_mem_available_mb_idle": 900, "min_mem_available_mb_load": load}


CASES = [
    ("inventory", lambda: P.gate_inventory({"a": True, "b": True}, ["a", "b"]),
     lambda: P.gate_inventory({"a": True, "b": False}, ["a", "b"])),
    ("power_mode_id", lambda: P.gate_power_mode(2), lambda: P.gate_power_mode(1)),
    ("coresident_min_mem_mb", lambda: P.gate_coresident(_mem(600)), lambda: P.gate_coresident(_mem(400))),
    ("warm_over_cold_load",
     lambda: P.gate_warm_over_cold({"bert": {"cold_load_s": 10, "warm_load_s": 2}}),
     lambda: P.gate_warm_over_cold({"bert": {"cold_load_s": 10, "warm_load_s": 8}})),
    ("gpu_lock_hold", lambda: P.gate_gpu_lock([1020] * 100, 1020),
     lambda: P.gate_gpu_lock([1020] * 90 + [612] * 10, 1020)),
    ("emc_lock_hold", lambda: P.gate_emc_lock([3199] * 100, [3199] * 100, 3199),
     lambda: P.gate_emc_lock([3199] * 100, [3199] * 90 + [2133] * 10, 3199)),
    ("switch_latency_s", lambda: P.gate_switch_latency([0.1, 0.2, 0.3]),
     lambda: P.gate_switch_latency([0.1, 0.2, 0.9])),
    ("ckpt_overhead_frac",
     lambda: P.gate_ckpt_overhead({"bert": {"with_s": [1.05], "without_s": [1.0]}}),
     lambda: P.gate_ckpt_overhead({"bert": {"with_s": [1.3], "without_s": [1.0]}})),
    ("yolo_solo_cv", lambda: P.gate_yolo_cv([1.0, 1.02, 0.98]), lambda: P.gate_yolo_cv([1.0, 2.0, 0.5])),
    ("energy_vs_f_h2",
     lambda: P.gate_energy_vs_f({"bert": {"0.6": _cells(90, 110)}}, 1020),
     lambda: P.gate_energy_vs_f({"bert": {"0.6": _cells(120, 100)}}, 1020)),
]


@pytest.mark.parametrize("gid,good,bad", CASES, ids=[c[0] for c in CASES])
def test_gate_pass_and_fail(gid, good, bad):
    g, b = good(), bad()
    assert g["pass"] is True and b["pass"] is False
    assert g["step"] == b["step"] == P.GATE_IDS.index(gid)
    assert set(g) == {"value", "threshold", "pass", "step"}


def test_missing_data_never_passes():
    assert P.gate_gpu_lock([], 1020)["pass"] is False
    assert P.gate_energy_vs_f({}, 1020)["value"] is None


def test_h2_uses_window_energy_including_idle():
    cells = {"612": {"window_energy_j": 100.0, "n_done": 5}, "1020": {"window_energy_j": 105.0, "n_done": 10}}
    assert P.gate_energy_vs_f({"bert": {"0.3": cells}}, 1020)["pass"] is False  # f_max cheaper per request
    cells["612"]["window_energy_j"] = 40.0
    assert P.gate_energy_vs_f({"bert": {"0.3": cells}}, 1020)["pass"] is True


def test_runbook_lists_every_gate_id_and_rules():
    text = RUNBOOK.read_text(encoding="utf-8")
    for gid in P.GATE_IDS:
        assert re.search("`%s`" % gid, text), gid
    for needle in ("getpass", "git bundle create exp2.bundle exp2-dev", "872e66d", "scp", "git worktree add"):
        assert needle in text


def test_dry_run_summary_keys_and_rc(tmp_path):
    rc = P.main(["--dry-run", "--out", str(tmp_path)])
    s = json.loads((tmp_path / "pilot_summary.json").read_text(encoding="utf-8"))
    assert rc == 0 and s["dry"] is True and s["schema"] == "exp2.pilot/1"
    assert list(s["gates"]) == list(P.GATE_IDS)
    assert all(g["pass"] for g in s["gates"].values())


def test_real_mode_missing_measurements_fails(tmp_path):
    cfg = json.loads(P.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    s = P.run_pilot(tmp_path, cfg, [2, 9], dry=False)
    assert not any(g["pass"] for g in s["gates"].values())
