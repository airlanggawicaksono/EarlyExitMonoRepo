import csv
import json
import os

import pytest

from exp2 import contracts as C
from exp2.model import backtest as B

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
TYPES = os.path.join(FIX, "types_from_run1.json")


def _heldout_dir(tmp_path, split="heldout"):
    d = tmp_path / "run1"
    d.mkdir()
    with open(os.path.join(FIX, "run1_csv", "corun.csv"), newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        rows = list(rd)
        cols = rd.fieldnames
    for r in rows:
        r["split"] = split
    _write(d, cols, rows)
    return d, rows, cols


def _write(d, cols, rows):
    with open(d / "corun.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _args(d, out):
    return ["--run1-csv", str(d), "--types", TYPES,
            "--slowdown", os.path.join(FIX, "slowdown_model.json"), "--power", os.path.join(FIX, "power_model.json"),
            "--out", str(out)]


def _run(tmp_path, d):
    assert B.main(_args(d, tmp_path / "out")) == 0
    with open(tmp_path / "out" / "backtest.json", encoding="utf-8") as fh:
        return json.load(fh)


def test_cli_writes_valid_backtest(tmp_path):
    d, _, _ = _heldout_dir(tmp_path)
    o = _run(tmp_path, d)
    C.validate_backtest(o)
    assert len(o["cells"]) == 4 and len(o["energy"]) == 2
    assert o["summary"]["tau_mdape"] >= 0


def test_ape_is_percent_of_measurement(tmp_path):
    d, rows, _ = _heldout_dir(tmp_path)
    o = _run(tmp_path, d)
    for c in o["cells"]:
        assert c["ape"] == pytest.approx(100 * abs(c["pred_tau_p50_s"] - c["meas_tau_p50_s"]) / c["meas_tau_p50_s"])
    r = [x for x in rows if x["type_id"] == "bert:5" and x["f_mhz"] == "612"][0]
    c = [x for x in o["cells"] if x["type_id"] == "bert:5" and x["f_mhz"] == 612][0]
    assert c["meas_tau_p50_s"] == pytest.approx(float(r["s_p50"]) * float(r["tau_solo_p50"]))
    e = [x for x in o["energy"] if x["f_mhz"] == 612][0]
    assert e["meas_j"] == pytest.approx(float(r["energy_j"]))


def test_energy_prediction_is_power_times_window(tmp_path):
    d, _, _ = _heldout_dir(tmp_path)
    o = _run(tmp_path, d)
    e = [x for x in o["energy"] if x["f_mhz"] == 1020][0]
    types = C.load_types(TYPES)
    u = types.get("bert:5").u[1020] + types.get("vision:24").u[1020]
    p = 5.62 + 7.98 * min(1.0, u / 0.91)
    assert e["pred_j"] == pytest.approx(p * 1.35365)


def test_train_rows_are_ignored(tmp_path):
    d, _, _ = _heldout_dir(tmp_path, split="train")
    o = _run(tmp_path, d)
    assert o["cells"] == [] and o["energy"] == []
    assert o["summary"] == {"tau_mdape": None, "energy_mdape": None}


def test_repeats_are_aggregated(tmp_path):
    d, rows, cols = _heldout_dir(tmp_path)
    _write(d, cols, rows + [dict(r, repeat="1") for r in rows])
    o = _run(tmp_path, d)
    assert len(o["cells"]) == 4 and len(o["energy"]) == 2


def test_bad_args_and_missing_file_exit_2(tmp_path):
    assert B.main([]) == 2
    d, _, _ = _heldout_dir(tmp_path)
    a = _args(d, tmp_path / "o")
    a[a.index("--power") + 1] = str(tmp_path / "nope.json")
    assert B.main(a) == 2
