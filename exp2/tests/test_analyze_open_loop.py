"""Tests for exp2.analyze.open_loop (slice F2): fixture reproduction, refusals, synthetic-grid arithmetic."""
import csv
import json
import shutil
from pathlib import Path

import pytest

from exp2 import contracts as C
from exp2 import stats as S
from exp2.analyze import open_loop as OL

FIX = Path(__file__).parent / "fixtures"
SESSION, FREEZE, PLAN = FIX / "session_open", FIX / "freeze.json", FIX / "plan_run2.json"


def _close(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            _close(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _close(x, y)
    elif isinstance(a, float) or isinstance(b, float):
        assert a == pytest.approx(b, rel=1e-9)
    else:
        assert a == b


def test_fixture_reproduced(tmp_path):
    assert OL.main(["--session", str(SESSION), "--freeze", str(FREEZE), "--plan", str(PLAN),
                    "--out", str(tmp_path)]) == 0
    _close(json.loads((tmp_path / "open_summary.json").read_text()),
           json.loads((FIX / "open_summary.json").read_text()))
    got = list(csv.DictReader(open(tmp_path / "open_conditions.csv")))
    exp = list(csv.DictReader(open(FIX / "open_conditions.csv")))
    assert len(got) == len(exp) == 20
    for g, e in zip(got, exp):
        for col in e:
            if g[col] == e[col]:
                continue
            assert float(g[col]) == pytest.approx(float(e[col]), rel=1e-9), col


def _copy(tmp_path, **edit):
    d = tmp_path / "s"
    shutil.copytree(SESSION, d)
    sj = json.loads((d / "session.json").read_text())
    sj.update(edit)
    (d / "session.json").write_text(json.dumps(sj))
    return d


def test_refuse_tampered_freeze(tmp_path):
    f = tmp_path / "freeze.json"
    shutil.copy(FREEZE, f)
    shutil.copy(str(FREEZE) + ".sha256", str(f) + ".sha256")
    f.write_bytes(f.read_bytes().replace(b'"f_grid_mhz"', b'"f_grid_mhz" '))
    assert OL.main(["--session", str(SESSION), "--freeze", str(f), "--out", str(tmp_path / "o")]) == 3


def test_refuse_wall_order_prints_both_clocks(tmp_path, capsys):
    frz = json.loads(FREEZE.read_text())["created_wall_ns"]
    d = _copy(tmp_path, t_wall_start_ns=frz)
    assert OL.main(["--session", str(d), "--freeze", str(FREEZE), "--out", str(tmp_path / "o")]) == 3
    err = capsys.readouterr().err
    assert str(frz) in err and "t_wall_start_ns" in err and "created_wall_ns" in err


def test_refuse_plan_hash_mismatch(tmp_path):
    d = _copy(tmp_path, plan_sha256="0" * 64)
    assert OL.main(["--session", str(d), "--freeze", str(FREEZE), "--plan", str(PLAN),
                    "--out", str(tmp_path / "o")]) == 3


def test_p95_nearest_rank_and_energy_window_covers_expired(tmp_path):
    out = tmp_path / "o"
    OL.analyze(SESSION, FREEZE, out)
    rows = {(r["condition_id"], r["type_id"]): r for r in csv.DictReader(open(out / "open_conditions.csv"))}
    cid = "W_hi_r060_serial_r0"
    reqs = C.read_requests(SESSION / cid / "requests.jsonl")
    live = [r for r in reqs if not r["is_warmup"] and r["type_id"] == "bert:5"]
    p95 = S.p_nearest_rank([(r["t_done_ns"] - r["t_arrival_ns"]) / 1e9 for r in live], 0.95)
    assert float(rows[(cid, "bert:5")]["p95_R_s"]) == pytest.approx(p95, rel=1e-9)
    assert any(r["outcome"] == "expired_in_queue" for r in live)
    t0, t1, n = S.request_window(reqs)
    assert t1 == max(r["t_done_ns"] for r in reqs if not r["is_warmup"])
    power = C.read_csv(SESSION / "telemetry" / "power.csv", "power")
    assert float(rows[(cid, "bert:5")]["energy_per_req_j"]) == pytest.approx(S.energy_j(power, t0, t1) / n)


def test_summary_arithmetic_on_synthetic_grid():
    fz = {"content_sha256": "a" * 64, "conditions": [{
        "condition_id": "c", "grid": [{"name": "g0"}, {"name": "g1"}],
        "game": {"worst": {"policy_key": "g1"}, "optimum": {"policy_key": "F"}},
        "predictions": {k: {"energy_per_req_j": 1.0, "per_type": {"t": {"complete_rate": 1.0}}}
                        for k in ("F", "g0", "g1", "x")}}]}
    cnt_ok, cnt_bad = {"t": (10, 10)}, {"t": (10, 9)}

    def man(k):
        return {"freeze_condition_id": "c", "policy_key": k}
    conds = [(man("F"), 2.0, cnt_ok), (man("F"), 2.2, cnt_ok), (man("g0"), 3.0, cnt_ok), (man("g1"), 1.0, cnt_bad),
             (man("x"), 1.5, cnt_ok), (man("x"), 1.7, cnt_ok)]
    s = OL._summarize(fz, conds, {})
    assert s["regret"] == {"F": 0.0, "grid_spread": pytest.approx(3.0 / 2.1 - 1.0), "best_key": "F"}
    assert s["oracle"]["c"] == {"policy_key": "x", "energy_per_req_j": pytest.approx(1.6)}
    assert s["per_condition"]["c_g1"]["feasible"] is False
    assert s["poa_measured"] == pytest.approx(1.0 / 2.1)
    d, lo, hi = S.welch_diff_ci([2.0, 2.2], [1.5, 1.7])
    assert s["energy_diff"]["x"] == {"diff_j": pytest.approx(d), "ci": [pytest.approx(lo), pytest.approx(hi)]}
    assert s["mape"]["energy"] == pytest.approx(S.mape([1.0] * 4, [2.1, 3.0, 1.0, 1.6]))
