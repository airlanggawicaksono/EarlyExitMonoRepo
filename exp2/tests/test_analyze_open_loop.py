"""Tests for exp2.analyze.open_loop (slice F2 + A3): fixture reproduction, refusals, multi-session, C2b arithmetic."""
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


def _run(tmp_path, sessions=(SESSION,), freeze=FREEZE, plans=(PLAN,), name="o"):
    args = ["--sessions", *map(str, sessions), "--freeze", str(freeze), "--out", str(tmp_path / name)]
    if plans:
        args += ["--plans", *map(str, plans)]
    return OL.main(args), tmp_path / name


def test_fixture_reproduced(tmp_path):
    rc, out = _run(tmp_path)
    assert rc == 0
    _close(json.loads((out / "open_summary.json").read_text()),
           json.loads((FIX / "open_summary.json").read_text()))
    got = list(csv.DictReader(open(out / "open_conditions.csv")))
    exp = list(csv.DictReader(open(FIX / "open_conditions.csv")))
    assert len(got) == len(exp) == 24       # 12 runs x 2 types (serial repeats 3 times since R30)
    assert list(got[0]) == list(C.OPEN_CONDITIONS_COLS)
    for g, e in zip(got, exp):
        for col in e:
            if g[col] == e[col]:
                continue
            assert float(g[col]) == pytest.approx(float(e[col]), rel=1e-9), col


def test_screen_csv_has_one_row_per_run(tmp_path):
    _, out = _run(tmp_path)
    rows = C.read_csv(out / "screen.csv", "screen")
    assert len(rows) == 12 and all(r["kept"] == 1 and r["mode"] == "open" for r in rows)
    assert json.loads((out / "open_summary.json").read_text())["counts"] == {
        "runs_valid": 12, "runs_invalid": 0, "runs_duplicate": 0,
        "runs_screen_flagged": sum(r["screen_ok"] == 0 for r in rows)}


def test_late_rate_and_overshoot_follow_the_deadlines(tmp_path):
    _, out = _run(tmp_path)
    rows = {(r["condition_id"], r["type_id"]): r for r in csv.DictReader(open(out / "open_conditions.csv"))}
    n_late = 0
    for (cid, tid), row in rows.items():
        live = [r for r in C.read_requests(SESSION / cid / "requests.jsonl")
                if not r["is_warmup"] and r["type_id"] == tid]
        over = [(r["t_done_ns"] - r["t_deadline_ns"]) / 1e9 for r in live
                if r["outcome"] in ("complete", "truncated", "miss") and r["t_done_ns"] is not None
                and r["t_done_ns"] > r["t_deadline_ns"]]
        assert float(row["late_rate"]) == pytest.approx(len(over) / len(live))
        if over:
            n_late += 1
            assert float(row["overshoot_max_s"]) == pytest.approx(max(over))
            assert float(row["overshoot_p95_s"]) == pytest.approx(S.p_nearest_rank(over, 0.95))
        else:
            assert row["overshoot_max_s"] == "" and row["overshoot_p95_s"] == ""
    assert n_late > 0                       # the fixture has late rows, so the branch above is exercised


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
    assert OL.main(["--sessions", str(SESSION), "--freeze", str(f), "--out", str(tmp_path / "o")]) == 3


def test_refuse_wall_order_prints_both_clocks(tmp_path, capsys):
    frz = json.loads(FREEZE.read_text())["created_wall_ns"]
    d = _copy(tmp_path, t_wall_start_ns=frz)
    assert OL.main(["--sessions", str(d), "--freeze", str(FREEZE), "--out", str(tmp_path / "o")]) == 3
    err = capsys.readouterr().err
    assert str(frz) in err and "t_wall_start_ns" in err and "created_wall_ns" in err


def test_refuse_plan_hash_mismatch(tmp_path):
    d = _copy(tmp_path, plan_sha256="0" * 64)
    assert OL.main(["--sessions", str(d), "--freeze", str(FREEZE), "--plans", str(PLAN),
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


# ------------------------------------------------------------------------------------------ synthetic arithmetic
def _spec(name, ks=1, kl=1, order="fcfs", f=1020):
    return C.PolicySpec(name, ks, kl, order, None, "static", f, {}).to_json()


def _fz(choices, grid, game=None, cid="c", keys=None):
    names = list(choices) + [g["name"] for g in grid]
    return {"condition_id": cid, "choices": choices, "grid": grid, "game": game or {"worst": {}, "optimum": {}},
            "predictions": {k: {"energy_per_req_j": 1.0, "per_type": {"t": {"complete_rate": 1.0}}}
                            for k in (keys or names)}}


def _conds(cid, rows):
    """rows: [(policy_key, repeat, energy, (n, k))]"""
    return [({"freeze_condition_id": cid, "policy_key": k, "repeat": r}, e, {"t": nk}) for k, r, e, nk in rows]


COUNTS = {"runs_valid": 0, "runs_invalid": 0, "runs_duplicate": 0, "runs_screen_flagged": 0}
OKC = (10, 10)


def test_summary_arithmetic_on_synthetic_grid():
    fz = {"content_sha256": "a" * 64, "conditions": [_fz(
        {"F": _spec("F", 2, 1, "stf", 612), "x": _spec("x", 2, 2, "stf", 408)},
        [_spec("g0", f=612), _spec("g1", 1, 2)],
        {"worst": {"policy_key": "g1"}, "optimum": {"policy_key": "F"}})]}
    bad = (10, 9)
    conds = _conds("c", [("F", 0, 2.0, OKC), ("F", 1, 2.2, OKC), ("g0", 0, 3.0, OKC), ("g1", 0, 1.0, bad),
                         ("x", 0, 1.5, OKC), ("x", 1, 1.7, OKC)])
    s = OL._summarize(fz, conds, {}, COUNTS)
    assert s["regret"] == {"F": 0.0, "grid_spread": pytest.approx(3.0 / 2.1 - 1.0), "best_key": "F"}
    assert s["oracle"]["c"] == {"policy_key": "x", "energy_per_req_j": pytest.approx(1.6)}
    assert s["per_condition"]["c_g1"]["feasible"] is False
    assert s["poa_measured"] == pytest.approx(1.0 / 2.1)
    assert s["poa_detail"]["c"] == {"worst_key": "g1", "optimum_key": "F", "ratio": pytest.approx(1.0 / 2.1),
                                    "complete_min_worst": 0.9, "complete_min_optimum": 1.0}
    d, lo, hi = S.paired_diff_ci([2.0, 2.2], [1.5, 1.7])
    assert s["energy_diff"]["c_x"] == {"diff_j": pytest.approx(d), "ci": [pytest.approx(lo), pytest.approx(hi)],
                                       "n_pairs": 2, "method": "paired_t"}
    assert "c_g0" not in s["energy_diff"]                 # one repeat only
    assert s["mape"]["energy"] == pytest.approx(S.mape([1.0] * 4, [2.1, 3.0, 1.0, 1.6]))
    assert s["per_condition"]["c_F"]["n_repeats"] == 2 and s["per_condition"]["c_F"]["measured_key"] == "F"
    checks = {"hash_ok": True, "wall_order_ok": True, "t_wall_start_ns": 2, "freeze_created_wall_ns": 1}
    C.validate_open_summary(dict(s, checks=checks))


def test_paired_difference_uses_repeat_pairs_not_group_means():
    fz = {"content_sha256": "a" * 64, "conditions": [_fz({"F": _spec("F"), "x": _spec("x", 2, 2)}, [])]}
    conds = _conds("c", [("F", 0, 1.0, OKC), ("F", 1, 2.0, OKC), ("F", 2, 3.0, OKC),
                         ("x", 2, 2.9, OKC), ("x", 0, 0.9, OKC), ("x", 1, 1.9, OKC)])       # shuffled repeats
    dd = OL._summarize(fz, conds, {}, COUNTS)["energy_diff"]["c_x"]
    assert dd["diff_j"] == pytest.approx(0.1) and dd["n_pairs"] == 3
    assert dd["ci"][1] - dd["ci"][0] < 1e-9                # identical pair differences: zero-width interval


def test_energy_diff_is_kept_per_freeze_condition():
    fz = {"content_sha256": "a" * 64, "conditions": [
        _fz({"F": _spec("F"), "x": _spec("x", 2, 2)}, [], cid="c1"),
        _fz({"F": _spec("F"), "x": _spec("x", 2, 2)}, [], cid="c2")]}
    conds = (_conds("c1", [("F", 0, 2.0, OKC), ("F", 1, 2.2, OKC), ("x", 0, 1.0, OKC), ("x", 1, 1.4, OKC)]) +
             _conds("c2", [("F", 0, 5.0, OKC), ("F", 1, 5.2, OKC), ("x", 0, 4.0, OKC), ("x", 1, 4.2, OKC)]))
    dd = OL._summarize(fz, conds, {}, COUNTS)["energy_diff"]
    assert set(dd) == {"c1_x", "c2_x"} and dd["c1_x"]["diff_j"] == pytest.approx(0.9)
    assert dd["c2_x"]["diff_j"] == pytest.approx(1.0)


def test_empty_worst_does_not_crash_and_poa_is_null():
    fz = {"content_sha256": "a" * 64, "conditions": [_fz({"F": _spec("F")}, [])]}
    s = OL._summarize(fz, _conds("c", [("F", 0, 1.0, OKC)]), {}, COUNTS)
    assert s["poa_detail"] == {} and s["poa_measured"] is None


def test_poa_detail_with_unmeasured_worst_is_null_ratio():
    fz = {"content_sha256": "a" * 64, "conditions": [_fz(
        {"F": _spec("F")}, [_spec("g0", 2, 2)], {"worst": {"policy_key": "g0"}, "optimum": {"policy_key": "F"}})]}
    s = OL._summarize(fz, _conds("c", [("F", 0, 1.0, OKC)]), {}, COUNTS)
    assert s["poa_detail"]["c"]["ratio"] is None and s["poa_detail"]["c"]["complete_min_worst"] is None
    assert s["poa_measured"] is None


def test_aliases_are_folded_and_dedup_grid_points_stay_regret_candidates():
    # default_pinned = default (alias); grid g0 equals F (alias of F); g1 is its own key
    fz = {"content_sha256": "a" * 64, "conditions": [_fz(
        {"F": _spec("F", 2, 1), "default": _spec("default", 1, 1), "default_pinned": _spec("default_pinned", 1, 1)},
        [_spec("g0", 2, 1), _spec("g1", 3, 3)])]}
    conds = _conds("c", [("F", 0, 2.0, OKC), ("default_pinned", 0, 1.0, OKC), ("default", 1, 1.2, OKC),
                         ("g1", 0, 3.0, OKC)])
    s = OL._summarize(fz, conds, {}, COUNTS)
    assert set(s["per_condition"]) == {"c_F", "c_default", "c_g1"}      # alias folded into its measured key
    assert s["per_condition"]["c_default"]["n_repeats"] == 2
    # candidates: F (also g0) and g1; default is a choice but no grid point, so not a candidate
    assert s["regret"]["best_key"] == "F" and s["regret"]["F"] == 0.0
    assert s["regret"]["grid_spread"] == pytest.approx(0.5)
    assert s["oracle"]["c"]["policy_key"] == "default"


@pytest.mark.parametrize("n,k,status", [(200, 200, "feasible"), (100, 95, "borderline"), (100, 90, "infeasible")])
def test_wilson_status(n, k, status):
    fz = {"content_sha256": "a" * 64, "conditions": [_fz({"F": _spec("F")}, [])]}
    s = OL._summarize(fz, _conds("c", [("F", 0, 1.0, (n, k))]), {}, COUNTS)["per_condition"]["c_F"]
    assert s["status"] == status and s["wilson_lo_min"] == pytest.approx(S.wilson_ci(k, n)[0])
    assert s["feasible"] == (k / n >= C.COMPLETE_TARGET)


# ------------------------------------------------------------------------------------------ multi-session
def _cond_list(d):
    return json.loads((d / "session.json").read_text())["conditions"]


def _edit_session(d, **edit):
    sj = json.loads((d / "session.json").read_text())
    sj.update(edit)
    (d / "session.json").write_text(json.dumps(sj))


def test_two_sessions_split_a_run_and_match_the_single_session(tmp_path):
    names = _cond_list(SESSION)
    a, b = tmp_path / "a", tmp_path / "b"
    shutil.copytree(SESSION, a)
    shutil.copytree(SESSION, b)
    _edit_session(a, conditions=names[:6], session_id="sA")
    _edit_session(b, conditions=names[6:], session_id="sB")
    rc, out = _run(tmp_path, sessions=(a, b), name="two")
    assert rc == 0
    _, one = _run(tmp_path, name="one")
    s2, s1 = (json.loads((d / "open_summary.json").read_text()) for d in (out, one))
    assert s2["counts"]["runs_valid"] == 12 and s2["counts"]["runs_duplicate"] == 0
    s2["counts"] = s1["counts"]
    _close(s2, s1)
    rows = C.read_csv(out / "screen.csv", "screen")
    assert {r["session_id"] for r in rows} == {"sA", "sB"} and len(rows) == 12


def test_duplicate_condition_in_a_later_session_is_skipped(tmp_path, capsys):
    b = tmp_path / "b"
    shutil.copytree(SESSION, b)
    _edit_session(b, session_id="sB")
    rc, out = _run(tmp_path, sessions=(SESSION, b))
    assert rc == 0 and "duplicate" in capsys.readouterr().err
    s = json.loads((out / "open_summary.json").read_text())
    assert s["counts"]["runs_valid"] == 12 and s["counts"]["runs_duplicate"] == 12
    assert len(list(csv.DictReader(open(out / "open_conditions.csv")))) == 24
    dup = [r for r in C.read_csv(out / "screen.csv", "screen") if r["kept"] == 0]
    assert len(dup) == 12 and {r["reason"] for r in dup} == {"duplicate"}


def test_invalid_condition_is_counted_and_left_out(tmp_path):
    d = tmp_path / "s"
    shutil.copytree(SESSION, d)
    mp = d / "W_hi_r060_g0_r0" / "manifest.json"
    m = json.loads(mp.read_text())
    m.update(valid=False, invalid_reason="ClockError: test")
    mp.write_text(json.dumps(m))
    rc, out = _run(tmp_path, sessions=(d,), plans=())
    assert rc == 0
    s = json.loads((out / "open_summary.json").read_text())
    assert s["counts"]["runs_invalid"] == 1 and s["counts"]["runs_valid"] == 11
    assert "W_hi_r060_g0" not in s["per_condition"]
    bad = [r for r in C.read_csv(out / "screen.csv", "screen") if r["kept"] == 0]
    assert len(bad) == 1 and bad[0]["reason"].startswith("invalid:")


def test_git_commit_mismatch_warns_but_does_not_refuse(tmp_path, capsys):
    d = tmp_path / "s"
    shutil.copytree(SESSION, d)
    sj = json.loads((d / "session.json").read_text())
    sj["git"]["commit"] = "f" * 40
    (d / "session.json").write_text(json.dumps(sj))
    rc, _ = _run(tmp_path, sessions=(d,), plans=())
    err = capsys.readouterr().err
    assert rc == 0 and "warning" in err and "git.commit" in err


def test_matching_git_commit_gives_no_warning(tmp_path, capsys):
    rc, _ = _run(tmp_path)
    assert rc == 0 and "warning" not in capsys.readouterr().err


def test_refuse_when_one_of_two_sessions_is_bad(tmp_path):
    frz = json.loads(FREEZE.read_text())["created_wall_ns"]
    bad = _copy(tmp_path, t_wall_start_ns=frz)
    assert OL.main(["--sessions", str(SESSION), str(bad), "--freeze", str(FREEZE),
                    "--out", str(tmp_path / "o")]) == 3
    assert not (tmp_path / "o").exists()               # nothing written before every session passes
