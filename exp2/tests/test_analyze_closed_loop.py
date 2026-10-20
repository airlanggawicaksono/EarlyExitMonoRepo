"""F1 closed-loop analyzer (DoD D12): session_closed -> run1_csv exactly, window drop, warmup energy, determinism."""
import json
import shutil
from pathlib import Path

import pytest

from exp2 import contracts as C
from exp2 import stats as S
from exp2.analyze import closed_loop as CL

FIX = Path(__file__).resolve().parent / "fixtures"
KINDS = tuple(C.RUN1_CSV_COLS)


@pytest.fixture(scope="module")
def produced(tmp_path_factory):
    out = tmp_path_factory.mktemp("run1")
    # the fixture has a 9-request solo condition; AM1 (minimum 10) is tested separately below
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(C, "MIN_MEASURED_REQ_PER_TENANT", 1)
        tables, skipped = CL.analyze_sessions([FIX / "session_closed"])
    CL.write_run1_csv(tables, out)
    return out, tables, skipped


@pytest.mark.parametrize("kind", KINDS)
def test_matches_fixture(produced, kind):
    got = C.read_csv(produced[0] / (kind + ".csv"), kind)
    want = C.read_csv(FIX / "run1_csv" / (kind + ".csv"), kind)
    assert len(got) == len(want) > 0
    for g, w in zip(got, want):
        for col in C.RUN1_CSV_COLS[kind]:
            if isinstance(w[col], float):
                assert g[col] == pytest.approx(w[col], rel=1e-9, abs=1e-12), (kind, col)
            else:
                assert g[col] == w[col], (kind, col)


def test_nothing_skipped_and_half_outside_samples_dropped(produced):
    _, tables, skipped = produced
    assert skipped == []
    assert any(r["n_window"] < r["n"] for r in tables["corun"])


def test_warmup_energy_is_power_above_idle_pre(produced):
    session = C.validate_session(__import__("json").load(open(FIX / "session_closed" / "session.json")))
    power = C.read_csv(FIX / "session_closed" / "telemetry" / "power.csv", "power")
    pre = session["idle"]["pre"]
    p_idle = S.energy_j(power, pre["t0_ns"], pre["t1_ns"]) / ((pre["t1_ns"] - pre["t0_ns"]) / 1e9)
    for r in produced[1]["warmup"]:
        sv = session["servers"][r["family"]]
        want = S.energy_j(power, sv["t_spawn_ns"], sv["t_ready_ns"]) - p_idle * (sv["t_ready_ns"] - sv["t_spawn_ns"]) / 1e9
        assert r["warm_energy_j"] == pytest.approx(want, rel=1e-9) and want > 0


def test_deterministic_with_seed():
    a, _ = CL.analyze_sessions([FIX / "session_closed"])
    b, _ = CL.analyze_sessions([FIX / "session_closed"])
    assert a == b


def test_cli_ok_and_missing_session(tmp_path):
    assert CL.main(["--sessions", str(FIX / "session_closed"), "--out", str(tmp_path / "o")]) == 0
    assert (tmp_path / "o" / "mem.csv").exists() and (tmp_path / "o" / "screen.csv").exists()
    assert CL.main(["--sessions", str(tmp_path / "nope"), "--out", str(tmp_path / "o2")]) == 2


# ----------------------------------------------------------------------------------------- AM1, screening, merge
def _copy_session(tmp_path, name, session_id):
    dst = tmp_path / name
    shutil.copytree(FIX / "session_closed", dst)
    meta = json.loads((dst / "session.json").read_text(encoding="utf-8"))
    meta["session_id"] = session_id
    (dst / "session.json").write_text(json.dumps(meta), encoding="utf-8")
    return dst


def _keep_measured(sess, cid, type_id, k):
    """Keep every warm-up row and only the first k measured rows of type_id in <cid>/requests.jsonl."""
    path = sess / cid / "requests.jsonl"
    out, seen = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["type_id"] == type_id and not r["is_warmup"]:
            seen += 1
            if seen > k:
                continue
        out.append(line)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _screen(tables):
    return {r["condition_id"]: r for r in tables["screen"]}


def test_am1_few_requests_solo_is_skipped_not_raised():
    tables, skipped = CL.analyze_sessions([FIX / "session_closed"])      # real minimum 10; vision:24 f612 has 9
    why = {c: w for _, c, w in skipped}
    assert why["solo_vision-24_f612_r0"] == C.few_requests_reason({"vision:24": 9}) == "few_requests:vision:24=9"
    scr = _screen(tables)
    assert scr["solo_vision-24_f612_r0"]["kept"] == 0
    assert scr["solo_vision-24_f612_r0"]["reason"] == "few_requests:vision:24=9"
    assert ("vision:24", 612, 0) not in {(r["type_id"], r["f_mhz"], r["repeat"]) for r in tables["solo"]}


def test_corun_without_solo_baseline_is_skipped_with_reason(tmp_path):
    sess = _copy_session(tmp_path, "a", "s_a")
    mpath = sess / "solo_vision-24_f612_r0" / "manifest.json"
    man = json.loads(mpath.read_text(encoding="utf-8"))
    man.update(valid=False, invalid_reason="board unlocked")
    mpath.write_text(json.dumps(man), encoding="utf-8")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(C, "MIN_MEASURED_REQ_PER_TENANT", 1)
        tables, skipped = CL.analyze_sessions([sess])
    why = {c: w for _, c, w in skipped}
    assert why["solo_vision-24_f612_r0"].startswith("manifest invalid")
    assert why["c001_f612_r0"] == "no_solo_baseline:vision:24@612"
    assert _screen(tables)["c001_f612_r0"]["kept"] == 0
    assert {r["f_mhz"] for r in tables["corun"]} == {1020}


def test_am1_corun_tenant_with_few_window_requests_is_skipped(tmp_path):
    sess = _copy_session(tmp_path, "a", "s_a")
    _keep_measured(sess, "c001_f1020_r0", "vision:24", 3)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(C, "MIN_MEASURED_REQ_PER_TENANT", 5)
        tables, skipped = CL.analyze_sessions([sess])
    why = {c: w for _, c, w in skipped}
    # window shrinks to the 3 vision requests; bert (sorted first) then has only 4 fully inside
    assert why["c001_f1020_r0"] == "few_requests:bert:5=4"
    assert _screen(tables)["c001_f1020_r0"]["kept"] == 0
    assert all(r["f_mhz"] != 1020 for r in tables["corun"])


def test_am1_tenant_with_no_requests_is_skipped_not_raised(tmp_path):
    sess = _copy_session(tmp_path, "a", "s_a")
    _keep_measured(sess, "c001_f1020_r0", "vision:24", 0)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(C, "MIN_MEASURED_REQ_PER_TENANT", 1)
        tables, skipped = CL.analyze_sessions([sess])
    assert {c: w for _, c, w in skipped}["c001_f1020_r0"] == "few_requests:vision:24=0"


def test_screen_rows_cover_every_condition_and_roundtrip(tmp_path):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(C, "MIN_MEASURED_REQ_PER_TENANT", 1)
        tables, skipped = CL.analyze_sessions([FIX / "session_closed"])
    assert len(tables["screen"]) == 8 and skipped == []
    assert all(r["session_id"] == "s20261020T101500" and r["kept"] == 1 and r["reason"] is None
               for r in tables["screen"])
    assert {r["mode"] for r in tables["screen"]} == {"idle", "closed"}
    CL.write_run1_csv(tables, tmp_path)
    back = C.read_csv(tmp_path / "screen.csv", "screen")
    assert len(back) == 8 and list(back[0]) == list(C.SCREEN_CSV_COLS)
    row = {r["condition_id"]: r for r in back}["c001_f1020_r0"]
    assert row["arm"] == "pinned" and row["gpu_mhz"] == 1020 and row["n_clock_rows"] > 0


def test_two_sessions_shift_repeats_and_use_same_session_baseline(tmp_path):
    a = _copy_session(tmp_path, "a", "s_a")
    b = _copy_session(tmp_path, "b", "s_b")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(C, "MIN_MEASURED_REQ_PER_TENANT", 1)
        one, _ = CL.analyze_sessions([a])
        both, skipped = CL.analyze_sessions([a, b])
    assert skipped == []
    assert len(both["solo"]) == 2 * len(one["solo"])
    assert sorted({r["repeat"] for r in both["solo"]}) == [0, 1]
    assert sorted({r["repeat"] for r in both["corun"]}) == [0, 1]
    assert len(both["corun"]) == 2 * len(one["corun"])
    assert {r["session_id"] for r in both["screen"]} == {"s_a", "s_b"} and len(both["screen"]) == 16
    # identical copies: same-session baseline equals the pooled one
    assert {r["tau_solo_p50"] for r in both["corun"]} == {r["tau_solo_p50"] for r in one["corun"]}


def test_am1_drop_happens_before_repeat_shift(tmp_path):
    a = _copy_session(tmp_path, "a", "s_a")
    b = _copy_session(tmp_path, "b", "s_b")
    _keep_measured(a, "solo_bert-5_f612_r0", "bert:5", 2)       # session a loses this solo
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(C, "MIN_MEASURED_REQ_PER_TENANT", 5)
        tables, skipped = CL.analyze_sessions([a, b])
    bert612 = [r for r in tables["solo"] if (r["type_id"], r["f_mhz"]) == ("bert:5", 612)]
    assert [r["repeat"] for r in bert612] == [0]
    assert [(s, c) for s, c, _ in skipped if c == "solo_bert-5_f612_r0"] == [("s_a", "solo_bert-5_f612_r0")]


def test_cli_prints_skip_summary(tmp_path, capsys):
    assert CL.main(["--sessions", str(FIX / "session_closed"), "--out", str(tmp_path / "o")]) == 0
    err = capsys.readouterr().err
    assert "few_requests:vision:24=9" in err and "kept 6 of 8 conditions" in err and "few_requests=2" in err
