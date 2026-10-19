"""F1 closed-loop analyzer (DoD D12): session_closed -> run1_csv exactly, window drop, warmup energy, determinism."""
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
    assert (tmp_path / "o" / "mem.csv").exists()
    assert CL.main(["--sessions", str(tmp_path / "nope"), "--out", str(tmp_path / "o2")]) == 2
