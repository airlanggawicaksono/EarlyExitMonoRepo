"""CellSlowdown: the simulator's 'real world' slowdown read from the measured Run 1 co-run cells.

Synthetic part: tiny corun.csv / corun_samples.csv in tmp_path, expected values worked out by hand below.
Frozen part: exp2/frozen/run2/run1_csv, expected values copied from the CSV rows (see the comments).
"""
import random
from pathlib import Path

import numpy as np
import pytest

from exp2 import contracts as C
from exp2.model.cell_slowdown import CellSlowdown

FROZEN = Path(__file__).resolve().parents[1] / "frozen" / "run2" / "run1_csv"


class StubFallback:
    """Records every call; predict -> 9.0, sample -> 8.0."""

    def __init__(self):
        self.calls = []

    def predict(self, i, A, f_mhz):
        self.calls.append(("predict", i, tuple(A), f_mhz))
        return 9.0

    def sample(self, i, A, f_mhz, rng):
        self.calls.append(("sample", i, tuple(A), f_mhz))
        return 8.0


def _write(path, kind, rows):
    cols = C.RUN1_CSV_COLS[kind]
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join("" if r.get(c) is None else str(r[c]) for c in cols))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _corun(cell, split, f, rep, tid, A, s):
    return {"cell_id": cell, "split": split, "f_mhz": f, "repeat": rep, "type_id": tid, "family": tid.split(":")[0],
            "A": A, "n": 10, "n_window": 10, "s_p50": s, "s_mean": s, "s_p95": s, "s_p50_lo": s, "s_p50_hi": s,
            "ess": 10.0, "tau_solo_p50": 0.01, "U_A": 1.0, "window_s": 1.0, "energy_j": 1.0, "p_board_w": 10.0}


def _smp(cell, f, rep, tid, s):
    return {"cell_id": cell, "f_mhz": f, "repeat": rep, "type_id": tid, "s": s}


# Hand-derived expectations for the synthetic directory:
#   key (("a:1", "b:2"), 1000, "a:1"): s_p50 1.3, 1.1, 1.2 (cell x1, train) + 1.5 (cell x2, heldout)
#       -> median of [1.1, 1.2, 1.3, 1.5] = (1.2 + 1.3) / 2 = 1.25
#       samples 1.05, 1.15 (x1 r0), 1.25 (x1 r1), 1.45 (x2 r0) -> draws only from these four
#   key (("a:1", "b:2"), 1000, "b:2"): s_p50 2.0 -> 2.0; samples 2.1 only
A_KEY = "a:1|b:2"
SAMPLES_A1 = {1.05, 1.15, 1.25, 1.45}


@pytest.fixture
def run1_dir(tmp_path):
    _write(tmp_path / "corun.csv", "corun", [
        _corun("x1", "train", 1000, 0, "a:1", A_KEY, 1.3),
        _corun("x1", "train", 1000, 1, "a:1", A_KEY, 1.1),
        _corun("x1", "train", 1000, 2, "a:1", A_KEY, 1.2),
        _corun("x2", "heldout", 1000, 0, "a:1", A_KEY, 1.5),
        _corun("x1", "train", 1000, 0, "b:2", A_KEY, 2.0),
    ])
    _write(tmp_path / "corun_samples.csv", "corun_samples", [
        _smp("x1", 1000, 0, "a:1", 1.05),
        _smp("x1", 1000, 0, "a:1", 1.15),
        _smp("x1", 1000, 1, "a:1", 1.25),
        _smp("x2", 1000, 0, "a:1", 1.45),
        _smp("x1", 1000, 0, "b:2", 2.1),
    ])
    return tmp_path


@pytest.fixture
def cells(run1_dir):
    fb = StubFallback()
    return CellSlowdown.from_run1_dir(run1_dir, fb), fb


# ------------------------------------------------------------------------------------------- predict
def test_predict_is_median_of_s_p50_over_every_split_and_repeat(cells):
    m, fb = cells
    assert m.predict("a:1", ("a:1", "b:2"), 1000) == pytest.approx(1.25, abs=1e-12)
    assert m.predict("b:2", ("a:1", "b:2"), 1000) == pytest.approx(2.0, abs=1e-12)
    assert fb.calls == []


def test_predict_ignores_order_of_A_and_accepts_float_level(cells):
    m, fb = cells
    assert m.predict("a:1", ["b:2", "a:1"], 1000.0) == pytest.approx(1.25, abs=1e-12)
    assert fb.calls == []


def test_predict_alone_is_one_without_fallback(cells):
    m, fb = cells
    assert m.predict("a:1", ("a:1",), 1000) == 1.0
    assert m.predict("z:9", ("z:9",), 1) == 1.0
    assert fb.calls == []


def test_predict_empty_A_raises(cells):
    m, _ = cells
    with pytest.raises(ValueError):
        m.predict("a:1", (), 1000)


@pytest.mark.parametrize("i,A,f", [
    ("a:1", ("a:1", "b:2"), 900),                 # unmeasured level
    ("a:1", ("a:1", "b:2", "c:3"), 1000),         # unmeasured set
    ("a:1", ("a:1", "a:1", "b:2"), 1000),         # two a:1 jobs is not the measured one-stream cell
    ("c:3", ("a:1", "c:3"), 1000),                # unmeasured type
])
def test_predict_unmeasured_key_goes_to_fallback(cells, i, A, f):
    m, fb = cells
    assert m.predict(i, A, f) == 9.0
    assert fb.calls == [("predict", i, A, f)]
    assert m.coverage()["fallback_calls"] == 1


# ------------------------------------------------------------------------------------------- sample
@pytest.mark.parametrize("make_rng", [lambda: np.random.default_rng(0), lambda: random.Random(0)],
                         ids=["numpy_generator", "random_Random"])
def test_sample_draws_only_measured_values_and_reaches_all(cells, make_rng):
    m, fb = cells
    rng = make_rng()
    draws = [m.sample("a:1", ("b:2", "a:1"), 1000, rng) for _ in range(400)]
    assert set(draws) == SAMPLES_A1
    assert {m.sample("b:2", ("a:1", "b:2"), 1000, rng) for _ in range(20)} == {2.1}
    assert fb.calls == []


def test_sample_uses_integers_when_present():
    class IntegersOnly:
        def __init__(self):
            self.n = []

        def integers(self, n):
            self.n.append(n)
            return 0

    class RandrangeOnly:
        def __init__(self):
            self.n = []

        def randrange(self, n):
            self.n.append(n)
            return n - 1

    m = CellSlowdown({(("a:1", "b:2"), 1000, "a:1"): [1.2]},
                     {(("a:1", "b:2"), 1000, "a:1"): [1.05, 1.45]}, StubFallback())
    r1, r2 = IntegersOnly(), RandrangeOnly()
    assert m.sample("a:1", ("a:1", "b:2"), 1000, r1) in (1.05, 1.45)
    assert m.sample("a:1", ("a:1", "b:2"), 1000, r2) in (1.05, 1.45)
    assert r1.n == [2] and r2.n == [2]


def test_sample_alone_is_one_without_fallback(cells):
    m, fb = cells
    assert m.sample("a:1", ("a:1",), 1000, random.Random(0)) == 1.0
    assert fb.calls == []


def test_sample_unmeasured_key_goes_to_fallback(cells):
    m, fb = cells
    rng = random.Random(0)
    assert m.sample("a:1", ("a:1", "b:2"), 900, rng) == 8.0
    assert fb.calls == [("sample", "a:1", ("a:1", "b:2"), 900)]


# ------------------------------------------------------------------------------------------- coverage, errors
def test_coverage_counts_measured_keys_and_fallback_calls(cells):
    m, _ = cells
    assert m.coverage() == {"measured_keys": 2, "fallback_calls": 0}
    m.predict("a:1", ("a:1", "b:2"), 1000)
    m.sample("a:1", ("a:1", "b:2"), 1000, random.Random(0))
    m.predict("a:1", ("a:1",), 1000)
    assert m.coverage() == {"measured_keys": 2, "fallback_calls": 0}
    m.predict("a:1", ("a:1", "b:2"), 900)
    m.sample("a:1", ("a:1", "b:2"), 900, random.Random(0))
    assert m.coverage() == {"measured_keys": 2, "fallback_calls": 2}


def test_sample_row_with_unknown_cell_raises(run1_dir):
    _write(run1_dir / "corun_samples.csv", "corun_samples", [_smp("nope", 1000, 0, "a:1", 1.1)])
    with pytest.raises(ValueError, match="nope"):
        CellSlowdown.from_run1_dir(run1_dir, StubFallback())


def test_bad_header_raises(run1_dir):
    (run1_dir / "corun_samples.csv").write_text("cell_id,f_mhz,type_id,s\n", encoding="utf-8")
    with pytest.raises(C.ContractError):
        CellSlowdown.from_run1_dir(run1_dir, StubFallback())


# ------------------------------------------------------------------------------------------- frozen Run 1
@pytest.fixture(scope="module")
def frozen():
    fb = StubFallback()
    return CellSlowdown.from_run1_dir(FROZEN, fb), fb


def test_frozen_llama15_next_to_bert5_at_1020(frozen):
    # corun.csv d_c002, train: s_p50 1.16369201927, 1.18002099021, 1.16698364497 -> median 1.16698364497
    m, fb = frozen
    s = m.predict("llama:15", ("bert:5", "llama:15"), 1020)
    assert s == pytest.approx(1.16698364497, abs=1e-9)
    assert 1.16 <= s <= 1.19
    assert fb.calls == []


def test_frozen_llama3_next_to_bert5_at_1020(frozen):
    # corun.csv c001, train: s_p50 1.61383861486, 1.60837359768, 1.64452139255 -> median 1.61383861486
    m, fb = frozen
    s = m.predict("llama:3", ("bert:5", "llama:3"), 1020)
    assert s == pytest.approx(1.61383861486, abs=1e-9)
    assert 1.60 <= s <= 1.65
    assert fb.calls == []


def test_frozen_samples_stay_in_the_measured_range(frozen):
    # corun_samples.csv pools: llama:15 key 72 samples in [1.15393328763, 1.20532134746];
    # llama:3 key 71 samples in [1.53303362667, 1.70638695533]
    m, fb = frozen
    rng = np.random.default_rng(1)
    d15 = [m.sample("llama:15", ("llama:15", "bert:5"), 1020, rng) for _ in range(300)]
    d3 = [m.sample("llama:3", ("llama:3", "bert:5"), 1020, rng) for _ in range(300)]
    assert 1.15393328763 <= min(d15) and max(d15) <= 1.20532134746
    assert 1.53303362667 <= min(d3) and max(d3) <= 1.70638695533
    assert fb.calls == []


def test_frozen_has_144_measured_keys(frozen):
    # 20 sets x (2 or 3 types) x 3 levels: 12 pairs x 2 x 3 + 8 triples x 3 x 3 = 72 + 72 = 144 (48 per level)
    m, _ = frozen
    assert m.coverage()["measured_keys"] == 144
