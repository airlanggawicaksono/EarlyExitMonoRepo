"""S4: end-to-end check of the CoExit pilot with the REAL simulator, ExitPolicy and runner (PLAN PL1 c, e, g).

Fake six-type table and stub slowdown/power, 60 requests, so every test runs in seconds. Checks: All-Final behaves
exactly like Policy(corun); All-Early runs only floor exits; EdgeServing never co-runs; every method sees the same
arrivals (common random numbers); the runner writes the frozen columns and the expected floor shares.
"""
import csv
from pathlib import Path

import pytest

from exp2 import contracts as C
from exp2 import exits as X
from exp2.contracts import PolicySpec
from exp2.model import coexit_pilot as P
from exp2.model.simulator import simulate
from exp2.tests import mocks

FROZEN = Path(__file__).resolve().parents[1] / "frozen" / "run2"   # only hashed by the runner, never loaded here
TYPES = {"bert:5": {"tau": 0.02, "klass": "short"}, "bert:23": {"tau": 0.08, "klass": "short"},
         "yolo:1": {"tau": 0.04, "klass": "short"}, "yolo:5": {"tau": 0.06, "klass": "short"},
         "llama:3": {"tau": 0.6, "klass": "long"}, "llama:15": {"tau": 1.6, "klass": "long"}}
DEADLINES = {"bert": 0.1, "yolo": 0.1, "llama": 3.35}
FLOORS = {"bert:5", "yolo:1", "llama:3"}
N, W, SEED = 60, 10, 20261022


@pytest.fixture(scope="module")
def T():
    return mocks.make_types(TYPES, f_levels=(612, 816, 1020))


@pytest.fixture(scope="module")
def SD():
    return mocks.StubSlowdown({1: 1.0, 2: 1.1, 3: 1.15})


@pytest.fixture(scope="module")
def PW():
    return mocks.StubPower(dp=0.0)


def _items(T, mix="B", rho=0.9):
    wl = P._workload(mix, "poisson", 1.0, rho, T)
    return wl, P._schedule(wl, N, SEED, DEADLINES, 1.0)


def _trace(T, SD, PW, wl, items, spec, policy):
    cfg = C.SimConfig(n_requests=len(items), n_warmup=W, seed=SEED, alpha=None, schedule=items)
    return simulate(wl, spec, T, SD, PW, cfg, policy=policy).trace


def _run(T, SD, PW, method, mix="B", rho=0.9):
    wl, items = _items(T, mix, rho)
    spec = X.pilot_specs(1020)[method]
    return _trace(T, SD, PW, wl, items, spec, X.ExitPolicy(spec, T, SD))


@pytest.mark.parametrize("mix", ["A", "B"])
def test_all_final_is_policy_corun(T, SD, PW, mix):
    wl, items = _items(T, mix)
    corun = PolicySpec.from_json({"name": "corun", "k_short": None, "k_long": None, "order": "fcfs",
                                  "lane_cap": None, "f_rule": "static", "f_mhz": 1020, "params": {}})
    old = _trace(T, SD, PW, wl, items, corun, None)
    spec = X.pilot_specs(1020)["all_final"]
    new = _trace(T, SD, PW, wl, items, spec, X.ExitPolicy(spec, T, SD))
    assert len(old) == len(new) == N
    assert [dict(r) for r in old] == [dict(r) for r in new]


def test_all_early_runs_only_floor_exits(T, SD, PW):
    rows = _run(T, SD, PW, "all_early")
    ran = [r["run_type_id"] for r in rows if r["run_type_id"] is not None]
    assert ran and set(ran) <= FLOORS


def test_edgeserving_never_co_runs(T, SD, PW):
    rows = _run(T, SD, PW, "edgeserving", rho=1.2)
    admitted = [r for r in rows if r["n_in_service_at_admit"] is not None]
    assert admitted and all(r["n_in_service_at_admit"] == 0 for r in admitted)


@pytest.mark.parametrize("method", ["greedy", "coexit"])
def test_greedy_and_coexit_run_menu_exits_of_the_right_family(T, SD, PW, method):
    rows = _run(T, SD, PW, method, rho=1.2)
    for r in rows:
        if r["run_type_id"] is not None:
            fam, ex = C.parse_type_id(r["run_type_id"])
            assert fam == r["family"] and ex in C.EXIT_MENU[fam]


def test_every_method_sees_the_same_arrivals(T, SD, PW):
    seen = {}
    for m in X.METHODS:
        rows = _run(T, SD, PW, m)
        seen[m] = [(r["req_id"], r["type_id"], r["t_arrival_ns"], r["t_deadline_ns"]) for r in rows]
    first = seen[X.METHODS[0]]
    assert all(v == first for v in seen.values())


def test_runner_end_to_end_with_the_real_simulator(T, SD, PW, tmp_path):
    a = P.parse_args(["--types", str(FROZEN / "types.json"), "--slowdown", str(FROZEN / "slowdown.json"),
                      "--power", str(FROZEN / "power.json"), "--out", str(tmp_path / "out"),
                      "--deadlines", "bert=0.1,yolo=0.1,llama=3.35", "--mixes", "B", "--rhos", "0.9",
                      "--gamma-mix", "", "--reps", "2", "--n-requests", str(N), "--n-warmup", str(W)])
    P._check(a)
    out = P.run_pilot(a, P.Inputs(T, SD, PW))
    with open(out / "reps.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert tuple(rows[0]) == C.COEXIT_PILOT_COLS
    assert len(rows) == len(X.METHODS) * 2 * 4            # 5 methods x 2 reps x (bert, llama, yolo, all)
    share = {(r["method"], r["family"], r["rep"]): r["share_floor"] for r in rows}
    for rep in ("0", "1"):
        assert float(share[("all_early", "all", rep)]) == 1.0
        assert float(share[("all_final", "all", rep)]) == 0.0
    for name in ("summary.csv", "meta.json"):
        assert (out / name).is_file()
