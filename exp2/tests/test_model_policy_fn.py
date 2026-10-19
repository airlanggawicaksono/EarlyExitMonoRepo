"""E3b tests: exp2.model.policy_fn.search_F with StubSim (tables worked out by hand)."""
import pytest

from exp2.contracts import ContractError, K_INF
from exp2.model.policy_fn import mem_footprint_mb, search_F
from exp2.tests.mocks import StubSim, make_types

SPEC = {"bert:5": {"tau": {612: 0.06, 1020: 0.03}}, "vision:24": {"tau": {612: 0.2, 1020: 0.1}}}
IDS = ["bert:5", "vision:24"]


def pt(cr, energy, mc=None):
    mc = cr if mc is None else mc
    row = {"complete_rate": cr, "truncation_rate": 1.0 - cr, "mean_completion": mc}
    return {"per_type": {t: dict(row) for t in IDS}, "energy_per_req_j": energy}


def stub(entries, base_cr=0.5):
    return StubSim(entries, default=pt(base_cr, 9.0))


def test_min_energy_feasible_wins_over_cheaper_infeasible():
    types = make_types(SPEC)
    sim = stub({(2, 2, "stf", 612): pt(0.97, 3.0), (K_INF, 1, "fcfs", 1020): pt(0.99, 2.0),
                (1, 1, "fcfs", 612): pt(0.90, 0.5)}, base_cr=0.1)
    r = search_F(types, IDS, sim)
    assert r.feasible
    assert (r.spec.k_short, r.spec.k_long, r.spec.order, r.spec.f_mhz) == (K_INF, 1, "fcfs", 1020)
    assert r.n_sim == 2 * 2 * 16 and len(sim.calls) == r.n_sim and r.compute_s >= 0


def test_infeasible_returns_highest_min_completion_with_flag():
    types = make_types(SPEC)
    sim = stub({(3, 1, "fcfs", 612): pt(0.8, 7.0), (1, 3, "stf", 1020): pt(0.7, 1.0)}, base_cr=0.2)
    r = search_F(types, IDS, sim)
    assert not r.feasible
    assert (r.spec.k_short, r.spec.k_long, r.spec.order, r.spec.f_mhz) == (3, 1, "fcfs", 612)
    assert r.min_metric == pytest.approx(0.8)


def test_constraint_choice_changes_the_point():
    types = make_types(SPEC)
    table = {(1, 1, "fcfs", 612): pt(0.97, 1.0, mc=0.90), (2, 2, "fcfs", 612): pt(0.96, 2.0, mc=0.97)}
    a = search_F(types, IDS, stub(table), constraint="complete_rate")
    b = search_F(types, IDS, stub(table), constraint="mean_completion")
    assert (a.spec.k_short, b.spec.k_short) == (1, 2)
    with pytest.raises(ContractError):
        search_F(types, IDS, stub(table), constraint="p95")


def test_memory_filter_removes_points_before_simulation():
    # resident 1300 MB for each of bert/vision/yolo; short = bert 38 + yolo 100, long = vision 51.
    types = make_types({"bert:5": {"tau": 0.03, "klass": "short", "act_mb": 38.0},
                        "yolo:2": {"tau": 0.03, "klass": "short", "act_mb": 100.0},
                        "vision:24": {"tau": 0.1, "klass": "long", "act_mb": 51.0}})
    ids = types.type_ids()
    assert mem_footprint_mb(types, ids, 1, 1) == pytest.approx(3900 + 100 + 51)
    assert mem_footprint_mb(types, ids, 2, K_INF) == pytest.approx(3900 + 138 + 51)
    sim = StubSim({}, default={"per_type": {t: {} for t in ids}, "energy_per_req_j": 1.0})
    r = search_F(types, ids, sim, mem_budget_mb=4060)
    assert r.n_skipped_mem == 48 and r.n_sim == 16     # k_short=1 only: 4 of 16 pairs x 4 (f, order)
    assert all(c.spec.k_short == 1 for c in sim.calls)


def test_levels_outside_corun_are_never_simulated():
    types = make_types(SPEC, f_levels=(612, 816, 1020), corun_levels=(612, 1020))
    sim = stub({}, base_cr=1.0)
    search_F(types, IDS, sim)
    assert {c.spec.f_mhz for c in sim.calls} == {612, 1020} and len(sim.calls) == 64
    bad = stub({}, base_cr=1.0)
    with pytest.raises(ContractError):
        search_F(types, IDS, bad, f_grid=(816,))
    assert bad.calls == []
