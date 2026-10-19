"""E3a: exp2.model.game with StubSim only (no real simulator)."""
import pytest

from exp2 import contracts as C
from exp2.model import game as G
from exp2.tests import mocks as M

INF = C.K_INF
K4 = (1, 2, 3, INF)
S, L = "bert:5", "vision:24"
TYPES = M.make_types({S: {"tau": 0.05, "klass": "short"}, L: {"tau": 0.5, "klass": "long"}})
SPEC = C.PolicySpec("g", 1, 1)


def run(fn, sim, **kw):
    return fn(None, SPEC, TYPES, None, None, sim_fn=sim, **kw)


def row(sf, r=0.05):
    return {"mean_shortfall": sf, "mean_R": r}


def full_row(sf, r=0.05):
    return {"n": 1000, "mean_R": r, "p50_R": r, "p95_R": r, "p99_R": r, "complete_rate": 1.0, "truncation_rate": 0.0,
            "miss_rate": 0.0, "expired_rate": 0.0, "mean_completion": 1.0 - sf, "mean_shortfall": sf}


def short_table(K, short_by_k, reps_by_k=None):
    """Long class is identical everywhere; the short class cost depends only on k_short."""
    table = {}
    for ks in K:
        for kl in K:
            entry = {"per_type": {S: row(*short_by_k[ks]), L: row(0.0, 0.5)}}
            if reps_by_k:
                entry["reps"] = [{"seed": i, "energy_per_req_j": 1.0, "n_slope": 0.0, "tagged": None,
                                  "per_type": {S: full_row(sf, short_by_k[ks][1]), L: full_row(0.0, 0.5)}}
                                 for i, sf in enumerate(reps_by_k[ks])]
            table[(ks, kl)] = entry
    return M.StubSim(table)


def test_maps_reach_the_known_fixed_point_from_all_16_starts():
    sim = M.StubSim.from_response_maps({k: 3 for k in K4}, {k: 2 for k in K4}, K4)
    eqs = run(G.equilibria, sim)
    assert len(eqs) == 16 and len({tuple(e["start"]) for e in eqs}) == 16
    assert all(e["converged"] and e["cycle"] is None and (e["k_short"], e["k_long"]) == (3, 2) for e in eqs)


def test_two_cycle_is_detected():
    sim = M.StubSim.from_response_maps({1: 2, 2: 1}, {1: 1, 2: 2}, (1, 2))
    e = run(G.best_response, sim, K=(1, 2), start=(1, 1))
    assert e["converged"] is False and e["cycle"] == [[1, 1], [2, 2]]


def test_shortfall_gap_of_0004_is_decided_by_rt():
    sim = short_table((1, 2), {1: (0.100, 0.05), 2: (0.104, 0.02)})
    assert run(G.best_response, sim, K=(1, 2), start=(1, 1))["k_short"] == 2
    # a gap of 0.006 is decided by shortfall alone
    sim = short_table((1, 2), {1: (0.100, 0.05), 2: (0.106, 0.02)})
    assert run(G.best_response, sim, K=(1, 2), start=(2, 1))["k_short"] == 1


def test_overlapping_cis_are_decided_by_rt():
    # shortfall 0.1 vs 0.2 but the rep CIs overlap -> rt decides (k=2 has the lower mean_R)
    sim = short_table((1, 2), {1: (0.1, 0.05), 2: (0.2, 0.02)}, {1: [0.0, 0.2], 2: [0.1, 0.3]})
    assert run(G.best_response, sim, K=(1, 2), start=(1, 1))["k_short"] == 2


def test_full_tie_keeps_current_then_smaller_k():
    same = {k: (0.0, 0.05) for k in (1, 2, 3)}
    assert run(G.best_response, short_table((1, 2, 3), same), K=(1, 2, 3), start=(2, 2))["k_short"] == 2
    worse3 = {1: (0.0, 0.05), 2: (0.0, 0.05), 3: (0.5, 0.05)}
    assert run(G.best_response, short_table((1, 2, 3), worse3), K=(1, 2, 3), start=(3, 1))["k_short"] == 1


def test_poa_energy_uses_the_worst_equilibrium():
    sim = M.StubSim.from_response_maps({k: k for k in K4}, {k: k for k in K4}, K4)   # equilibria (k, k)
    g = run(G.poa, sim)
    assert g["worst"] == {"k_short": INF, "k_long": INF} and g["optimum"] == {"k_short": 1, "k_long": 1}
    assert g["poa_E"] == pytest.approx(1.45 / 1.0)
    assert g["poa_R"] is None


def test_poa_r_only_for_rt_without_alpha():
    sim = M.StubSim.from_response_maps({1: 1, 2: 2}, {1: 1, 2: 2}, (1, 2))
    kw = dict(K=(1, 2), player_cost="rt")
    assert run(G.poa, sim, cfg=C.SimConfig(alpha=None), **kw)["poa_R"] == pytest.approx(1.0)
    assert run(G.poa, sim, cfg=C.SimConfig(alpha=5.0), **kw)["poa_R"] is None
    assert run(G.poa, sim, cfg=C.SimConfig(alpha=None), K=(1, 2))["poa_R"] is None
