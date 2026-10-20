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
    eqs = run(G.equilibria, sim, K=K4)
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
    g = run(G.poa, sim, K=K4)
    assert g["worst"] == {"k_short": INF, "k_long": INF} and g["optimum"] == {"k_short": 1, "k_long": 1}
    assert g["poa_E"] == pytest.approx(1.45 / 1.0)
    assert g["poa_R"] is None
    assert g["equilibria"] == [{"k_short": k, "k_long": k} for k in K4]


def test_poa_r_only_for_rt_without_alpha():
    sim = M.StubSim.from_response_maps({1: 1, 2: 2}, {1: 1, 2: 2}, (1, 2))
    kw = dict(K=(1, 2), player_cost="rt")
    assert run(G.poa, sim, cfg=C.SimConfig(alpha=None), **kw)["poa_R"] == pytest.approx(1.0)
    assert run(G.poa, sim, cfg=C.SimConfig(alpha=5.0), **kw)["poa_R"] is None
    assert run(G.poa, sim, cfg=C.SimConfig(alpha=None), K=(1, 2))["poa_R"] is None


# ---- D32-8: 9 profiles, pure Nash with tolerance tau, constrained optimum, poa_eps
def test_defaults_are_the_contract_k_set_and_tolerance():
    assert G.DEFAULT_K == C.K_SET and len(C.K_SET) == 3 and G.SHORTFALL_TIE == C.GAME_TIE_TOL == 0.005


def test_enumerates_all_nine_profiles_once():
    sim = M.StubSim.from_response_maps({k: k for k in C.K_SET}, {k: k for k in C.K_SET}, C.K_SET)
    g = run(G.poa, sim)
    assert len({c.key[:2] for c in sim.calls}) == 9 and len(sim.calls) == 9      # memoised
    assert g["equilibria"] == [{"k_short": k, "k_long": k} for k in C.K_SET]
    assert g["tie_tol"] == C.GAME_TIE_TOL


def test_all_equilibria_are_reported_and_worst_has_highest_energy():
    # two pure equilibria (1,1) and (INF,INF): energy 1.0 and 1.3
    sim = M.StubSim.from_response_maps({1: 1, 2: 2, INF: INF}, {1: 1, 2: 2, INF: INF}, C.K_SET)
    g = run(G.poa, sim)
    assert len(g["equilibria"]) == 3
    assert g["worst"] == {"k_short": INF, "k_long": INF} and g["optimum"] == {"k_short": 1, "k_long": 1}
    assert g["poa_E"] == pytest.approx(1.3)


def test_no_pure_equilibrium_gives_empty_block_values():
    sim = M.StubSim.from_response_maps({1: 2, 2: 1}, {1: 1, 2: 2}, (1, 2))
    g = run(G.poa, sim, K=(1, 2))
    assert g["equilibria"] == [] and g["worst"] == {} and g["poa_E"] is None and g["poa_eps"] is None
    assert g["optimum_constrained"] is None      # stub complete_rate = 1 - shortfall < 0.95 in every profile


def table9(energy, complete, sf=lambda ks, kl: (0.0, 0.0), reps=None):
    t = {}
    for ks in C.K_SET:
        for kl in C.K_SET:
            s_sf, l_sf = sf(ks, kl)
            c = complete(ks, kl)
            t[(ks, kl)] = {"per_type": {S: {"mean_shortfall": s_sf, "mean_R": 0.05, "complete_rate": c,
                                            "truncation_rate": 1.0 - c},
                                        L: {"mean_shortfall": l_sf, "mean_R": 0.5, "complete_rate": 1.0}},
                           "energy_per_req_j": energy(ks, kl), **({"reps": reps(ks, kl)} if reps else {})}
    return M.StubSim(t)


def test_gap_within_tau_is_not_an_improvement_but_above_tau_is():
    # long class never cares; short class: shortfall 0.004 more at k=1 than at k=2 -> tie, rt equal -> (1, *) stays
    def sf(ks, kl):
        return ({1: 0.004, 2: 0.0, INF: 0.0}[ks], 0.0)
    g = run(G.poa, table9(lambda a, b: 1.0, lambda a, b: 1.0, sf))
    assert {"k_short": 1, "k_long": 1} in g["equilibria"]
    # a gap of 0.006 is an improvement: (1, *) is no longer an equilibrium
    def sf2(ks, kl):
        return ({1: 0.006, 2: 0.0, INF: 0.0}[ks], 0.0)
    g = run(G.poa, table9(lambda a, b: 1.0, lambda a, b: 1.0, sf2))
    assert all(e["k_short"] != 1 for e in g["equilibria"])


def test_constrained_optimum_and_poa_eps():
    # equilibrium (INF, INF) at energy 2.0 meets the rule; cheapest profile (1,1) at 1.0 fails it (complete 0.90)
    def energy(a, b):
        return 1.0 if (a, b) == (1, 1) else (2.0 if (a, b) == (INF, INF) else 1.5)

    def complete(a, b):
        return 0.90 if (a, b) == (1, 1) else 1.0
    sim = M.StubSim.from_response_maps({k: INF for k in C.K_SET}, {k: INF for k in C.K_SET}, C.K_SET)
    # reuse the maps for shortfall, override energy and complete rate on top
    t = {}
    for key, entry in sim.table.items():
        e = dict(entry)
        e["energy_per_req_j"] = energy(*key)
        e["per_type"] = {tid: {**v, "complete_rate": complete(*key), "truncation_rate": 1.0 - complete(*key)} for tid, v in e["per_type"].items()}
        t[key] = e
    g = run(G.poa, M.StubSim(t))
    assert g["equilibria"] == [{"k_short": INF, "k_long": INF}]
    assert g["optimum"] == {"k_short": 1, "k_long": 1} and g["poa_E"] == pytest.approx(2.0)
    oc = g["optimum_constrained"]
    assert oc is not None and (oc["k_short"], oc["k_long"]) != (1, 1)
    assert g["poa_eps"] == pytest.approx(2.0 / 1.5)


def test_no_feasible_profile_gives_null_poa_eps():
    g = run(G.poa, table9(lambda a, b: 1.0, lambda a, b: 0.5))
    assert g["equilibria"] and g["optimum_constrained"] is None and g["poa_eps"] is None
    assert g["tie_tol"] == C.GAME_TIE_TOL


def test_one_sided_lower_bound_uses_replications():
    # point estimate 0.96 meets 0.95, but replications 0.99/0.93 give mean 0.96, sd .0424 -> lower bound < 0.95
    def rep(ks, kl):
        return [{"seed": i, "energy_per_req_j": 1.0, "n_slope": 0.0, "tagged": None,
                 "per_type": {S: {**full_row(0.0), "complete_rate": c, "truncation_rate": 1.0 - c}, L: full_row(0.0)}}
                for i, c in enumerate((0.99, 0.93))]
    sim = table9(lambda a, b: 1.0, lambda a, b: 0.96, reps=rep)
    assert run(G.poa, sim)["optimum_constrained"] is None
    # without replications the point estimate is used
    assert run(G.poa, table9(lambda a, b: 1.0, lambda a, b: 0.96))["optimum_constrained"] is not None


def test_block_passes_the_freeze_game_validator_shape():
    g = run(G.poa, M.StubSim.from_response_maps({k: k for k in C.K_SET}, {k: k for k in C.K_SET}, C.K_SET))
    assert set(g) == {"player_cost", "equilibria", "worst", "optimum", "poa_E", "poa_R", "optimum_constrained",
                      "poa_eps", "tie_tol"}
    assert all(set(e) == {"k_short", "k_long"} for e in g["equilibria"])


def test_class_without_requests_is_not_a_player():
    # mix A of Run 2 has no long-class type: the long class never improves, only k_short decides
    table = {(ks, kl): {"per_type": {S: row(0.0, 0.05 if ks == 2 else 0.08)}} for ks in C.K_SET for kl in C.K_SET}
    eqs = run(G.nash_equilibria, M.StubSim(table), K=C.K_SET, player_cost="shortfall_then_rt")
    assert eqs and all(e["k_short"] == 2 for e in eqs) and len(eqs) == len(C.K_SET)


def test_rt_tie_within_one_percent_is_not_an_improvement():
    table = {(ks, kl): {"per_type": {S: row(0.0, 0.0500 if ks == 1 else 0.0498), L: row(0.0, 0.5)}}
             for ks in C.K_SET for kl in C.K_SET}
    eqs = run(G.nash_equilibria, M.StubSim(table), K=C.K_SET, player_cost="shortfall_then_rt")
    assert {e["k_short"] for e in eqs} == set(C.K_SET)      # a 0.4% faster mean R is a tie
