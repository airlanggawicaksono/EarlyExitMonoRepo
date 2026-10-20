"""E3b tests: exp2.model.baselines.select_baseline (hand-computed choices, StubSim only)."""
import pytest

from exp2.contracts import ContractError, K_INF
from exp2.model.baselines import BASELINES, select_baseline
from exp2.policies import registry
from exp2.tests.mocks import StubSim, make_types

SPEC = {"bert:5": {"tau": {612: 0.06, 1020: 0.03}}, "vision:24": {"tau": {612: 0.2, 1020: 0.1}}}
IDS = ["bert:5", "vision:24"]


def entry(energy, cr=1.0, mean_R=0.05):
    row = {"complete_rate": cr, "truncation_rate": 1.0 - cr, "mean_completion": cr, "mean_R": mean_R}
    return {"per_type": {t: dict(row) for t in IDS}, "energy_per_req_j": energy}


def key(sel):
    s = sel.spec
    return (s.k_short, s.k_long, s.order, s.f_mhz)


def test_static_rules_are_the_registry_specs():
    types = make_types(SPEC)
    reg = registry(types)
    for n in ("default", "default_stock", "serial", "serial_stf", "serial_edgeserving", "corun", "greenllm_lanes",
              "occupancy_f"):
        assert select_baseline(n, types, IDS).spec == reg[n]
    ps = select_baseline("priority_serial", types, IDS).spec
    assert (ps.k_short, ps.k_long, ps.order, ps.f_mhz) == (1, 1, "class", 1020)
    with pytest.raises(ContractError):
        select_baseline("nope", types, IDS)
    assert len(set(BASELINES)) == len(BASELINES)


def test_solo_profile_freq_lowest_level_meeting_every_deadline():
    # alpha 2.5: D = 0.075 / 0.25 ; at 612 solo p95 = 0.06 / 0.2 -> ok.  alpha 1.5: D = 0.045 / 0.15 -> 612 fails.
    assert key(select_baseline("solo_profile_freq", make_types(SPEC, alpha=2.5), IDS)) == (1, 1, "fcfs", 612)
    assert key(select_baseline("solo_profile_freq", make_types(SPEC, alpha=1.5), IDS)) == (1, 1, "fcfs", 1020)


def test_probe_margin_formula():
    types = make_types(SPEC, alpha=3.0)                 # D = 0.09 / 0.30
    # 612: bert 1.035*0.06 + 0.01*1020/612 = 0.0788 <= 0.09 ; vision 1.035*0.2 + 0.02*1.6667 = 0.2403 <= 0.30
    sel = select_baseline("probe_margin", types, IDS, probe_margins={"bert:5": 0.01, "vision:24": 0.02})
    assert key(sel) == (K_INF, K_INF, "fcfs", 612) and sel.feasible
    # vision M = 0.1: 612 gives 0.207 + 0.1667 = 0.3737 > 0.30 ; 1020 gives 0.1035 + 0.1 = 0.2035 ok
    sel = select_baseline("probe_margin", types, IDS, probe_margins={"bert:5": 0.01, "vision:24": 0.1})
    assert key(sel)[3] == 1020
    with pytest.raises(ContractError):
        select_baseline("probe_margin", types, IDS)


def test_search_rules_restrict_the_grid():
    types = make_types(SPEC)
    sim = StubSim({}, default=entry(1.0))
    sel = select_baseline("freq_only", types, IDS, sim_fn=sim)
    assert {(c.spec.k_short, c.spec.k_long) for c in sim.calls} == {(a, b) for a in (1, K_INF) for b in (1, K_INF)}
    sim = StubSim({}, default=entry(1.0))
    select_baseline("conc_only", types, IDS, sim_fn=sim)
    assert {c.spec.f_mhz for c in sim.calls} == {1020}
    sim = StubSim({(2, K_INF, "stf", 612): entry(0.4)}, default=entry(1.0, cr=0.5))   # K_SET has no k=3 (R30)
    sel = select_baseline("F_M3", types, IDS, sim_fn=sim)
    assert sel.name == "F_M3" and key(sel) == (2, K_INF, "stf", 612) and sel.fresult.feasible


def test_capacity_one_uses_mean_response_at_fmax():
    types = make_types(SPEC)                            # D = 0.15 / 0.5 at alpha 5
    sim = StubSim({(1, 1, "fcfs", 1020): entry(0.1, mean_R=0.6),       # cheapest but mean R > D
                   (2, 1, "stf", 1020): entry(0.7, mean_R=0.1)}, default=entry(2.0, mean_R=0.2))
    sel = select_baseline("capacity_one", types, IDS, sim_fn=sim)
    assert {c.spec.f_mhz for c in sim.calls} == {1020}
    assert key(sel) == (2, 1, "stf", 1020) and sel.feasible
