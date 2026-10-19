"""S1b tests: Policy.admit / Policy.desired_f / AdmissionCore / registry (BRIEF_v1 K4, K20).

Every expected value in the inline tests is derived by hand from the K4 text, NOT by calling the
code under test.  The golden replays (admission_cases.json, admission_traces.json, policyspecs.json)
are written by S0a/S0b and are replayed here without edits.
"""
import copy
import dataclasses
import heapq
import json
import time
from pathlib import Path

import pytest

from exp2 import contracts as C
from exp2.contracts import ContractError, DispatchView, InServiceReq, PolicySpec, QueuedReq
from exp2.policies import AdmissionCore, Decision, Policy, registry

REPO = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
NS = 1_000_000_000

# --------------------------------------------------------------------------------------
# Synthetic type table.  (p50@1020, p50@612, p95@1020, p95@612) seconds; mean == p50.
# median of the six p50@1020 is (0.03 + 0.08) / 2 = 0.055 -> short: bert:5 vision:6 yolo:1.
# --------------------------------------------------------------------------------------
BASE = {
    "bert:5": (0.030, 0.045, 0.040, 0.060),
    "bert:23": (0.120, 0.180, 0.150, 0.220),
    "vision:6": (0.020, 0.030, 0.025, 0.040),
    "vision:24": (0.080, 0.120, 0.100, 0.150),
    "yolo:1": (0.010, 0.015, 0.012, 0.018),
    "llama:15": (0.800, 1.200, 0.900, 1.300),
}
F_LEVELS = [408, 612, 714, 816, 1020]


def _interp(f, v612, v1020):
    if f <= 612:
        return v612 * 612.0 / f
    return v612 + (v1020 - v612) * (f - 612.0) / (1020.0 - 612.0)


def types_doc(base=None, alpha=5.0, levels=None, corun=None, mean_override=None):
    levels = list(levels or F_LEVELS)
    base = dict(base or BASE)
    means = sorted(v[0] for v in base.values())
    n = len(means)
    median = means[n // 2] if n % 2 else 0.5 * (means[n // 2 - 1] + means[n // 2])
    tys = []
    for tid, (p50_hi, p50_lo, p95_hi, p95_lo) in base.items():
        fam, ex = C.parse_type_id(tid)
        ckpts = C.default_checkpoints(fam, ex)
        m = len(ckpts)
        head = 0.0 if fam == "llama" else 0.10
        top = 1.0 - head
        tau = {}
        for f in levels:
            p50 = round(_interp(f, p50_lo, p50_hi), 9)
            p95 = round(_interp(f, p95_lo, p95_hi), 9)
            mean = (mean_override or {}).get(tid, p50)
            tau[str(f)] = {"p50": p50, "mean": mean, "p95": p95, "samples": [p50, p50, p95]}
        tys.append({
            "type_id": tid, "family": fam, "exit": ex,
            "klass": "short" if p50_hi <= median else "long",
            "ckpts": ckpts, "tau_s": tau,
            "ckpt_frac": {str(c): round(top * (i + 1) / m, 6) for i, c in enumerate(ckpts)},
            "head_frac": head, "u": {str(f): 0.6 for f in levels}, "act_mb": 10.0,
        })
    fams = sorted({t["family"] for t in tys})
    return {
        "schema": "exp2.types/1", "synthetic": True, "alpha": alpha,
        "f_levels_mhz": levels, "corun_levels_mhz": list(corun or [612, 1020]), "heldout_level_mhz": 714,
        "f_max_mhz": max(levels),
        "llama_tokens": 32, "llama_prompt_tokens": 128,
        "families": {f: {"resident_mb": 1000.0} for f in fams},
        "types": tys,
        "provenance": {"klass_rule": "short iff tau_mean(f_max) <= median over types",
                       "sources": ["synthetic"], "sha256": {"synthetic": "0" * 64}},
    }


def make_types(tmp_path, base=None, alpha=5.0, name="types.json", **kw):
    p = tmp_path / name
    p.write_text(json.dumps(types_doc(base, alpha, **kw)), encoding="utf-8")
    return C.load_types(str(p))


@pytest.fixture(scope="module")
def T(tmp_path_factory):
    return make_types(tmp_path_factory.mktemp("types"))


def lanes(short, long):
    """lane_cap value as the contract represents it (single place to adapt)."""
    return {"short": short, "long": long}


def spec_of(**kw):
    d = {"name": "t", "k_short": None, "k_long": None, "order": "fcfs", "lane_cap": None,
         "f_rule": "static", "f_mhz": 1020, "params": {}}
    d.update(kw)
    return PolicySpec.from_json(d)


def Q(T, rid, tid, a_ns, dl=None, ko=None):
    t = T.get(tid)
    return QueuedReq(req_id=rid, type_id=tid, family=t.family, klass=t.klass, t_arrival_ns=a_ns,
                     t_deadline_ns=dl, k_override=ko)


def S(T, rid, tid, t_admit_ns=0):
    t = T.get(tid)
    return InServiceReq(req_id=rid, type_id=tid, family=t.family, klass=t.klass, t_admit_ns=t_admit_ns)


def V(queue, in_service=(), now=1000, f=1020):
    return DispatchView(now_ns=now, queue=list(queue), in_service=list(in_service), f_current_mhz=f)


def admit(T, spec, view, **kw):
    return list(Policy(spec, T, **kw).admit(view))


# --------------------------------------------------------------------------------------
# Policy.admit
# --------------------------------------------------------------------------------------
class TestAdmitBasics:
    def test_empty_queue_admits_nothing(self, T):
        assert admit(T, spec_of(), V([])) == []

    def test_k1_idle_admits_only_the_head(self, T):
        q = [Q(T, "r1", "bert:5", 1), Q(T, "r2", "vision:6", 2)]
        assert admit(T, spec_of(k_short=1, k_long=1), V(q)) == ["r1"]

    def test_k1_busy_admits_nothing(self, T):
        q = [Q(T, "r1", "bert:5", 1)]
        assert admit(T, spec_of(k_short=1, k_long=1), V(q, [S(T, "r0", "vision:24")])) == []

    def test_busy_family_head_does_not_block_those_behind(self, T):
        # spec example k2_head_family_busy_nonblocking: n=1 -> r2 skipped (bert busy), r3 admitted
        # (1<2, n becomes 2), r4 refused (2<2 is false).
        q = [Q(T, "r2", "bert:23", 100), Q(T, "r3", "vision:6", 200), Q(T, "r4", "yolo:1", 300)]
        assert admit(T, spec_of(k_short=2, k_long=2), V(q, [S(T, "r1", "bert:5")])) == ["r3"]

    def test_second_request_of_same_family_waits_inside_one_call(self, T):
        q = [Q(T, "r1", "bert:5", 1), Q(T, "r2", "bert:23", 2), Q(T, "r3", "vision:6", 3)]
        assert admit(T, spec_of(), V(q)) == ["r1", "r3"]

    def test_unbounded_k_four_families_six_queued(self, T):
        q = [Q(T, "r1", "bert:5", 1), Q(T, "r2", "vision:6", 2), Q(T, "r3", "llama:15", 3),
             Q(T, "r4", "yolo:1", 4), Q(T, "r5", "bert:23", 5), Q(T, "r6", "vision:24", 6)]
        assert admit(T, spec_of(), V(q)) == ["r1", "r2", "r3", "r4"]

    def test_k_per_class_uses_the_class_of_the_candidate(self, T):
        # k_short=3, k_long=1, n=0.  r1 long: 0<1 admit (n=1).  r2 short: 1<3 admit (n=2).
        # r3 llama is long: 2<1 false.  r4 yolo short: 2<3 admit (n=3).
        q = [Q(T, "r1", "bert:23", 1), Q(T, "r2", "vision:6", 2), Q(T, "r3", "llama:15", 3),
             Q(T, "r4", "yolo:1", 4)]
        assert admit(T, spec_of(k_short=3, k_long=1), V(q)) == ["r1", "r2", "r4"]

    def test_long_candidate_blocked_by_class_limit_while_short_passes(self, T):
        # n=1 in service.  long limit 1 -> 1<1 false; short limit 3 -> 1<3 true.
        q = [Q(T, "r1", "bert:23", 1), Q(T, "r2", "vision:6", 2)]
        assert admit(T, spec_of(k_short=3, k_long=1), V(q, [S(T, "r0", "yolo:1")])) == ["r2"]

    def test_n_grows_inside_the_call(self, T):
        q = [Q(T, "r1", "bert:5", 1), Q(T, "r2", "vision:6", 2), Q(T, "r3", "yolo:1", 3)]
        assert admit(T, spec_of(k_short=2, k_long=2), V(q)) == ["r1", "r2"]

    def test_family_exclusive_false_allows_same_family_twice(self, T):
        q = [Q(T, "r1", "bert:5", 1), Q(T, "r2", "bert:23", 2), Q(T, "r3", "vision:6", 3)]
        assert admit(T, spec_of(), V(q), family_exclusive=False) == ["r1", "r2", "r3"]

    def test_family_exclusive_false_still_obeys_k(self, T):
        q = [Q(T, "r1", "bert:5", 1), Q(T, "r2", "bert:23", 2), Q(T, "r3", "vision:6", 3)]
        assert admit(T, spec_of(k_short=2, k_long=2), V(q), family_exclusive=False) == ["r1", "r2"]

    def test_family_exclusive_default_is_true(self, T):
        q = [Q(T, "r1", "bert:5", 1), Q(T, "r2", "bert:23", 2)]
        assert admit(T, spec_of(), V(q)) == ["r1"]

    def test_in_service_family_blocks_same_family_candidate(self, T):
        q = [Q(T, "r1", "bert:5", 1)]
        assert admit(T, spec_of(), V(q, [S(T, "r0", "bert:23")])) == []


class TestKOverride:
    def test_override_raises_the_limit(self, T):
        # spec k=1, n=1 -> r1 normally refused; k_override=2 -> 1<2 admitted.
        q = [Q(T, "r1", "bert:5", 1, ko=2)]
        assert admit(T, spec_of(k_short=1, k_long=1), V(q, [S(T, "r0", "vision:24")])) == ["r1"]

    def test_override_lowers_the_limit(self, T):
        # spec k=inf, n=1 -> r1 with k_override=1 -> 1<1 false.  r2 without override is admitted.
        q = [Q(T, "r1", "bert:5", 1, ko=1), Q(T, "r2", "yolo:1", 2)]
        assert admit(T, spec_of(), V(q, [S(T, "r0", "vision:24")])) == ["r2"]

    def test_override_only_applies_to_its_own_request(self, T):
        q = [Q(T, "r1", "bert:5", 1, ko=5), Q(T, "r2", "vision:6", 2)]
        # k=1, n=0: r1 0<5 admit (n=1); r2 1<1 false.
        assert admit(T, spec_of(k_short=1, k_long=1), V(q)) == ["r1"]


class TestOrders:
    def test_stf_orders_by_p50_at_fmax(self, T):
        # p50@1020: yolo:1 .010 < vision:6 .020 < bert:5 .030 < vision:24 .080 < bert:23 .120
        q = [Q(T, "r1", "bert:23", 1), Q(T, "r2", "vision:24", 2), Q(T, "r3", "vision:6", 3),
             Q(T, "r4", "bert:5", 4), Q(T, "r5", "yolo:1", 5)]
        # order: r5 yolo, r3 vision, r4 bert, r2, r1.  k=3: r5 (n0), r3 (n1), r4 (n2), then n=3 stop.
        assert admit(T, spec_of(order="stf", k_short=3, k_long=3), V(q)) == ["r5", "r3", "r4"]

    def test_stf_key_is_p50_not_p95_nor_mean(self, tmp_path):
        # vision:6 p50 .020 / p95 .100 / mean .500 ; yolo:1 p50 .030 / p95 .040 / mean .001.
        # by p50 vision:6 is first ; by p95 yolo:1 would be first ; by mean yolo:1 would be first.
        base = dict(BASE)
        base["vision:6"] = (0.020, 0.030, 0.100, 0.120)
        base["yolo:1"] = (0.030, 0.040, 0.040, 0.050)
        T2 = make_types(tmp_path, base, mean_override={"vision:6": 0.5, "yolo:1": 0.001})
        q = [Q(T2, "r1", "yolo:1", 1), Q(T2, "r2", "vision:6", 2)]
        assert admit(T2, spec_of(order="stf", k_short=1, k_long=1), V(q)) == ["r2"]

    def test_stf_uses_fmax_not_the_current_level(self, tmp_path):
        # yolo:1 .010@1020 / .015@612 ; vision:6 .005@1020 / 5.0@612.  At f_current=612 the order by the
        # current level would put yolo first; by f_max (the contract) vision:6 first.
        base = dict(BASE)
        base["vision:6"] = (0.005, 5.0, 0.006, 6.0)
        T2 = make_types(tmp_path, base)
        q = [Q(T2, "r1", "yolo:1", 1), Q(T2, "r2", "vision:6", 2)]
        assert admit(T2, spec_of(order="stf", k_short=1, k_long=1), V(q, f=612)) == ["r2"]

    def test_stf_tie_falls_back_to_arrival(self, tmp_path):
        base = dict(BASE)
        base["yolo:1"] = (0.020, 0.030, 0.025, 0.040)   # identical to vision:6
        T2 = make_types(tmp_path, base)
        spec = spec_of(order="stf", k_short=1, k_long=1)
        q1 = [Q(T2, "r1", "yolo:1", 1), Q(T2, "r2", "vision:6", 2)]
        q2 = [Q(T2, "r1", "vision:6", 1), Q(T2, "r2", "yolo:1", 2)]
        assert admit(T2, spec, V(q1)) == ["r1"]
        assert admit(T2, spec, V(q2)) == ["r1"]

    def test_class_puts_short_before_long_then_arrival(self, T):
        q = [Q(T, "r1", "bert:23", 1), Q(T, "r2", "vision:6", 2), Q(T, "r3", "llama:15", 3),
             Q(T, "r4", "yolo:1", 4)]
        assert admit(T, spec_of(order="class", k_short=2, k_long=2), V(q)) == ["r2", "r4"]
        # fcfs would have taken r1, r2
        assert admit(T, spec_of(order="fcfs", k_short=2, k_long=2), V(q)) == ["r1", "r2"]

    def test_class_keeps_arrival_order_inside_a_class(self, T):
        q = [Q(T, "r1", "vision:24", 1), Q(T, "r2", "bert:23", 2)]
        assert admit(T, spec_of(order="class", k_short=1, k_long=1), V(q)) == ["r1"]


GREEN = dict(k_short=2, k_long=2, f_rule="greenllm", f_mhz=None)


class TestLaneCap:
    def test_lane_cap_limits_in_service_per_class(self, T):
        spec = spec_of(lane_cap=lanes(1, 1), **GREEN)
        # one short already in service -> short lane full; long lane empty.
        q = [Q(T, "r1", "bert:5", 1), Q(T, "r2", "vision:24", 2)]
        assert admit(T, spec, V(q, [S(T, "r0", "yolo:1")])) == ["r2"]

    def test_lane_cap_counts_admits_inside_the_call(self, T):
        spec = spec_of(lane_cap=lanes(1, 1), **GREEN)
        q = [Q(T, "r1", "bert:5", 1), Q(T, "r2", "vision:6", 2), Q(T, "r3", "vision:24", 3)]
        # r1 short admitted (short lane 1/1); r2 short refused by lane cap; r3 long admitted (n=1<2).
        assert admit(T, spec, V(q)) == ["r1", "r3"]

    def test_lane_cap_larger_than_one(self, T):
        spec = spec_of(lane_cap=lanes(2, 1), **dict(GREEN, k_short=3, k_long=3))
        q = [Q(T, "r1", "bert:5", 1), Q(T, "r2", "vision:6", 2), Q(T, "r3", "yolo:1", 3)]
        assert admit(T, spec, V(q)) == ["r1", "r2"]

    def test_no_lane_cap_means_no_lane_limit(self, T):
        q = [Q(T, "r1", "bert:5", 1), Q(T, "r2", "vision:6", 2)]
        assert admit(T, spec_of(k_short=3, k_long=3), V(q)) == ["r1", "r2"]


# --------------------------------------------------------------------------------------
# EdgeServing
# --------------------------------------------------------------------------------------
ES = dict(order="edgeserving", k_short=1, k_long=1, params={"C": 10})
NOW = 50_000_000       # 0.05 s


class TestEdgeServing:
    def test_three_heads_smallest_total_urgency_wins(self, T):
        # (a, deadline) s: A bert:5 (0,.2)  B vision:6 (.02,.12)  C yolo:1 (.04,.09)  -> D = .2 .1 .05
        # p95@1020: A .040, B .025, C .012 ; waits at now=.05: .05 .03 .01.
        # S(A)=exp((.03+.04)/.1-1)+exp((.01+.04)/.05-1)=.7408+1      =1.7408
        # S(B)=exp((.05+.025)/.2-1)+exp((.01+.025)/.05-1)=.5353+.7408=1.2761
        # S(C)=exp((.05+.012)/.2-1)+exp((.03+.012)/.1-1)=.5016+.5599 =1.0615  -> C
        q = [Q(T, "rA", "bert:5", 0, dl=200_000_000), Q(T, "rB", "vision:6", 20_000_000, dl=120_000_000),
             Q(T, "rC", "yolo:1", 40_000_000, dl=90_000_000)]
        assert admit(T, spec_of(**ES), V(q, now=NOW)) == ["rC"]

    def test_winner_changes_when_the_urgency_changes(self, T):
        # rC relaxed (D=2 s, deadline 2.04 s): S(A)=.7408+exp(-.975)=1.1180 ; S(B)=.5353+exp(-.9825)=.9096 ;
        # S(C)=1.0615 -> B.
        q = [Q(T, "rA", "bert:5", 0, dl=200_000_000), Q(T, "rB", "vision:6", 20_000_000, dl=120_000_000),
             Q(T, "rC", "yolo:1", 40_000_000, dl=2_040_000_000)]
        assert admit(T, spec_of(**ES), V(q, now=NOW)) == ["rB"]

    def test_only_per_type_heads_are_candidates(self, T):
        # r1 bert:5 a=0 D=.2 ; r2 vision:6 a=.02 D=.05 ; r4 bert:5 a=.03 D=.03 (NOT a head).
        # Hand S: r1=4.2101 r2=2.1840 r4=2.0688.  If r4 were a candidate it would win.
        q = [Q(T, "r1", "bert:5", 0, dl=200_000_000), Q(T, "r2", "vision:6", 20_000_000, dl=70_000_000),
             Q(T, "r4", "bert:5", 30_000_000, dl=60_000_000)]
        assert admit(T, spec_of(**ES), V(q, now=NOW)) == ["r2"]

    def test_non_head_still_counts_in_the_sums(self, T):
        # now=.05 ; r1 bert:5 a=0 D=.06 ; r2 yolo:1 a=.02 D=.2 ; r4 bert:5 a=.05 D=.015 (not a head).
        # Without r4: S(r1)=exp((.03+.040)/.2-1)=.5220 < S(r2)=exp((.05+.012)/.06-1)=1.0339 -> r1.
        # With r4 in the sums: S(r1)=.5220+exp(.040/.015-1)=5.8165 ; S(r2)=1.0339+exp(.012/.015-1)=1.8526 -> r2.
        q = [Q(T, "r1", "bert:5", 0, dl=60_000_000), Q(T, "r2", "yolo:1", 20_000_000, dl=220_000_000)]
        assert admit(T, spec_of(**ES), V(q, now=NOW)) == ["r1"]
        q.append(Q(T, "r4", "bert:5", 50_000_000, dl=65_000_000))
        assert admit(T, spec_of(**ES), V(q, now=NOW)) == ["r2"]

    def test_every_queued_request_needs_a_deadline(self, T):
        # contracts docstring: "Edgeserving needs deadlines on every queued request."  No silent default.
        q = [Q(T, "rA", "bert:5", 0, dl=200_000_000), Q(T, "rB", "vision:6", 20_000_000)]
        with pytest.raises(ContractError):
            admit(T, spec_of(**ES), V(q, now=NOW))
        q2 = [Q(T, "rA", "bert:5", 0), Q(T, "rB", "vision:6", 20_000_000, dl=120_000_000)]
        with pytest.raises(ContractError):
            admit(T, spec_of(**ES), V(q2, now=NOW))

    def test_term_is_clipped_at_C(self, T):
        # now=.01 ; L llama:15 a=0 D=.04 ; B bert:23 a=.005 D=.04.
        # S(B)=exp((.01+.150)/.04-1)=exp(3)=20.09 ; S(L)=exp((.005+.9)/.04-1)=exp(21.6)=2.4e9.
        # C=10 clips both to 10 -> exact tie -> earliest arrival L.  C=30 clips only S(L) to 30 -> B.
        q = [Q(T, "L", "llama:15", 0, dl=40_000_000), Q(T, "B", "bert:23", 5_000_000, dl=45_000_000)]
        view = V(q, now=10_000_000)
        assert admit(T, spec_of(**dict(ES, params={"C": 10.0})), view) == ["L"]
        assert admit(T, spec_of(**dict(ES, params={"C": 30.0})), view) == ["B"]

    def test_default_C_is_the_contract_constant(self, T):
        assert C.EDGESERVING_C == 10.0
        q = [Q(T, "L", "llama:15", 0, dl=40_000_000), Q(T, "B", "bert:23", 5_000_000, dl=45_000_000)]
        assert admit(T, spec_of(**dict(ES, params={})), V(q, now=10_000_000)) == ["L"]

    def test_exact_tie_prefers_earliest_arrival(self, tmp_path):
        base = dict(BASE)
        base["yolo:1"] = (0.020, 0.030, 0.025, 0.040)   # p95 equal to vision:6
        T2 = make_types(tmp_path, base)
        # identical a and D -> identical S ; same arrival -> first in the queue wins both ways round.
        for first, second, want in (("vision:6", "yolo:1", "r1"), ("yolo:1", "vision:6", "r1")):
            q = [Q(T2, "r1", first, 0, dl=100_000_000), Q(T2, "r2", second, 0, dl=100_000_000)]
            assert admit(T2, spec_of(**ES), V(q, now=NOW)) == [want]

    def test_equal_urgency_with_different_arrivals_picks_the_earlier(self, tmp_path):
        base = dict(BASE)
        base["yolo:1"] = (0.020, 0.030, 0.025, 0.040)
        T2 = make_types(tmp_path, base)
        # r1 a=0 D=.15 ; r2 a=.02 D=.11 : (.05+.025)/.15 == (.03+.025)/.11 == .5 -> S equal.
        q = [Q(T2, "r1", "vision:6", 0, dl=150_000_000), Q(T2, "r2", "yolo:1", 20_000_000, dl=130_000_000)]
        assert admit(T2, spec_of(**ES), V(q, now=NOW)) == ["r1"]

    def test_p95_is_read_at_f_current_and_f_max_when_none(self, T):
        # h1 bert:23 (a=0, D=.06), h2 yolo:1 (a=.02, D=.2), now=.05
        #  f=1020: S(h1)=exp((.03+.150)/.2-1)=.9048 < S(h2)=exp((.05+.012)/.06-1)=1.0339 -> h1
        #  f=612 : S(h1)=exp((.03+.220)/.2-1)=1.2840 > S(h2)=exp((.05+.018)/.06-1)=1.1426 -> h2
        q = [Q(T, "h1", "bert:23", 0, dl=60_000_000), Q(T, "h2", "yolo:1", 20_000_000, dl=220_000_000)]
        assert admit(T, spec_of(**ES), V(q, now=NOW, f=1020)) == ["h1"]
        assert admit(T, spec_of(**ES), V(q, now=NOW, f=612)) == ["h2"]
        assert admit(T, spec_of(**ES), V(q, now=NOW, f=None)) == ["h1"]

    def test_edgeserving_with_k_other_than_one_cannot_be_built(self, T):
        for kw in ({"k_short": 2, "k_long": 1}, {"k_short": 1, "k_long": 2}, {"k_short": None, "k_long": None}):
            with pytest.raises(ContractError):
                Policy(spec_of(**dict(ES, **kw)), T)

    def test_busy_system_admits_nothing(self, T):
        q = [Q(T, "rA", "bert:5", 0, dl=200_000_000)]
        assert admit(T, spec_of(**ES), V(q, [S(T, "r0", "vision:24")], now=NOW)) == []

    def test_empty_queue(self, T):
        assert admit(T, spec_of(**ES), V([], now=NOW)) == []

    def test_family_busy_head_is_skipped_not_blocking(self, T):
        # k=1 so with n=0 nothing is busy; the non-blocking scan is exercised with family_exclusive off
        # being irrelevant.  Here: single head admitted, the other heads are left queued.
        q = [Q(T, "rA", "bert:5", 0, dl=200_000_000), Q(T, "rB", "vision:6", 0, dl=200_000_000)]
        assert len(admit(T, spec_of(**ES), V(q, now=NOW))) == 1


# --------------------------------------------------------------------------------------
# desired_f
# --------------------------------------------------------------------------------------
def desired(T, spec, view, **kw):
    return Policy(spec, T, **kw).desired_f(view)


class TestDesiredFStaticGovernor:
    def test_static_returns_none(self, T):
        assert desired(T, spec_of(f_rule="static", f_mhz=612), V([Q(T, "r1", "bert:5", 1)])) is None

    def test_governor_returns_none(self, T):
        assert desired(T, spec_of(f_rule="governor", f_mhz=None), V([Q(T, "r1", "bert:5", 1)], f=None)) is None


G_NOW = 1 * NS


class TestGreenLLM:
    def _spec(self):
        return spec_of(lane_cap=lanes(1, 1), **GREEN)

    def test_lowest_feasible_level_is_chosen(self, T):
        # r1 bert:5 a=.99 s deadline 1.10 s.  612: 1.0+.045=1.045 <= 1.10 -> 612.
        q = [Q(T, "r1", "bert:5", 990_000_000, dl=1_100_000_000)]
        assert desired(T, self._spec(), V(q, now=G_NOW)) == 612

    def test_only_fmax_feasible(self, T):
        # deadline 1.035: 612 -> 1.045 > 1.035 ; 1020 -> 1.030 <= 1.035.
        q = [Q(T, "r1", "bert:5", 990_000_000, dl=1_035_000_000)]
        assert desired(T, self._spec(), V(q, now=G_NOW)) == 1020

    def test_boundary_equality_is_feasible(self, T):
        # deadline exactly 1.045: 612 -> 1.045 <= 1.045.
        q = [Q(T, "r1", "bert:5", 990_000_000, dl=1_045_000_000)]
        assert desired(T, self._spec(), V(q, now=G_NOW)) == 612

    def test_infeasible_everywhere_falls_back_to_fmax(self, T):
        # deadline 1.020: even 1020 gives 1.030.
        q = [Q(T, "r1", "bert:5", 990_000_000, dl=1_020_000_000)]
        assert desired(T, self._spec(), V(q, now=G_NOW)) == T.f_max_mhz == 1020

    def test_every_queued_request_must_be_feasible(self, T):
        # r1 relaxed, r2 (yolo, other lane? yolo:1 short, same lane as r1 bert:5):
        # lane short: r1 first then r2.  r2: wait .0 + (.045 + .015) = 1.060 at 612 > deadline 1.05.
        q = [Q(T, "r1", "bert:5", G_NOW, dl=2 * NS), Q(T, "r2", "yolo:1", G_NOW, dl=1_050_000_000)]
        # 1020: .030 + .010 = 1.040 <= 1.05 -> 1020
        assert desired(T, self._spec(), V(q, now=G_NOW)) == 1020

    def test_sum_over_requests_ahead_in_the_same_lane(self, T):
        # rA vision:24 long (.12@612 / .08@1020), rB bert:23 long (.18 / .12), same lane, rA ahead.
        # rB at 612: 1.0 + .12 + .18 = 1.30 ; at 1020: 1.0 + .08 + .12 = 1.20.  deadline 1.25 -> only 1020.
        q = [Q(T, "rA", "vision:24", 900_000_000, dl=5 * NS), Q(T, "rB", "bert:23", 950_000_000, dl=1_250_000_000)]
        assert desired(T, self._spec(), V(q, now=G_NOW)) == 1020

    def test_other_lane_does_not_add_to_the_sum(self, T):
        # a short request (yolo:1 .015@612) sits between rA and rB but is in the other lane.
        # rB at 612 = 1.0 + .12 + .18 = 1.30 <= 1.31 ; wrongly adding .015 would give 1.315 > 1.31.
        q = [Q(T, "rA", "vision:24", 900_000_000, dl=5 * NS), Q(T, "rS", "yolo:1", 920_000_000, dl=5 * NS),
             Q(T, "rB", "bert:23", 950_000_000, dl=1_310_000_000)]
        assert desired(T, self._spec(), V(q, now=G_NOW)) == 612

    def test_residual_of_the_in_service_request_counts(self, T):
        # llama:15 (long) admitted .4 s ago at f_current=1020 (p50 .8 s): residual(f) = tau_in(f)*(1-.4/.8)
        # = .6 @612 (1.2*.5) and .4 @1020 (.8*.5).  r1 bert:23 long: 612 -> 1.0+.18+.6=1.78 ; 1020 -> 1.52.
        ins = [S(T, "r0", "llama:15", t_admit_ns=600_000_000)]
        q = [Q(T, "r1", "bert:23", 900_000_000, dl=1_600_000_000)]
        assert desired(T, self._spec(), V(q, ins, now=G_NOW, f=1020)) == 1020      # 612 would miss by .18
        q = [Q(T, "r1", "bert:23", 900_000_000, dl=1_800_000_000)]
        assert desired(T, self._spec(), V(q, ins, now=G_NOW, f=1020)) == 612       # 1.78 <= 1.80

    def test_residual_is_zero_when_in_service_is_overdue(self, T):
        # admitted at 0, now 1.0 s > tau_in(1020)=.8 s -> max(0, 1-1/.8)=0.  r1: 612 -> 1.0+.18=1.18 <= 1.20.
        ins = [S(T, "r0", "llama:15", t_admit_ns=0)]
        q = [Q(T, "r1", "bert:23", 900_000_000, dl=1_200_000_000)]
        assert desired(T, self._spec(), V(q, ins, now=G_NOW, f=1020)) == 612

    def test_residual_denominator_is_tau_in_at_f_current(self, T):
        # f_current=612, elapsed .6 s: factor = 1-.6/1.2 = .5 -> residual(612) = 1.2*.5 = .6, so
        # r1 (bert:23, dl 1.70): 612 -> 1.0+.18+.6 = 1.78 > 1.70 ; 1020 -> 1.0+.12+.4 = 1.52 <= 1.70 -> 1020.
        # (a denominator taken at f_max, .8, would give residual(612)=.3 and pick 612.)
        ins = [S(T, "r0", "llama:15", t_admit_ns=400_000_000)]
        q = [Q(T, "r1", "bert:23", 900_000_000, dl=1_700_000_000)]
        assert desired(T, self._spec(), V(q, ins, now=G_NOW, f=612)) == 1020

    def test_residual_of_the_other_lane_counts_too(self, T):
        # in service is SHORT (yolo:1, just admitted, f_current 1020: residual(f) = tau_in(f) * 1),
        # queued r1 is LONG: the GPU is shared, so .015 @612 / .010 @1020 still delays r1 (golden k20c, T6).
        # dl 1.19: 612 -> 1.0 + .18 + .015 = 1.195 > 1.19 ; 1020 -> 1.0 + .12 + .010 = 1.13 -> 1020.
        ins = [S(T, "r0", "yolo:1", t_admit_ns=G_NOW)]
        q = [Q(T, "r1", "bert:23", 900_000_000, dl=1_190_000_000)]
        assert desired(T, self._spec(), V(q, ins, now=G_NOW, f=1020)) == 1020
        # dl 1.20: 612 -> 1.195 <= 1.20 -> 612
        q = [Q(T, "r1", "bert:23", 900_000_000, dl=1_200_000_000)]
        assert desired(T, self._spec(), V(q, ins, now=G_NOW, f=1020)) == 612

    def test_residuals_of_several_in_service_requests_add_up(self, T):
        # two in service, both just admitted: residual(612) = .015 (yolo:1) + .045 (bert:5) = .060 ;
        # residual(1020) = .010 + .030 = .040.  r1 vision:24 (.12 @612, .08 @1020) a=.9:
        #   dl 1.17: 612 -> 1.0 + .12 + .06 = 1.18 > 1.17 ; 1020 -> 1.0 + .08 + .04 = 1.12 -> 1020
        #   dl 1.18: 612 -> 1.18 <= 1.18 -> 612
        ins = [S(T, "r0", "yolo:1", t_admit_ns=G_NOW), S(T, "r9", "bert:5", t_admit_ns=G_NOW)]
        assert desired(T, self._spec(), V([Q(T, "r1", "vision:24", 900_000_000, dl=1_170_000_000)], ins,
                                          now=G_NOW, f=1020)) == 1020
        assert desired(T, self._spec(), V([Q(T, "r1", "vision:24", 900_000_000, dl=1_180_000_000)], ins,
                                          now=G_NOW, f=1020)) == 612

    def test_request_without_deadline_is_unconstrained(self, T):
        q = [Q(T, "r1", "bert:5", 990_000_000, dl=None)]
        assert desired(T, self._spec(), V(q, now=G_NOW)) == 612

    def test_empty_queue_picks_the_lowest_level(self, T):
        # every (zero) queued request is feasible -> the lowest corun level.
        assert desired(T, self._spec(), V([], now=G_NOW)) == 612


OCC = dict(f_rule="occupancy", f_mhz=None)


class TestOccupancy:
    def _spec(self, f_base=612, alpha_p=2):
        return spec_of(params={"f_base_mhz": f_base, "alpha_p": alpha_p}, **OCC)

    def test_n_sys_one_gives_the_base_level(self, T):
        assert desired(T, self._spec(), V([], [S(T, "r0", "bert:5")])) == 612

    def test_n_sys_two_rounds_up_to_next_level(self, T):
        # 612*2**.5 = 865.5 -> smallest corun level >= 865.5 is 1020.
        q = [Q(T, "r1", "bert:23", 1)]
        assert desired(T, self._spec(), V(q, [S(T, "r0", "vision:6")])) == 1020

    def test_no_level_above_threshold_uses_the_highest(self, T):
        # n_sys=3 -> 612*3**.5 = 1060.0 -> none >= -> highest = 1020  (golden case 21).
        q = [Q(T, "r1", "bert:23", 1), Q(T, "r2", "vision:24", 2)]
        assert desired(T, self._spec(), V(q, [S(T, "r0", "yolo:1")])) == 1020

    def test_threshold_equal_to_a_level_selects_that_level(self, T):
        # 153 * 4**1 = 612.0 exactly -> level 612 satisfies >=.
        q = [Q(T, "r%d" % i, "bert:5", i) for i in range(3)]
        assert desired(T, self._spec(153, 1), V(q, [S(T, "r9", "yolo:1")])) == 612

    def test_counts_queue_plus_in_service(self, T):
        # 2 queued + 0 in service = 2 -> 865.5 -> 1020 ; 1 queued = 1 -> 612.
        q2 = [Q(T, "r1", "bert:5", 1), Q(T, "r2", "bert:23", 2)]
        assert desired(T, self._spec(), V(q2)) == 1020
        assert desired(T, self._spec(), V(q2[:1])) == 612

    def test_empty_system_is_the_lowest_level(self, T):
        assert desired(T, self._spec(), V([])) == 612

    def test_alpha_p_changes_the_curve(self, T):
        # n_sys=2.  alpha_p=64 : f_base*2**(1/64) = f_base*1.01089.
        #  f_base 600 -> 606.5 <= 612 -> 612 ; f_base 612 -> 618.7 > 612 -> 1020.
        ins = [S(T, "r0", "vision:6")]
        q = [Q(T, "r1", "bert:23", 1)]
        assert desired(T, self._spec(600, 64), V(q, ins)) == 612
        assert desired(T, self._spec(612, 64), V(q, ins)) == 1020


# --------------------------------------------------------------------------------------
# Purity, determinism, construction
# --------------------------------------------------------------------------------------
class TestPurity:
    SPECS = [
        dict(),
        dict(order="stf", k_short=2, k_long=2),
        dict(order="class", k_short=2, k_long=2),
        ES,
        dict(lane_cap=lanes(1, 1), **GREEN),
        dict(params={"f_base_mhz": 612, "alpha_p": 2}, **OCC),
    ]

    def _view(self, T):
        return V([Q(T, "r1", "bert:5", 0, dl=200_000_000), Q(T, "r2", "vision:6", 20_000_000, dl=120_000_000),
                  Q(T, "r3", "yolo:1", 40_000_000, dl=90_000_000), Q(T, "r4", "bert:23", 45_000_000, dl=400_000_000)],
                 [S(T, "r0", "llama:15", 10_000_000)], now=NOW)

    @pytest.mark.parametrize("kw", SPECS, ids=lambda k: k.get("order", "fcfs") + str(k.get("f_rule", "")))
    def test_admit_and_desired_f_do_not_mutate_and_repeat(self, T, kw):
        pol = Policy(spec_of(**kw), T)
        view = self._view(T)
        before = copy.deepcopy(dataclasses.asdict(view))
        a1, f1 = list(pol.admit(view)), pol.desired_f(view)
        assert dataclasses.asdict(view) == before
        a2, f2 = list(pol.admit(view)), pol.desired_f(view)
        assert (a1, f1) == (a2, f2)
        fresh = Policy(spec_of(**kw), T)
        assert (list(fresh.admit(self._view(T))), fresh.desired_f(self._view(T))) == (a1, f1)

    def test_admit_does_not_depend_on_previous_calls(self, T):
        pol = Policy(spec_of(order="stf", k_short=2, k_long=2), T)
        v1 = V([Q(T, "r1", "bert:23", 1), Q(T, "r2", "yolo:1", 2)])
        v2 = V([Q(T, "r1", "bert:5", 1)])
        first = list(pol.admit(v1))
        pol.admit(v2)
        assert list(pol.admit(v1)) == first

    def test_large_queue_is_fast_and_correct(self, T):
        q = [Q(T, "r%d" % i, ["bert:5", "vision:6", "llama:15", "yolo:1"][i % 4], i) for i in range(20_000)]
        t0 = time.perf_counter()
        got = admit(T, spec_of(), V(q))
        assert time.perf_counter() - t0 < 2.0
        assert got == ["r0", "r1", "r2", "r3"]


class TestConstruction:
    def test_policy_keeps_spec_and_family_exclusive(self, T):
        sp = spec_of(k_short=1, k_long=1)
        p = Policy(sp, T, family_exclusive=False)
        assert p.spec == sp and p.family_exclusive is False
        assert Policy(sp, T).family_exclusive is True

    def test_policy_satisfies_the_contract_protocol(self, T):
        assert isinstance(Policy(spec_of(), T), C.Policy)

    def test_family_exclusive_is_keyword_only(self, T):
        with pytest.raises(TypeError):
            Policy(spec_of(), T, False)

    def test_edgeserving_policy_requires_k_one(self, T):
        with pytest.raises(ContractError):
            Policy(spec_of(**dict(ES, k_short=3)), T)


# --------------------------------------------------------------------------------------
# AdmissionCore
# --------------------------------------------------------------------------------------
def core_of(T, kw=None, f0=1020, **pk):
    return AdmissionCore(Policy(spec_of(**(kw or dict(k_short=1, k_long=1))), T, **pk), T, f0)


class TestAdmissionCore:
    def test_arrival_into_idle_system_is_admitted_at_once(self, T):
        core = core_of(T)
        d = core.arrive(Q(T, "r1", "bert:5", 100), 100)
        assert isinstance(d, Decision)
        assert list(d.admitted) == ["r1"] and list(d.n_in_service_at_admit) == [0]
        assert list(d.expired) == [] and d.commanded_f is None
        v = core.view(100)
        assert [r.req_id for r in v.in_service] == ["r1"] and v.in_service[0].t_admit_ns == 100
        assert list(v.queue) == [] and v.now_ns == 100 and v.f_current_mhz == 1020

    def test_second_arrival_waits_and_done_admits_it_at_the_done_time(self, T):
        core = core_of(T)
        core.arrive(Q(T, "r1", "bert:5", 100), 100)
        d = core.arrive(Q(T, "r2", "vision:6", 200), 200)
        assert list(d.admitted) == [] and [r.req_id for r in core.view(200).queue] == ["r2"]
        d = core.done("r1", 300)
        assert list(d.admitted) == ["r2"] and list(d.n_in_service_at_admit) == [0]
        v = core.view(300)
        assert [r.req_id for r in v.in_service] == ["r2"] and v.in_service[0].t_admit_ns == 300

    def test_n_in_service_at_admit_counts_the_others_in_service(self, T):
        core = core_of(T, dict(k_short=2, k_long=2))
        core.arrive(Q(T, "r1", "bert:5", 1), 1)
        d = core.arrive(Q(T, "r2", "vision:6", 2), 2)
        assert list(d.admitted) == ["r2"] and list(d.n_in_service_at_admit) == [1]
        d = core.arrive(Q(T, "r3", "yolo:1", 3), 3)          # n=2 -> queued
        assert list(d.admitted) == []
        d = core.done("r1", 4)                                  # n=1 -> r3 admitted
        assert list(d.admitted) == ["r3"] and list(d.n_in_service_at_admit) == [1]

    def test_k_override_of_the_arriving_request_is_honoured(self, T):
        core = core_of(T)                                  # serial: k = 1
        core.arrive(Q(T, "r1", "bert:5", 0), 0)
        d = core.arrive(Q(T, "r2", "vision:6", 1, ko=2), 1)          # 1 < 2 -> admitted although serial
        assert list(d.admitted) == ["r2"] and list(d.n_in_service_at_admit) == [1]
        d = core.arrive(Q(T, "r3", "yolo:1", 2), 2)                  # no override: 2 < 1 false
        assert list(d.admitted) == []

    def test_decision_is_immutable(self, T):
        d = core_of(T).arrive(Q(T, "r1", "bert:5", 0), 0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            d.commanded_f = 612

    def test_deadline_equal_to_now_is_dropped(self, T):
        core = core_of(T)
        core.arrive(Q(T, "r1", "bert:5", 0), 0)
        core.arrive(Q(T, "r2", "vision:6", 10, dl=500), 10)
        d = core.done("r1", 500)                  # now >= deadline -> expired, not admitted
        assert list(d.expired) == ["r2"] and list(d.admitted) == []
        assert list(core.view(500).queue) == [] and list(core.view(500).in_service) == []

    def test_deadline_one_ns_later_is_not_dropped(self, T):
        core = core_of(T)
        core.arrive(Q(T, "r1", "bert:5", 0), 0)
        core.arrive(Q(T, "r2", "vision:6", 10, dl=500), 10)
        d = core.done("r1", 499)
        assert list(d.expired) == [] and list(d.admitted) == ["r2"]

    def test_expired_request_is_never_admitted_even_if_it_is_the_head(self, T):
        # drop happens BEFORE admit: r1 (head, deadline 500) is dropped, r2 takes the slot at 500.
        core = core_of(T)
        core.arrive(Q(T, "r0", "yolo:1", 0), 0)
        core.arrive(Q(T, "r1", "vision:6", 10, dl=500), 10)
        core.arrive(Q(T, "r2", "bert:5", 20, dl=None), 20)
        d = core.done("r0", 500)
        assert list(d.expired) == ["r1"] and list(d.admitted) == ["r2"]

    def test_arrival_with_deadline_not_after_now_expires_at_once(self, T):
        core = core_of(T)
        d = core.arrive(Q(T, "r1", "bert:5", 100, dl=100), 100)
        assert list(d.expired) == ["r1"] and list(d.admitted) == []
        assert list(core.view(100).in_service) == []

    def test_several_expiries_in_queue_order(self, T):
        core = core_of(T)
        core.arrive(Q(T, "r0", "yolo:1", 0), 0)
        core.arrive(Q(T, "r1", "bert:5", 1, dl=50), 1)
        core.arrive(Q(T, "r2", "vision:6", 2, dl=40), 2)
        core.arrive(Q(T, "r3", "yolo:1", 3, dl=900), 3)
        d = core.done("r0", 60)
        assert list(d.expired) == ["r1", "r2"] and list(d.admitted) == ["r3"]

    def test_later_deadlines_still_expire_after_earlier_ones_were_removed(self, T):
        core = core_of(T)
        core.arrive(Q(T, "r0", "yolo:1", 0), 0)
        core.arrive(Q(T, "r1", "bert:5", 1, dl=50), 1)
        core.arrive(Q(T, "r2", "vision:6", 2, dl=500), 2)
        d = core.done("r0", 60)                       # r1 expires, r2 (dl 500) is admitted -> nothing left
        assert list(d.expired) == ["r1"] and list(d.admitted) == ["r2"]
        core.arrive(Q(T, "r3", "bert:23", 61, dl=700), 61)      # r2 in service, serial k=1 -> queued
        core.arrive(Q(T, "r4", "llama:15", 62, dl=640), 62)
        d = core.done("r2", 650)                      # r4 (dl 640) expired, r3 (dl 700) admitted
        assert list(d.expired) == ["r4"] and list(d.admitted) == ["r3"]

    def test_queued_deadline_expires_on_a_later_event_after_other_removals(self, T):
        core = core_of(T)
        core.arrive(Q(T, "r0", "yolo:1", 0), 0)
        core.arrive(Q(T, "r1", "bert:5", 1, dl=50), 1)
        core.arrive(Q(T, "r2", "vision:6", 2, dl=500), 2)
        core.arrive(Q(T, "r3", "bert:23", 3, dl=520), 3)
        d = core.done("r0", 60)                       # r1 expires; serial admits r2; r3 stays queued
        assert list(d.expired) == ["r1"] and list(d.admitted) == ["r2"]
        d = core.done("r2", 520)                      # r3: now >= dl 520 -> expired, not admitted
        assert list(d.expired) == ["r3"] and list(d.admitted) == []

    def test_queue_keeps_arrival_order_after_removals(self, T):
        core = core_of(T)
        core.arrive(Q(T, "r0", "yolo:1", 0), 0)
        for i, tid in enumerate(["bert:5", "vision:6", "bert:23", "vision:24"], start=1):
            core.arrive(Q(T, "r%d" % i, tid, i), i)
        assert [r.req_id for r in core.view(10).queue] == ["r1", "r2", "r3", "r4"]
        core.done("r0", 11)            # serial fcfs: r1 admitted
        assert [r.req_id for r in core.view(11).queue] == ["r2", "r3", "r4"]

    def test_unknown_and_double_done_raise(self, T):
        core = core_of(T)
        core.arrive(Q(T, "r1", "bert:5", 0), 0)
        with pytest.raises(KeyError):
            core.done("nope", 1)
        core.done("r1", 2)
        with pytest.raises(KeyError):
            core.done("r1", 3)

    def test_done_for_a_queued_request_raises(self, T):
        core = core_of(T)
        core.arrive(Q(T, "r1", "bert:5", 0), 0)
        core.arrive(Q(T, "r2", "vision:6", 1), 1)
        with pytest.raises(KeyError):
            core.done("r2", 2)

    def test_duplicate_req_id_raises(self, T):
        core = core_of(T)
        core.arrive(Q(T, "r1", "bert:5", 0), 0)
        with pytest.raises(ValueError):
            core.arrive(Q(T, "r1", "vision:6", 1), 1)        # in service
        core.arrive(Q(T, "r2", "vision:6", 2), 2)             # queued
        with pytest.raises(ValueError):
            core.arrive(Q(T, "r2", "yolo:1", 3), 3)

    def test_view_is_a_snapshot(self, T):
        core = core_of(T)
        core.arrive(Q(T, "r1", "bert:5", 0), 0)
        v0 = core.view(0)
        core.arrive(Q(T, "r2", "vision:6", 1), 1)
        assert list(v0.queue) == [] and [r.req_id for r in v0.in_service] == ["r1"]

    def test_view_does_not_change_state(self, T):
        core = core_of(T)
        core.arrive(Q(T, "r1", "bert:5", 0), 0)
        core.arrive(Q(T, "r2", "vision:6", 1, dl=10), 1)
        core.view(10_000)                      # far past r2's deadline: a plain view() must not drop it
        assert [r.req_id for r in core.view(2).queue] == ["r2"]

    def test_independent_cores_do_not_share_state(self, T):
        a, b = core_of(T), core_of(T)
        a.arrive(Q(T, "r1", "bert:5", 0), 0)
        assert list(b.view(0).in_service) == []


class _Stub:
    """Duck-typed policy: records the views the core passes in, returns scripted answers."""

    def __init__(self, admits=(), fs=()):
        self.admits, self.fs = list(admits), list(fs)
        self.admit_views, self.f_views = [], []

    def admit(self, view):
        self.admit_views.append(view)
        return self.admits.pop(0) if self.admits else []

    def desired_f(self, view):
        self.f_views.append(view)
        return self.fs.pop(0) if self.fs else None


class TestAdmissionCoreWithStub:
    def test_desired_f_sees_the_view_after_admit(self, T):
        stub = _Stub(admits=[["r1"]], fs=[None])
        core = AdmissionCore(stub, T, 1020)
        core.arrive(Q(T, "r1", "bert:5", 7), 7)
        assert [r.req_id for r in stub.admit_views[0].queue] == ["r1"]
        assert list(stub.admit_views[0].in_service) == []
        fv = stub.f_views[0]
        assert list(fv.queue) == [] and [(r.req_id, r.t_admit_ns) for r in fv.in_service] == [("r1", 7)]
        assert fv.now_ns == 7

    def test_admit_sees_the_view_after_drop(self, T):
        stub = _Stub()
        core = AdmissionCore(stub, T, 1020)
        core.arrive(Q(T, "r1", "bert:5", 0, dl=5), 4)           # not yet expired at 4
        core.arrive(Q(T, "r2", "vision:6", 1), 6)               # r1 dropped at 6 before admit is called
        assert [r.req_id for r in stub.admit_views[-1].queue] == ["r2"]

    def test_multiple_admits_report_growing_n(self, T):
        stub = _Stub(admits=[[], ["r1", "r2"]])
        core = AdmissionCore(stub, T, 1020)
        core.arrive(Q(T, "r1", "bert:5", 0), 0)
        d = core.arrive(Q(T, "r2", "vision:6", 1), 1)
        assert list(d.admitted) == ["r1", "r2"] and list(d.n_in_service_at_admit) == [0, 1]
        assert {r.req_id: r.t_admit_ns for r in core.view(1).in_service} == {"r1": 1, "r2": 1}

    def test_commanded_f_only_on_change_and_none_never_commands(self, T):
        stub = _Stub(fs=[1020, 612, 612, None, 1020, 612])
        core = AdmissionCore(stub, T, 1020)
        got = []
        for i in range(6):
            got.append(core.arrive(Q(T, "r%d" % i, "bert:5", i), i).commanded_f)
        # desired: 1020 (== initial, no cmd), 612 (cmd), 612 (same), None, 1020 (cmd back), 612 (cmd)
        assert got == [None, 612, None, None, 1020, 612]

    def test_f_current_is_the_last_commanded_level_not_the_desired_one(self, T):
        stub = _Stub(admits=[["r0"]], fs=[612, None, None])
        core = AdmissionCore(stub, T, 1020)
        core.arrive(Q(T, "r0", "bert:5", 0), 0)
        assert stub.admit_views[0].f_current_mhz == 1020
        core.arrive(Q(T, "r1", "bert:5", 1), 1)
        assert stub.admit_views[1].f_current_mhz == 612 and core.view(1).f_current_mhz == 612
        core.done("r0", 2)
        assert stub.admit_views[2].f_current_mhz == 612

    def test_initial_f_none_is_kept_for_governor(self, T):
        stub = _Stub()
        core = AdmissionCore(stub, T, None)
        core.arrive(Q(T, "r0", "bert:5", 0), 0)
        assert stub.admit_views[0].f_current_mhz is None and core.view(0).f_current_mhz is None

    def test_initial_f_equal_to_desired_is_not_a_command(self, T):
        stub = _Stub(fs=[612])
        core = AdmissionCore(stub, T, 612)
        assert core.arrive(Q(T, "r0", "bert:5", 0), 0).commanded_f is None

    def test_policy_admitting_an_unknown_id_is_an_error(self, T):
        core = AdmissionCore(_Stub(admits=[["ghost"]]), T, 1020)
        with pytest.raises(RuntimeError):
            core.arrive(Q(T, "r1", "bert:5", 0), 0)

    def test_policy_admitting_the_same_id_twice_is_an_error(self, T):
        core = AdmissionCore(_Stub(admits=[["r1", "r1"]]), T, 1020)
        with pytest.raises(RuntimeError):
            core.arrive(Q(T, "r1", "bert:5", 0), 0)


class TestAdmissionCoreWithGreenLLM:
    def test_commanded_sequence_follows_the_feasibility(self, T):
        spec = dict(lane_cap=lanes(1, 1), **GREEN)
        core = AdmissionCore(Policy(spec_of(**spec), T), T, 1020)
        # r1 bert:5 arrives at .99 s, system idle -> admitted.  After the admit the queue is empty, so
        # every queued request (none) is feasible at the lowest corun level -> command 612.
        d = core.arrive(Q(T, "r1", "bert:5", 990_000_000, dl=1_035_000_000), 990_000_000)
        assert list(d.admitted) == ["r1"] and d.commanded_f == 612
        # r2 bert:23 (bert busy -> queued, long lane).  At 612: 1.0 + .18 = 1.18 <= 1.30 -> stays 612.
        d = core.arrive(Q(T, "r2", "bert:23", NS, dl=1_300_000_000), NS)
        assert list(d.admitted) == [] and d.commanded_f is None
        # r3 bert:23 (bert busy -> queued behind r2 in the long lane).  At 612: 1.0 + .18 + .18 = 1.36 > 1.30;
        # at 1020: 1.0 + .12 + .12 = 1.24 <= 1.30 -> 1020.
        d = core.arrive(Q(T, "r3", "bert:23", NS, dl=1_300_000_000), NS)
        assert list(d.admitted) == [] and d.commanded_f == 1020
        assert core.view(NS).f_current_mhz == 1020


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------
KEYS = ["default", "default_stock", "serial", "serial_stf", "serial_class", "serial_edgeserving", "corun",
        "greenllm_lanes", "occupancy_f"]


class TestRegistry:
    def test_has_exactly_the_contract_keys(self, T):
        assert sorted(registry(T)) == sorted(KEYS)

    def test_values_are_valid_policyspecs_and_buildable(self, T):
        for key, sp in registry(T).items():
            assert isinstance(sp, PolicySpec) and sp.name == key
            Policy(sp, T)

    def test_hand_listed_fields(self, T):
        r = registry(T)
        assert (r["serial"].k_short, r["serial"].k_long, r["serial"].order) == (1, 1, "fcfs")
        assert (r["serial"].f_rule, r["serial"].f_mhz) == ("static", T.f_max_mhz)
        assert r["serial_stf"].order == "stf" and r["serial_class"].order == "class"
        assert r["serial_edgeserving"].order == "edgeserving" and r["serial_edgeserving"].params == {"C": 10}
        assert (r["default"].f_rule, r["default"].f_mhz) == ("governor", None)
        assert (C.k_of(r["default"], "short"), C.k_of(r["default"], "long")) == (C.K_INF, C.K_INF)
        assert (C.k_of(r["corun"], "short"), r["corun"].f_rule, r["corun"].f_mhz) == (C.K_INF, "static", 1020)
        g = r["greenllm_lanes"]
        assert (g.k_short, g.k_long, g.f_rule) == (2, 2, "greenllm") and g.lane_cap == lanes(1, 1)
        assert r["occupancy_f"].params == {"f_base_mhz": 612, "alpha_p": 2}

    def test_registry_follows_f_max_of_the_type_table(self, tmp_path):
        T2 = make_types(tmp_path, levels=[408, 612, 714, 816], corun=[612, 816])
        assert T2.f_max_mhz == 816
        r = registry(T2)
        assert r["serial"].f_mhz == 816 and r["corun"].f_mhz == 816 and r["serial_edgeserving"].f_mhz == 816

    def test_registry_returns_fresh_dicts(self, T):
        a = registry(T)
        a.pop("serial")
        assert "serial" in registry(T)

    def test_matches_policyspecs_fixture(self, T):
        raw = json.loads((FIXTURES / "policyspecs.json").read_text(encoding="utf-8"))
        want = raw.get("policies", raw) if isinstance(raw, dict) else {s["name"]: s for s in raw}
        got = {k: v.to_json() for k, v in registry(T).items()}
        assert got == want


# --------------------------------------------------------------------------------------
# Golden replays (S0a / S0b fixtures, never edited here)
# --------------------------------------------------------------------------------------
def _golden(name, key):
    """Fixture entries; a missing file becomes ONE failing entry (never a skip, never a collection error)."""
    path = FIXTURES / name
    if not path.exists():
        return [{"id": "MISSING_" + name, "__missing__": str(path)}]
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw.get(key, raw) if isinstance(raw, dict) else raw


def _golden_types(case):
    return C.load_types(str(REPO / case["types_ref"]))


def _view_of(d):
    return DispatchView.from_json(d)


ADMISSION = _golden("admission_cases.json", "cases")
TRACES = _golden("admission_traces.json", "traces")


def test_golden_admission_has_at_least_20_cases():
    assert len(ADMISSION) >= 20 and not any("__missing__" in c for c in ADMISSION)


@pytest.mark.parametrize("case", ADMISSION, ids=[c["id"] for c in ADMISSION])
def test_golden_admission_case(case):
    assert "__missing__" not in case, case.get("__missing__")
    types = _golden_types(case)
    fx = case.get("family_exclusive", True)
    if case.get("expect_error"):
        with pytest.raises(getattr(C, case["expect_error"])):
            Policy(PolicySpec.from_json(case["spec"]), types, family_exclusive=fx)
        return
    pol = Policy(PolicySpec.from_json(case["spec"]), types, family_exclusive=fx)
    view = _view_of(case["view"])
    assert list(pol.admit(view)) == case["expect_admit"]
    assert pol.desired_f(view) == case["expect_desired_f"]


def test_golden_traces_has_at_least_6():
    assert len(TRACES) >= 6 and not any("__missing__" in t for t in TRACES)


def replay(trace):
    """Deterministic service: done = admit + service_ns[type_id] (slowdown s = 1)."""
    types = C.load_types(str(REPO / trace["types_ref"]))
    spec = PolicySpec.from_json(trace["spec"])
    f0 = trace.get("f_initial_mhz", spec.f_mhz)
    core = AdmissionCore(Policy(spec, types), types, f0)
    heap, seq = [], 0
    by_id = {}
    for a in trace["arrivals"]:
        by_id[a["req_id"]] = a
        heapq.heappush(heap, (a["t_arrival_ns"], 1, seq, "arrive", a["req_id"]))   # done (0) before arrive (1)
        seq += 1
    out, cmds = {}, []
    while heap:
        t, _, _, kind, rid = heapq.heappop(heap)
        if kind == "arrive":
            a = by_id[rid]
            t_ = types.get(a["type_id"])
            dec = core.arrive(QueuedReq(req_id=rid, type_id=a["type_id"], family=t_.family, klass=t_.klass,
                                        t_arrival_ns=a["t_arrival_ns"], t_deadline_ns=a["t_deadline_ns"],
                                        k_override=a["k_override"]), t)
        else:
            dec = core.done(rid, t)
        for r in dec.expired:
            out[r] = {"t_admit_ns": None, "outcome": "expired_in_queue", "t_done_ns": by_id[r]["t_deadline_ns"]}
        for r in dec.admitted:
            out[r] = {"t_admit_ns": t, "outcome": "admitted"}
            heapq.heappush(heap, (t + trace["service_ns"][by_id[r]["type_id"]], 0, seq, "done", r))
            seq += 1
        if dec.commanded_f is not None:
            cmds.append(dec.commanded_f)
    return out, cmds


@pytest.mark.parametrize("trace", TRACES, ids=[t["id"] for t in TRACES])
def test_golden_trace(trace):
    assert "__missing__" not in trace, trace.get("__missing__")
    out, cmds = replay(trace)
    for rid, want in trace["expect"].items():
        assert rid in out, "request %s never admitted or expired" % rid
        for key, val in want.items():
            assert out[rid][key] == val, (rid, key)
    assert cmds == trace["expect_commanded_f"]
