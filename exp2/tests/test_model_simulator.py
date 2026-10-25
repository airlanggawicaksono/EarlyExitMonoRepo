"""E2 tests: exp2.model.simulator (BRIEF_v1 sec. 4 'E2', K14).

    TestAnalytic*     closed-form checks (D6, `-k analytic`). The expected numbers come from queueing theory; each
                      derivation is in the comment above its test. Sizes and tolerances are the ones of the E2 table.
    TestGolden*       D7: admission_traces.json replayed through simulate(cfg.schedule=...) (`-k golden`).
    everything else   small deterministic scenarios whose answer is worked out by hand in the comments.

Time is in seconds here and in the comments; the simulator and the trace use int ns.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from exp2 import contracts as C
from exp2.contracts import PolicySpec, SimConfig, TagSpec, Workload
from exp2.model import simulator as SIM
from exp2.model.simulator import simulate
from exp2.policies import AdmissionCore, Policy
from exp2.tests.mocks import StubSlowdown, UnitSlowdown, make_types

REPO = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
LEVELS = (612, 1020)
# Big simulations (the closed-form cases) run on the Jetson / with EXP2_SLOW=1, never in the default laptop suite.
slow = pytest.mark.skipif(os.environ.get("EXP2_SLOW") != "1", reason="slow: set EXP2_SLOW=1 (Jetson only)")
N100 = dict(n_requests=100000, n_warmup=5000, n_reps=5)      # E2 table: 4.54, 3.12, 3.02 and the instability cases
N40 = dict(n_requests=40000, n_warmup=2000, n_reps=5)        # every other analytic case


# ================================================================================================ helpers
class FlatPower:
    """PowerModelP with p(A, f) = idle_w for an empty A and busy_w otherwise. O(1) per call (the K13 sum over A is
    O(|A|), which dominates the unstable runs with thousands of jobs in service)."""

    def __init__(self, idle=5.0, busy=12.0):
        self.idle, self.busy = idle, busy

    def p(self, A, f_mhz):
        return self.idle if not A else self.busy

    def to_json(self):
        return {}


class CountPower(FlatPower):
    """p = idle + per_job * |A|."""

    def __init__(self, idle=5.0, per_job=3.0):
        super().__init__(idle)
        self.per_job = per_job

    def p(self, A, f_mhz):
        return self.idle + self.per_job * len(A)


class Recorder:
    """SlowdownModel with a call log; s(A) = table[len(A)] (default 1)."""
    name = "rec"

    def __init__(self, table=None, noise=0.0):
        self.table, self.noise = table or {}, noise
        self.calls, self.predicts = [], []

    def predict(self, i, A, f_mhz):
        self.predicts.append((i, A, f_mhz))
        return self.table.get(len(A), 1.0)

    def sample(self, i, A, f_mhz, rng):
        assert isinstance(rng, np.random.Generator)
        self.calls.append((i, A, f_mhz))
        return self.table.get(len(A), 1.0) * (1.0 + self.noise * rng.random())


def det(v):
    return {"p50": v, "mean": v, "p95": v, "samples": [v]}


def exp_stat(n=20000):
    """tau stats whose samples are the midpoint quantile grid of an exponential, rescaled to mean exactly 1."""
    q = -np.log(1.0 - (np.arange(n) + 0.5) / n)
    q = q / q.mean()
    return {"p50": float(np.percentile(q, 50)), "mean": 1.0, "p95": float(np.percentile(q, 95)), "samples": q.tolist()}


EXP = exp_stat()


def table(types, **kw):
    """TypeTable from {type_id: tau-stat dict | {f: stat}} + options (klass, ckpt_frac, head_frac)."""
    spec = {}
    for tid, v in types.items():
        v = dict(v)
        tau = v.pop("tau")
        v["tau"] = tau if all(isinstance(k, int) for k in tau) else {f: tau for f in LEVELS}
        v.setdefault("head_frac", 0.0)
        spec[tid] = v
    return make_types(spec, alpha=None, f_levels=LEVELS, **kw)


def pspec(k_short=None, k_long=None, order="fcfs", f_rule="static", f_mhz=1020, lane_cap=None, params=None, name="p"):
    f = f_mhz if f_rule == "static" else None
    return PolicySpec(name, k_short, k_long, order, lane_cap, f_rule, f, params or {})


def wload(ids, probs, rho, mean_service=1.0):
    return Workload("w", tuple(ids), tuple(probs), rho, rho / mean_service)


def item(rid, tid, t_s, dl_s=None, ko=None):
    return {"req_id": rid, "type_id": tid, "t_arrival_ns": int(round(t_s * 1e9)),
            "t_deadline_ns": None if dl_s is None else int(round(dl_s * 1e9)), "k_override": ko}


def replay(items, spec, types, *, slowdown=None, power=None, policy=None, n_warmup=0, lag=0.0, tag=None, alpha=None,
           exclusive=True, sample=False):
    ids = sorted({it["type_id"] for it in items})
    wl = Workload("replay", tuple(ids), tuple(1.0 / len(ids) for _ in ids), 1.0, 1.0)
    cfg = SimConfig(n_warmup=n_warmup, family_exclusive=exclusive, sample_slowdown=sample, alpha=alpha,
                    switch_lag_s=lag, tag=tag, schedule=tuple(items))
    return simulate(wl, spec, types, slowdown or UnitSlowdown(), power or FlatPower(), cfg, policy)


def rows(res):
    return {r["req_id"]: r for r in res.trace}


def ns(x_s):
    return int(round(x_s * 1e9))


def near_ns(got, want_s):
    """Hand-computed seconds vs a simulated int ns: the clock resolution is 1 ns."""
    assert got is not None and abs(got - want_s * 1e9) <= 1.5, (got, want_s * 1e9)


@pytest.fixture
def rt(record_property):
    """D16: record the wall time of a simulate() call (junit property `runtime_s`, printed with -s)."""
    def note(res):
        record_property("runtime_s", round(res.runtime_s, 2))
        print("runtime_s=%.2f" % res.runtime_s)
        RUNTIMES.append(res.runtime_s)
        return res
    return note


RUNTIMES = []
N_ANALYTIC_TESTS = 15        # keep equal to the number of test_analytic_* tests below (the budget test counts them)


def mean_R(res):
    n = sum(s.n for s in res.per_type.values())
    return sum(s.n * s.mean_R for s in res.per_type.values()) / n


def one_type(stat=EXP, tid="bert:3"):
    return table({tid: {"tau": stat, "klass": "short"}})


def mix_types():
    """90% short (size 0.357) / 10% long (6.786): E[S] = 0.9999, E[S^2] = 4.7197."""
    return table({"yolo:0": {"tau": det(0.357), "klass": "short"}, "bert:3": {"tau": det(6.786), "klass": "long"}})


MIX_MEAN = 0.9 * 0.357 + 0.1 * 6.786


# ================================================================================================ D6 analytic
@pytest.mark.slow
@slow
class TestAnalyticClosedForms:
    """Means over 5 reps against queueing theory. Unit of time = mean service."""

    def test_analytic_mm1_fcfs(self, rt):
        # M/M/1: R = 1 / (1 - rho) = 2.5 at rho = 0.6
        res = rt(simulate(wload(["bert:3"], [1.0], 0.6), pspec(1, 1), one_type(), UnitSlowdown(), FlatPower(),
                          SimConfig(family_exclusive=False, alpha=None, **N40)))
        assert res.per_type["bert:3"].mean_R == pytest.approx(2.50, rel=0.04)

    def test_analytic_mg1_fcfs_mix_90_10(self, rt):
        # Pollaczek-Khinchine: W = lam E[S^2] / (2 (1 - rho)) = 0.6 / 0.9999 * 4.7197 / 0.8 = 3.54 ; R = W + E[S] = 4.54
        res = rt(simulate(wload(["yolo:0", "bert:3"], [0.9, 0.1], 0.6, MIX_MEAN), pspec(1, 1), mix_types(),
                          UnitSlowdown(), FlatPower(), SimConfig(family_exclusive=False, alpha=None, **N100)))
        assert mean_R(res) == pytest.approx(4.54, rel=0.06)

    def test_analytic_corun_processor_sharing_c093(self, rt):
        # k = inf, s = n / c(n), c(1) = 1, c(n>=2) = 0.93, exponential sizes: N is a birth-death chain with birth
        # rate lam and death rate c(n) -> E[N] = 1.7709 at lam = 0.6, R = E[N] / lam = 2.951 (Little)
        res = rt(simulate(wload(["bert:3"], [1.0], 0.6), pspec(), one_type(), StubSlowdown({1: 1.0, 2: 0.93}),
                          FlatPower(), SimConfig(family_exclusive=False, alpha=None, **N40)))
        assert res.per_type["bert:3"].mean_R == pytest.approx(2.951, rel=0.04)

    def test_analytic_corun_processor_sharing_c07(self, rt):
        # same chain with c(n>=2) = 0.7 at lam = 0.3: R = 2.008
        res = rt(simulate(wload(["bert:3"], [1.0], 0.3), pspec(), one_type(), StubSlowdown({1: 1.0, 2: 0.7}),
                          FlatPower(), SimConfig(family_exclusive=False, alpha=None, **N40)))
        assert res.per_type["bert:3"].mean_R == pytest.approx(2.008, rel=0.04)

    def test_analytic_priority_stf_short_class_rho085(self, rt):
        # non-preemptive priority M/G/1, short first: W0 = lam E[S^2] / 2 = 2.0059, rho1 = 0.2731
        # W_short = W0 / (1 - rho1) = 2.759 ; R_short = W_short + 0.357 = 3.117
        res = rt(simulate(wload(["yolo:0", "bert:3"], [0.9, 0.1], 0.85, MIX_MEAN), pspec(1, 1, order="stf"),
                          mix_types(), UnitSlowdown(), FlatPower(),
                          SimConfig(family_exclusive=False, alpha=None, **N100)))
        assert res.per_type["yolo:0"].mean_R == pytest.approx(3.12, rel=0.05)

    def test_analytic_priority_stf_overall_rho06(self, rt):
        # rho = 0.6: R_short = 2.111, W_long = W0 / ((1 - rho1)(1 - rho)) = 4.385, R_long = 11.17 ; mean = 3.017
        res = rt(simulate(wload(["yolo:0", "bert:3"], [0.9, 0.1], 0.6, MIX_MEAN), pspec(1, 1, order="stf"),
                          mix_types(), UnitSlowdown(), FlatPower(),
                          SimConfig(family_exclusive=False, alpha=None, **N100)))
        assert mean_R(res) == pytest.approx(3.02, rel=0.05)
        assert res.per_type["bert:3"].mean_R == pytest.approx(11.17, rel=0.10)      # the long class waits behind shorts

    def test_analytic_family_exclusive_single_family_is_mm1(self, rt):
        # k = inf but one family: family exclusivity makes it a single server -> M/M/1, R = 2.5
        res = rt(simulate(wload(["bert:3"], [1.0], 0.6), pspec(), one_type(), UnitSlowdown(), FlatPower(),
                          SimConfig(family_exclusive=True, alpha=None, **N40)))
        assert res.per_type["bert:3"].mean_R == pytest.approx(2.50, rel=0.04)

    def test_analytic_infinite_server_when_not_family_exclusive(self, rt):
        # the contrast: the same system with family_exclusive=False and s = 1 is M/G/inf, R = E[S] = 1
        res = rt(simulate(wload(["bert:3"], [1.0], 0.6), pspec(), one_type(), UnitSlowdown(), FlatPower(),
                          SimConfig(family_exclusive=False, alpha=None, **N40)))
        assert res.per_type["bert:3"].mean_R == pytest.approx(1.0, rel=0.03)

    def test_analytic_two_families_do_not_interfere(self, rt):
        # s = 1, two families each carrying rho = 0.3 -> two independent M/M/1: R = 1 / 0.7 = 1.4286 each
        types = table({"bert:3": {"tau": EXP, "klass": "short"}, "yolo:0": {"tau": EXP, "klass": "short"}})
        res = rt(simulate(wload(["bert:3", "yolo:0"], [0.5, 0.5], 0.6), pspec(), types, UnitSlowdown(), FlatPower(),
                          SimConfig(family_exclusive=True, alpha=None, **N40)))
        for tid in ("bert:3", "yolo:0"):
            assert res.per_type[tid].mean_R == pytest.approx(1.4286, rel=0.04)

    def test_analytic_mm1_with_deadline_expired_rate(self, rt):
        # M/M/1 + D, one checkpoint at the end of service, D = 2, rho = 0.6. An arrival whose virtual wait is >= D is
        # dropped; workload density f(v) = lam P0 exp((lam-1) v) on (0, D), P0 [1 + lam/(lam-1) (e^{(lam-1)D} - 1)
        # + lam e^{(lam-1)D}] = 1 -> P0 = 0.47719, expired = lam P0 e^{(lam-1)D} = 0.12865
        types = one_type()
        alpha = 2.0 / EXP["p50"]                                      # D = alpha * p50 = 2 mean services
        (probe,) = C.with_deadlines([item("p", "bert:3", 0.0)], types, alpha)
        assert probe["t_deadline_ns"] == 2_000_000_000
        res = rt(simulate(wload(["bert:3"], [1.0], 0.6), pspec(1, 1), types, UnitSlowdown(), FlatPower(),
                          SimConfig(family_exclusive=False, alpha=alpha, **N40)))
        st = res.per_type["bert:3"]
        assert st.expired_rate == pytest.approx(0.1286, abs=0.01)
        assert st.truncation_rate == 0.0                              # one checkpoint: late means miss, never truncated
        assert st.miss_rate > 0.05                                    # admitted with wait w < 2 but w + S > 2
        assert st.complete_rate + st.miss_rate + st.expired_rate == pytest.approx(1.0)
        assert not res.unstable                                       # alpha is set

    @pytest.mark.parametrize("case", ["kinf_c093", "k2_c093"])
    def test_analytic_prop1_overload_is_flagged_unstable(self, case, rt):
        # lam = 0.95 > capacity c = 0.93 when two or more jobs run: N(t) drifts up at lam - c = 0.02 jobs/s, so the OLS
        # slope on the second half is about 0.02 > 0.25 (lam - c) = 0.005 (and > the detector's 0.005 lam = 0.00475)
        k = None if case == "kinf_c093" else 2
        res = rt(simulate(wload(["bert:3"], [1.0], 0.95), pspec(k, k), one_type(), StubSlowdown({1: 1.0, 2: 0.93}),
                          FlatPower(), SimConfig(family_exclusive=False, alpha=None, **N100)))
        slopes = [r["n_slope"] for r in res.reps]
        assert sum(s > 0.25 * (0.95 - 0.93) for s in slopes) >= 4, slopes
        assert res.unstable is True

    @pytest.mark.parametrize("case", ["k1_rho095", "kinf_rho085", "kinf_rho095_alpha5"])
    def test_analytic_prop1_stable_or_deadlined_is_not_flagged(self, case, rt):
        # k = 1, rho = 0.95: capacity 1 > 0.95, stable ; k = inf, rho = 0.85 < 0.93: stable ;
        # k = inf, rho = 0.95 with alpha = 5: N still drifts up, but with deadlines the detector is off by definition
        k, rho, alpha = {"k1_rho095": (1, 0.95, None), "kinf_rho085": (None, 0.85, None),
                         "kinf_rho095_alpha5": (None, 0.95, 5.0)}[case]
        res = rt(simulate(wload(["bert:3"], [1.0], rho), pspec(k, k), one_type(), StubSlowdown({1: 1.0, 2: 0.93}),
                          FlatPower(), SimConfig(family_exclusive=False, alpha=alpha, **N100)))
        slopes = [r["n_slope"] for r in res.reps]
        assert res.unstable is False
        if alpha is None:
            assert sum(s > C.UNSTABLE_SLOPE_FRAC * rho for s in slopes) <= 1, slopes     # truly stable, not just gated
        else:
            assert sum(s > 0.25 * (0.95 - 0.93) for s in slopes) >= 4, slopes             # the gate hides a real drift


@pytest.mark.slow
@slow
def test_analytic_zz_total_runtime_within_ten_minutes():
    """E2 DoD: `-k analytic` must finish in 10 minutes. Counts only when every analytic case ran in this session."""
    if len(RUNTIMES) < N_ANALYTIC_TESTS:
        pytest.skip("only %d of %d analytic simulations ran in this session" % (len(RUNTIMES), N_ANALYTIC_TESTS))
    print("analytic total simulate() time: %.1f s over %d runs" % (sum(RUNTIMES), len(RUNTIMES)))
    assert sum(RUNTIMES) <= 600.0


class TestSmoke:
    def test_smoke_mm1_sanity_bounds(self):
        # 2000 requests x 2 reps: loose bounds only (the exact targets live in the slow analytic tests)
        res = simulate(wload(["bert:3"], [1.0], 0.5), pspec(1, 1), one_type(), UnitSlowdown(), FlatPower(),
                       SimConfig(n_requests=2000, n_warmup=100, n_reps=2, alpha=None, family_exclusive=False))
        st = res.per_type["bert:3"]
        assert 1.2 < st.mean_R < 3.5 and st.complete_rate == 1.0 and res.unstable is False
        assert 0.3 < res.utilization < 0.7


# ================================================================================================ D7 golden replay
def _golden():
    return json.loads((FIXTURES / "admission_traces.json").read_text(encoding="utf-8"))


class _RecordingCore(AdmissionCore):
    commands, initial = [], []

    def __init__(self, policy, types, f_initial_mhz):
        super().__init__(policy, types, f_initial_mhz)
        _RecordingCore.initial.append(f_initial_mhz)

    def _log(self, d):
        if d.commanded_f is not None:
            _RecordingCore.commands.append(d.commanded_f)
        return d

    def arrive(self, req, now_ns):
        return self._log(super().arrive(req, now_ns))

    def done(self, req_id, now_ns):
        return self._log(super().done(req_id, now_ns))


class TestGoldenReplay:
    @pytest.mark.parametrize("tr", _golden(), ids=lambda t: t["id"])
    def test_golden_trace_replay_matches_admission_traces(self, tr, monkeypatch):
        # service is deterministic (done = admit + service_ns) and independent of f, exactly as in the S1b replay:
        # job sizes come from a table whose every tau sample is the service time, the policy decides on the example
        # table (its tau p50/p95 at 612/1020 drive stf, edgeserving and greenllm).
        monkeypatch.setattr(_RecordingCore, "commands", [])
        monkeypatch.setattr(_RecordingCore, "initial", [])
        monkeypatch.setattr(SIM, "AdmissionCore", _RecordingCore)
        decide_on = C.load_types(str(REPO / tr["types_ref"]))
        doc = decide_on.to_json()
        for t in doc["types"]:
            if t["type_id"] in tr["service_ns"]:
                t["tau_s"] = {f: det(tr["service_ns"][t["type_id"]] / 1e9) for f in t["tau_s"]}
        sized_by = C.TypeTable.from_json(doc)
        spec = PolicySpec.from_json(tr["spec"])
        res = replay(tr["arrivals"], spec, sized_by, policy=Policy(spec, decide_on))
        got = rows(res)
        assert set(got) == set(tr["expect"])
        for rid, want in tr["expect"].items():
            row = got[rid]
            assert row["t_admit_ns"] == want["t_admit_ns"], (tr["id"], rid)
            if want["outcome"] == "expired_in_queue":
                assert row["outcome"] == "expired_in_queue"
                assert row["t_done_ns"] == want["t_done_ns"]
                assert row["completion"] == 0.0 and row["reached"] is None and row["t_end_ns"] is None
            else:
                assert row["outcome"] in ("complete", "truncated", "miss")
                assert row["t_done_ns"] - row["t_admit_ns"] == tr["service_ns"][row["type_id"]]
        assert _RecordingCore.commands == tr["expect_commanded_f"]
        assert _RecordingCore.initial == [tr["f_initial_mhz"]]

    def test_golden_replay_forces_a_single_rep_and_fills_the_trace(self):
        tr = _golden()[0]
        res = replay(tr["arrivals"], PolicySpec.from_json(tr["spec"]), C.load_types(str(REPO / tr["types_ref"])))
        assert len(res.reps) == 1 and len(res.trace) == len(tr["arrivals"])
        assert [r["req_id"] for r in res.trace] == [a["req_id"] for a in tr["arrivals"]]


# ================================================================================================ truncation (K7)
TRUNC = table({"bert:11": {"tau": det(10.0), "klass": "long", "head_frac": 0.1,
                           "ckpt_frac": {"3": 0.3, "7": 0.6, "11": 0.9}}})
# bert:11 has checkpoints [3, 7, 11] reached at 3 s, 6 s, 9 s after admit; the head takes 1 s; total 10 s.


class TestTruncation:
    @pytest.mark.parametrize("dl, outcome, reached, t_done, ckpt_times", [
        (9.0, "complete", 11, 10.0, [3, 6, 9]),               # last checkpoint at 9 == deadline: on time ; head after
        (8.999, "truncated", 7, 10.0, [3, 6, 9]),             # third checkpoint 1 ms late: stop there, head of 7 (1 s)
        (5.0, "truncated", 3, 7.0, [3, 6]),                   # second checkpoint (6 s) late: stop at 6 s, head -> 7 s
        (3.0, "truncated", 3, 7.0, [3, 6]),                   # first checkpoint at 3 == deadline is on time
        (2.0, "miss", None, 3.0, [3]),                        # first checkpoint late: miss, no head, leaves at 3 s
        (100.0, "complete", 11, 10.0, [3, 6, 9]),
    ])
    def test_deadline_between_checkpoints(self, dl, outcome, reached, t_done, ckpt_times):
        # r2 arrives at 1 s into a serial system: it is admitted when r1 leaves, so t_admit(r2) == t_done(r1)
        items = [item("r1", "bert:11", 0.0, dl), item("r2", "bert:11", 1.0)]
        got = rows(replay(items, pspec(1, 1), TRUNC))
        r1 = got["r1"]
        assert r1["outcome"] == outcome and r1["reached"] == reached
        near_ns(r1["t_done_ns"], t_done)
        near_ns(r1["t_end_ns"], t_done)
        assert r1["completion"] == (0.0 if reached is None else pytest.approx(C.completion("bert", 11, reached)))
        assert [c for c, _ in r1["ckpt"]] == [3, 7, 11][:len(ckpt_times)]
        for (_, t), want in zip(r1["ckpt"], ckpt_times):
            near_ns(t, want)
        near_ns(got["r2"]["t_admit_ns"], t_done)                  # the late segment and the head kept the GPU

    def test_truncation_outcome_is_the_shared_function_and_its_answer_is_used(self, monkeypatch):
        calls = []
        real = C.truncation_outcome

        def spy(ckpts, t_ckpt_ns, deadline_ns):
            calls.append((list(ckpts), list(t_ckpt_ns), deadline_ns))
            return real(ckpts, t_ckpt_ns, deadline_ns)

        monkeypatch.setattr(C, "truncation_outcome", spy)
        res = replay([item("r1", "bert:11", 0.0, 5.0)], pspec(1, 1), TRUNC)
        assert len(calls) == 1
        ckpts, times, dl = calls[0]
        assert ckpts == [3, 7, 11] and dl == 5_000_000_000
        assert len(times) == 2 and all(type(t) is int for t in times)       # on-time checkpoint + the first late one
        near_ns(times[0], 3.0)
        near_ns(times[1], 6.0)
        assert rows(res)["r1"]["outcome"] == "truncated"
        # the verdict is not recomputed by the simulator:
        monkeypatch.setattr(C, "truncation_outcome", lambda c, t, d: (None, "miss"))
        res = replay([item("r1", "bert:11", 0.0, 100.0)], pspec(1, 1), TRUNC)
        assert rows(res)["r1"]["outcome"] == "miss" and rows(res)["r1"]["reached"] is None

    def test_head_share_is_head_frac_after_a_stop_and_a_complete_job_ends_at_one(self):
        # ckpt_frac[last] + head_frac = 0.98 (inside the 0.05 contract slack). Truncated at the 6 s checkpoint: the head
        # takes head_frac * tau = 0.8 s -> leaves at 6.8. A complete job is a whole 10 s long.
        types = table({"bert:11": {"tau": det(10.0), "klass": "long", "head_frac": 0.08,
                                   "ckpt_frac": {"3": 0.3, "7": 0.6, "11": 0.9}}})
        got = rows(replay([item("r1", "bert:11", 0.0, 5.0), item("r2", "bert:11", 20.0, 100.0)], pspec(1, 1), types))
        near_ns(got["r1"]["t_done_ns"], 6.8)
        assert got["r2"]["outcome"] == "complete"
        near_ns(got["r2"]["t_done_ns"], 30.0)

    def test_no_deadline_means_complete_and_never_truncates(self):
        res = replay([item("r1", "bert:11", 0.0)], pspec(1, 1), TRUNC)
        r = rows(res)["r1"]
        assert (r["outcome"], r["reached"], r["completion"]) == ("complete", 11, 1.0)
        near_ns(r["t_done_ns"], 10.0)

    def test_alpha_none_never_truncates_even_under_overload(self):
        types = table({"bert:11": {"tau": det(1.0), "klass": "long", "head_frac": 0.1,
                                   "ckpt_frac": {"3": 0.3, "7": 0.6, "11": 0.9}}})
        wl = wload(["bert:11"], [1.0], 3.0)
        cfg = dict(n_requests=400, n_warmup=0, n_reps=1, seed=3, family_exclusive=False)
        none = simulate(wl, pspec(1, 1), types, UnitSlowdown(), FlatPower(), SimConfig(alpha=None, **cfg))
        st = none.per_type["bert:11"]
        assert (st.complete_rate, st.expired_rate, st.miss_rate, st.truncation_rate) == (1.0, 0.0, 0.0, 0.0)
        tight = simulate(wl, pspec(1, 1), types, UnitSlowdown(), FlatPower(), SimConfig(alpha=2.0, **cfg))
        assert tight.per_type["bert:11"].complete_rate < 0.5          # the same overload with deadlines fails requests

    def test_all_four_outcomes_in_one_run_give_the_right_rates_and_completion(self):
        # r1 complete (R 10) ; r2 truncated at 3, completion 4/12, leaves at 27 (R 7) ; r3 miss, leaves at 43 (R 3) ;
        # r4 arrives at 41 while r3 is in service, deadline 42 -> dropped at 43, logged at 42 (R 1)
        items = [item("r1", "bert:11", 0.0, 100.0), item("r2", "bert:11", 20.0, 25.0),
                 item("r3", "bert:11", 40.0, 42.0), item("r4", "bert:11", 41.0, 42.0), item("r5", "bert:11", 60.0)]
        res = replay(items, pspec(1, 1), TRUNC, n_warmup=0)
        got = rows(res)
        outcomes = [got[r]["outcome"] for r in ("r1", "r2", "r3", "r4")]
        assert outcomes == ["complete", "truncated", "miss", "expired_in_queue"]
        assert got["r4"]["t_admit_ns"] is None and got["r4"]["t_done_ns"] == ns(42.0) and got["r4"]["ckpt"] == []
        near_ns(got["r2"]["t_done_ns"], 27.0)
        near_ns(got["r3"]["t_done_ns"], 43.0)
        # r5 arrives alone at 60 and completes (R 10): five measured requests in all
        st = res.per_type["bert:11"]
        assert st.n == 5
        rates = (st.complete_rate, st.truncation_rate, st.miss_rate, st.expired_rate)
        assert rates == pytest.approx((0.4, 0.2, 0.2, 0.2))
        assert st.mean_completion == pytest.approx((1.0 + 4.0 / 12.0 + 0.0 + 0.0 + 1.0) / 5)
        assert st.mean_shortfall == pytest.approx(1.0 - st.mean_completion)
        assert st.mean_R == pytest.approx((10 + 7 + 3 + 1 + 10) / 5)
        C.validate_simresult(res.to_json())


# ================================================================================================ expiry in the queue
class TestExpiry:
    def test_expired_request_is_never_served_and_does_not_delay_the_next_one(self):
        types = table({"bert:3": {"tau": det(10.0), "klass": "long"}, "yolo:0": {"tau": det(4.0), "klass": "short"}})
        items = [item("r1", "bert:3", 0.0, 100.0), item("r2", "yolo:0", 1.0, 4.0), item("r3", "yolo:0", 2.0, 100.0)]
        got = rows(replay(items, pspec(1, 1), types))
        r2 = got["r2"]
        assert r2["outcome"] == "expired_in_queue" and r2["t_admit_ns"] is None and r2["t_end_ns"] is None
        assert r2["t_done_ns"] == ns(4.0)                           # logged at its deadline, not when it was noticed
        assert r2["completion"] == 0.0 and r2["reached"] is None and r2["ckpt"] == []
        near_ns(got["r3"]["t_admit_ns"], 10.0)                      # r2 consumed no service: r3 starts when r1 leaves
        near_ns(got["r3"]["t_done_ns"], 14.0)

    def test_deadline_equal_to_the_event_time_is_dropped(self):
        # r1 leaves at 10 s == r2's deadline: now >= deadline -> dropped before the admit
        types = table({"bert:3": {"tau": det(10.0), "klass": "long"}, "yolo:0": {"tau": det(1.0), "klass": "short"}})
        items = [item("r1", "bert:3", 0.0), item("r2", "yolo:0", 1.0, 10.0), item("r3", "yolo:0", 2.0)]
        got = rows(replay(items, pspec(1, 1), types))
        assert got["r2"]["outcome"] == "expired_in_queue" and got["r3"]["t_admit_ns"] == ns(10.0)

    def test_response_time_of_an_expired_request_is_its_deadline_minus_arrival(self):
        types = table({"bert:3": {"tau": det(10.0), "klass": "long"}, "yolo:0": {"tau": det(1.0), "klass": "short"}})
        res = replay([item("r1", "bert:3", 0.0), item("r2", "yolo:0", 1.0, 4.0)], pspec(1, 1), types)
        st = res.per_type["yolo:0"]
        assert (st.n, st.expired_rate, st.mean_R, st.mean_completion, st.mean_shortfall) == (1, 1.0, 3.0, 0.0, 1.0)


# =========================================================================================== per-family exclusivity
class TestFamilyExclusivity:
    TYPES = table({"bert:3": {"tau": det(4.0), "klass": "short"}, "bert:5": {"tau": det(4.0), "klass": "short"},
                   "yolo:0": {"tau": det(4.0), "klass": "short"}})

    def test_same_family_serialises_other_family_overlaps(self):
        items = [item("r1", "bert:3", 0.0), item("r2", "bert:5", 1.0), item("r3", "yolo:0", 2.0)]
        got = rows(replay(items, pspec(), self.TYPES, exclusive=True))
        near_ns(got["r2"]["t_admit_ns"], 4.0)           # bert busy until r1 is done
        near_ns(got["r3"]["t_admit_ns"], 2.0)           # yolo is a different family: admitted at once
        assert got["r3"]["n_in_service_at_admit"] == 1

    def test_without_exclusivity_the_same_family_runs_concurrently(self):
        items = [item("r1", "bert:3", 0.0), item("r2", "bert:5", 1.0)]
        got = rows(replay(items, pspec(), self.TYPES, exclusive=False))
        near_ns(got["r2"]["t_admit_ns"], 1.0)
        assert got["r2"]["n_in_service_at_admit"] == 1


# ================================================================================================ slowdown sampling
class TestSlowdown:
    TYPES = table({"bert:3": {"tau": det(10.0), "klass": "short"}, "yolo:0": {"tau": det(10.0), "klass": "short"}})

    def test_slowdown_is_resampled_every_time_the_running_set_changes(self):
        # r1 (bert) alone 0..4 at s = 1: 4 s of 10 done. r2 (yolo) joins at 4: A has two jobs, s = 2 for both.
        # r1: 6 s left at rate 1/2 -> 12 s -> done at 16. r2: 6 s done by 16 (12 s at 1/2), 4 s left -> done at 20.
        model = Recorder({2: 2.0})
        res = replay([item("r1", "bert:3", 0.0), item("r2", "yolo:0", 4.0)], pspec(), self.TYPES, slowdown=model,
                     sample=True)
        got = rows(res)
        near_ns(got["r1"]["t_done_ns"], 16.0)
        near_ns(got["r2"]["t_done_ns"], 20.0)
        assert model.calls == [("bert:3", ("bert:3",), 1020),                                   # r1 admitted
                               ("bert:3", ("bert:3", "yolo:0"), 1020),                         # r2 admitted
                               ("yolo:0", ("bert:3", "yolo:0"), 1020),
                               ("yolo:0", ("yolo:0",), 1020)]                                   # r1 left: A shrinks
        assert model.predicts == []

    def test_sample_slowdown_false_uses_predict_only(self):
        model = Recorder({2: 2.0})
        res = replay([item("r1", "bert:3", 0.0), item("r2", "yolo:0", 4.0)], pspec(), self.TYPES, slowdown=model,
                     sample=False)
        near_ns(rows(res)["r1"]["t_done_ns"], 16.0)
        assert model.calls == [] and len(model.predicts) == 4

    def test_a_multiset_of_the_same_type_is_passed_with_multiplicity(self):
        types = table({"bert:3": {"tau": det(10.0), "klass": "short"}})
        model = Recorder({1: 1.0, 2: 1.0, 3: 1.0})
        replay([item("r1", "bert:3", 0.0), item("r2", "bert:3", 1.0), item("r3", "bert:3", 2.0)], pspec(), types,
               slowdown=model, exclusive=False, sample=True)
        assert max(len(a) for _, a, _ in model.calls) == 3 and ("bert:3",) * 3 in {a for _, a, _ in model.calls}

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_a_non_positive_or_non_finite_slowdown_is_an_error(self, bad):
        with pytest.raises(ValueError, match="slowdown"):
            replay([item("r1", "bert:3", 0.0)], pspec(), self.TYPES, slowdown=Recorder({1: bad}), sample=True)

    def test_same_seed_same_result_other_seed_other_result(self):
        types = one_type()
        wl = wload(["bert:3"], [1.0], 0.7)
        cfg = dict(n_requests=400, n_warmup=20, n_reps=2, alpha=None, family_exclusive=False)

        def run(seed):
            r = simulate(wl, pspec(), types, Recorder({2: 1.3}, noise=0.4), FlatPower(), SimConfig(seed=seed, **cfg))
            j = r.to_json()
            j["runtime_s"] = 0.0
            return j

        assert run(5) == run(5)
        assert run(5) != run(6)


# ================================================================================================ power and energy
class TestEnergy:
    TYPES = table({"bert:3": {"tau": det(3.0), "klass": "short"}, "yolo:0": {"tau": det(1.0), "klass": "short"}})

    def test_energy_is_the_integral_of_power_including_idle_gaps(self):
        # r1: 2 -> 5 (3 s busy), idle 5 -> 8 (3 s), r2: 8 -> 9 (1 s busy). Window [2, 9] = 7 s.
        # E = 4 s * 12 W + 3 s * 5 W = 63 J ; per request 31.5 J ; mean power 9 W
        items = [item("r1", "bert:3", 2.0), item("r2", "yolo:0", 8.0)]
        res = replay(items, pspec(1, 1), self.TYPES, power=FlatPower(5.0, 12.0))
        assert res.energy_per_req_j == pytest.approx(31.5, rel=1e-9)
        assert res.mean_power_w == pytest.approx(9.0, rel=1e-9)
        assert res.utilization == pytest.approx(4.0 / 7.0, rel=1e-9)
        assert res.mean_in_system == pytest.approx(4.0 / 7.0, rel=1e-9)
        assert res.throughput == pytest.approx(2.0 / 7.0, rel=1e-9)

    def test_warmup_requests_leave_the_window_and_the_count(self):
        # r1 is warmup: the window starts at r2's arrival (8 s) and ends at 9 s: 1 s busy = 12 J, one request
        items = [item("r1", "bert:3", 2.0), item("r2", "yolo:0", 8.0)]
        res = replay(items, pspec(1, 1), self.TYPES, power=FlatPower(5.0, 12.0), n_warmup=1)
        assert res.energy_per_req_j == pytest.approx(12.0, rel=1e-9)
        assert sum(s.n for s in res.per_type.values()) == 1

    def test_power_follows_the_running_set(self):
        # r1 0..3 alone, r2 (yolo, 1 s) joins at 1, s = 1: A = 1 job 0..1 (8 W), 2 jobs 1..2 (11 W), 1 job 2..3 (8 W)
        # window [0, 3]: E = 8 + 11 + 8 = 27 J over 2 requests
        res = replay([item("r1", "bert:3", 0.0), item("r2", "yolo:0", 1.0)], pspec(), self.TYPES,
                     power=CountPower(5.0, 3.0))
        assert res.energy_per_req_j == pytest.approx(13.5, rel=1e-9)

    def test_window_ends_at_the_deadline_of_an_expired_request(self):
        # r1 (warmup) holds the GPU 0..3 ; r2 arrives at 1 and expires at 2 (never served). Window = [1, 2]: 1 s * 12 W
        res = replay([item("r1", "bert:3", 0.0), item("r2", "yolo:0", 1.0, 2.0)], pspec(1, 1), self.TYPES,
                     power=FlatPower(5.0, 12.0), n_warmup=1)
        assert res.energy_per_req_j == pytest.approx(12.0, rel=1e-9)
        assert res.per_type["yolo:0"].expired_rate == 1.0

    def test_power_model_sees_the_effective_level_and_the_empty_set(self):
        seen = []

        class FPower(FlatPower):
            def p(self, A, f_mhz):
                seen.append((tuple(A), f_mhz))
                return super().p(A, f_mhz)

        replay([item("r1", "bert:3", 0.0)], pspec(), self.TYPES, power=FPower())
        assert seen[0] == ((), 1020) and ((("bert:3",), 1020) in seen)


# ================================================================================================ tagged deviators
class TestTaggedDeviators:
    TYPES = table({"bert:3": {"tau": det(10.0), "klass": "long"}, "yolo:0": {"tau": det(1.0), "klass": "short"}})

    def test_every_second_short_request_deviates_and_its_statistics_are_separate(self):
        # serial (k = 1). r1 long 0..10. shorts r2 (t=1), r3 (2), r4 (3), r5 (4); every=2 tags r3 and r5 (k_dev = inf).
        # tagged r3 is admitted at 2 (n = 1 < inf, yolo family idle) -> done 3, R 1 ; tagged r5 admitted at 4 -> done 5
        # untagged: r2 admitted at 10 -> done 11 (R 10) ; r4 admitted at 11 -> done 12 (R 9). yolo mean R = 5.25
        items = [item("r1", "bert:3", 0.0), item("r2", "yolo:0", 1.0), item("r3", "yolo:0", 2.0),
                 item("r4", "yolo:0", 3.0), item("r5", "yolo:0", 4.0)]
        res = replay(items, pspec(1, 1), self.TYPES, tag=TagSpec("short", C.K_INF, 2))
        got = rows(res)
        assert [got[r]["k_override"] for r in ("r1", "r2", "r3", "r4", "r5")] == [None, None, C.K_INF, None, C.K_INF]
        near_ns(got["r3"]["t_admit_ns"], 2.0)
        near_ns(got["r5"]["t_admit_ns"], 4.0)
        near_ns(got["r2"]["t_admit_ns"], 10.0)
        near_ns(got["r4"]["t_admit_ns"], 11.0)
        assert res.per_type["yolo:0"].mean_R == pytest.approx(5.25)
        tg = res.tagged
        assert (tg["klass"], tg["k_dev"], tg["n"]) == ("short", C.K_INF, 2)
        assert tg["mean_R"] == pytest.approx(1.0) and tg["complete_rate"] == 1.0
        assert set(tg) == {"klass", "k_dev", "n", "mean_R", "p50_R", "p95_R", "p99_R", "complete_rate",
                           "truncation_rate", "miss_rate", "expired_rate", "mean_completion", "mean_shortfall"}
        assert res.reps[0]["tagged"]["mean_R"] == pytest.approx(1.0)
        C.validate_simresult(res.to_json())

    def test_tag_counts_requests_of_its_class_only_and_skips_warmup_in_the_statistics(self):
        # 1000 spaced shorts (each alone), every 50th is tagged: 20 tagged ; the first 100 are warmup -> 18 measured
        items = [item("s%04d" % i, "yolo:0", 5.0 * i) for i in range(1000)]
        items.insert(500, item("long1", "bert:3", 5.0 * 500 - 2.0))
        res = replay(items, pspec(), self.TYPES, tag=TagSpec("short", 3, 50), n_warmup=100, exclusive=True)
        assert res.tagged["n"] == 18 and res.tagged["k_dev"] == 3
        flagged = [r["req_id"] for r in res.trace if r["k_override"] == 3]
        assert len(flagged) == 20 and flagged[0] == "s0049" and flagged[1] == "s0099"

    def test_no_tag_means_no_tagged_block(self):
        res = replay([item("r1", "bert:3", 0.0)], pspec(1, 1), self.TYPES)
        assert res.tagged is None and res.reps[0]["tagged"] is None


# =========================================================================================== dynamic f and switch lag
class TestSwitchLag:
    # tau(1020) = 10 s, tau(612) = 20 s. occupancy with f_base = 612, alpha_p = 1 -> f(n_sys) = 612 for n_sys = 1 and
    # the top level (1020) for n_sys >= 2. The initial level is f_max = 1020.
    TYPES = table({tid: {"tau": {612: det(20.0), 1020: det(10.0)}, "klass": "short"} for tid in ("bert:3", "yolo:0")})
    SPEC = pspec(f_rule="occupancy", params={"f_base_mhz": 612, "alpha_p": 1}, name="occ")
    ITEMS = [item("r1", "bert:3", 0.0), item("r2", "yolo:0", 3.0)]

    def test_lag_zero_timeline(self):
        # t=0 r1 admitted -> command 612 (r1 at 1/20 per s). t=3: r1 at 0.15 ; r2 admitted -> command 1020 (both 1/10).
        # r1 needs 0.85 * 10 = 8.5 s -> done 11.5 ; r2 has 0.85 -> n_sys = 1 -> command 612: 0.15 * 20 = 3 s -> 14.5
        got = rows(replay(self.ITEMS, self.SPEC, self.TYPES, lag=0.0))
        near_ns(got["r1"]["t_done_ns"], 11.5)
        near_ns(got["r2"]["t_done_ns"], 14.5)

    def test_lag_four_timeline_applies_only_the_newest_command_after_the_one_in_flight(self):
        # t=0 command 612 -> applies at 4 (r1 runs at 1020 for 4 s: 0.4). t=3 r2 joins at 1020, command 1020 while the
        # first switch is in flight -> waits. t=4: 612 applies (r1 0.4, r2 0.1), the pending 1020 starts, lands at 8.
        # 4..8 at 612: r1 0.6, r2 0.3. 8..: 1020: r1 needs 4 s -> done 12 ; r2 0.7 at 12 -> command 612 (applies at 16);
        # r2 needs 0.3 * 10 = 3 s at 1020 -> done 15, before the switch lands
        got = rows(replay(self.ITEMS, self.SPEC, self.TYPES, lag=4.0))
        near_ns(got["r1"]["t_done_ns"], 12.0)
        near_ns(got["r2"]["t_done_ns"], 15.0)

    def test_lag_changes_the_result(self):
        a = rows(replay(self.ITEMS, self.SPEC, self.TYPES, lag=0.0))
        b = rows(replay(self.ITEMS, self.SPEC, self.TYPES, lag=4.0))
        assert a["r2"]["t_done_ns"] < b["r2"]["t_done_ns"]

    def test_static_policy_never_switches_whatever_the_lag(self):
        a = rows(replay(self.ITEMS, pspec(f_mhz=612), self.TYPES, lag=0.0))
        b = rows(replay(self.ITEMS, pspec(f_mhz=612), self.TYPES, lag=4.0))
        assert a == b
        near_ns(a["r1"]["t_done_ns"], 20.0)                         # always 612

    def test_energy_follows_the_applied_level_not_the_commanded_one(self):
        class FPower(FlatPower):
            def p(self, A, f_mhz):
                return 0.0 if not A else (4.0 if f_mhz == 612 else 10.0)

        # one job, lag 4, command 612 at t=0 applies at t=4: 4 s at 1020 (10 W), then 612: 0.6 * 20 = 12 s at 4 W
        res = replay([item("r1", "bert:3", 0.0)], self.SPEC, self.TYPES, lag=4.0, power=FPower())
        assert res.energy_per_req_j == pytest.approx(4 * 10.0 + 12 * 4.0, rel=1e-9)


# ================================================================================================ instability detector
class TestInstabilityDetector:
    @staticmethod
    def dd1(service_s, n=20000, alpha=None):
        types = table({"bert:3": {"tau": det(service_s), "klass": "short"}})
        items = [item("r%05d" % i, "bert:3", float(i)) for i in range(n)]
        return replay(items, pspec(1, 1), types, alpha=alpha)

    def test_slope_is_lambda_minus_service_rate_and_the_threshold_is_half_a_percent_of_lambda(self):
        # D/D/1 with lam = 1/s. service 1.006 s: N grows at 1 - 1/1.006 = 0.005964 > 0.005 lam -> unstable ;
        # service 1.004 s: 0.003984 < 0.005 lam -> not flagged
        over = self.dd1(1.006)
        assert over.reps[0]["n_slope"] == pytest.approx(1 - 1 / 1.006, abs=3e-4)
        assert over.unstable is True
        near = self.dd1(1.004)
        assert near.reps[0]["n_slope"] == pytest.approx(1 - 1 / 1.004, abs=3e-4)
        assert near.unstable is False

    def test_a_stable_system_is_not_flagged(self):
        res = self.dd1(0.5, n=4000)
        assert abs(res.reps[0]["n_slope"]) < 1e-3 and res.unstable is False

    def test_a_deadlined_run_is_never_flagged(self):
        res = self.dd1(1.5, n=4000, alpha=5.0)             # slope 1/3 per s, but cfg.alpha is set
        assert res.reps[0]["n_slope"] > 0.3 and res.unstable is False

    @pytest.mark.parametrize("slopes, flagged", [
        ([0.0051, 0.0051, 0.0051, 0.0051, 0.0], True),     # 4 of 5 reps above 0.005 * lam (lam = 1)
        ([0.0051, 0.0051, 0.0051, 0.0, 0.0], False),       # 3 of 5: below ceil(0.8 * 5) = 4
        ([0.005, 0.005, 0.005, 0.005, 0.005], False),      # exactly at the threshold is not above it
        ([0.0051] * 5, True),
    ])
    def test_unstable_needs_ceil_0_8_n_reps_reps_above_the_threshold(self, slopes, flagged, monkeypatch):
        it = iter(slopes)
        monkeypatch.setattr(SIM._Rep, "n_slope", lambda self: next(it))
        res = simulate(wload(["bert:3"], [1.0], 1.0), pspec(1, 1), one_type(det(0.1)), UnitSlowdown(), FlatPower(),
                       SimConfig(n_requests=60, n_warmup=5, n_reps=5, alpha=None))
        assert [r["n_slope"] for r in res.reps] == slopes
        assert res.unstable is flagged


# ================================================================================================ statistics
class TestStatistics:
    def test_percentiles_are_nearest_rank_and_mean_is_exact(self):
        # 20 requests at t=0, serial, 1 s each -> R = 1..20 : mean 10.5, p50 = 10, p95 = 19, p99 = 20
        types = table({"bert:3": {"tau": det(1.0), "klass": "short"}})
        res = replay([item("r%02d" % i, "bert:3", 0.0) for i in range(20)], pspec(1, 1), types)
        st = res.per_type["bert:3"]
        assert (st.n, st.mean_R, st.p50_R, st.p95_R, st.p99_R) == (20, pytest.approx(10.5), 10.0, 19.0, 20.0)

    def test_reps_are_independent_seeded_runs_and_the_aggregate_is_their_mean(self):
        wl = wload(["bert:3"], [1.0], 0.6)
        cfg = SimConfig(n_requests=300, n_warmup=50, n_reps=4, seed=7, alpha=None, family_exclusive=False)
        res = simulate(wl, pspec(1, 1), one_type(), UnitSlowdown(), FlatPower(), cfg)
        assert [r["seed"] for r in res.reps] == [7, 8, 9, 10]
        means = [r["per_type"]["bert:3"]["mean_R"] for r in res.reps]
        assert len(set(means)) == 4                                       # different arrivals per rep
        assert res.per_type["bert:3"].mean_R == pytest.approx(np.mean(means))
        assert res.per_type["bert:3"].n == 4 * 250                        # warmup excluded, total over reps
        es = [r["energy_per_req_j"] for r in res.reps]
        assert res.energy_per_req_j == pytest.approx(np.mean(es))
        assert res.energy_ci[0] < res.energy_per_req_j < res.energy_ci[1]
        C.validate_simresult(res.to_json())
        assert C.SimResult.from_json(res.to_json()).to_json() == res.to_json()

    def test_one_rep_has_a_degenerate_energy_interval(self):
        res = simulate(wload(["bert:3"], [1.0], 0.6), pspec(1, 1), one_type(), UnitSlowdown(), FlatPower(),
                       SimConfig(n_requests=200, n_warmup=10, n_reps=1, alpha=None))
        assert res.energy_ci == (res.energy_per_req_j, res.energy_per_req_j)

    def test_runtime_is_recorded(self):
        res = simulate(wload(["bert:3"], [1.0], 0.6), pspec(1, 1), one_type(), UnitSlowdown(), FlatPower(),
                       SimConfig(n_requests=200, n_warmup=10, n_reps=1, alpha=None))
        assert 0.0 < res.runtime_s < 60.0


# ================================================================================================ CRN and plumbing
class TestCommonRandomNumbers:
    SAMPLES = [0.1 * k for k in range(1, 21)]

    def test_job_sizes_depend_on_the_request_not_on_the_admission_order(self):
        # two types with the same size distribution but p50 0.5 < 1.0, so stf and fcfs admit in different orders.
        # Each request's service time must be identical under both policies (a size is a function of seed and index).
        stat_a = dict(p50=0.5, mean=1.05, p95=1.9, samples=self.SAMPLES)
        stat_b = dict(p50=1.0, mean=1.05, p95=1.9, samples=self.SAMPLES)
        types = table({"bert:3": {"tau": stat_b, "klass": "long"}, "yolo:0": {"tau": stat_a, "klass": "short"}})
        items = [item("r%02d" % i, "bert:3" if i % 2 == 0 else "yolo:0", 0.0) for i in range(12)]

        def run(order):
            got = rows(replay(items, pspec(1, 1, order=order), types))
            service = {rid: r["t_end_ns"] - r["t_admit_ns"] for rid, r in got.items()}
            admit_order = [rid for rid, _ in sorted(got.items(), key=lambda kv: kv[1]["t_admit_ns"])]
            return service, admit_order

        service_fcfs, order_fcfs = run("fcfs")
        service_stf, order_stf = run("stf")
        assert order_fcfs != order_stf                              # the admission order really differs
        assert service_fcfs == service_stf                         # yet every request has the same size
        assert len(set(service_fcfs.values())) > 4                 # and the sizes do vary

    def test_arrival_mix_is_identical_across_policies_for_one_seed(self):
        types = table({"bert:3": {"tau": EXP, "klass": "short"}, "yolo:0": {"tau": EXP, "klass": "short"}})
        wl = wload(["bert:3", "yolo:0"], [0.7, 0.3], 0.5)
        cfg = SimConfig(n_requests=500, n_warmup=0, n_reps=2, seed=4, alpha=None)
        a = simulate(wl, pspec(1, 1), types, UnitSlowdown(), FlatPower(), cfg)
        b = simulate(wl, pspec(), types, UnitSlowdown(), FlatPower(), cfg)
        assert {t: s.n for t, s in a.per_type.items()} == {t: s.n for t, s in b.per_type.items()}

    def test_other_seed_other_sizes(self):
        types = table({"bert:3": {"tau": dict(p50=1.0, mean=1.05, p95=1.9, samples=self.SAMPLES), "klass": "short"}})
        items = [item("r%02d" % i, "bert:3", 100.0 * i) for i in range(30)]

        def sizes(seed):
            cfg = SimConfig(n_warmup=0, seed=seed, alpha=None, schedule=tuple(items))
            res = simulate(wload(["bert:3"], [1.0], 0.1), pspec(), types, UnitSlowdown(), FlatPower(), cfg)
            return [r["t_end_ns"] - r["t_admit_ns"] for r in res.trace]

        assert sizes(1) == sizes(1) and sizes(1) != sizes(2)


class TestPlumbing:
    def test_every_arrival_and_every_departure_goes_through_admission_core(self, monkeypatch):
        counts = {"arrive": 0, "done": 0}

        class Spy(AdmissionCore):
            def arrive(self, req, now_ns):
                counts["arrive"] += 1
                return super().arrive(req, now_ns)

            def done(self, req_id, now_ns):
                counts["done"] += 1
                return super().done(req_id, now_ns)

        monkeypatch.setattr(SIM, "AdmissionCore", Spy)
        types = table({"bert:3": {"tau": det(10.0), "klass": "long"}, "yolo:0": {"tau": det(1.0), "klass": "short"}})
        items = [item("r1", "bert:3", 0.0), item("r2", "yolo:0", 1.0, 4.0), item("r3", "yolo:0", 2.0)]
        replay(items, pspec(1, 1), types)
        assert counts == {"arrive": 3, "done": 2}                      # r2 expired in the queue and never ran

    def test_default_policy_is_built_from_spec_types_and_family_exclusive(self, monkeypatch):
        built = []

        class Rec(Policy):
            def __init__(self, spec, types, *, family_exclusive=True):
                built.append((spec.name, family_exclusive))
                super().__init__(spec, types, family_exclusive=family_exclusive)

        monkeypatch.setattr(SIM, "Policy", Rec)
        types = table({"bert:3": {"tau": det(1.0), "klass": "short"}})
        replay([item("r1", "bert:3", 0.0)], pspec(name="zz"), types, exclusive=False)
        replay([item("r1", "bert:3", 0.0)], pspec(name="zz"), types, exclusive=True)
        assert built == [("zz", False), ("zz", True)]

    def test_an_explicit_policy_is_used_as_given(self):
        class NeverAdmit(Policy):
            def admit(self, view):
                return []

        types = table({"bert:3": {"tau": det(1.0), "klass": "short"}})
        spec = pspec()
        with pytest.raises(RuntimeError, match="admits nothing"):
            replay([item("r1", "bert:3", 0.0)], spec, types, policy=NeverAdmit(spec, types))

    def test_input_errors(self):
        types = table({"bert:3": {"tau": det(1.0), "klass": "short"}})
        wl = wload(["bert:3"], [1.0], 0.5)
        with pytest.raises(ValueError, match="nothing to measure"):
            simulate(wl, pspec(), types, UnitSlowdown(), FlatPower(), SimConfig(n_requests=10, n_warmup=10, n_reps=1))
        with pytest.raises(ValueError, match="must not decrease"):
            replay([item("a", "bert:3", 2.0), item("b", "bert:3", 1.0)], pspec(), types)
        with pytest.raises(ValueError, match="unique"):
            replay([item("a", "bert:3", 1.0), item("a", "bert:3", 2.0)], pspec(), types)
        with pytest.raises(C.ContractError, match="no tau"):
            replay([item("a", "bert:3", 1.0)], pspec(f_mhz=714), types)       # 714 is not in the table
        with pytest.raises(ValueError, match="TagSpec"):
            simulate(wl, pspec(), types, UnitSlowdown(), FlatPower(),
                     SimConfig(n_requests=20, n_warmup=0, n_reps=1, tag=TagSpec("medium", 2, 5)))

    def test_governor_and_dynamic_rules_start_at_f_max(self):
        seen = []

        class FP(FlatPower):
            def p(self, A, f_mhz):
                seen.append(f_mhz)
                return super().p(A, f_mhz)

        types = table({"bert:3": {"tau": {612: det(20.0), 1020: det(10.0)}, "klass": "short"}})
        got = rows(replay([item("r1", "bert:3", 0.0)], pspec(f_rule="governor"), types, power=FP()))
        near_ns(got["r1"]["t_done_ns"], 10.0)
        assert set(seen) == {1020}

    def test_trace_rows_have_the_k8_names_and_are_json_serialisable(self):
        types = table({"bert:3": {"tau": det(1.0), "klass": "short"}})
        res = replay([item("r1", "bert:3", 0.0, 5.0)], pspec(1, 1), types)
        row = res.trace[0]
        assert {"req_id", "type_id", "t_arrival_ns", "t_deadline_ns", "t_admit_ns", "t_end_ns", "t_done_ns", "outcome",
                "reached", "completion", "f_mhz", "n_in_service_at_admit", "queue_len_at_arrival", "ckpt"} <= set(row)
        assert row["outcome"] in C.OUTCOMES and row["f_mhz"] == 1020
        json.dumps(res.to_json())


# ================================================================================================ edgeserving_lp (B1)
# Owner 2026-10-25: "baseline lama, di sela per token ini bisa running lagi kan ?". EdgeServing-LP pauses a running
# LLaMA request at its next token boundary when a short request waits. llama:15 runs 3.2 s solo over 32 tokens with
# ckpt_frac k/32 (make_types, head 0), so token k ends at 0.1 k s. bert:23 runs 0.08 s solo. Slowdown 1 everywhere.
LP_TYPES = {"llama:3": {"tau": det(1.6)}, "llama:15": {"tau": det(3.2)},
            "bert:5": {"tau": det(0.02)}, "bert:23": {"tau": det(0.08)}}


def near_lp(got, want_s):
    """Up to 50 ns: every checkpoint event rounds the clock to 1 ns (bert:23 has 6, llama 32)."""
    assert got is not None and abs(got - want_s * 1e9) <= 50, (got, want_s * 1e9)


def _lp_run(method, items):
    from exp2 import exits
    types = table(LP_TYPES)
    spec = exits.sim_specs(1020)[method]
    return rows(replay(items, spec, types, policy=exits.ExitPolicy(spec, types)))


class TestEdgeservingLP:
    # r1 llama arrives at 0 (deadline 10): alone, 0 + 3.2 <= 10, so it runs at its target llama:15 from 0 to 3.2.
    # r2 bert arrives at 0.25 (deadline 5.25), in the middle of token 3 (0.2 .. 0.3).
    ITEMS = [item("r1", "llama:15", 0.0, 10.0), item("r2", "bert:23", 0.25, 5.25)]

    def test_edgeserving_makes_the_short_request_wait_for_the_whole_llama_request(self):
        # k = 1: r2 starts when r1 ends at 3.2; 3.2 + 0.08 <= 5.25 -> target bert:23, done 3.28.
        got = _lp_run("edgeserving", self.ITEMS)
        near_lp(got["r1"]["t_done_ns"], 3.2)
        near_lp(got["r2"]["t_admit_ns"], 3.2)
        near_lp(got["r2"]["t_done_ns"], 3.28)

    def test_edgeserving_lp_runs_the_short_request_at_the_next_token_boundary(self):
        # r1 pauses after token 3 at 0.3; r2 runs alone 0.3 .. 0.38 (0.3 + 0.08 <= 5.25 -> bert:23); r1 resumes at
        # 0.38 with tokens 4..32 (2.9 s) and ends at 3.28 = 3.2 + 0.08, later by exactly the service time of r2.
        es, lp = _lp_run("edgeserving", self.ITEMS), _lp_run("edgeserving_lp", self.ITEMS)
        near_lp(lp["r2"]["t_admit_ns"], 0.3)
        near_lp(lp["r2"]["t_done_ns"], 0.38)
        assert lp["r2"]["run_type_id"] == "bert:23" and lp["r2"]["outcome"] == "complete"
        near_lp(lp["r1"]["t_done_ns"] - es["r1"]["t_done_ns"], 0.08)
        near_lp(lp["r1"]["t_done_ns"], 3.28)
        ck = dict(lp["r1"]["ckpt"])
        near_lp(ck[3], 0.3)
        near_lp(ck[4], 0.48)                          # token 4 = 0.1 s of solo time after the resume at 0.38
        # LLaMA keeps its exit and its admit time, and still completes
        assert (lp["r1"]["run_type_id"], lp["r1"]["outcome"], lp["r1"]["reached"]) == ("llama:15", "complete", 32)
        assert lp["r1"]["t_admit_ns"] == es["r1"]["t_admit_ns"] == 0

    def test_a_queued_llama_request_never_runs_while_the_llama_family_is_paused(self):
        # r3 llama arrives at 0.26 with deadline 0.36. At the 0.3 tick the queue holds r2 and r3. Ranked by S alone,
        # r3 would win: S(r2) = exp((0.04 + 0.08) / 0.1 - 1) = 1.22 > S(r3) = exp((0.05 + 1.6) / 5 - 1) = 0.51
        # (r3 at its floor llama:3, 0.3 + 1.6 > 0.36). The paused family is not a candidate, so r2 runs; r3 expires.
        got = _lp_run("edgeserving_lp", self.ITEMS + [item("r3", "llama:15", 0.26, 0.36)])
        near_lp(got["r2"]["t_admit_ns"], 0.3)
        assert got["r3"]["outcome"] == "expired_in_queue" and got["r3"]["run_type_id"] is None
        near_lp(got["r1"]["t_done_ns"], 3.28)

    def test_without_llama_edgeserving_lp_is_edgeserving(self):
        items = [item("b%d" % k, "bert:23", 0.03 * k, 0.03 * k + 0.5) for k in range(6)]
        assert _lp_run("edgeserving_lp", items) == _lp_run("edgeserving", items)
