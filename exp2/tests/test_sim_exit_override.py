"""S2 tests: the simulator runs Decision.run_type_ids (PLAN PL1 c "Detail S2", d "Kontrak S2", b4).

A policy that offers admit_exits() picks the type each admitted request runs as. The simulator must then take tau,
checkpoints and head from that run type, put the run type in the slowdown set A, and write it to the trace row as
"run_type_id" (None for expired_in_queue). Every expected number is worked out by hand in the comment above it.
Time is in seconds in the comments; the trace uses int ns.
"""
import numpy as np
import pytest

from exp2.contracts import PolicySpec, SimConfig, Workload
from exp2.model.simulator import simulate
from exp2.tests.mocks import StubPower, UnitSlowdown, make_types

# bert:5 has checkpoints [3, 5], bert:23 has [3, 7, 11, 15, 19, 23]; head_frac 0, so ckpt_frac is k / n_ckpts.
TYPES = make_types({"bert:5": {"tau": 0.02, "klass": "short", "head_frac": 0.0},
                    "bert:23": {"tau": 0.08, "klass": "short", "head_frac": 0.0}}, f_levels=(612, 1020))
SPEC = PolicySpec("fake", None, None, "fcfs", None, "static", 1020, {})
OLD_KEYS = {"req_id", "type_id", "family", "klass", "t_arrival_ns", "t_deadline_ns", "t_admit_ns", "t_end_ns",
            "t_done_ns", "outcome", "reached", "completion", "f_mhz", "n_in_service_at_admit", "queue_len_at_arrival",
            "k_override", "ckpt"}


class RunAs:
    """Exit policy stub: admits every queued request at once, as run[req_id] (default: its queued type)."""
    family_exclusive = False

    def __init__(self, run):
        self.run = dict(run)

    def admit_exits(self, view):
        return [(q.req_id, self.run.get(q.req_id, q.type_id)) for q in view.queue]

    def admit(self, view):
        return [rid for rid, _ in self.admit_exits(view)]

    def desired_f(self, view):
        return None


class RecSlowdown:
    """SlowdownModel that logs every A it is asked about; s = table[len(A)] (default 1)."""
    name = "rec"

    def __init__(self, table=None):
        self.table = table or {}
        self.seen = []

    def predict(self, i, A, f_mhz):
        assert i in A
        self.seen.append(A)
        return self.table.get(len(A), 1.0)

    def sample(self, i, A, f_mhz, rng):
        return self.predict(i, A, f_mhz)


def item(rid, tid, t_s, dl_s=None):
    return {"req_id": rid, "type_id": tid, "t_arrival_ns": int(round(t_s * 1e9)),
            "t_deadline_ns": None if dl_s is None else int(round(dl_s * 1e9)), "k_override": None}


def replay(items, *, policy=None, spec=SPEC, slowdown=None, types=TYPES):
    ids = sorted({it["type_id"] for it in items})
    wl = Workload("replay", tuple(ids), tuple(1.0 / len(ids) for _ in ids), 1.0, 1.0)
    cfg = SimConfig(n_warmup=0, family_exclusive=True, sample_slowdown=False, schedule=tuple(items))
    return simulate(wl, spec, types, slowdown or UnitSlowdown(), StubPower(dp=0.0), cfg, policy)


def rows(res):
    return {r["req_id"]: r for r in res.trace}


def near_ns(got, want_s):
    assert got is not None and abs(got - want_s * 1e9) <= 1.5, (got, want_s * 1e9)


# ------------------------------------------------------------------------------------------ Kontrak S2
def test_kontrak_s2_the_job_runs_as_the_chosen_type():
    # r1 queues as bert:23 (0.08 s) but runs as bert:5 (0.02 s, head 0): checkpoint 3 at 0.5 x 0.02 = 0.01 s,
    # checkpoint 5 at 0.02 s, both before the 1.0 s deadline -> complete, reached 5, t_done = t_end = 0.02 s.
    res = replay([item("r1", "bert:23", 0.0, 1.0)], policy=RunAs({"r1": "bert:5"}))
    r = rows(res)["r1"]
    want = {"req_id": "r1", "type_id": "bert:23", "run_type_id": "bert:5", "outcome": "complete", "reached": 5,
            "t_done_ns": 20_000_000}
    assert {k: r[k] for k in want} == want
    assert r["t_admit_ns"] == 0 and r["t_end_ns"] == 20_000_000
    assert r["ckpt"] == [[3, 10_000_000], [5, 20_000_000]]          # the run type's checkpoints, not bert:23's six
    assert r["completion"] == 1.0                                   # completion("bert", 5, 5); bert:23 would give 0.25
    assert r["family"] == "bert" and r["klass"] == "short"
    assert set(r) == OLD_KEYS | {"run_type_id"}


def test_kontrak_s2_the_slowdown_set_holds_the_run_type():
    rec = RecSlowdown()
    replay([item("r1", "bert:23", 0.0, 1.0)], policy=RunAs({"r1": "bert:5"}), slowdown=rec)
    assert rec.seen and all(A == ("bert:5",) for A in rec.seen)


def test_summary_completion_uses_the_run_type():
    # per_type stays keyed by the queued type; its completion is that of the run type: completion("bert", 5, 5) = 1.
    res = replay([item("r1", "bert:23", 0.0, 1.0)], policy=RunAs({"r1": "bert:5"}))
    assert set(res.per_type) == {"bert:23"}
    st = res.per_type["bert:23"]
    assert st.n == 1 and st.complete_rate == 1.0 and st.mean_completion == 1.0
    assert st.mean_R == pytest.approx(0.02, abs=1e-9)


def test_corun_uses_the_run_tau_and_puts_both_run_types_in_A():
    # s = 2 with two jobs in service, 1 alone. r1 (queued bert:23, runs bert:5, tau 0.02) starts at 0.
    # r2 (bert:23, runs bert:23, tau 0.08) arrives at 0.005: r1 has 0.005 solo-s done, 0.015 left.
    # Both at s = 2: r1 ends at 0.005 + 2 x 0.015 = 0.035; r2 did 0.030 / 2 = 0.015 by then, 0.065 left alone
    # -> r2 ends at 0.035 + 0.065 = 0.100.
    rec = RecSlowdown({2: 2.0})
    res = replay([item("r1", "bert:23", 0.0, 1.0), item("r2", "bert:23", 0.005, 1.0)],
                 policy=RunAs({"r1": "bert:5"}), slowdown=rec)
    got = rows(res)
    assert got["r1"]["run_type_id"] == "bert:5" and got["r2"]["run_type_id"] == "bert:23"
    near_ns(got["r1"]["t_done_ns"], 0.035)
    near_ns(got["r2"]["t_done_ns"], 0.100)
    assert got["r1"]["reached"] == 5 and got["r2"]["reached"] == 23
    assert got["r2"]["n_in_service_at_admit"] == 1
    assert set(rec.seen) == {("bert:5",), ("bert:23", "bert:5"), ("bert:23",)}


# ------------------------------------------------------------------------------------------ ordinary policies
def test_ordinary_policy_runs_the_queued_type_on_every_row():
    # serial FCFS (policies.Policy): r1 bert:23 0 -> 0.08, r2 bert:5 0.08 -> 0.10, r3 bert:23 0.10 -> 0.18
    spec = PolicySpec("serial", 1, 1, "fcfs", None, "static", 1020, {})
    res = replay([item("r1", "bert:23", 0.0), item("r2", "bert:5", 0.01), item("r3", "bert:23", 0.02)], spec=spec)
    assert len(res.trace) == 3
    for r in res.trace:
        assert r["run_type_id"] == r["type_id"]
        assert set(r) == OLD_KEYS | {"run_type_id"}
    got = rows(res)
    near_ns(got["r2"]["t_done_ns"], 0.10)
    near_ns(got["r3"]["t_done_ns"], 0.18)


def test_expired_in_queue_has_no_run_type():
    # serial: r1 runs 0 -> 0.08; r2 (deadline 0.05) is still queued at the next event (0.08 >= 0.05) and expires.
    spec = PolicySpec("serial", 1, 1, "fcfs", None, "static", 1020, {})
    got = rows(replay([item("r1", "bert:23", 0.0), item("r2", "bert:5", 0.01, 0.05)], spec=spec))
    assert got["r2"]["outcome"] == "expired_in_queue"
    assert got["r2"]["run_type_id"] is None and got["r2"]["ckpt"] == [] and got["r2"]["completion"] == 0.0
    assert got["r1"]["run_type_id"] == "bert:23"


# ------------------------------------------------------------------------------------------ bit identity
class NoisySlowdown:
    """s = n x (1 + 0.1 u), u from the simulator's rng: a result depends on the order of the rng draws."""
    name = "noisy"

    def predict(self, i, A, f_mhz):
        return float(len(A))

    def sample(self, i, A, f_mhz, rng):
        return len(A) * (1.0 + 0.1 * rng.random())


def test_unused_types_in_the_table_do_not_change_a_sampled_run():
    # Every type in the table now has a slot. Types the schedule never uses must not move a single rng draw.
    base = {"bert:5": {"tau": {612: 0.03, 1020: 0.02}, "klass": "short", "head_frac": 0.1},
            "bert:23": {"tau": {612: 0.12, 1020: 0.08}, "klass": "long", "head_frac": 0.1}}
    extra = dict(base, **{"bert:11": {"tau": {612: 0.06, 1020: 0.04}, "klass": "short", "head_frac": 0.1},
                          "yolo:1": {"tau": {612: 0.05, 1020: 0.03}, "klass": "short", "head_frac": 0.1}})
    wl = Workload("w", ("bert:5", "bert:23"), (0.5, 0.5), 0.6, 0.6 / 0.05)
    spec = PolicySpec("corun", None, None, "fcfs", None, "static", 1020, {})
    cfg = SimConfig(n_requests=400, n_warmup=20, n_reps=2, seed=7, family_exclusive=False, alpha=3.0)
    out = []
    for spec_types in (base, extra):
        t = make_types(spec_types, f_levels=(612, 1020))
        res = simulate(wl, spec, t, NoisySlowdown(), StubPower(dp=4.0, u=0.5), cfg).to_json()
        res.pop("runtime_s")
        out.append(res)
    assert out[0] == out[1]
    assert np.isfinite(out[0]["energy_per_req_j"])
