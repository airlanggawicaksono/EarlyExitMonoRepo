"""S1 tests (Run 3, PLAN PL2 e): the dispatcher runs the exit an ExitPolicy chooses.

Rig: VirtualDispatcher from test_serve_dispatcher (the real Dispatcher, AdmissionCore and wire; only _poll is virtual,
every `run` is answered at t_admit + tau p50 of the RUN type).  Six mock types with the solo times of the brief;
klass per family as in exp2/frozen/run2/types.json.  Every expected value is derived by hand from the contract
(exp2/contracts.py, "Exit choice on the board"), not by calling the code under test.
"""
from pathlib import Path

import pytest

from exp2 import arrivals
from exp2 import contracts as C
from exp2.exits import ExitPolicy, pilot_specs
from exp2.policies import Policy
from exp2.tests import mocks
from exp2.tests.test_serve_dispatcher import MS, VirtualDispatcher, by_id, cond_of, item, run_open, spec_of

TAU_S = {"bert:5": 0.02, "bert:23": 0.08, "yolo:1": 0.04, "yolo:5": 0.06, "llama:3": 0.6, "llama:15": 1.6}
KLASS = {"bert": "short", "yolo": "short", "llama": "long"}
DEADLINES_S = {"bert": 0.1, "yolo": 0.1, "llama": 3.35}
EXIT_KEYS = set(C.REQUEST_KEYS) | {"req_type_id"}
SPECS = pilot_specs(1020)


@pytest.fixture(scope="module")
def types():
    return mocks.make_types({tid: {"tau": tau, "klass": KLASS[C.parse_type_id(tid)[0]]} for tid, tau in TAU_S.items()})


def nothing_sent(d):
    return d.runs == [] and all(not a.sent for a, _b in d.pairs.values())


# ================================================================================================ run type on admit
def test_all_early_runs_the_floor_exit_and_keeps_the_arrival_type(types, tmp_path):
    # r1 bert:23, deadline 100 ms -> all_early runs bert:5 (tau 20 ms): ckpts [3, 5] at 10 and 20 ms, complete.
    d = VirtualDispatcher(types)
    pol = ExitPolicy(SPECS["all_early"], types)
    summary, rows, _ = run_open(d, [item("r1", "bert:23", 0, 100)], pol, tmp_path)
    r, t0 = rows[0], summary["t0_ns"]
    assert set(r) == EXIT_KEYS
    assert (r["type_id"], r["family"], r["exit"], r["target"], r["klass"]) == ("bert:5", "bert", 5, 5, "short")
    assert r["req_type_id"] == "bert:23"
    assert (r["outcome"], r["reached"], r["completion"]) == ("complete", 5, 1.0)
    assert r["t_admit_ns"] == t0 and r["t_deadline_ns"] == t0 + 100 * MS
    assert r["t_done_ns"] - r["t_arrival_ns"] == 20 * MS
    assert r["ckpt"] == [[3, t0 + 10 * MS], [5, t0 + 20 * MS]]
    assert r["policy_spec"] == SPECS["all_early"].to_json()
    m = d.runs[0]
    assert (m["req_id"], m["type_id"], m["exit"], m["deadline_ns"]) == ("r1", "bert:5", 5, t0 + 100 * MS)


def test_all_final_runs_the_arrival_type(types, tmp_path):
    # Three families, k unbounded: all admitted at arrival, each at its own (target) type.
    d = VirtualDispatcher(types)
    sched = [item("r1", "bert:23", 0, 100), item("r2", "yolo:5", 1, 101), item("r3", "llama:15", 2, 3352)]
    summary, rows, _ = run_open(d, sched, ExitPolicy(SPECS["all_final"], types), tmp_path)
    g = by_id(rows)
    for rid, tid in (("r1", "bert:23"), ("r2", "yolo:5"), ("r3", "llama:15")):
        assert set(g[rid]) == EXIT_KEYS
        assert g[rid]["type_id"] == g[rid]["req_type_id"] == tid, rid
        assert g[rid]["outcome"] == "complete", rid
    assert [g[r]["t_done_ns"] - g[r]["t_arrival_ns"] for r in ("r1", "r2", "r3")] == [80 * MS, 60 * MS, 1600 * MS]
    assert (g["r3"]["exit"], g["r3"]["target"], g["r3"]["reached"]) == (15, 32, 32)
    assert [(m["req_id"], m["type_id"], m["exit"]) for m in d.runs] == [
        ("r1", "bert:23", 23), ("r2", "yolo:5", 5), ("r3", "llama:15", 15)]


def test_an_expired_request_keeps_its_arrival_type(types, tmp_path):
    # all_early: r1 runs bert:5 0..20 ms.  r2 bert:23 (arrives 5, deadline 15) waits for the bert server and is
    # dropped at the 20 ms event, logged at its deadline, never sent.
    d = VirtualDispatcher(types)
    sched = [item("r1", "bert:23", 0, 100), item("r2", "bert:23", 5, 15)]
    summary, rows, _ = run_open(d, sched, ExitPolicy(SPECS["all_early"], types), tmp_path)
    r2, t0 = by_id(rows)["r2"], summary["t0_ns"]
    assert r2["outcome"] == "expired_in_queue" and r2["t_admit_ns"] is None
    assert (r2["type_id"], r2["exit"], r2["target"], r2["req_type_id"]) == ("bert:23", 23, 23, "bert:23")
    assert r2["t_done_ns"] == r2["t_deadline_ns"] == t0 + 15 * MS
    assert [m["req_id"] for m in d.runs] == ["r1"]


def test_rows_of_an_ordinary_policy_have_exactly_the_run2_keys(types, tmp_path):
    # corun: r1 bert:23 0..80, r2 yolo:5 1..61, r3 bert:5 waits for bert -> admitted at 80 ms at its own type.
    d = VirtualDispatcher(types)
    sched = [item("r1", "bert:23", 0), item("r2", "yolo:5", 1), item("r3", "bert:5", 2)]
    summary, rows, _ = run_open(d, sched, spec_of(name="corun"), tmp_path)
    g, t0 = by_id(rows), summary["t0_ns"]
    assert all(set(r) == set(C.REQUEST_KEYS) for r in rows)
    assert [g[r]["type_id"] for r in ("r1", "r2", "r3")] == ["bert:23", "yolo:5", "bert:5"]
    assert g["r3"]["t_admit_ns"] == t0 + 80 * MS
    assert [(m["req_id"], m["type_id"]) for m in d.runs] == [("r1", "bert:23"), ("r2", "yolo:5"), ("r3", "bert:5")]


# ================================================================================================ deadlines_s
def test_deadlines_s_sets_the_deadline_per_family_and_keeps_a_given_one(types, tmp_path):
    # r1 bert 3 ms + 0.1 s = 103 ms; r2 keeps its 777 ms; r3 llama 12 ms + 3.35 s = 3362 ms.
    d = VirtualDispatcher(types)
    sched = [item("r1", "bert:23", 3), item("r2", "yolo:5", 9, 777), item("r3", "llama:15", 12)]
    summary, rows, _ = run_open(d, sched, ExitPolicy(SPECS["all_final"], types), tmp_path, deadlines_s=DEADLINES_S)
    g, t0 = by_id(rows), summary["t0_ns"]
    want = {"r1": t0 + 103 * MS, "r2": t0 + 777 * MS, "r3": t0 + 3362 * MS}
    assert {rid: g[rid]["t_deadline_ns"] for rid in want} == want
    assert [m["deadline_ns"] for m in d.runs] == [want["r1"], want["r2"], want["r3"]]
    assert all(r["outcome"] == "complete" for r in rows)


def test_alpha_and_deadlines_s_together_are_rejected_before_anything_is_sent(types, tmp_path):
    d = VirtualDispatcher(types)
    out = tmp_path / "x"
    with pytest.raises(ValueError, match="alpha"):
        d.run_open(cond_of(alpha=5.0, deadlines_s=DEADLINES_S), [item("r1", "bert:23", 0)],
                   ExitPolicy(SPECS["all_final"], types), str(out))
    assert nothing_sent(d) and not out.exists()


@pytest.mark.parametrize("wrap", [True, False], ids=["Policy", "PolicySpec"])
def test_an_exit_rule_spec_without_an_exit_policy_is_rejected(types, tmp_path, wrap):
    spec = SPECS["coexit"]
    d = VirtualDispatcher(types)
    out = tmp_path / "x"
    with pytest.raises(ValueError, match="needs exits.ExitPolicy"):
        d.run_open(cond_of(), [item("r1", "bert:23", 0, 100)], Policy(spec, types) if wrap else spec, str(out))
    assert nothing_sent(d) and not out.exists()


# ================================================================================================ run type outcomes
def test_a_floor_run_past_its_deadline_is_truncated_at_the_floor_checkpoints(types, tmp_path):
    # bert:5 runs 0..20 ms, ckpts 3 at 10 ms and 5 at 20 ms; deadline 15 ms -> first late is 5, reached 3,
    # completion depth(3) / depth(5) = 4 / 6.
    d = VirtualDispatcher(types, truncate=True)
    summary, rows, _ = run_open(d, [item("r1", "bert:23", 0, 15)], ExitPolicy(SPECS["all_early"], types), tmp_path)
    r, t0 = rows[0], summary["t0_ns"]
    assert (r["type_id"], r["exit"], r["target"], r["req_type_id"]) == ("bert:5", 5, 5, "bert:23")
    assert (r["outcome"], r["reached"]) == ("truncated", 3)
    assert r["completion"] == pytest.approx(4 / 6)
    assert r["ckpt"] == [[3, t0 + 10 * MS], [5, t0 + 20 * MS]] and r["t_end_ns"] == t0 + 20 * MS


def test_coexit_on_mix_b_gives_valid_rows_with_menu_run_types(types, tmp_path):
    wl = {"name": "mixB", "types": ["llama:15", "bert:23", "yolo:5"], "probs": [0.2, 0.4, 0.4], "rho": 0.9,
          "arrival": "poisson", "cv_a": 1.0}
    wl["rate_per_s"] = C.rate_for_load(0.9, wl, types)
    sched = arrivals.schedule(wl, 40, 20261022)
    pol = ExitPolicy(SPECS["coexit"], types, mocks.StubSlowdown({1: 1.0, 2: 1.1, 3: 1.15}))
    d = VirtualDispatcher(types)
    summary, rows, out = run_open(d, sched, pol, tmp_path, condition_id="B_p_rho090_coexit_r0", seed=20261022,
                                  n_warmup=5, deadlines_s=DEADLINES_S)
    assert summary["n_requests"] == summary["n_rows"] == len(rows) == 40
    assert "error" not in summary["outcomes"]
    arrival = {s["req_id"]: s for s in sched}
    sent = {m["req_id"]: m for m in d.runs}
    for r in rows:
        C.validate_request_row(r)
        fam, ex = C.parse_type_id(r["type_id"])
        assert set(r) == EXIT_KEYS
        assert ex in C.EXIT_MENU[fam], r["type_id"]
        assert r["req_type_id"] == arrival[r["req_id"]]["type_id"]
        assert r["t_deadline_ns"] - r["t_arrival_ns"] == int(round(DEADLINES_S[fam] * 1e9))
        if r["t_admit_ns"] is None:
            assert r["type_id"] == r["req_type_id"] and r["req_id"] not in sent
        else:
            assert (sent[r["req_id"]]["type_id"], sent[r["req_id"]]["exit"]) == (r["type_id"], r["exit"])
    assert sorted(sent) == sorted(r["req_id"] for r in rows if r["t_admit_ns"] is not None)
    assert len(C.read_requests(out / "requests.jsonl")) == 40


# ================================================================================================ errors
def test_a_server_error_reply_gives_an_error_row_with_the_run_type(types, tmp_path):
    def hook(run, done):
        if run["req_id"] == "r1":
            done = dict(done, outcome="error", err="CUDA out of memory", reached=None, completion=0.0, ckpt=[],
                        tokens_done=None)
        return [done]

    # all_early: r1 runs bert:5 0..20 ms and fails; r2 waits for bert and runs bert:5 20..40 ms.
    d = VirtualDispatcher(types, hook=hook)
    sched = [item("r1", "bert:23", 0, 100), item("r2", "bert:23", 1, 200)]
    summary, rows, _ = run_open(d, sched, ExitPolicy(SPECS["all_early"], types), tmp_path)
    g, t0 = by_id(rows), summary["t0_ns"]
    r1 = g["r1"]
    assert (r1["outcome"], r1["err"], r1["reached"], r1["completion"]) == ("error", "CUDA out of memory", None, 0.0)
    assert (r1["type_id"], r1["exit"], r1["target"], r1["req_type_id"]) == ("bert:5", 5, 5, "bert:23")
    assert r1["t_admit_ns"] == t0
    assert g["r2"]["outcome"] == "complete" and g["r2"]["t_admit_ns"] == t0 + 20 * MS
    assert g["r2"]["type_id"] == "bert:5" and summary["dead_servers"] == {}


def test_a_dead_server_gives_error_rows_with_run_type_or_arrival_type(types, tmp_path):
    # r1's connection breaks at 20 ms: bert is dead.  r2 (queued) is admitted at that event at bert:5, the send
    # fails -> error row with the run type.  r3 arrives at 30 ms to a dead server: never admitted, arrival type.
    def hook(run, done):
        return [EOFError("pipe closed")] if run["req_id"] == "r1" else [done]

    d = VirtualDispatcher(types, hook=hook)
    sched = [item("r1", "bert:23", 0, 100), item("r2", "bert:23", 1, 200), item("r3", "bert:23", 30, 300)]
    summary, rows, _ = run_open(d, sched, ExitPolicy(SPECS["all_early"], types), tmp_path)
    g, t0 = by_id(rows), summary["t0_ns"]
    assert [g[r]["outcome"] for r in ("r1", "r2", "r3")] == ["error"] * 3
    assert (g["r1"]["type_id"], g["r1"]["req_type_id"]) == ("bert:5", "bert:23") and "EOFError" in g["r1"]["err"]
    assert (g["r2"]["type_id"], g["r2"]["exit"], g["r2"]["req_type_id"]) == ("bert:5", 5, "bert:23")
    assert g["r2"]["t_admit_ns"] == t0 + 20 * MS and "down" in g["r2"]["err"]
    assert (g["r3"]["type_id"], g["r3"]["req_type_id"], g["r3"]["t_admit_ns"]) == ("bert:23", "bert:23", None)
    assert "down" in g["r3"]["err"]
    assert [m["req_id"] for m in d.runs] == ["r1"] and "bert" in summary["dead_servers"]
