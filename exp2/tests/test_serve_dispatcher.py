"""C2a tests: Dispatcher event loop and FreqActuator (BRIEF_v1 K4, K5, K8; slice C2a).

Two rigs, both around the REAL Dispatcher, Policy, AdmissionCore and wire:

  VirtualDispatcher   the Dispatcher with ONLY the blocking wait (_poll) replaced.  FakeClock time, FakeConn pairs,
                      a scripted server that answers every `run` at admit + service.  Exact-time checks and the
                      golden traces (D7, run with -k golden).
  ThreadServer        a server thread behind a real transport (multiprocessing.Pipe, which goes through
                      connection.wait, or FakeConn, which goes through the poll fallback).  Real clock, real threads.

Every expected value is derived by hand from K4/K8, not by calling the code under test.
"""
import csv
import heapq
import json
import multiprocessing
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from exp2 import arrivals
from exp2 import contracts as C
from exp2 import wire
from exp2.contracts import ClockError
from exp2.policies import AdmissionCore, Policy, registry
from exp2.serve import dispatcher as dispatcher_mod
from exp2.serve.dispatcher import Dispatcher
from exp2.serve.freq_actuator import FreqActuator, write_freq_switch_csv
from exp2.tests import mocks
from exp2.tests.mocks import FakeBoard, FakeClock, FakeConn

REPO = Path(__file__).resolve().parents[2]
MS = 1_000_000
NS = 1_000_000_000
ORIGIN = 7_000_000_000          # virtual clock origin (t0 of a run is wherever the clock stands when it starts)


@pytest.fixture(scope="module")
def types():
    return C.load_types(str(REPO / "exp2" / "configs" / "example_types.json"))


def spec_of(**kw):
    d = {"name": "t", "k_short": None, "k_long": None, "order": "fcfs", "lane_cap": None,
         "f_rule": "static", "f_mhz": 1020, "params": {}}
    d.update(kw)
    return C.PolicySpec.from_json(d)


SERIAL = dict(name="serial", k_short=1, k_long=1)


def item(rid, tid, t_ms, dl_ms=None, ko=None):
    return {"req_id": rid, "type_id": tid, "t_arrival_ns": int(t_ms * MS),
            "t_deadline_ns": None if dl_ms is None else int(dl_ms * MS), "k_override": ko}


def cond_of(**kw):
    c = {"condition_id": "c1", "mode": "open", "arm": "pinned", "gpu_mhz": 1020, "alpha": None, "seed": 7,
         "n_warmup": 0}
    c.update(kw)
    return c


# ================================================================================================ server emulation
def emulate_done(run, t_recv, t_start, service_ns, truncate=True):
    """What a K5 server answers to `run`: checkpoints spread evenly over service_ns; with a deadline the first late
    checkpoint is the last one recorded (the segment keeps the GPU).  Returns (done message, t_end_ns)."""
    fam, ex = C.parse_type_id(run["type_id"])
    ckpts = C.default_checkpoints(fam, ex)
    times = [t_start + (service_ns * (i + 1)) // len(ckpts) for i in range(len(ckpts))]
    dl = run["deadline_ns"] if truncate else None
    keep = len(ckpts)
    if dl is not None:
        for i, t in enumerate(times):
            if t > dl:
                keep = i + 1
                break
    reached, outcome = C.truncation_outcome(ckpts, times[:keep], dl)
    t_end = times[keep - 1]
    return {"op": "done", "req_id": run["req_id"], "type_id": run["type_id"], "t_recv_ns": t_recv,
            "t_start_ns": t_start, "t_end_ns": t_end, "reached": reached, "target": ckpts[-1],
            "completion": C.completion(fam, ex, reached), "outcome": outcome,
            "ckpt": [[c, t] for c, t in zip(ckpts[:keep], times[:keep])],
            "tokens_done": reached if fam == "llama" and reached else None, "err": None}, t_end


class VirtualDispatcher(Dispatcher):
    """The real Dispatcher; only _poll is virtual.  Every `run` the dispatcher sends is picked up from its FakeConn
    (conn.sent) and answered at t_admit + service.  Ties: a completion is delivered before an arrival that is due
    at the same instant (the same order the golden replay uses)."""

    def __init__(self, types, *, service=None, truncate=False, board=None, n_inputs=32, lag_ns=0, families=None,
                 hook=None):
        self.vclock = FakeClock(start_ns=ORIGIN)
        self.pairs = {f: FakeConn.pair() for f in (families or C.FAMILIES)}
        super().__init__({f: a for f, (a, _b) in self.pairs.items()}, types, board=board, clock=self.vclock,
                         n_inputs=n_inputs)
        self._service = service or {}
        self._truncate = truncate
        self._lag = lag_ns
        self._hook = hook
        self._heap = []
        self._seq = 0
        self._scanned = {f: 0 for f in self.pairs}
        self.runs = []                       # every run message, in send order

    def service_ns(self, type_id):
        s = self._service
        if callable(s):
            return s(type_id)
        if type_id in s:
            return s[type_id]
        return int(round(self.types.tau(type_id, self.types.f_max_mhz, "p50") * NS))

    def _scan(self):
        for f, (a, _b) in self.pairs.items():
            msgs = a.sent_msgs()
            for m in msgs[self._scanned[f]:]:
                if m["op"] != "run":
                    continue
                C.validate_run(m)
                self.runs.append(m)
                done, t_end = emulate_done(m, m["t_admit_ns"], m["t_admit_ns"], self.service_ns(m["type_id"]),
                                           self._truncate)
                replies = [done]
                if self._hook is not None:
                    replies = self._hook(m, done)
                for r in replies:
                    heapq.heappush(self._heap, (max(t_end, m["t_admit_ns"]), self._seq, f, r))
                    self._seq += 1
            self._scanned[f] = len(msgs)

    def _poll(self, timeout_ns):
        self._scan()
        now = self.vclock()
        limit = now + timeout_ns
        if self._heap and self._heap[0][0] <= limit:
            t, _, fam, msg = heapq.heappop(self._heap)
            self.vclock.set_ns(max(t, now))
            return [(fam, msg, self.vclock())]
        self.vclock.set_ns(limit + self._lag)
        return []


def run_open(d, schedule, policy, tmp_path, name="c1", **cond_kw):
    if isinstance(policy, C.PolicySpec):
        policy = Policy(policy, d.types)
    out = Path(tmp_path) / name
    summary = d.run_open(cond_of(**cond_kw), schedule, policy, str(out))
    rows = C.read_requests(out / "requests.jsonl")          # validate_request_row on every row
    return summary, rows, out


def by_id(rows):
    return {r["req_id"]: r for r in rows}


# ================================================================================================ FreqActuator
class GatedBoard(FakeBoard):
    """FakeBoard whose set_gpu_level blocks until `release` is set; `entered` gets the level when a call starts."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.entered = []
        self._entered_cv = threading.Condition()
        self.release = threading.Event()

    def set_gpu_level(self, mhz):
        with self._entered_cv:
            self.entered.append(mhz)
            self._entered_cv.notify_all()
        assert self.release.wait(10), "test never released the board"
        return super().set_gpu_level(mhz)

    def wait_entered(self, n, timeout=10):
        with self._entered_cv:
            assert self._entered_cv.wait_for(lambda: len(self.entered) >= n, timeout), self.entered


def levels_called(board):
    return [c.args[0] for c in board.calls if c.name == "set_gpu_level"]


def test_actuator_applies_a_single_target_and_records_the_row():
    clock = FakeClock(start_ns=100)
    board = FakeBoard()
    act = FreqActuator(board, clock)
    act.start()
    act.request(612)
    act.close()
    assert levels_called(board) == [612]
    assert act.rows == [{"t_req_ns": 100, "t_done_ns": 100, "target_mhz": 612, "readback_mhz": 612}]
    assert act.errors == []


def test_actuator_row_holds_the_board_readback_not_the_target():
    board = FakeBoard(readback={612: 510})
    act = FreqActuator(board, FakeClock())
    act.start()
    act.request(612)
    act.close()
    assert act.rows[0]["target_mhz"] == 612 and act.rows[0]["readback_mhz"] == 510


def test_actuator_three_fast_targets_apply_only_the_last_after_the_running_one():
    board = GatedBoard()
    act = FreqActuator(board, FakeClock())
    act.start()
    act.request(408)
    board.wait_entered(1)                  # 408 is now being applied (blocked inside the board)
    act.request(510)
    act.request(612)
    act.request(816)                       # 510 and 612 are superseded
    board.release.set()
    act.close()
    assert levels_called(board) == [408, 816]
    assert [r["target_mhz"] for r in act.rows] == [408, 816]
    assert act.n_requests == 4 and act.n_superseded == 2


def test_actuator_request_times_are_those_of_the_applied_target():
    clock = FakeClock(start_ns=1000)
    board = GatedBoard()
    act = FreqActuator(board, clock)
    act.start()
    act.request(408)
    board.wait_entered(1)
    clock.advance_ns(50)
    act.request(510)
    clock.advance_ns(50)
    act.request(612)                       # supersedes 510, requested at t=1100
    clock.advance_ns(900)
    board.release.set()
    act.close()
    assert act.rows[0]["t_req_ns"] == 1000
    assert act.rows[1]["t_req_ns"] == 1100 and act.rows[1]["target_mhz"] == 612
    assert act.rows[1]["t_done_ns"] >= act.rows[1]["t_req_ns"]


def test_actuator_close_applies_the_pending_target_and_waits_for_the_running_call():
    board = GatedBoard()
    act = FreqActuator(board, FakeClock())
    act.start()
    act.request(408)
    board.wait_entered(1)
    act.request(816)
    threading.Timer(0.1, board.release.set).start()
    act.close()                            # must not return while 408 is still blocked in the board
    assert board.release.is_set()
    assert levels_called(board) == [408, 816]


def test_actuator_clock_error_is_recorded_and_the_thread_keeps_going():
    board = FakeBoard(fail_on={"set_gpu_level": {1: ClockError("cur_freq stale")}})
    act = FreqActuator(board, FakeClock())
    act.start()
    act.request(408)
    deadline = time.monotonic() + 5
    while len(act.rows) < 1 and time.monotonic() < deadline:
        time.sleep(0.001)
    act.request(816)
    act.close()
    assert levels_called(board) == [408, 816]
    assert act.rows[0]["target_mhz"] == 408 and act.rows[0]["readback_mhz"] is None
    assert act.rows[1]["readback_mhz"] == 816
    assert len(act.errors) == 1
    assert "ClockError" in act.errors[0]["error"] and "cur_freq stale" in act.errors[0]["error"]
    assert act.errors[0]["target_mhz"] == 408


def test_actuator_level_the_board_does_not_offer_is_recorded_not_raised():
    board = FakeBoard()
    act = FreqActuator(board, FakeClock())
    act.start()
    act.request(999)                       # FakeBoard raises ValueError
    act.close()
    assert act.rows[0]["readback_mhz"] is None
    assert "ValueError" in act.errors[0]["error"]


def test_actuator_request_before_start_or_after_close_raises():
    act = FreqActuator(FakeBoard(), FakeClock())
    with pytest.raises(RuntimeError):
        act.request(612)
    act.start()
    act.close()
    with pytest.raises(RuntimeError):
        act.request(612)


def test_actuator_close_is_idempotent_and_close_without_start_is_a_noop():
    act = FreqActuator(FakeBoard(), FakeClock())
    act.close()
    act.start()
    act.close()
    act.close()


def test_actuator_every_request_is_applied_or_superseded():
    board = FakeBoard()
    act = FreqActuator(board, FakeClock())
    act.start()
    for i in range(200):
        act.request(612 if i % 2 else 1020)
    act.close()
    assert act.n_requests == 200
    assert len(act.rows) + act.n_superseded == 200
    assert act.rows[-1]["target_mhz"] == 612 and levels_called(board)[-1] == 612     # the last request (i=199) wins


def test_write_freq_switch_csv_is_readable_by_the_contract_reader(tmp_path):
    rows = [{"t_req_ns": 5, "t_done_ns": 9, "target_mhz": 612, "readback_mhz": 612},
            {"t_req_ns": 10, "t_done_ns": 12, "target_mhz": 1020, "readback_mhz": None}]
    p = tmp_path / "freq_switch.csv"
    write_freq_switch_csv(p, rows)
    assert p.read_text(encoding="utf-8").splitlines()[0] == "t_req_ns,t_done_ns,target_mhz,readback_mhz"
    assert C.read_csv(p, "freq_switch") == rows
    write_freq_switch_csv(p, [])
    assert C.read_csv(p, "freq_switch") == []


# ================================================================================================ golden traces (D7)
TRACES = mocks.load_golden("admission_traces")


@pytest.mark.parametrize("trace", TRACES, ids=[t["id"] for t in TRACES])
def test_golden_trace_through_dispatcher(trace, tmp_path):
    types = mocks.golden_types(trace)
    spec = C.PolicySpec.from_json(trace["spec"])
    d = VirtualDispatcher(types, service=trace["service_ns"], board=FakeBoard())
    f0 = trace.get("f_initial_mhz", spec.f_mhz)
    summary, rows, _ = run_open(d, trace["arrivals"], Policy(spec, types), tmp_path, gpu_mhz=f0, alpha=None)
    t0, got = summary["t0_ns"], by_id(rows)
    assert sorted(got) == sorted(a["req_id"] for a in trace["arrivals"])          # one row per request
    for rid, want in trace["expect"].items():
        row = got[rid]
        if want["outcome"] == "expired_in_queue":
            assert row["outcome"] == "expired_in_queue", rid
        else:
            assert row["outcome"] != "expired_in_queue", rid
        if want["t_admit_ns"] is None:
            assert row["t_admit_ns"] is None, rid
        else:
            assert row["t_admit_ns"] - t0 == want["t_admit_ns"], rid
        if "t_done_ns" in want:
            assert row["t_done_ns"] - t0 == want["t_done_ns"], rid
    assert summary["freq"]["commanded"] == trace["expect_commanded_f"]


# ================================================================================================ inline traces
def test_serial_trace_admits_in_order_and_drops_the_expired_request(types, tmp_path):
    # r1 vision:24 0..40ms; r2 waits, admitted at 40; r3 (deadline 38) is dropped at the 40 ms event, logged at 38.
    d = VirtualDispatcher(types, service={"vision:24": 40 * MS, "bert:5": 10 * MS})
    sched = [item("r1", "vision:24", 0, 200), item("r2", "bert:5", 5, 55), item("r3", "bert:5", 8, 38)]
    summary, rows, _ = run_open(d, sched, spec_of(**SERIAL), tmp_path)
    t0, g = summary["t0_ns"], by_id(rows)
    assert g["r1"]["t_admit_ns"] == t0 and g["r2"]["t_admit_ns"] == t0 + 40 * MS
    assert g["r1"]["t_done_ns"] == t0 + 40 * MS and g["r2"]["t_done_ns"] == t0 + 50 * MS
    assert g["r3"]["outcome"] == "expired_in_queue"
    assert g["r3"]["t_done_ns"] == g["r3"]["t_deadline_ns"] == t0 + 38 * MS
    assert [(r["req_id"]) for r in d.runs] == ["r1", "r2"]


def test_two_families_run_together_and_a_same_family_request_waits_for_its_server(types, tmp_path):
    # corun (k unbounded). r1 bert 0..10, r2 vision 1..21, r3 bert:23 (arrives 2) waits for bert -> 10..40,
    # r4 vision:6 (arrives 3) waits for vision -> 21..41.
    d = VirtualDispatcher(types, service={"bert:5": 10 * MS, "vision:6": 20 * MS, "bert:23": 30 * MS})
    sched = [item("r1", "bert:5", 0), item("r2", "vision:6", 1), item("r3", "bert:23", 2), item("r4", "vision:6", 3)]
    summary, rows, _ = run_open(d, sched, spec_of(name="corun"), tmp_path)
    t0, g = summary["t0_ns"], by_id(rows)
    assert [g[r]["t_admit_ns"] - t0 for r in ("r1", "r2", "r3", "r4")] == [0, 1 * MS, 10 * MS, 21 * MS]
    assert [g[r]["n_in_service_at_admit"] for r in ("r1", "r2", "r3", "r4")] == [0, 1, 1, 1]
    assert [g[r]["queue_len_at_arrival"] for r in ("r1", "r2", "r3", "r4")] == [0, 0, 0, 1]


def test_stf_admits_the_shorter_type_first(types, tmp_path):
    d = VirtualDispatcher(types, service={"vision:24": 40 * MS, "bert:23": 30 * MS, "bert:5": 10 * MS})
    sched = [item("r1", "vision:24", 0), item("r2", "bert:23", 1), item("r3", "bert:5", 2)]
    summary, rows, _ = run_open(d, sched, spec_of(name="stf", k_short=1, k_long=1, order="stf"), tmp_path)
    t0, g = summary["t0_ns"], by_id(rows)
    assert g["r3"]["t_admit_ns"] - t0 == 40 * MS and g["r2"]["t_admit_ns"] - t0 == 50 * MS


def test_k_override_of_the_schedule_item_reaches_the_policy(types, tmp_path):
    # serial, but r2 carries k_override 2: admitted at arrival next to r1 (different family).
    d = VirtualDispatcher(types, service={"vision:24": 40 * MS, "bert:5": 10 * MS})
    sched = [item("r1", "vision:24", 0), item("r2", "bert:5", 5, ko=2)]
    summary, rows, _ = run_open(d, sched, spec_of(**SERIAL), tmp_path)
    assert by_id(rows)["r2"]["t_admit_ns"] - summary["t0_ns"] == 5 * MS
    assert by_id(rows)["r2"]["n_in_service_at_admit"] == 1


# ================================================================================================ rows (K8)
def test_expired_request_is_never_sent_and_is_logged_at_its_deadline(types, tmp_path):
    d = VirtualDispatcher(types, service={"vision:24": 40 * MS, "bert:5": 10 * MS})
    sched = [item("r1", "vision:24", 0, 200), item("r2", "bert:5", 8, 38)]
    summary, rows, _ = run_open(d, sched, spec_of(**SERIAL), tmp_path)
    r2 = by_id(rows)["r2"]
    t0 = summary["t0_ns"]
    assert [m["req_id"] for m in d.runs] == ["r1"]                      # r2 never reached a server
    for conn_a, conn_b in d.pairs.values():
        assert all(json.loads(b)["req_id"] != "r2" for b in conn_a.sent if json.loads(b).get("op") == "run")
    assert r2["outcome"] == "expired_in_queue" and r2["completion"] == 0.0 and r2["reached"] is None
    assert r2["t_done_ns"] == r2["t_deadline_ns"] == t0 + 38 * MS
    assert r2["t_arrival_ns"] == t0 + 8 * MS
    for k in ("t_admit_ns", "t_srv_recv_ns", "t_start_ns", "t_end_ns", "n_in_service_at_admit", "tokens_done", "err"):
        assert r2[k] is None, k
    assert r2["ckpt"] == []
    assert summary["outcomes"] == {"complete": 1, "expired_in_queue": 1}


def test_row_fields_of_a_completed_request(types, tmp_path):
    d = VirtualDispatcher(types, service={"bert:23": 60 * MS})
    spec = spec_of(**SERIAL)
    summary, rows, _ = run_open(d, [item("r1", "bert:23", 4, 500)], spec, tmp_path, name="cond-x", seed=11,
                                condition_id="W_hi_r060_serial_r0", policy_key="serial")
    r, t0 = rows[0], summary["t0_ns"]
    assert set(r) == set(C.REQUEST_KEYS)
    assert r["condition_id"] == "W_hi_r060_serial_r0" and r["req_id"] == "r1" and r["type_id"] == "bert:23"
    assert (r["family"], r["exit"], r["klass"], r["target"], r["server"]) == ("bert", 23, "long", 23, "bert")
    assert r["t_arrival_ns"] == t0 + 4 * MS and r["t_arrival_actual_ns"] == t0 + 4 * MS
    assert r["t_deadline_ns"] == t0 + 500 * MS and r["t_admit_ns"] == t0 + 4 * MS
    assert r["t_srv_recv_ns"] == t0 + 4 * MS and r["t_start_ns"] == t0 + 4 * MS
    assert r["t_end_ns"] == t0 + 64 * MS and r["t_done_ns"] == t0 + 64 * MS
    assert r["policy"] == "serial" and r["policy_spec"] == spec.to_json()
    assert r["f_mhz"] == 1020 and r["n_in_service_at_admit"] == 0 and r["queue_len_at_arrival"] == 0
    assert (r["reached"], r["completion"], r["outcome"]) == (23, 1.0, "complete")
    assert (r["tokens_done"], r["err"]) == (None, None)
    assert [p[0] for p in r["ckpt"]] == [3, 7, 11, 15, 19, 23]
    assert (r["is_warmup"], r["seed"], r["mode"]) == (False, 11, "open")


def test_response_time_uses_the_scheduled_arrival_not_the_actual_one(types, tmp_path):
    # The dispatcher wakes 3 ms late; the row keeps the scheduled arrival, so R = t_done - t_arrival includes the lag.
    d = VirtualDispatcher(types, service={"bert:5": 10 * MS}, lag_ns=3 * MS)
    summary, rows, _ = run_open(d, [item("r1", "bert:5", 10, 500)], spec_of(**SERIAL), tmp_path)
    r, t0 = rows[0], summary["t0_ns"]
    assert r["t_arrival_ns"] == t0 + 10 * MS
    assert r["t_arrival_actual_ns"] == t0 + 13 * MS
    assert r["t_admit_ns"] == t0 + 13 * MS
    assert r["t_done_ns"] - r["t_arrival_ns"] == 13 * MS                       # 3 ms lag + 10 ms service
    assert summary["max_arrival_lag_ns"] == 3 * MS


def test_arrival_already_past_its_deadline_is_expired_at_once_and_never_sent(types, tmp_path):
    d = VirtualDispatcher(types, service={"bert:5": 10 * MS}, lag_ns=3 * MS)
    summary, rows, _ = run_open(d, [item("r1", "bert:5", 10, 12)], spec_of(**SERIAL), tmp_path)
    r, t0 = rows[0], summary["t0_ns"]
    assert r["outcome"] == "expired_in_queue" and d.runs == []
    assert r["t_arrival_ns"] == t0 + 10 * MS and r["t_arrival_actual_ns"] == t0 + 13 * MS
    assert r["t_done_ns"] == r["t_deadline_ns"] == t0 + 12 * MS


def test_truncated_and_miss_rows_pass_through_what_the_server_reported(types, tmp_path):
    # bert:23 has 6 checkpoints; service 60 ms -> checkpoints every 10 ms from the start.
    d = VirtualDispatcher(types, service={"bert:23": 60 * MS}, truncate=True)
    sched = [item("r1", "bert:23", 0, 35), item("r2", "bert:23", 100, 105)]
    summary, rows, _ = run_open(d, sched, spec_of(**SERIAL), tmp_path)
    g, t0 = by_id(rows), summary["t0_ns"]
    # r1: checkpoints at 10,20,30,40 ms; first late is 40 -> reached = ckpt of 30 ms = 11 (3rd), truncated, t_end 40
    assert g["r1"]["outcome"] == "truncated" and g["r1"]["reached"] == 11
    assert g["r1"]["completion"] == C.completion("bert", 23, 11)
    assert [p[0] for p in g["r1"]["ckpt"]] == [3, 7, 11, 15] and g["r1"]["t_end_ns"] == t0 + 40 * MS
    # r2: first checkpoint at 110 ms > deadline 105 -> miss
    assert g["r2"]["outcome"] == "miss" and g["r2"]["reached"] is None and g["r2"]["completion"] == 0.0
    assert [p[0] for p in g["r2"]["ckpt"]] == [3]
    assert summary["outcomes"] == {"truncated": 1, "miss": 1}


def test_server_error_reply_becomes_an_error_row_and_the_queue_moves_on(types, tmp_path):
    def hook(run, done):
        if run["req_id"] == "r1":
            done = dict(done, outcome="error", err="CUDA out of memory", reached=None, completion=0.0, ckpt=[],
                        tokens_done=None)
        return [done]

    d = VirtualDispatcher(types, service={"bert:5": 10 * MS}, hook=hook)
    sched = [item("r1", "bert:5", 0), item("r2", "bert:5", 1)]
    summary, rows, _ = run_open(d, sched, spec_of(**SERIAL), tmp_path)
    g = by_id(rows)
    assert g["r1"]["outcome"] == "error" and g["r1"]["err"] == "CUDA out of memory"
    assert g["r1"]["reached"] is None and g["r1"]["completion"] == 0.0
    assert g["r2"]["outcome"] == "complete" and g["r2"]["t_admit_ns"] == g["r1"]["t_done_ns"]
    assert summary["dead_servers"] == {}


def test_invalid_done_message_becomes_an_error_row_not_a_crash(types, tmp_path):
    def hook(run, done):
        return [dict(done, completion=0.25)] if run["req_id"] == "r1" else [done]      # complete but completion 0.25

    d = VirtualDispatcher(types, service={"bert:5": 10 * MS}, hook=hook)
    summary, rows, _ = run_open(d, [item("r1", "bert:5", 0), item("r2", "bert:5", 1)], spec_of(**SERIAL), tmp_path)
    g = by_id(rows)
    assert g["r1"]["outcome"] == "error" and "invalid done" in g["r1"]["err"]
    assert g["r2"]["outcome"] == "complete"


def test_stray_message_is_counted_and_ignored(types, tmp_path):
    def hook(run, done):
        stray = dict(done, req_id="r999")
        return [stray, done]                 # the stray arrives while r1 is still in flight

    d = VirtualDispatcher(types, service={"bert:5": 10 * MS}, hook=hook)
    summary, rows, _ = run_open(d, [item("r1", "bert:5", 0)], spec_of(**SERIAL), tmp_path)
    assert len(rows) == 1 and rows[0]["outcome"] == "complete"
    assert summary["unexpected_msgs"] == 1


def test_warmup_flag_marks_the_first_n_warmup_requests_of_the_schedule(types, tmp_path):
    d = VirtualDispatcher(types, service={"bert:5": 1 * MS})
    sched = [item("r%d" % i, "bert:5", i * 5) for i in range(1, 6)]
    summary, rows, _ = run_open(d, sched, spec_of(**SERIAL), tmp_path, n_warmup=2)
    flags = sorted((r["req_id"], r["is_warmup"]) for r in rows)
    assert flags == [("r1", True), ("r2", True), ("r3", False), ("r4", False), ("r5", False)]


def test_missing_deadlines_are_filled_from_cond_alpha_with_contracts_with_deadlines(types, tmp_path):
    d = VirtualDispatcher(types, service={"bert:5": 1 * MS, "vision:24": 1 * MS})
    sched = [item("r1", "bert:5", 3), item("r2", "vision:24", 9, 777)]      # r2 keeps the deadline it carries
    summary, rows, _ = run_open(d, sched, spec_of(name="corun"), tmp_path, alpha=5.0)
    g, t0 = by_id(rows), summary["t0_ns"]
    want = C.with_deadlines(sched, types, 5.0)
    assert g["r1"]["t_deadline_ns"] == t0 + want[0]["t_deadline_ns"]
    assert g["r2"]["t_deadline_ns"] == t0 + 777 * MS
    assert [m["deadline_ns"] for m in d.runs] == [g["r1"]["t_deadline_ns"], g["r2"]["t_deadline_ns"]]


def test_without_alpha_there_is_no_deadline_and_nothing_expires(types, tmp_path):
    d = VirtualDispatcher(types, service={"vision:24": 5000 * MS, "bert:5": 1 * MS})
    sched = [item("r1", "vision:24", 0), item("r2", "bert:5", 1)]
    summary, rows, _ = run_open(d, sched, spec_of(**SERIAL), tmp_path, alpha=None)
    assert all(r["t_deadline_ns"] is None for r in rows) and all(m["deadline_ns"] is None for m in d.runs)
    assert summary["outcomes"] == {"complete": 2}


def test_input_idx_cycles_per_family_modulo_n_inputs(types, tmp_path):
    d = VirtualDispatcher(types, n_inputs=3, service=lambda tid: 1 * MS)
    sched = [item("r%d" % i, "bert:5" if i % 2 else "vision:6", i * 4) for i in range(1, 12)]
    run_open(d, sched, spec_of(name="corun"), tmp_path)
    bert = [m["input_idx"] for m in d.runs if m["type_id"] == "bert:5"]
    vision = [m["input_idx"] for m in d.runs if m["type_id"] == "vision:6"]
    assert bert == [0, 1, 2, 0, 1, 2] and vision == [0, 1, 2, 0, 1]


def test_input_idx_restarts_for_every_run(types, tmp_path):
    d = VirtualDispatcher(types, n_inputs=4, service=lambda tid: 1 * MS)
    sched = [item("r%d" % i, "bert:5", i * 3) for i in range(1, 4)]
    run_open(d, sched, spec_of(**SERIAL), tmp_path, name="a")
    run_open(d, sched, spec_of(**SERIAL), tmp_path, name="b")
    assert [m["input_idx"] for m in d.runs] == [0, 1, 2, 0, 1, 2]


def test_run_message_is_a_valid_k5_run(types, tmp_path):
    d = VirtualDispatcher(types, service=lambda tid: 1 * MS)
    summary, rows, _ = run_open(d, [item("r1", "yolo:5", 2, 90)], spec_of(**SERIAL), tmp_path)
    m = d.runs[0]
    assert set(m) == {"op", "req_id", "type_id", "exit", "input_idx", "deadline_ns", "t_admit_ns"}
    assert m["op"] == "run" and m["exit"] == 5 and m["t_admit_ns"] == rows[0]["t_admit_ns"]
    assert m["deadline_ns"] == rows[0]["t_deadline_ns"] == summary["t0_ns"] + 90 * MS


def test_f_mhz_label_is_governor_for_a_governor_policy_and_arm(types, tmp_path):
    d = VirtualDispatcher(types, service=lambda tid: 1 * MS)
    gov = spec_of(name="default", f_rule="governor", f_mhz=None)
    _, rows, _ = run_open(d, [item("r1", "bert:5", 0)], gov, tmp_path, arm="governor", gpu_mhz=None)
    assert rows[0]["f_mhz"] == "governor"
    d2 = VirtualDispatcher(types, service=lambda tid: 1 * MS)
    _, rows2, _ = run_open(d2, [item("r1", "bert:5", 0)], spec_of(name="corun"), tmp_path, name="z", arm="stock",
                           gpu_mhz=None)
    assert rows2[0]["f_mhz"] == "governor"


# ================================================================================================ AdmissionCore
class SpyCore(AdmissionCore):
    log = []

    def arrive(self, req, now_ns):
        SpyCore.log.append(("arrive", req.req_id, now_ns))
        return super().arrive(req, now_ns)

    def done(self, req_id, now_ns):
        SpyCore.log.append(("done", req_id, now_ns))
        return super().done(req_id, now_ns)


class SpyPolicy(Policy):
    admit_calls = 0
    views = []

    def admit(self, view):
        SpyPolicy.admit_calls += 1
        return super().admit(view)

    def desired_f(self, view):
        SpyPolicy.views.append(view)
        return super().desired_f(view)


def test_every_decision_goes_through_admission_core(types, tmp_path, monkeypatch):
    SpyCore.log, SpyPolicy.admit_calls, SpyPolicy.views = [], 0, []
    monkeypatch.setattr(dispatcher_mod, "AdmissionCore", SpyCore)
    d = VirtualDispatcher(types, service=lambda tid: 3 * MS)
    sched = [item("r%d" % i, ["bert:5", "vision:24", "yolo:1"][i % 3], i * 2, 4000) for i in range(1, 31)]
    summary, rows, _ = run_open(d, sched, SpyPolicy(spec_of(name="serial_x", k_short=1, k_long=1), types), tmp_path)
    arrives = [e for e in SpyCore.log if e[0] == "arrive"]
    dones = [e for e in SpyCore.log if e[0] == "done"]
    assert [e[1] for e in arrives] == [s["req_id"] for s in sched]
    assert sorted(e[1] for e in dones) == sorted(r["req_id"] for r in rows if r["t_admit_ns"] is not None)
    # Policy.admit ran exactly once per core event: the dispatcher never admits on its own.
    assert SpyPolicy.admit_calls == len(SpyCore.log)
    assert sorted(m["req_id"] for m in d.runs) == sorted(r["req_id"] for r in rows if r["t_admit_ns"] is not None)


def test_policy_view_f_current_is_the_commanded_level_not_the_board_readback(types, tmp_path):
    SpyCore.log, SpyPolicy.admit_calls, SpyPolicy.views = [], 0, []
    trace = next(t for t in TRACES if t["id"].startswith("T6"))
    spec = C.PolicySpec.from_json(trace["spec"])
    board = FakeBoard(readback=lambda mhz: 408)                  # the board always claims to sit at 408
    d = VirtualDispatcher(types, service=trace["service_ns"], board=board)
    summary, rows, out = run_open(d, trace["arrivals"], SpyPolicy(spec, types), tmp_path, gpu_mhz=1020)
    seen = [v.f_current_mhz for v in SpyPolicy.views]
    assert 408 not in seen
    assert seen[0] == 1020 and set(seen) <= {1020, 612}
    commanded = summary["freq"]["commanded"]
    assert commanded == [612, 1020, 612]
    # the view after each command shows that command: 1020 -> (612) -> 612 -> (1020) -> 1020 -> (612) -> 612
    assert seen[:2] == [1020, 612] and 1020 in seen[1:]
    freq = C.read_csv(out / "freq_switch.csv", "freq_switch")
    assert freq and all(r["readback_mhz"] == 408 for r in freq)
    assert [r["target_mhz"] for r in freq] == levels_called(board)
    assert set(r["target_mhz"] for r in freq) <= set(commanded)


def test_actuator_errors_do_not_stop_the_run_and_are_reported(types, tmp_path):
    trace = next(t for t in TRACES if t["id"].startswith("T6"))
    spec = C.PolicySpec.from_json(trace["spec"])
    board = FakeBoard(fail_on={"set_gpu_level": ClockError("cur_freq stale")})
    d = VirtualDispatcher(types, service=trace["service_ns"], board=board)
    summary, rows, out = run_open(d, trace["arrivals"], Policy(spec, types), tmp_path, gpu_mhz=1020)
    assert [r["outcome"] for r in rows] == ["complete"] * 3
    assert summary["freq"]["commanded"] == [612, 1020, 612]
    assert summary["freq"]["errors"] and all("ClockError" in e["error"] for e in summary["freq"]["errors"])
    assert all(r["readback_mhz"] is None for r in C.read_csv(out / "freq_switch.csv", "freq_switch"))


def test_freq_summary_balances_commands_applied_and_superseded(types, tmp_path):
    trace = next(t for t in TRACES if t["id"].startswith("T7"))
    spec = C.PolicySpec.from_json(trace["spec"])
    d = VirtualDispatcher(types, service=trace["service_ns"], board=FakeBoard())
    summary, _, out = run_open(d, trace["arrivals"], Policy(spec, types), tmp_path, gpu_mhz=1020)
    f = summary["freq"]
    assert f["commanded"] == [612, 1020, 612]
    assert f["applied"] + f["superseded"] == len(f["commanded"])
    rows = C.read_csv(out / "freq_switch.csv", "freq_switch")
    assert len(rows) == f["applied"]
    assert rows[-1]["target_mhz"] == 612 and d.board.gpu_mhz == 612       # the board ends at the last command


def test_without_a_board_commands_are_still_tracked_and_the_csv_has_a_header(types, tmp_path):
    trace = next(t for t in TRACES if t["id"].startswith("T7"))
    spec = C.PolicySpec.from_json(trace["spec"])
    d = VirtualDispatcher(types, service=trace["service_ns"], board=None)
    summary, _, out = run_open(d, trace["arrivals"], Policy(spec, types), tmp_path, gpu_mhz=1020)
    assert summary["freq"]["commanded"] == [612, 1020, 612] and summary["freq"]["applied"] == 0
    assert C.read_csv(out / "freq_switch.csv", "freq_switch") == []


# ================================================================================================ conservation
def _workload(types, ids, probs, rho=0.7):
    w = {"name": "W", "types": ids, "probs": probs, "rho": rho, "arrival": "poisson", "cv_a": 1.0}
    w["rate_per_s"] = C.rate_for_load(rho, w, types)
    return w


def _intervals_ok(rows, key):
    """No two admitted requests of one `key` value overlap in [t_admit, t_done]."""
    last = {}
    for r in sorted((r for r in rows if r["t_admit_ns"] is not None), key=lambda r: r["t_admit_ns"]):
        k = r[key] if key else 0
        if k in last and r["t_admit_ns"] < last[k]:
            return False
        last[k] = max(last.get(k, 0), r["t_done_ns"])
    return True


@pytest.mark.parametrize("key", ["serial", "serial_stf", "serial_class", "serial_edgeserving", "corun",
                                 "greenllm_lanes", "occupancy_f", "default"])
def test_every_request_becomes_exactly_one_row_for_every_registry_policy(types, tmp_path, key):
    ids = ["bert:5", "bert:23", "vision:6", "vision:24", "yolo:1", "llama:3"]
    w = _workload(types, ids, [0.3, 0.1, 0.2, 0.1, 0.2, 0.1], rho=1.3)           # overloaded: forces queueing + expiry
    sched = C.with_deadlines(arrivals.schedule(w, 250, 3), types, 5.0)
    spec = registry(types)[key]
    d = VirtualDispatcher(types, board=FakeBoard(), truncate=True)
    summary, rows, out = run_open(d, sched, Policy(spec, types), tmp_path, alpha=5.0, policy_key=key,
                                  gpu_mhz=spec.f_mhz or 1020, arm="governor" if spec.f_rule == "governor" else "pinned")
    ids_in = [s["req_id"] for s in sched]
    assert sorted(r["req_id"] for r in rows) == sorted(ids_in) and len(set(r["req_id"] for r in rows)) == 250
    assert summary["n_requests"] == summary["n_rows"] == 250 and sum(summary["outcomes"].values()) == 250
    assert summary["outcomes"].get("expired_in_queue", 0) == sum(1 for r in rows if r["outcome"] == "expired_in_queue")
    admitted = [r for r in rows if r["t_admit_ns"] is not None]
    assert sorted(m["req_id"] for m in d.runs) == sorted(r["req_id"] for r in admitted)
    assert all(r["t_done_ns"] >= r["t_admit_ns"] for r in admitted)
    assert all(r["t_done_ns"] >= r["t_arrival_actual_ns"] for r in rows if r["outcome"] != "expired_in_queue")
    assert _intervals_ok(rows, "family")                                           # one request per server
    if spec.k_short == 1 and spec.k_long == 1:
        assert _intervals_ok(rows, None)                                           # serial really is serial
    if key == "serial_edgeserving":
        assert summary["freq"]["commanded"] == []


def test_two_runs_on_one_dispatcher_do_not_mix_rows(types, tmp_path):
    d = VirtualDispatcher(types, service=lambda tid: 2 * MS)
    s1 = [item("r%d" % i, "bert:5", i * 3) for i in range(1, 6)]
    s2 = [item("r%d" % i, "bert:23", i * 3) for i in range(1, 4)]
    a = run_open(d, s1, spec_of(**SERIAL), tmp_path, name="a", condition_id="ca")
    b = run_open(d, s2, spec_of(**SERIAL), tmp_path, name="b", condition_id="cb")
    assert len(a[1]) == 5 and len(b[1]) == 3
    assert {r["condition_id"] for r in a[1]} == {"ca"} and {r["condition_id"] for r in b[1]} == {"cb"}
    assert {r["type_id"] for r in b[1]} == {"bert:23"}


def test_a_policy_that_never_admits_ends_in_error_rows_not_a_hang(types, tmp_path):
    class Never(Policy):
        def admit(self, view):
            return []

    d = VirtualDispatcher(types, service=lambda tid: 1 * MS)
    summary, rows, _ = run_open(d, [item("r1", "bert:5", 0), item("r2", "vision:6", 1)],
                                Never(spec_of(name="corun"), types), tmp_path)
    assert sorted(r["outcome"] for r in rows) == ["error", "error"]
    assert all("never admitted" in r["err"] for r in rows)
    assert d.runs == [] and summary["n_rows"] == 2


def test_a_completion_and_an_arrival_at_the_same_instant_are_handled_completion_first(types, tmp_path):
    # r1 runs 0..10 ms (serial); r2 arrives at 5 and waits; r3 arrives at exactly 10 ms, the instant r1 is done.
    # Completion first: r2 is admitted at 10, then r3 arrives to an EMPTY queue (queue_len 0) and waits behind r2.
    d = VirtualDispatcher(types, service={"bert:5": 10 * MS})
    sched = [item("r1", "bert:5", 0), item("r2", "bert:5", 5), item("r3", "bert:5", 10)]
    summary, rows, _ = run_open(d, sched, spec_of(**SERIAL), tmp_path)
    g, t0 = by_id(rows), summary["t0_ns"]
    assert g["r2"]["t_admit_ns"] == t0 + 10 * MS
    assert g["r3"]["queue_len_at_arrival"] == 0 and g["r3"]["t_admit_ns"] == t0 + 20 * MS


def test_all_completions_due_at_an_instant_are_handled_before_an_arrival_due_then(types, tmp_path):
    # r1 (bert) and r2 (vision) both finish at 10 ms; r5 (vision) queues behind r2; r4 arrives at exactly 10 ms.
    # Both completions first: r2's completion admits r5, so r4 meets an EMPTY queue.  An arrival handled between the
    # two completions would see r5 still queued (queue_len 1).
    d = VirtualDispatcher(types, service={"bert:5": 10 * MS, "vision:6": 10 * MS, "yolo:1": 3 * MS})
    sched = [item("r1", "bert:5", 0), item("r2", "vision:6", 0), item("r5", "vision:6", 5), item("r4", "yolo:1", 10)]
    summary, rows, _ = run_open(d, sched, spec_of(name="corun"), tmp_path)
    g, t0 = by_id(rows), summary["t0_ns"]
    assert g["r5"]["t_admit_ns"] == t0 + 10 * MS
    assert g["r4"]["queue_len_at_arrival"] == 0


def test_invalid_n_inputs_is_rejected(types):
    with pytest.raises(ValueError):
        Dispatcher({}, types, n_inputs=0)


def test_a_row_that_breaks_the_contract_is_still_written_as_an_error_row(types, tmp_path, monkeypatch):
    real = C.validate_request_row

    def strict(row):                      # stands in for a contract the row builder does not satisfy
        if row["outcome"] == "complete":
            raise C.ContractError("request_row: outcome 'complete' refused by this test")
        return real(row)

    monkeypatch.setattr(C, "validate_request_row", strict)
    d = VirtualDispatcher(types, service=lambda tid: 1 * MS)
    summary, rows, _ = run_open(d, [item("r1", "bert:5", 0)], spec_of(name="corun"), tmp_path)
    assert len(rows) == 1 and rows[0]["outcome"] == "error" and "invalid row (complete)" in rows[0]["err"]
    assert rows[0]["reached"] is None and rows[0]["completion"] == 0.0 and rows[0]["ckpt"] == []
    assert summary["n_rows"] == 1 and summary["outcomes"] == {"error": 1}


def test_a_policy_that_raises_propagates_and_leaves_no_actuator_thread(types, tmp_path):
    class Boom(Policy):
        def admit(self, view):
            raise RuntimeError("policy bug")

    d = VirtualDispatcher(types, service=lambda tid: 1 * MS, board=FakeBoard())
    with pytest.raises(RuntimeError, match="policy bug"):
        d.run_open(cond_of(), [item("r1", "bert:5", 0)], Boom(spec_of(name="corun"), types), str(tmp_path / "x"))
    assert not [t for t in threading.enumerate() if t.name == "freq-actuator"]


# ================================================================================================ argument checks
def test_duplicate_request_ids_in_a_schedule_are_rejected_before_anything_is_sent(types, tmp_path):
    d = VirtualDispatcher(types)
    with pytest.raises(ValueError, match="duplicate"):
        d.run_open(cond_of(), [item("r1", "bert:5", 0), item("r1", "bert:5", 1)], Policy(spec_of(name="corun"), types),
                   str(tmp_path / "x"))
    assert d.runs == [] and all(not a.sent for a, _b in d.pairs.values())


def test_a_family_without_a_connection_is_rejected_up_front(types, tmp_path):
    d = VirtualDispatcher(types, families=("bert",))
    with pytest.raises(ValueError, match="vision"):
        d.run_open(cond_of(), [item("r1", "vision:6", 0)], Policy(spec_of(name="corun"), types), str(tmp_path / "x"))


def test_unknown_type_id_is_rejected_up_front(types, tmp_path):
    d = VirtualDispatcher(types)
    with pytest.raises(ValueError):
        d.run_open(cond_of(), [item("r1", "bert:99", 0)], Policy(spec_of(name="corun"), types), str(tmp_path / "x"))


def test_policy_without_family_exclusivity_is_rejected(types, tmp_path):
    d = VirtualDispatcher(types)
    pol = Policy(spec_of(name="corun"), types, family_exclusive=False)
    with pytest.raises(ValueError, match="family_exclusive"):
        d.run_open(cond_of(), [item("r1", "bert:5", 0)], pol, str(tmp_path / "x"))


def test_a_spec_instead_of_a_policy_is_wrapped(types, tmp_path):
    d = VirtualDispatcher(types, service=lambda tid: 1 * MS)
    out = tmp_path / "y"
    summary = d.run_open(cond_of(), [item("r1", "bert:5", 0)], spec_of(name="corun"), str(out))
    assert summary["n_rows"] == 1


def test_mode_mismatch_is_rejected(types, tmp_path):
    d = VirtualDispatcher(types)
    with pytest.raises(ValueError, match="mode"):
        d.run_open(cond_of(mode="closed"), [item("r1", "bert:5", 0)], spec_of(name="corun"), str(tmp_path / "x"))
    with pytest.raises(ValueError, match="mode"):
        d.run_closed(cond_of(mode="open", cell=["bert:5"], duration_s=0.1), str(tmp_path / "x"))


def test_closed_cell_with_two_types_of_one_family_is_rejected(types, tmp_path):
    d = VirtualDispatcher(types)
    with pytest.raises(ValueError, match="family"):
        d.run_closed(cond_of(mode="closed", cell=["bert:5", "bert:23"], duration_s=0.1), str(tmp_path / "x"))


def test_output_directory_is_created(types, tmp_path):
    d = VirtualDispatcher(types, service=lambda tid: 1 * MS)
    out = tmp_path / "a" / "b" / "c"
    d.run_open(cond_of(), [item("r1", "bert:5", 0)], spec_of(name="corun"), str(out))
    assert (out / "requests.jsonl").is_file() and (out / "freq_switch.csv").is_file()


# ================================================================================================ closed loop (virtual)
def closed_cond(cell, duration_s, **kw):
    return cond_of(mode="closed", cell=list(cell), duration_s=duration_s, **kw)


def run_closed(d, cond, tmp_path, name="cc"):
    out = Path(tmp_path) / name
    summary = d.run_closed(cond, str(out))
    return summary, C.read_requests(out / "requests.jsonl"), out


def test_closed_loop_runs_each_type_back_to_back_with_a_common_start(types, tmp_path):
    d = VirtualDispatcher(types, service={"bert:5": 29 * MS, "vision:24": 60 * MS})
    cond = closed_cond(["bert:5", "vision:24"], 0.5, n_warmup=2, split="train", repeat=0)
    summary, rows, out = run_closed(d, cond, tmp_path)
    t0 = summary["t0_ns"]
    bert = [r for r in rows if r["type_id"] == "bert:5"]
    vis = [r for r in rows if r["type_id"] == "vision:24"]
    assert len(bert) == 18 and len(vis) == 9          # launches at k*29 ms < 500 ms (k<=17) and k*60 ms < 500 ms (k<=8)
    # barrier: both tenants start at the same instant, before any completion
    assert bert[0]["t_admit_ns"] == vis[0]["t_admit_ns"] == t0
    assert [m["type_id"] for m in d.runs[:2]] == ["bert:5", "vision:24"]
    # back to back: next admit == previous done (virtual clock, no overhead); one in flight per type
    for tenant in (bert, vis):
        for a, b in zip(tenant, tenant[1:]):
            assert b["t_admit_ns"] == a["t_done_ns"]
        assert [r["req_id"] for r in tenant] == sorted(r["req_id"] for r in tenant)
    assert _intervals_ok(rows, "type_id")
    # no launch at or after the stop time; the last request of each tenant ran to completion
    assert all(r["t_admit_ns"] < t0 + 500 * MS for r in rows)
    assert all(r["outcome"] == "complete" for r in rows)
    assert max(r["t_done_ns"] for r in bert) == t0 + 18 * 29 * MS


def test_closed_loop_rows_follow_the_closed_contract(types, tmp_path):
    d = VirtualDispatcher(types, service={"bert:5": 29 * MS, "vision:24": 60 * MS})
    cond = closed_cond(["bert:5", "vision:24"], 0.2, n_warmup=1, gpu_mhz=612, seed=5)
    summary, rows, out = run_closed(d, cond, tmp_path)
    assert len({r["req_id"] for r in rows}) == len(rows)                      # unique within the condition
    for r in rows:
        assert r["mode"] == "closed" and r["policy"] == "closed" and r["policy_spec"] is None
        assert r["t_deadline_ns"] is None
        assert r["t_arrival_ns"] == r["t_admit_ns"] == r["t_arrival_actual_ns"]
        assert r["queue_len_at_arrival"] == 0 and r["f_mhz"] == 612 and r["seed"] == 5
    # warm-up is per type: the first request of each type only
    for tid in ("bert:5", "vision:24"):
        flags = [r["is_warmup"] for r in rows if r["type_id"] == tid]
        assert flags[0] is True and not any(flags[1:])
    # n_in_service_at_admit: the first launch sees nobody, the second sees the first; later ones see the other tenant
    assert [r["n_in_service_at_admit"] for r in rows[:2]] == [0, 1]
    assert all(r["n_in_service_at_admit"] == 1 for r in rows[2:])
    assert summary["n_requests"] == summary["n_rows"] == len(rows)


def test_closed_loop_solo_cell_and_governor_arm_label(types, tmp_path):
    d = VirtualDispatcher(types, service={"yolo:1": 15 * MS})
    summary, rows, _ = run_closed(d, closed_cond(["yolo:1"], 0.1, arm="governor", gpu_mhz=None), tmp_path)
    assert len(rows) == 7                                                     # launches at k*15 ms < 100 ms: k = 0..6
    assert all(r["f_mhz"] == "governor" for r in rows) and all(r["n_in_service_at_admit"] == 0 for r in rows)


def test_closed_loop_does_not_launch_at_the_stop_instant(types, tmp_path):
    # yolo:1 takes 50 ms: launches at 0, 50, ..., 450; a request would start at exactly 500 = t_stop and must not.
    d = VirtualDispatcher(types, service={"yolo:1": 50 * MS})
    summary, rows, _ = run_closed(d, closed_cond(["yolo:1"], 0.5), tmp_path)
    assert len(rows) == 10 and max(r["t_admit_ns"] for r in rows) == summary["t0_ns"] + 450 * MS


def test_closed_loop_with_zero_duration_still_runs_one_request_per_type(types, tmp_path):
    d = VirtualDispatcher(types, service=lambda tid: 5 * MS)
    summary, rows, _ = run_closed(d, closed_cond(["bert:5", "vision:6"], 0.0), tmp_path)
    assert sorted(r["type_id"] for r in rows) == ["bert:5", "vision:6"]


def test_closed_loop_server_error_does_not_end_the_tenant(types, tmp_path):
    def hook(run, done):
        if run["req_id"] == "r000002":
            done = dict(done, outcome="error", err="boom", reached=None, completion=0.0, ckpt=[], tokens_done=None)
        return [done]

    d = VirtualDispatcher(types, service=lambda tid: 10 * MS, hook=hook)
    summary, rows, _ = run_closed(d, closed_cond(["bert:5"], 0.05), tmp_path)
    assert [r["outcome"] for r in rows] == ["complete", "error", "complete", "complete", "complete"]


# ================================================================================================ real transports
class ThreadServer(threading.Thread):
    """Answers `run` with a done after service_s of real time, one request at a time like the real server.
    die_after=n closes the connection after answering n requests; hang=True never answers."""

    def __init__(self, conn, family, service_s=0.01, die_after=None, hang=False):
        super().__init__(daemon=True)
        self.conn, self.family, self.service_s = conn, family, service_s
        self.die_after, self.hang = die_after, hang
        self.runs = []
        self.overlap = False
        self.error = None

    def run(self):
        n = 0
        try:
            while True:
                msg = wire.recv(self.conn)
                if msg["op"] == "stop":
                    return
                if msg["op"] != "run":
                    continue
                self.runs.append((msg, time.monotonic_ns()))
                if self.hang:
                    continue
                t_recv = time.monotonic_ns()
                time.sleep(self.service_s)
                done, _ = emulate_done(msg, t_recv, t_recv, time.monotonic_ns() - t_recv, truncate=True)
                if self.conn.poll(0):
                    self.overlap = True              # a second run arrived while this one was in service
                wire.send(self.conn, done)
                n += 1
                if self.die_after is not None and n >= self.die_after:
                    self.conn.close()
                    return
        except (EOFError, OSError):
            return
        except BaseException as exc:                  # surfaced by the test, never swallowed
            self.error = exc


def make_transport(kind, families):
    """-> ({family: dispatcher_end}, {family: server_end})"""
    if kind == "socket":                                    # the real wire: loopback TCP, as run_session uses
        listener = wire.listen()
        port = wire.listener_port(listener)
        pairs = {}
        for f in families:
            client = wire.connect(port)
            pairs[f] = (client, listener.accept())
        listener.close()
    elif kind == "pipe":
        pairs = {f: multiprocessing.Pipe() for f in families}
    else:
        pairs = {f: FakeConn.pair() for f in families}
    return {f: a for f, (a, _b) in pairs.items()}, {f: b for f, (_a, b) in pairs.items()}


TRANSPORTS = ["socket", "pipe", "fakeconn"]


@pytest.fixture
def servers_of():
    started = []

    def make(kind, spec):
        """spec {family: kwargs}; -> (dispatcher conns, {family: ThreadServer})"""
        conns, ends = make_transport(kind, list(spec))
        srv = {f: ThreadServer(ends[f], f, **kw) for f, kw in spec.items()}
        for s in srv.values():
            s.start()
        started.append((conns, ends, srv))
        return conns, srv

    yield make
    for conns, ends, srv in started:
        for c in list(conns.values()) + list(ends.values()):
            try:
                c.close()
            except OSError:
                pass
        for s in srv.values():
            s.join(5)
            assert s.error is None, s.error


@pytest.mark.parametrize("kind", TRANSPORTS)
def test_open_loop_over_a_real_transport(types, tmp_path, servers_of, kind):
    conns, srv = servers_of(kind, {"bert": {"service_s": 0.01}, "vision": {"service_s": 0.015}})
    d = Dispatcher(conns, types, board=FakeBoard())
    now = time.monotonic_ns()
    sched = [item("r%d" % i, "bert:5" if i % 2 else "vision:6", i * 12, None) for i in range(1, 13)]
    out = tmp_path / "rt"
    t_before = time.monotonic_ns()
    summary = d.run_open(cond_of(), sched, Policy(spec_of(name="corun"), types), str(out))
    t_after = time.monotonic_ns()
    rows = C.read_requests(out / "requests.jsonl")
    assert len(rows) == 12 and {r["outcome"] for r in rows} == {"complete"}
    assert sorted(r["req_id"] for r in rows) == sorted(s["req_id"] for s in sched)
    assert not srv["bert"].overlap and not srv["vision"].overlap
    for r in rows:
        assert t_before <= r["t_arrival_actual_ns"] <= r["t_admit_ns"] <= r["t_srv_recv_ns"] <= r["t_start_ns"]
        assert r["t_start_ns"] <= r["t_end_ns"] <= r["t_done_ns"] <= t_after
        assert r["t_arrival_actual_ns"] >= r["t_arrival_ns"] - 1                     # never early
        assert r["t_done_ns"] - r["t_end_ns"] < 200 * MS
    assert t_after - t_before >= 12 * 12 * MS
    assert summary["dead_servers"] == {} and summary["unexpected_msgs"] == 0


@pytest.mark.parametrize("kind", TRANSPORTS)
def test_arrivals_are_not_early_and_not_much_late(types, tmp_path, servers_of, kind):
    conns, srv = servers_of(kind, {"bert": {"service_s": 0.002}})
    d = Dispatcher(conns, types)
    sched = [item("r%d" % i, "bert:5", 20 * i) for i in range(1, 9)]
    summary = d.run_open(cond_of(), sched, Policy(spec_of(name="corun"), types), str(tmp_path / "o"))
    rows = C.read_requests(tmp_path / "o" / "requests.jsonl")
    lag = [r["t_arrival_actual_ns"] - r["t_arrival_ns"] for r in rows]
    assert min(lag) >= 0
    assert summary["max_arrival_lag_ns"] == max(lag)
    assert sorted(lag)[len(lag) // 2] < 15 * MS                       # median lag: generous for a loaded machine


@pytest.mark.parametrize("kind", TRANSPORTS)
def test_server_that_dies_gives_error_rows_and_the_other_server_is_unaffected(types, tmp_path, servers_of, kind):
    conns, srv = servers_of(kind, {"bert": {"service_s": 0.005, "die_after": 2}, "vision": {"service_s": 0.005}})
    d = Dispatcher(conns, types)
    sched = [item("r%d" % i, "bert:5" if i % 2 else "vision:6", i * 15) for i in range(1, 11)]
    t = time.monotonic()
    summary = d.run_open(cond_of(), sched, Policy(spec_of(name="corun"), types), str(tmp_path / "o"))
    assert time.monotonic() - t < 10
    rows = C.read_requests(tmp_path / "o" / "requests.jsonl")
    assert len(rows) == 10
    bert = [r for r in rows if r["family"] == "bert"]
    vision = [r for r in rows if r["family"] == "vision"]
    assert [r["outcome"] for r in bert[:2]] == ["complete", "complete"]
    assert [r["outcome"] for r in bert[2:]] == ["error"] * 3 and all(r["err"] for r in bert[2:])
    assert [r["outcome"] for r in vision] == ["complete"] * 5
    assert "bert" in summary["dead_servers"] and "vision" not in summary["dead_servers"]
    assert summary["outcomes"] == {"complete": 7, "error": 3}


@pytest.mark.parametrize("kind", TRANSPORTS)
def test_request_in_flight_when_the_server_dies_is_an_error_row(types, tmp_path, servers_of, kind):
    conns, srv = servers_of(kind, {"bert": {"hang": True}})          # takes the run, never answers
    d = Dispatcher(conns, types)
    threading.Timer(0.1, lambda: srv["bert"].conn.close()).start()   # then the process dies
    summary = d.run_open(cond_of(), [item("r1", "bert:5", 0), item("r2", "bert:5", 30)],
                         Policy(spec_of(name="corun"), types), str(tmp_path / "o"))
    rows = by_id(C.read_requests(tmp_path / "o" / "requests.jsonl"))
    assert rows["r1"]["outcome"] == "error" and rows["r1"]["t_admit_ns"] is not None
    assert rows["r2"]["outcome"] == "error"
    assert "bert" in summary["dead_servers"]


def test_connection_closed_on_our_side_gives_error_rows_without_hanging(types, tmp_path):
    a, b = multiprocessing.Pipe()
    a.close()
    d = Dispatcher({"bert": a}, types)
    summary = d.run_open(cond_of(), [item("r1", "bert:5", 0), item("r2", "bert:5", 5)],
                         Policy(spec_of(name="corun"), types), str(tmp_path / "o"))
    rows = C.read_requests(tmp_path / "o" / "requests.jsonl")
    assert [r["outcome"] for r in rows] == ["error", "error"] and "bert" in summary["dead_servers"]
    b.close()


def test_dead_server_stays_dead_for_the_next_condition(types, tmp_path, servers_of):
    conns, srv = servers_of("pipe", {"bert": {"hang": True}})
    d = Dispatcher(conns, types)
    srv["bert"].conn.close()
    d.run_open(cond_of(), [item("r1", "bert:5", 0)], Policy(spec_of(name="corun"), types), str(tmp_path / "a"))
    summary = d.run_open(cond_of(condition_id="c2"), [item("r1", "bert:5", 0), item("r2", "bert:5", 5)],
                         Policy(spec_of(name="corun"), types), str(tmp_path / "b"))
    rows = C.read_requests(tmp_path / "b" / "requests.jsonl")
    assert [r["outcome"] for r in rows] == ["error", "error"] and "bert" in summary["dead_servers"]


@pytest.mark.parametrize("kind", TRANSPORTS)
def test_silent_server_is_declared_dead_after_the_hang_timeout(types, tmp_path, servers_of, monkeypatch, kind):
    monkeypatch.setattr(dispatcher_mod, "HANG_TIMEOUT_NS", 150 * MS)
    conns, srv = servers_of(kind, {"bert": {"hang": True}, "vision": {"service_s": 0.005}})
    d = Dispatcher(conns, types)
    t = time.monotonic()
    summary = d.run_open(cond_of(), [item("r1", "bert:5", 0), item("r2", "vision:6", 10), item("r3", "bert:5", 20)],
                         Policy(spec_of(name="corun"), types), str(tmp_path / "o"))
    assert 0.1 < time.monotonic() - t < 5
    rows = by_id(C.read_requests(tmp_path / "o" / "requests.jsonl"))
    assert rows["r1"]["outcome"] == "error" and "no reply" in rows["r1"]["err"]
    assert rows["r2"]["outcome"] == "complete"
    assert rows["r3"]["outcome"] == "error"
    assert "bert" in summary["dead_servers"]


@pytest.mark.parametrize("kind", TRANSPORTS)
def test_closed_loop_over_a_real_transport(types, tmp_path, servers_of, kind):
    conns, srv = servers_of(kind, {"bert": {"service_s": 0.01}, "vision": {"service_s": 0.02}})
    d = Dispatcher(conns, types, board=FakeBoard())
    out = tmp_path / "cl"
    summary = d.run_closed(closed_cond(["bert:5", "vision:6"], 0.25, n_warmup=1, gpu_mhz=1020), str(out))
    rows = C.read_requests(out / "requests.jsonl")
    bert = [r for r in rows if r["family"] == "bert"]
    vis = [r for r in rows if r["family"] == "vision"]
    assert len(bert) >= 8 and len(vis) >= 5
    assert not srv["bert"].overlap and not srv["vision"].overlap                  # one in flight per type
    assert abs(bert[0]["t_admit_ns"] - vis[0]["t_admit_ns"]) < 20 * MS            # barrier: common start
    assert min(r["t_done_ns"] for r in rows) > max(bert[0]["t_admit_ns"], vis[0]["t_admit_ns"])
    for tenant in (bert, vis):
        for a, b in zip(tenant, tenant[1:]):
            assert b["t_admit_ns"] >= a["t_done_ns"]
            assert b["t_admit_ns"] - a["t_done_ns"] < 50 * MS
    assert all(r["t_deadline_ns"] is None and r["outcome"] == "complete" for r in rows)
    assert summary["n_requests"] == len(rows)


def test_closed_loop_dead_server_ends_its_tenant_and_the_other_continues(types, tmp_path, servers_of):
    conns, srv = servers_of("pipe", {"bert": {"service_s": 0.01, "die_after": 3}, "vision": {"service_s": 0.01}})
    d = Dispatcher(conns, types)
    out = tmp_path / "cl"
    summary = d.run_closed(closed_cond(["bert:5", "vision:6"], 0.2), str(out))
    rows = C.read_requests(out / "requests.jsonl")
    bert = [r for r in rows if r["family"] == "bert"]
    assert [r["outcome"] for r in bert] == ["complete"] * 3 + ["error"]
    assert len([r for r in rows if r["family"] == "vision"]) >= 8
    assert "bert" in summary["dead_servers"]


def test_poll_fallback_handles_a_fake_conn_that_has_no_file_descriptor():
    a, b = FakeConn.pair()
    d = Dispatcher({"bert": a}, mocks.make_types())
    assert not hasattr(a, "fileno")
    assert d._poll(2 * MS) == []
    wire.send(b, {"op": "pong", "t_ns": 5})
    ev = d._poll(100 * MS)
    assert len(ev) == 1 and ev[0][0] == "bert" and ev[0][1] == {"op": "pong", "t_ns": 5}
    b.close()
    ev = d._poll(100 * MS)
    assert isinstance(ev[0][1], EOFError)


def test_poll_with_a_real_pipe_returns_after_the_timeout_and_on_data():
    a, b = multiprocessing.Pipe()
    d = Dispatcher({"vision": a}, mocks.make_types())
    t = time.monotonic_ns()
    assert d._poll(30 * MS) == []
    assert 25 * MS <= time.monotonic_ns() - t < 500 * MS
    threading.Timer(0.03, lambda: wire.send(b, {"op": "pong", "t_ns": 1})).start()
    ev = d._poll(2 * NS)
    assert ev and ev[0][1] == {"op": "pong", "t_ns": 1}
    a.close()
    b.close()


# ================================================================================================ plumbing
def _imports_of(path):
    import ast
    names = set()
    for node in ast.walk(ast.parse(Path(path).read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(node.module + "." + a.name for a in node.names)
    return names


def test_dispatcher_policies_and_actuator_never_import_telemetry_or_torch():
    """Static closure over the exp2 modules the dispatcher reaches (S0b's test_import_hygiene does the real import)."""
    seen, todo, banned_hits = set(), ["exp2.serve.dispatcher", "exp2.serve.freq_actuator", "exp2.policies"], []
    while todo:
        mod = todo.pop()
        if mod in seen:
            continue
        seen.add(mod)
        path = REPO / (mod.replace(".", os.sep) + ".py")
        if not path.is_file():
            continue                                               # a package or a name imported from a module
        for name in _imports_of(path):
            top = name.split(".")[0]
            if name.startswith("exp2.board.telemetry") or top in ("torch", "transformers", "cv2", "shared", "numpy"):
                banned_hits.append((mod, name))
            if top == "exp2":
                todo.append(name)
    assert not banned_hits, banned_hits
    assert {"exp2.contracts", "exp2.wire", "exp2.policies"} <= seen


def test_summary_keys(types, tmp_path):
    d = VirtualDispatcher(types, service=lambda tid: 1 * MS, board=FakeBoard())
    summary, _, _ = run_open(d, [item("r1", "bert:5", 0)], spec_of(name="corun"), tmp_path)
    assert {"condition_id", "mode", "n_requests", "n_rows", "outcomes", "t0_ns", "t_end_ns", "max_arrival_lag_ns",
            "unexpected_msgs", "dead_servers", "freq"} <= set(summary)
    assert {"commanded", "applied", "superseded", "errors"} <= set(summary["freq"])
    assert summary["mode"] == "open" and summary["t_end_ns"] >= summary["t0_ns"]
    json.dumps(summary)                                                           # JSON-able for the manifest


def test_requests_file_has_one_json_object_per_line_and_only_contract_keys(types, tmp_path):
    d = VirtualDispatcher(types, service=lambda tid: 1 * MS)
    _, rows, out = run_open(d, [item("r%d" % i, "bert:5", i) for i in range(1, 6)], spec_of(**SERIAL), tmp_path)
    lines = (out / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5 and all(set(json.loads(l)) == set(C.REQUEST_KEYS) for l in lines)
