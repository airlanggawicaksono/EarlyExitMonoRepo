"""Tests for limited preemption in exp2.serve.server (board LP plan, slice S1; owner 2026-10-26).

The llama server reads waiting messages after every on-time, non-last token. A `pause` for the running request
stops it at that token boundary until its `resume`. Everything runs on a virtual FakeAdapter (100 ms per token)
and a TimedConn: each incoming message arrives at an exact virtual time, so every timestamp below is exact.
No torch, no GPU, no Jetson.
"""
import collections
import json

import pytest

from exp2 import contracts
from exp2.serve import server
from exp2.serve.fake_adapter import FakeAdapter, default_spec
from exp2.tests.mocks import FakeClock, FakeConn

T0 = 1_000_000_000
MS = 1_000_000
CFG = {"preprocess": "per_request", "n_inputs": 4, "seed": 0, "sync": "event", "compile": False,
       "input_source": "synthetic"}


class TimedConn(FakeConn):
    """FakeConn on a virtual clock. at(ms, msg) delivers msg at T0 + ms. n_poll counts poll() calls."""

    def __init__(self, clock=None, latency_ns=0):
        super().__init__(clock, latency_ns)
        self.n_poll = 0

    def at(self, ms, msg):
        data = msg if isinstance(msg, bytes) else json.dumps(msg).encode("utf-8")
        self._inbox.append((T0 + round(ms * MS), data))
        self._inbox = collections.deque(sorted(self._inbox, key=lambda m: m[0]))   # stable: same time keeps order
        return self

    def poll(self, timeout=0.0):
        self.n_poll += 1
        return super().poll(timeout)


def make(family="llama", exit_=15, seg_ms=100.0, adapter_cls=FakeAdapter):
    """(ExitServer, server end of a TimedConn pair). The clock starts at T0; load and first inference cost 0."""
    clock = FakeClock(T0)
    spec = default_spec(family, [exit_], clock="virtual", seg_ms=seg_ms, head_ms=0.0, load_s=0.0, first_infer_s=0.0)
    srv = server.ExitServer(adapter_cls(spec, clock=clock), family, [exit_], n_inputs=4, clock=clock)
    srv.load(CFG, 0, "synthetic")
    conn, _ = TimedConn.pair(clock)
    return srv, conn


def run(req="r1", type_id="llama:15", deadline_ms=10_000):
    _, exit_ = contracts.parse_type_id(type_id)
    return {"op": "run", "req_id": req, "type_id": type_id, "exit": exit_, "input_idx": 0,
            "deadline_ns": None if deadline_ms is None else T0 + deadline_ms * MS, "t_admit_ns": T0}


def pause(req="r1"):
    return {"op": "pause", "req_id": req}


def resume(req="r1"):
    return {"op": "resume", "req_id": req}


STOP = {"op": "stop"}


def ops(conn):
    return [m["op"] for m in conn.sent_msgs()]


# ------------------------------------------------------------------------------------------------ pause and resume
def test_pause_stops_at_the_next_token_boundary_and_resume_continues_from_the_next_token():
    srv, conn = make()
    conn.at(0, run()).at(250, pause()).at(400, resume()).at(5000, STOP)
    assert srv.serve(conn) == "stop"
    paused, done = conn.sent_msgs()
    assert paused == {"op": "paused", "req_id": "r1", "type_id": "llama:15", "tokens_done": 3,
                      "t_pause_ns": T0 + 300 * MS}
    contracts.validate_server_msg(paused)
    contracts.validate_server_msg(done)
    contracts.validate_done(done)
    assert (done["outcome"], done["reached"], done["tokens_done"]) == ("complete", 32, 32)
    assert done["t_end_ns"] == T0 + 3300 * MS
    assert done["pauses"] == [[3, T0 + 300 * MS, T0 + 400 * MS]]
    assert done["ckpt"][2] == [3, T0 + 300 * MS] and done["ckpt"][3] == [4, T0 + 500 * MS]


@pytest.mark.parametrize("pause_ms,resume_ms,k", [(50, 150, 1), (3050, 3150, 31)])
def test_pause_points_range_from_the_prefill_token_to_the_token_before_the_last(pause_ms, resume_ms, k):
    srv, conn = make()
    conn.at(0, run()).at(pause_ms, pause()).at(resume_ms, resume()).at(9000, STOP)
    srv.serve(conn)
    paused, done = conn.sent_msgs()
    assert paused["tokens_done"] == k and paused["t_pause_ns"] == T0 + k * 100 * MS
    assert done["pauses"] == [[k, T0 + k * 100 * MS, T0 + resume_ms * MS]]
    assert done["t_end_ns"] == T0 + (resume_ms + (32 - k) * 100) * MS
    contracts.validate_done(done)


def test_no_pause_at_the_last_token_and_the_stale_pause_is_ignored_when_idle():
    srv, conn = make()
    conn.at(0, run()).at(3150, pause()).at(5000, STOP)          # token 31 ends at 3100, token 32 (last) at 3200
    assert srv.serve(conn) == "stop"
    (done,) = conn.sent_msgs()
    assert done["outcome"] == "complete" and done["t_end_ns"] == T0 + 3200 * MS and "pauses" not in done
    assert conn.n_poll == 31                                     # one poll after every non-last token
    contracts.validate_done(done)


def test_no_pause_after_a_late_token():
    srv, conn = make()
    conn.at(0, run(deadline_ms=250)).at(220, pause()).at(5000, STOP)    # token 3 ends at 300 > 250: late, no poll
    assert srv.serve(conn) == "stop"
    (done,) = conn.sent_msgs()
    assert (done["outcome"], done["reached"]) == ("truncated", 2) and "pauses" not in done
    assert conn.n_poll == 2
    contracts.validate_done(done)


def test_resume_after_the_deadline_runs_one_more_token_then_truncates():
    srv, conn = make()
    conn.at(0, run(deadline_ms=1000)).at(250, pause()).at(1500, resume()).at(5000, STOP)
    srv.serve(conn)
    paused, done = conn.sent_msgs()
    assert (done["outcome"], done["reached"], done["tokens_done"]) == ("truncated", 3, 3)
    assert done["ckpt"][-1] == [4, T0 + 1600 * MS] and done["t_end_ns"] == T0 + 1600 * MS
    assert done["pauses"] == [[3, T0 + 300 * MS, T0 + 1500 * MS]]
    contracts.validate_done(done)


def test_two_pauses_in_one_request_are_both_recorded():
    srv, conn = make()
    conn.at(0, run()).at(250, pause()).at(400, resume()).at(650, pause()).at(1000, resume()).at(9000, STOP)
    srv.serve(conn)
    assert ops(conn) == ["paused", "paused", "done"]
    done = conn.sent_msgs()[-1]
    assert done["pauses"] == [[3, T0 + 300 * MS, T0 + 400 * MS], [6, T0 + 700 * MS, T0 + 1000 * MS]]
    assert done["t_end_ns"] == T0 + (1000 + 26 * 100) * MS
    contracts.validate_done(done)


# ------------------------------------------------------------------------------------------------ stale and races
def test_pause_and_resume_for_other_requests_are_ignored_at_a_boundary_and_when_idle():
    srv, conn = make()
    conn.at(0, pause("r0")).at(0, run()).at(150, pause("r0")).at(150, resume("r0")).at(5000, STOP)
    assert srv.serve(conn) == "stop"
    (done,) = conn.sent_msgs()
    assert done["req_id"] == "r1" and done["t_end_ns"] == T0 + 3200 * MS and "pauses" not in done


def test_duplicate_pause_and_foreign_resume_are_ignored_while_paused():
    srv, conn = make()
    conn.at(0, run()).at(250, pause()).at(350, pause()).at(360, resume("r0")).at(400, resume()).at(5000, STOP)
    srv.serve(conn)
    assert ops(conn) == ["paused", "done"]
    assert conn.sent_msgs()[-1]["pauses"] == [[3, T0 + 300 * MS, T0 + 400 * MS]]


def test_resume_without_a_pause_is_stale_and_ignored():
    srv, conn = make()
    conn.at(0, run()).at(150, resume()).at(5000, STOP)
    srv.serve(conn)
    (done,) = conn.sent_msgs()
    assert done["t_end_ns"] == T0 + 3200 * MS and "pauses" not in done


def test_ping_is_answered_at_a_boundary_and_while_paused():
    srv, conn = make()
    conn.at(0, run()).at(150, {"op": "ping"}).at(250, pause()).at(350, {"op": "ping"}).at(400, resume())
    conn.at(5000, STOP)
    srv.serve(conn)
    assert ops(conn) == ["pong", "paused", "pong", "done"]
    first, _, second, done = conn.sent_msgs()
    assert first == {"op": "pong", "t_ns": T0 + 200 * MS} and second == {"op": "pong", "t_ns": T0 + 350 * MS}
    assert done["pauses"] == [[3, T0 + 300 * MS, T0 + 400 * MS]] and done["t_end_ns"] == T0 + 3300 * MS


def test_stop_while_paused_exits_without_a_done():
    srv, conn = make()
    conn.at(0, run()).at(250, pause()).at(350, STOP)
    assert srv.serve(conn) == "stop"
    assert ops(conn) == ["paused"]


def test_dispatcher_gone_while_paused_is_a_disconnect():
    srv, conn = make()
    conn.at(0, run()).at(250, pause())
    conn._peer.close()                                           # the paused reply finds no dispatcher
    assert srv.serve(conn) == "disconnect"


# ------------------------------------------------------------------------------------------------ backlog
def test_run_and_stop_read_at_a_boundary_wait_for_the_request_in_order():
    srv, conn = make()
    conn.at(0, run()).at(150, run("r2")).at(150, STOP)
    assert srv.serve(conn) == "stop"
    first, second = conn.sent_msgs()
    assert (first["req_id"], second["req_id"]) == ("r1", "r2")
    assert first["t_end_ns"] == T0 + 3200 * MS
    assert second["t_recv_ns"] == second["t_start_ns"] == T0 + 3200 * MS
    assert second["t_end_ns"] == T0 + 6400 * MS


def test_unknown_op_read_at_a_boundary_stops_the_server_after_the_request():
    srv, conn = make()
    conn.at(0, run()).at(150, {"op": "launch"}).at(5000, STOP)
    with pytest.raises(ValueError, match="unknown op"):
        srv.serve(conn)
    assert ops(conn) == ["done"] and conn.sent_msgs()[0]["outcome"] == "complete"


def test_unparseable_message_read_at_a_boundary_stops_the_server_after_the_request():
    srv, conn = make()
    conn.at(0, run()).at(150, b"not json").at(5000, STOP)
    with pytest.raises(ValueError):
        srv.serve(conn)
    assert ops(conn) == ["done"] and conn.sent_msgs()[0]["outcome"] == "complete"


class BoomAt5(FakeAdapter):
    def advance(self, state, ckpt):
        if ckpt == 5:
            raise RuntimeError("boom at token 5")
        super().advance(state, ckpt)


def test_adapter_error_after_a_resume_keeps_the_pause_in_the_error_done():
    srv, conn = make(adapter_cls=BoomAt5)
    conn.at(0, run()).at(250, pause()).at(400, resume()).at(5000, STOP)
    srv.serve(conn)
    done = conn.sent_msgs()[-1]
    assert done["outcome"] == "error" and "boom at token 5" in done["err"]
    assert done["pauses"] == [[3, T0 + 300 * MS, T0 + 400 * MS]]
    contracts.validate_done(done)


# ------------------------------------------------------------------------------------------------ non-llama, no conn
def test_bert_server_never_polls_and_answers_ping_after_the_request():
    srv, conn = make("bert", 11, seg_ms=10.0)
    conn.at(0, run(type_id="bert:11", deadline_ms=None)).at(5, {"op": "ping"}).at(100, STOP)
    assert srv.serve(conn) == "stop"
    assert conn.n_poll == 0 and ops(conn) == ["done", "pong"]


def test_bert_server_refuses_a_pause_as_an_unknown_op():
    srv, conn = make("bert", 11, seg_ms=10.0)
    conn.at(0, pause()).at(100, STOP)
    with pytest.raises(ValueError, match="unknown op"):
        srv.serve(conn)


def test_handle_run_without_serve_never_polls():
    srv, conn = make()
    done = srv.handle_run(run())
    assert done["outcome"] == "complete" and "pauses" not in done and conn.n_poll == 0
