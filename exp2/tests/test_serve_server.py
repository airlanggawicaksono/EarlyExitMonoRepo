"""Tests for exp2.serve.server (slice C1): K5 wire messages, K6 adapter driving, K7 truncation, K19 CLI.

Almost everything runs on the real FakeAdapter in "virtual" mode with a FakeClock: time moves only when the
adapter says a segment took time, so every timestamp below is exact. The process-level tests use real sockets
(in-process thread) and one real subprocess with `--adapter fake`. No torch, no GPU, no Jetson.
"""
import json
import os
import subprocess
import sys
import threading
import types
from pathlib import Path

import pytest

from exp2 import contracts, wire
from exp2.serve import fake_adapter, server
from exp2.serve.fake_adapter import FakeAdapter, default_spec
from exp2.tests.mocks import FakeClock, FakeConn

ROOT = Path(__file__).resolve().parents[2]
T0 = 1_000_000_000
MS = 1_000_000
CFG = {"preprocess": "per_request", "n_inputs": 4, "seed": 0, "sync": "event", "compile": False,
       "input_source": "synthetic"}


# ------------------------------------------------------------------------------------------------ helpers
def make_server(exits=(11,), *, family="bert", checkpoints=True, seg_ms=10.0, head_ms=5.0, first_s=0.0, load_s=0.0,
                adapter_cls=FakeAdapter, n_inputs=4):
    """(ExitServer, adapter, clock) on a loaded virtual FakeAdapter. Adapter spends no time outside segments/head."""
    clock = FakeClock(T0)
    spec = default_spec(family, list(exits), clock="virtual", seg_ms=seg_ms, head_ms=head_ms, load_s=load_s,
                        first_infer_s=first_s)
    adapter = adapter_cls(spec, clock=clock)
    srv = server.ExitServer(adapter, family, list(exits), use_checkpoints=checkpoints, n_inputs=n_inputs, clock=clock)
    srv.load(dict(CFG, n_inputs=n_inputs), 0, "synthetic")
    return srv, adapter, clock


def run_msg(type_id="bert:11", deadline=None, idx=0, req="r1"):
    _, exit_ = contracts.parse_type_id(type_id)
    return {"op": "run", "req_id": req, "type_id": type_id, "exit": exit_, "input_idx": idx,
            "deadline_ns": deadline, "t_admit_ns": T0}


def request_calls(adapter):
    """The adapter calls a request makes, minus bookkeeping calls."""
    return [c for c in adapter.calls if c[0] in ("begin", "advance", "sync", "finish")]


def scrub_env(monkeypatch, *names):
    """Make `names` absent now and restored to their old state when the test ends, even if the code sets them."""
    for n in names:
        monkeypatch.setenv(n, "x")
        monkeypatch.delenv(n)


INDUCTOR_VARS = ("TORCHINDUCTOR_CACHE_DIR", "TORCHINDUCTOR_FX_GRAPH_CACHE", "TORCHINDUCTOR_COMPILE_THREADS")


# ------------------------------------------------------------------------------------------------ K7 truncation
# (deadline offset from T0 in ms | None, outcome, reached, recorded ckpt times in ms, t_end offset in ms)
TRUNC_CASES = [
    pytest.param(None, "complete", 11, [10, 20, 30], 35, id="no_deadline_complete"),
    pytest.param(30, "complete", 11, [10, 20, 30], 35, id="deadline_equals_last_checkpoint_is_on_time"),
    pytest.param(25, "truncated", 7, [10, 20, 30], 35, id="between_checkpoint_2_and_3"),
    pytest.param(20, "truncated", 7, [10, 20, 30], 35, id="deadline_equals_checkpoint_2_is_on_time"),
    pytest.param(10, "truncated", 3, [10, 20], 25, id="deadline_equals_checkpoint_1_is_on_time"),
    pytest.param(9, "miss", None, [10], 10, id="before_first_checkpoint_is_miss"),
    pytest.param(-50, "miss", None, [10], 10, id="deadline_already_past_is_miss"),
]


@pytest.mark.parametrize("deadline_ms,outcome,reached,ckpt_ms,t_end_ms", TRUNC_CASES)
def test_truncation_table(deadline_ms, outcome, reached, ckpt_ms, t_end_ms):
    srv, _, _ = make_server()
    deadline = None if deadline_ms is None else T0 + deadline_ms * MS
    done = srv.handle_run(run_msg(deadline=deadline))
    contracts.validate_done(done)
    assert done["outcome"] == outcome
    assert done["reached"] == reached
    assert done["target"] == 11
    assert done["completion"] == pytest.approx(0.0 if reached is None else (reached + 1) / 12)
    assert done["ckpt"] == [[c, T0 + t * MS] for c, t in zip([3, 7, 11], ckpt_ms)]
    assert (done["t_recv_ns"], done["t_start_ns"]) == (T0, T0)
    assert done["t_end_ns"] == T0 + t_end_ms * MS
    assert done["err"] is None and done["tokens_done"] is None and done["req_id"] == "r1"


def test_late_segment_still_runs_and_finish_gets_reached():
    """K7: the overrun segment occupies the GPU, then the head for `reached` runs. Sync after every checkpoint."""
    srv, adapter, _ = make_server()
    srv.handle_run(run_msg(deadline=T0 + 25 * MS))
    assert request_calls(adapter) == [("begin", 0, 11), ("advance", 3), ("sync",), ("advance", 7), ("sync",),
                                      ("advance", 11), ("sync",), ("finish", 7)]


def test_miss_runs_first_segment_then_finish_none_and_stops():
    srv, adapter, _ = make_server()
    done = srv.handle_run(run_msg(deadline=T0 + 5 * MS))
    assert request_calls(adapter) == [("begin", 0, 11), ("advance", 3), ("sync",), ("finish", None)]
    assert done["outcome"] == "miss" and done["completion"] == 0.0 and done["reached"] is None


def test_complete_runs_every_checkpoint_in_order_then_head_of_target():
    srv, adapter, _ = make_server(exits=(5, 11))
    srv.handle_run(run_msg("bert:11"))
    assert request_calls(adapter) == [("begin", 0, 11), ("advance", 3), ("sync",), ("advance", 7), ("sync",),
                                      ("advance", 11), ("sync",), ("finish", 11)]


def test_head_cost_is_measured_into_t_end_not_charged_to_the_deadline():
    srv, _, _ = make_server(head_ms=40.0)
    done = srv.handle_run(run_msg(deadline=T0 + 30 * MS))          # last checkpoint exactly on time
    assert done["outcome"] == "complete"                           # the 40 ms head does not make it late
    assert done["t_end_ns"] - done["ckpt"][-1][1] == 40 * MS


def test_uses_contracts_truncation_outcome_with_the_measured_times(monkeypatch):
    seen = []
    real = contracts.truncation_outcome

    def spy(ckpts, t_ckpt_ns, deadline_ns):
        seen.append((list(ckpts), list(t_ckpt_ns), deadline_ns))
        return real(ckpts, t_ckpt_ns, deadline_ns)

    monkeypatch.setattr(contracts, "truncation_outcome", spy)
    srv, _, _ = make_server()
    srv.handle_run(run_msg(deadline=T0 + 25 * MS))
    assert seen == [([3, 7, 11], [T0 + 10 * MS, T0 + 20 * MS, T0 + 30 * MS], T0 + 25 * MS)]


def test_the_outcome_comes_from_contracts_not_from_a_private_rule(monkeypatch):
    monkeypatch.setattr(contracts, "truncation_outcome", lambda ckpts, t, d: (3, "truncated"))
    srv, adapter, _ = make_server()
    done = srv.handle_run(run_msg(deadline=None))
    assert (done["reached"], done["outcome"]) == (3, "truncated")
    assert ("finish", 3) in adapter.calls and done["completion"] == pytest.approx(4 / 12)


def test_no_checkpoints_means_one_advance_one_sync():
    srv, adapter, _ = make_server(checkpoints=False)
    done = srv.handle_run(run_msg(deadline=None))
    assert request_calls(adapter) == [("begin", 0, 11), ("advance", 11), ("sync",), ("finish", 11)]
    assert done["outcome"] == "complete" and done["ckpt"] == [[11, T0 + 30 * MS]]
    assert done["target"] == 11
    contracts.validate_done(done)


def test_no_checkpoints_late_is_a_miss_never_truncated():
    srv, adapter, _ = make_server(checkpoints=False)
    done = srv.handle_run(run_msg(deadline=T0 + 25 * MS))
    assert done["outcome"] == "miss" and done["reached"] is None and done["completion"] == 0.0
    assert [c for c in request_calls(adapter) if c[0] == "sync"] == [("sync",)]
    assert ("finish", None) in adapter.calls
    contracts.validate_done(done)


def test_llama_checkpoints_are_tokens_and_tokens_done_is_reported():
    srv, adapter, _ = make_server(exits=(3,), family="llama", seg_ms=2.0, head_ms=0.0)
    done = srv.handle_run(run_msg("llama:3", deadline=T0 + 19 * MS))   # token k ends at 2k ms: 9 on time, 10th late
    contracts.validate_done(done)
    assert (done["outcome"], done["reached"], done["target"], done["tokens_done"]) == ("truncated", 9, 32, 9)
    assert done["completion"] == pytest.approx(9 / 32)
    assert len(done["ckpt"]) == 10 and ("finish", 9) in adapter.calls
    full = srv.handle_run(run_msg("llama:3", req="r2"))
    assert (full["outcome"], full["reached"], full["tokens_done"], full["completion"]) == ("complete", 32, 32, 1.0)


def test_yolo_completion_uses_backbone_depth():
    srv, _, _ = make_server(exits=(5,), family="yolo", seg_ms=10.0)
    done = srv.handle_run(run_msg("yolo:5", deadline=T0 + 35 * MS))    # ckpt k ends at 10(k+1) ms: 0,1,2 on time
    contracts.validate_done(done)
    assert (done["outcome"], done["reached"], done["target"]) == ("truncated", 2, 5)
    assert done["completion"] == pytest.approx(10 / 22)
    assert done["tokens_done"] is None


# ------------------------------------------------------------------------------------------------ errors in `done`
class BoomAdapter(FakeAdapter):
    """Raises on the second advance of the request whose input_idx is 3."""

    def advance(self, state, ckpt):
        if state.input_idx == 3 and ckpt == 7:
            raise RuntimeError("boom in segment")
        super().advance(state, ckpt)


def test_adapter_exception_becomes_error_done_and_server_keeps_serving():
    srv, _, _ = make_server(adapter_cls=BoomAdapter)
    bad = srv.handle_run(run_msg(idx=3))
    contracts.validate_done(bad)
    assert bad["outcome"] == "error" and "boom in segment" in bad["err"] and "RuntimeError" in bad["err"]
    assert bad["reached"] is None and bad["completion"] == 0.0 and bad["target"] == 11
    assert bad["t_start_ns"] is not None and bad["t_end_ns"] is not None
    ok = srv.handle_run(run_msg(idx=0, req="r2"))
    assert ok["outcome"] == "complete" and ok["err"] is None


@pytest.mark.parametrize("msg_edit,why", [
    ({"type_id": "vision:6", "exit": 6}, "family"),
    ({"type_id": "bert:5", "exit": 5}, "exit"),
    ({"input_idx": 4}, "input_idx"),
    ({"input_idx": -1}, "input_idx"),
])
def test_requests_the_server_cannot_serve_get_an_error_done(msg_edit, why):
    srv, adapter, _ = make_server(n_inputs=4)
    done = srv.handle_run(dict(run_msg(), **msg_edit))
    contracts.validate_done(done)
    assert done["outcome"] == "error" and why in done["err"]
    assert not [c for c in adapter.calls if c[0] in ("begin", "advance", "finish")]      # nothing ran


class LaxAdapter(FakeAdapter):
    """An adapter that does not police input_idx (the server must)."""

    def begin(self, input_idx, exit):
        self.calls.append(("begin", input_idx, exit))
        return fake_adapter.FakeState(exit, input_idx, contracts.default_checkpoints(self.family, exit))


def test_server_itself_refuses_an_input_idx_outside_the_loaded_inputs():
    srv, adapter, _ = make_server(adapter_cls=LaxAdapter, n_inputs=4)
    done = srv.handle_run(run_msg(idx=4))
    assert done["outcome"] == "error" and "input_idx" in done["err"]
    assert not [c for c in adapter.calls if c[0] in ("begin", "advance", "finish")]
    assert srv.handle_run(run_msg(idx=3, req="r2"))["outcome"] == "complete"


def test_malformed_run_message_gets_an_error_done_not_a_crash():
    srv, _, _ = make_server()
    done = srv.handle_run({"op": "run", "req_id": "rX", "type_id": "nonsense"})
    assert done["op"] == "done" and done["req_id"] == "rX"
    assert done["outcome"] == "error" and done["err"] and done["reached"] is None and done["completion"] == 0.0


def test_stale_state_never_leaks_between_requests():
    srv, adapter, _ = make_server()
    srv.handle_run(run_msg(deadline=T0 + 5 * MS, req="a"))
    second = srv.handle_run(run_msg(req="b", deadline=None))
    assert second["outcome"] == "complete" and second["ckpt"][0][1] > second["t_start_ns"]
    assert [c for c in adapter.calls if c[0] == "begin"] == [("begin", 0, 11), ("begin", 0, 11)]


# ------------------------------------------------------------------------------------------------ serve loop
def test_serve_answers_ping_run_and_stops():
    srv, _, clock = make_server()
    server_end, client_end = FakeConn.pair()
    wire.send(client_end, {"op": "ping"})
    wire.send(client_end, run_msg(deadline=None))
    wire.send(client_end, {"op": "stop"})
    assert srv.serve(server_end) == "stop"
    pong, done = wire.recv(client_end, 1), wire.recv(client_end, 1)
    contracts.validate_server_msg(pong)
    assert pong == {"op": "pong", "t_ns": T0}
    contracts.validate_server_msg(done)
    assert done["outcome"] == "complete" and not client_end.poll(0)         # stop has no reply


def test_serve_returns_when_the_dispatcher_goes_away():
    srv, _, _ = make_server()
    server_end, client_end = FakeConn.pair()
    client_end.close()                                                     # EOF with nothing queued
    assert srv.serve(server_end) == "disconnect"


def test_serve_survives_a_broken_pipe_while_replying():
    srv, _, _ = make_server()
    server_end, client_end = FakeConn.pair()
    wire.send(client_end, run_msg())
    client_end.closed = True                                               # the reply send sees a dead peer
    assert srv.serve(server_end) == "disconnect"


@pytest.mark.parametrize("bad", [{"op": "launch"}, {"nop": 1}, {"op": "ready"}])
def test_unknown_op_stops_the_server_loudly(bad):
    srv, _, _ = make_server()
    server_end, client_end = FakeConn.pair()
    wire.send(client_end, bad)
    wire.send(client_end, {"op": "stop"})                                  # a lenient server would reach this
    with pytest.raises(ValueError, match="unknown op"):
        srv.serve(server_end)


def test_serve_handles_requests_one_at_a_time_in_order():
    srv, adapter, _ = make_server()
    server_end, client_end = FakeConn.pair()
    for i in range(3):
        wire.send(client_end, run_msg(req="r%d" % i, idx=i))
    wire.send(client_end, {"op": "stop"})
    srv.serve(server_end)
    dones = [wire.recv(client_end, 1) for _ in range(3)]
    assert [d["req_id"] for d in dones] == ["r0", "r1", "r2"]
    assert [c[1] for c in adapter.calls if c[0] == "begin"] == [0, 1, 2]
    assert dones[0]["t_end_ns"] <= dones[1]["t_start_ns"] <= dones[1]["t_end_ns"] <= dones[2]["t_start_ns"]


# ------------------------------------------------------------------------------------------------ load + warm-up
def test_cv_of_known_samples():
    assert server.cv([1.0] * 5) == 0.0
    assert server.cv([9.0, 11.0, 10.0, 10.0, 10.0]) == pytest.approx(0.5 ** 0.5 / 10.0)
    assert server.cv([0.0, 0.0, 0.0]) == 0.0                                # mean 0: defined as 0, not a ZeroDivision
    assert server.cv([3.0]) == 0.0


def test_load_passes_cfg_and_inputs_to_the_adapter_and_times_it():
    srv, adapter, clock = make_server(load_s=0.25)
    assert ("load", (11,)) in adapter.calls and adapter.cfg["n_inputs"] == 4
    assert ("make_inputs", 4, 0, "synthetic") in adapter.calls
    assert clock() == T0 + 250 * MS                                         # the virtual load really took 0.25 s


def test_load_reports_load_seconds_from_the_injected_clock():
    clock = FakeClock(T0)
    adapter = FakeAdapter(default_spec("bert", [11], clock="virtual", load_s=0.25), clock=clock)
    srv = server.ExitServer(adapter, "bert", [11], clock=clock)
    assert srv.load(CFG, 0, "synthetic") == pytest.approx(0.25)


class WrongCkptAdapter(FakeAdapter):
    def checkpoints(self, exit):
        return super().checkpoints(exit)[:-1]


def test_adapter_checkpoints_must_equal_contracts_default():
    with pytest.raises(ValueError, match="checkpoints"):
        make_server(adapter_cls=WrongCkptAdapter)


def test_adapter_family_must_match_server_family():
    clock = FakeClock(T0)
    adapter = FakeAdapter(default_spec("vision", [6], clock="virtual"), clock=clock)
    with pytest.raises(ValueError, match="family"):
        server.ExitServer(adapter, "bert", [6], clock=clock)


def test_warm_up_covers_every_exit_with_exact_first_inference_and_activation():
    srv, adapter, _ = make_server(exits=(11, 5), first_s=0.02)
    warm = srv.warm_up(6)
    assert set(warm) == {"first_infer_s", "n_warm", "cv_last5", "act_mb"}
    for key in warm:
        assert sorted(warm[key]) == ["11", "5"]
    # first inference = first_infer_s 20 ms + segments + head
    assert warm["first_infer_s"]["5"] == pytest.approx(0.020 + 0.020 + 0.005)
    assert warm["first_infer_s"]["11"] == pytest.approx(0.020 + 0.030 + 0.005)
    assert warm["n_warm"] == {"11": 6, "5": 6}
    assert warm["cv_last5"] == {"11": 0.0, "5": 0.0}
    # act = FakeAdapter activation 2 + 0.5 * depth; exit 5 is warmed AFTER 11, so it needs its own reset_peak
    assert warm["act_mb"]["11"] == pytest.approx(2 + 0.5 * 12)
    assert warm["act_mb"]["5"] == pytest.approx(2 + 0.5 * 6)
    begins = [c for c in adapter.calls if c[0] == "begin"]
    assert {c[2] for c in begins} == {5, 11} and len(begins) == 12
    assert [c for c in adapter.calls if c[0] == "reset_peak"]


def test_warm_up_uses_the_serving_path_with_sync_at_every_checkpoint():
    srv, adapter, _ = make_server()
    srv.warm_up(5)
    assert [c for c in adapter.calls if c[0] == "advance"] == [("advance", c) for c in (3, 7, 11)] * 5
    assert len([c for c in adapter.calls if c[0] == "sync"]) == 15


def test_warm_up_cycles_input_indices_within_n_inputs():
    srv, adapter, _ = make_server(n_inputs=3)
    srv.warm_up(7)
    idx = [c[1] for c in adapter.calls if c[0] == "begin"]
    assert idx == [i % 3 for i in range(7)]


class NoisyAdapter(FakeAdapter):
    """Adds 0 / 20 ms alternately to begin() for the first `noisy` calls (all calls when noisy is None)."""
    noisy = None

    def begin(self, input_idx, exit):
        n = sum(1 for c in self.calls if c[0] == "begin")
        if self.noisy is None or n < self.noisy:
            self._wait_ms(20.0 if n % 2 else 0.0)
        return super().begin(input_idx, exit)


class SettlingAdapter(NoisyAdapter):
    noisy = 8


def test_warm_up_keeps_going_until_the_last_five_are_stable():
    srv, _, _ = make_server(adapter_cls=SettlingAdapter)
    warm = srv.warm_up(6)
    assert warm["n_warm"]["11"] == 13            # first 5-window that avoids the 8 noisy runs ends at run 13
    assert warm["cv_last5"]["11"] < server.CV_MAX


def test_warm_up_stops_at_the_cap_and_reports_the_unsettled_cv():
    srv, _, _ = make_server(adapter_cls=NoisyAdapter)
    warm = srv.warm_up(6)
    assert warm["n_warm"]["11"] == 6 * server.WARMUP_MAX_FACTOR
    assert warm["cv_last5"]["11"] >= server.CV_MAX        # not hidden: the caller sees it in ready


def test_ready_message_is_valid_and_complete():
    srv, adapter, clock = make_server(exits=(5, 11), first_s=0.02, load_s=0.05)
    load_s = 0.05
    warm = srv.warm_up(5)
    t_before = clock()
    msg = srv.ready_message(load_s=load_s, warm=warm, compile_state="eager_fallback", cache_files=7,
                            input_source="per_request")
    contracts.validate_ready(msg)
    assert msg["op"] == "ready" and msg["family"] == "bert" and msg["pid"] == os.getpid()
    assert msg["exits_warmed"] == [5, 11]
    assert msg["warmup"]["load_s"] == load_s and msg["warmup"]["inductor_cache_files_at_start"] == 7
    assert msg["warmup"]["n_warm"] == {"5": 6, "11": 6}        # the cold run sits in the first 5-window, so one more
    assert msg["versions"]["compile"] == "eager_fallback"          # a hard fallback is reported, never silent
    assert msg["input_source"] == "per_request" and msg["t_ready_ns"] == t_before
    assert msg["mem"] == {"torch_reserved_mb": 1020.0, "rss_mb": 2000.0}
    assert msg["versions"]["python"] == ".".join(str(v) for v in sys.version_info[:3])
    json.dumps(msg, allow_nan=False)


class NoRssAdapter(FakeAdapter):
    def mem_stats(self):
        return dict(super().mem_stats(), rss_mb=None)


def test_ready_message_refuses_to_send_an_invalid_ready():
    srv, _, _ = make_server(adapter_cls=NoRssAdapter)
    warm = srv.warm_up(5)
    with pytest.raises(contracts.ContractError, match="rss_mb"):
        srv.ready_message(load_s=0.0, warm=warm, compile_state="off", cache_files=0, input_source="per_request")


def test_versions_come_from_loaded_modules_only(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "transformers", raising=False)
    v = server.versions_info("off")
    assert (v["torch"], v["transformers"], v["cuda"], v["compile"]) == (None, None, None, "off")
    torch = types.ModuleType("torch")
    torch.__version__ = "9.9.0"
    torch.version = types.SimpleNamespace(cuda="12.6")
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(__version__="4.1"))
    v = server.versions_info("on")
    assert (v["torch"], v["transformers"], v["cuda"], v["compile"]) == ("9.9.0", "4.1", "12.6", "on")


# ------------------------------------------------------------------------------------------------ inductor env
def test_inductor_env_defaults_forced_values_and_counts_existing_cache_files(monkeypatch, tmp_path):
    scrub_env(monkeypatch, *INDUCTOR_VARS)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert server.prepare_inductor_env() == 0                       # no cache dir yet
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(tmp_path / ".cache" / "exp2_inductor")
    assert os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] == "1" and os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "1"


def test_inductor_env_counts_files_recursively_and_forces_single_thread(monkeypatch, tmp_path):
    scrub_env(monkeypatch, *INDUCTOR_VARS)
    cache = tmp_path / "ind"
    (cache / "sub" / "deeper").mkdir(parents=True)
    for rel in ("a.bin", "sub/b.py", "sub/deeper/c.json"):
        (cache / rel).write_text("x")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(cache))
    monkeypatch.setenv("TORCHINDUCTOR_FX_GRAPH_CACHE", "0")
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "8")
    assert server.prepare_inductor_env() == 3
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(cache)       # an explicit dir is kept
    assert os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] == "1"
    assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "1"


# ------------------------------------------------------------------------------------------------ K19 CLI
def base_argv(tmp_path, *extra):
    return ["--family", "bert", "--exits", "5,11", "--adapter", "fake", "--out", str(tmp_path), *extra]


def write_spec(tmp_path, family="bert", exits=(5, 11), **kw):
    path = tmp_path / "fake_spec.json"
    path.write_text(json.dumps(default_spec(family, list(exits), **kw)))
    return str(path)


def test_parse_args_defaults_and_types(tmp_path):
    args = server.parse_args(base_argv(tmp_path, "--fake-spec", write_spec(tmp_path)))
    assert (args.family, args.exits, args.adapter) == ("bert", [5, 11], "fake")
    got = (args.preprocess, args.n_inputs, args.warmup_iters, args.sync, args.seed)
    assert got == ("per_request", 32, 20, "event", 0)
    assert args.no_checkpoints is False and args.out == str(tmp_path)


def test_parse_args_accepts_every_option(tmp_path):
    spec = write_spec(tmp_path, family="llama", exits=(3, 15), clock="sleep")
    args = server.parse_args(["--family", "llama", "--exits", "3,15", "--adapter", "fake", "--fake-spec", spec,
                              "--out", str(tmp_path), "--preprocess", "preload", "--n-inputs", "8",
                              "--warmup-iters", "9", "--no-checkpoints", "--sync", "spin", "--seed", "4"])
    assert (args.preprocess, args.n_inputs, args.warmup_iters, args.sync, args.seed) == ("preload", 8, 9, "spin", 4)
    assert args.no_checkpoints is True and args.exits == [3, 15]


@pytest.mark.parametrize("argv_edit", [
    pytest.param(lambda a: a + ["--family", "gpt"], id="unknown_family"),
    pytest.param(lambda a: a + ["--exits", "99"], id="exit_out_of_range_for_family"),
    pytest.param(lambda a: a + ["--exits", "5,5"], id="duplicate_exit"),
    pytest.param(lambda a: a + ["--exits", ""], id="empty_exits"),
    pytest.param(lambda a: a + ["--exits", "five"], id="non_numeric_exit"),
    pytest.param(lambda a: a + ["--warmup-iters", "4"], id="warmup_below_cv_window"),
    pytest.param(lambda a: a + ["--n-inputs", "0"], id="zero_inputs"),
    pytest.param(lambda a: a + ["--sync", "busy"], id="bad_sync"),
    pytest.param(lambda a: a + ["--preprocess", "lazy"], id="bad_preprocess"),
    pytest.param(lambda a: a + ["--adapter", "gpu"], id="bad_adapter"),
    pytest.param(lambda a: a + ["--adapter", "real"], id="fake_spec_with_real_adapter"),
])
def test_usage_errors_exit_2(tmp_path, argv_edit):
    argv = argv_edit(base_argv(tmp_path, "--fake-spec", write_spec(tmp_path)))
    with pytest.raises(SystemExit) as exc:
        server.parse_args(argv)
    assert exc.value.code == 2


def test_missing_required_args_exit_2(tmp_path):
    for drop in ("--family", "--exits", "--out"):
        argv = base_argv(tmp_path, "--fake-spec", write_spec(tmp_path))
        i = argv.index(drop)
        del argv[i:i + 2]
        with pytest.raises(SystemExit) as exc:
            server.parse_args(argv)
        assert exc.value.code == 2, drop


def test_fake_adapter_needs_a_spec_file_for_the_same_family(tmp_path):
    with pytest.raises(SystemExit) as exc:
        server.build_adapter(server.parse_args(base_argv(tmp_path)))
    assert exc.value.code == 2
    other = write_spec(tmp_path, family="vision", exits=(6,))
    with pytest.raises(SystemExit) as exc:
        server.build_adapter(server.parse_args(base_argv(tmp_path, "--fake-spec", other)))
    assert exc.value.code == 2
    missing_exit = write_spec(tmp_path, exits=(5,))
    with pytest.raises(SystemExit) as exc:
        server.build_adapter(server.parse_args(base_argv(tmp_path, "--fake-spec", missing_exit)))
    assert exc.value.code == 2


def test_fake_adapter_is_built_from_the_spec(tmp_path):
    args = server.parse_args(base_argv(tmp_path, "--fake-spec", write_spec(tmp_path)))
    adapter = server.build_adapter(args)
    assert isinstance(adapter, fake_adapter.FakeAdapter) and adapter.family == "bert"


def test_real_adapter_comes_from_families(monkeypatch, tmp_path):
    asked = []
    sentinel = object()
    monkeypatch.setattr(server.families, "get_adapter", lambda family, cfg=None: asked.append(family) or sentinel)
    args = server.parse_args(["--family", "vision", "--exits", "6", "--adapter", "real", "--out", str(tmp_path)])
    assert server.build_adapter(args) is sentinel and asked == ["vision"]


# ------------------------------------------------------------------------------------------- process flow (sockets)
class Recorder(FakeAdapter):
    """Sleep-mode FakeAdapter that snapshots the environment at load() and the call log at write_port_file()."""
    env_at_load = None
    log = None

    def load(self, exits, cfg):
        type(self).env_at_load = {k: os.environ.get(k) for k in INDUCTOR_VARS}
        self.log["order"].append("load")
        return super().load(exits, cfg)


def drive_main(tmp_path, monkeypatch, argv_extra=(), *, compile_state="eager_fallback", client=None):
    """Run server.main in a thread against a Recorder adapter; `client(conn, ready)` talks to it. Returns a dict."""
    scrub_env(monkeypatch, *INDUCTOR_VARS)
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "ind"))
    spec = default_spec("bert", [5, 11], clock="sleep", seg_ms=1.0, head_ms=1.0, load_s=0.0, first_infer_s=0.0)
    adapter = Recorder(spec)
    log = {"setup_compile": [], "port_snapshot": None, "order": []}
    adapter.log = log

    def fake_setup_compile(enable):
        log["env_at_compile"] = {k: os.environ.get(k) for k in INDUCTOR_VARS}
        log["setup_compile"].append(enable)
        log["order"].append("setup_compile")
        return compile_state

    real_write = wire.write_port_file

    def write_port_file(path, port):
        log["port_snapshot"] = list(adapter.calls)
        log["order"].append("port_file")
        real_write(path, port)

    monkeypatch.setattr(server.families, "setup_compile", fake_setup_compile)
    monkeypatch.setattr(server, "build_adapter", lambda args: log["order"].append("build_adapter") or adapter)
    monkeypatch.setattr(server.wire, "write_port_file", write_port_file)
    argv = ["--family", "bert", "--exits", "5,11", "--adapter", "real", "--out", str(tmp_path), "--warmup-iters", "5",
            "--n-inputs", "4", *argv_extra]
    result = {}

    def target():
        try:
            result["rc"] = server.main(argv)
        except BaseException as e:      # surfaced to the test thread below
            result["exc"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    port_path = tmp_path / "server_bert.port"
    conn = None
    try:
        port = wire.wait_port_file(port_path, 60.0)
        conn = wire.connect(port)
        ready = wire.recv(conn, 30.0)
        log["ready"] = ready
        if client:
            client(conn, ready)
    finally:
        if conn is not None:
            try:
                wire.send(conn, {"op": "stop"})
            except OSError:
                pass
            conn.close()
        t.join(30.0)
    assert not t.is_alive(), "server thread did not stop"
    assert "exc" not in result, repr(result.get("exc"))
    log.update(result, adapter=adapter, port_path=port_path)
    return log


def test_main_order_warm_everything_then_port_file_then_ready(tmp_path, monkeypatch):
    log = drive_main(tmp_path, monkeypatch)
    assert log["rc"] == 0
    # setup_compile(True) happens first for the real adapter and the env is final before the adapter loads
    assert log["setup_compile"] == [True]
    assert log["order"] == ["setup_compile", "build_adapter", "load", "port_file"]
    final_env = {"TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "ind"),
                 "TORCHINDUCTOR_FX_GRAPH_CACHE": "1", "TORCHINDUCTOR_COMPILE_THREADS": "1"}
    assert Recorder.env_at_load == final_env
    assert log["env_at_compile"] == final_env                        # already final when torch.compile is probed
    snap = log["port_snapshot"]
    assert snap[0][0] == "load" and snap[0][1] == (5, 11)
    warmed = {c[2] for c in snap if c[0] == "begin"}
    assert warmed == {5, 11}                                          # every configured exit ran before the port file
    assert snap.index(("make_inputs", 4, 0, "dataset")) < min(i for i, c in enumerate(snap) if c[0] == "begin")
    ready = log["ready"]
    contracts.validate_ready(ready)
    assert ready["exits_warmed"] == [5, 11] and ready["versions"]["compile"] == "eager_fallback"
    assert ready["input_source"] == "per_request"
    assert ready["warmup"]["inductor_cache_files_at_start"] == 0


@pytest.mark.parametrize("state,flag", [("on", True), ("eager_fallback", False), ("off", False)])
def test_compile_flag_reaches_the_adapter_only_when_compile_is_on(tmp_path, monkeypatch, state, flag):
    log = drive_main(tmp_path, monkeypatch, compile_state=state)
    assert log["adapter"].cfg["compile"] is flag and log["ready"]["versions"]["compile"] == state


def test_cfg_given_to_load_carries_every_cli_setting(tmp_path, monkeypatch):
    log = drive_main(tmp_path, monkeypatch, ("--preprocess", "preload", "--sync", "spin", "--seed", "3"))
    assert log["adapter"].cfg == {"preprocess": "preload", "n_inputs": 4, "seed": 3, "sync": "spin",
                                  "compile": False, "input_source": "dataset"}
    assert ("make_inputs", 4, 3, "dataset") in log["adapter"].calls
    assert log["ready"]["input_source"] == "preload"


def test_main_serves_requests_over_a_real_socket_then_stops(tmp_path, monkeypatch):
    got = {}

    def client(conn, ready):
        wire.send(conn, {"op": "ping"})
        got["pong"] = wire.recv(conn, 10.0)
        wire.send(conn, {"op": "run", "req_id": "r1", "type_id": "bert:11", "exit": 11, "input_idx": 1,
                         "deadline_ns": None, "t_admit_ns": 1})
        got["done"] = wire.recv(conn, 10.0)
        wire.send(conn, {"op": "run", "req_id": "r2", "type_id": "bert:5", "exit": 5, "input_idx": 2,
                         "deadline_ns": 0, "t_admit_ns": 1})                      # deadline long past -> miss
        got["miss"] = wire.recv(conn, 10.0)

    log = drive_main(tmp_path, monkeypatch, client=client)
    assert log["rc"] == 0
    contracts.validate_server_msg(got["pong"])
    contracts.validate_server_msg(got["done"])
    assert got["done"]["outcome"] == "complete" and got["done"]["reached"] == 11
    contracts.validate_done(got["miss"])
    assert got["miss"]["outcome"] == "miss" and got["miss"]["completion"] == 0.0


def test_main_exits_cleanly_when_the_dispatcher_drops(tmp_path, monkeypatch):
    log = drive_main(tmp_path, monkeypatch, client=lambda conn, ready: conn.close())
    assert log["rc"] == 0


def test_main_no_checkpoints_serves_with_a_single_checkpoint(tmp_path, monkeypatch):
    got = {}

    def client(conn, ready):
        wire.send(conn, {"op": "run", "req_id": "r1", "type_id": "bert:11", "exit": 11, "input_idx": 0,
                         "deadline_ns": None, "t_admit_ns": 1})
        got["done"] = wire.recv(conn, 10.0)

    drive_main(tmp_path, monkeypatch, ("--no-checkpoints",), client=client)
    assert [c for c, _ in got["done"]["ckpt"]] == [11]
    contracts.validate_done(got["done"])


# ------------------------------------------------------------------------------------------------ subprocess
def server_cmd(tmp_path, spec, *extra):
    return [sys.executable, "-m", "exp2.serve.server", "--family", "bert", "--exits", "5,11", "--adapter", "fake",
            "--fake-spec", spec, "--out", str(tmp_path), "--warmup-iters", "5", "--n-inputs", "4", *extra]


def test_subprocess_fake_adapter_end_to_end(tmp_path):
    spec = write_spec(tmp_path, clock="sleep", seg_ms=2.0, head_ms=1.0, load_s=0.0, first_infer_s=0.0)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TORCHINDUCTOR_CACHE_DIR=str(tmp_path / "ind"))
    err_path = tmp_path / "server.err"
    port_path = tmp_path / "server_bert.port"
    with open(err_path, "w") as err:
        proc = subprocess.Popen(server_cmd(tmp_path, spec), cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL,
                                stderr=err)
    conn = None
    try:
        port = wire.wait_port_file(port_path, 120.0, proc)
        conn = wire.connect(port)
        ready = wire.recv(conn, 60.0)
        contracts.validate_ready(ready)
        assert ready["family"] == "bert" and ready["exits_warmed"] == [5, 11] 
        assert ready["versions"]["torch"] is None and ready["versions"]["transformers"] is None   # torch never loaded
        assert ready["versions"]["compile"] == "off"
        wire.send(conn, {"op": "ping"})
        assert wire.recv(conn, 10.0)["op"] == "pong"
        wire.send(conn, {"op": "run", "req_id": "r1", "type_id": "bert:11", "exit": 11, "input_idx": 1,
                         "deadline_ns": None, "t_admit_ns": 1})
        done = wire.recv(conn, 10.0)
        contracts.validate_done(done)
        assert done["outcome"] == "complete" and done["reached"] == 11 and len(done["ckpt"]) == 3
        wire.send(conn, {"op": "run", "req_id": "r2", "type_id": "bert:5", "exit": 5, "input_idx": 0,
                         "deadline_ns": 1, "t_admit_ns": 1})
        miss = wire.recv(conn, 10.0)
        contracts.validate_done(miss)
        assert miss["outcome"] == "miss" and miss["completion"] == 0.0
        wire.send(conn, {"op": "stop"})
        assert proc.wait(timeout=30) == 0
    finally:
        if conn is not None:
            conn.close()
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=30)
    assert "Traceback" not in err_path.read_text()


def test_subprocess_usage_error_exits_2_without_a_port_file(tmp_path):
    proc = subprocess.run([sys.executable, "-m", "exp2.serve.server", "--family", "bert", "--exits", "99",
                           "--adapter", "fake", "--out", str(tmp_path)], cwd=str(ROOT), capture_output=True,
                          text=True, timeout=60)
    assert proc.returncode == 2 and not list(tmp_path.glob("*.port"))


def test_importing_the_server_does_not_load_torch():
    code = ("import sys; import exp2.serve.server; "
            "bad = [m for m in ('torch', 'transformers', 'cv2') if m in sys.modules]; "
            "sys.exit(1 if bad else 0)")
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr


def test_server_source_never_imports_heavy_modules_at_top_level():
    import ast

    tree = ast.parse((ROOT / "exp2" / "serve" / "server.py").read_text(encoding="utf-8"))
    top = {a.name.split(".")[0] for n in tree.body if isinstance(n, ast.Import) for a in n.names}
    top |= {n.module.split(".")[0] for n in tree.body if isinstance(n, ast.ImportFrom) and n.module}
    assert not top & {"torch", "transformers", "cv2", "shared"}
