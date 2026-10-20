"""S0b gate: the golden fixtures validate against exp2.contracts, fail when a key is removed, agree with each
other, and the mocks that make them (exp2/tests/mocks.py, exp2/serve/fake_adapter.py) behave as documented.

Sections
    A. mocks: FakeClock, FakeConn, FakeBoard, StubSlowdown, UnitSlowdown, StubPower, make_types, StubSim
    B. FakeAdapter (K6)
    C. fixtures: validators, key-removal mutation tests, cross-file consistency, regeneration
"""
import json
import math
import threading

import pytest

from exp2 import contracts as C
from exp2.tests import mocks as M


# ===================================================================================================== A. mocks
class TestFakeClock:
    def test_starts_at_start_ns_and_is_callable(self):
        clock = M.FakeClock(start_ns=5)
        assert clock() == 5 and isinstance(clock(), int)

    def test_advance_accumulates_in_ns_and_seconds(self):
        clock = M.FakeClock()
        clock.advance_ns(1_500)
        clock.advance_s(0.25)
        assert clock() == 1_500 + 250_000_000

    def test_sleep_advances_instead_of_waiting(self):
        clock = M.FakeClock()
        clock.sleep(2.0)
        assert clock() == 2_000_000_000
        assert clock.monotonic() == 2.0

    def test_negative_advance_and_backwards_set_are_refused(self):
        clock = M.FakeClock(100)
        with pytest.raises(ValueError):
            clock.advance_ns(-1)
        with pytest.raises(ValueError):
            clock.set_ns(99)
        clock.set_ns(100)
        clock.set_ns(250)
        assert clock() == 250

    def test_wall_clock_follows_monotonic_with_fixed_offset(self):
        clock = M.FakeClock(10, wall_offset_ns=1_000_000)
        clock.advance_ns(5)
        assert clock.wall_ns() == 1_000_015

    def test_concurrent_advances_are_not_lost(self):
        clock = M.FakeClock()
        threads = [threading.Thread(target=lambda: [clock.advance_ns(1) for _ in range(2000)]) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert clock() == 8000


class TestFakeConn:
    def test_bytes_round_trip_in_order_both_ways(self):
        a, b = M.FakeConn.pair()
        a.send_bytes(b"one")
        a.send_bytes(b"two")
        b.send_bytes(b"back")
        assert [b.recv_bytes(), b.recv_bytes()] == [b"one", b"two"]
        assert a.recv_bytes() == b"back"

    def test_poll_reports_pending_data_without_consuming_it(self):
        a, b = M.FakeConn.pair()
        assert b.poll(0) is False
        a.send_bytes(b"x")
        assert b.poll(0) is True and b.poll(0) is True
        assert b.recv_bytes() == b"x"
        assert b.poll(0) is False

    def test_poll_waits_for_a_late_sender_in_real_time(self):
        a, b = M.FakeConn.pair()
        threading.Timer(0.05, lambda: a.send_bytes(b"late")).start()
        assert b.poll(5.0) is True
        assert b.recv_bytes() == b"late"

    def test_blocked_recv_is_released_by_send_from_another_thread(self):
        a, b = M.FakeConn.pair()
        got = []
        t = threading.Thread(target=lambda: got.append(b.recv_bytes()))
        t.start()
        a.send_bytes(b"wake")
        t.join(5.0)
        assert got == [b"wake"]

    def test_peer_close_drains_then_raises_eof(self):
        a, b = M.FakeConn.pair()
        a.send_bytes(b"last")
        a.close()
        assert b.poll(0) is True
        assert b.recv_bytes() == b"last"
        assert b.poll(0) is True            # a closed pipe is "readable": the read then raises EOFError
        with pytest.raises(EOFError):
            b.recv_bytes()

    def test_send_on_closed_end_raises_oserror(self):
        a, b = M.FakeConn.pair()
        a.close()
        with pytest.raises(OSError):
            a.send_bytes(b"x")
        assert a.closed is True and b.closed is False

    def test_sent_log_keeps_every_payload_and_decodes_json(self):
        a, b = M.FakeConn.pair()
        a.send_bytes(json.dumps({"op": "ping"}).encode())
        a.send_bytes(json.dumps({"op": "stop"}).encode())
        assert a.sent_msgs() == [{"op": "ping"}, {"op": "stop"}]
        assert b.sent == []

    def test_virtual_clock_poll_advances_time_to_the_delivery(self):
        clock = M.FakeClock()
        a, b = M.FakeConn.pair(clock=clock, latency_ns=5_000_000)
        a.send_bytes(b"m")
        assert b.poll(0.001) is False and clock() == 1_000_000       # waited 1 ms, still in flight
        assert b.poll(0.010) is True and clock() == 5_000_000        # jumped to the delivery time, not +10 ms
        assert b.recv_bytes() == b"m"

    def test_virtual_clock_empty_poll_burns_the_whole_timeout(self):
        clock = M.FakeClock()
        a, b = M.FakeConn.pair(clock=clock)
        assert b.poll(0.25) is False
        assert clock() == 250_000_000

    def test_virtual_clock_recv_with_nothing_pending_fails_loudly_instead_of_hanging(self):
        clock = M.FakeClock()
        a, b = M.FakeConn.pair(clock=clock)
        with pytest.raises(RuntimeError, match="deadlock"):
            b.recv_bytes()

    def test_virtual_clock_delivers_in_send_order(self):
        clock = M.FakeClock()
        a, b = M.FakeConn.pair(clock=clock, latency_ns=100)
        a.send_bytes(b"1")
        clock.advance_ns(50)
        a.send_bytes(b"2")
        assert b.recv_bytes() == b"1" and clock() == 100
        assert b.recv_bytes() == b"2" and clock() == 150


class TestFakeBoard:
    def test_implements_the_boardcontrol_protocol(self):
        assert isinstance(M.FakeBoard(), C.BoardControl)

    def test_every_call_is_recorded_in_order_with_arguments(self):
        board = M.FakeBoard()
        board.set_gpu_level(612)
        board.lock_emc(3199.0)
        board.set_gpu_level(1020)
        assert board.names() == ["set_gpu_level", "lock_emc", "set_gpu_level"]
        assert [c.args for c in board.calls] == [(612,), (3199.0,), (1020,)]

    def test_set_gpu_level_returns_the_configured_readback_not_the_command(self):
        board = M.FakeBoard(readback={612: 510})
        assert board.set_gpu_level(612) == 510
        assert board.set_gpu_level(1020) == 1020
        assert board.gpu_mhz == 1020                                  # commanded level, independent of read-back

    def test_readback_may_be_a_function(self):
        assert M.FakeBoard(readback=lambda mhz: mhz - 6).set_gpu_level(612) == 606

    def test_unknown_level_is_value_error_and_changes_nothing(self):
        board = M.FakeBoard()
        board.set_gpu_level(612)
        with pytest.raises(ValueError):
            board.set_gpu_level(777)
        assert board.gpu_mhz == 612

    def test_emc_level_must_be_lockable(self):
        board = M.FakeBoard()
        assert board.lock_emc(3199.0) == 3199.0
        with pytest.raises(ValueError):
            board.lock_emc(1000.0)
        assert board.emc_mhz == 3199.0
        board.unlock_emc()
        assert board.emc_mhz is None

    def test_fail_on_exception_fails_every_call_of_that_method(self):
        board = M.FakeBoard(fail_on={"set_gpu_level": C.ClockError("stuck")})
        for _ in range(2):
            with pytest.raises(C.ClockError):
                board.set_gpu_level(612)
        assert board.names() == ["set_gpu_level", "set_gpu_level"]       # the failed calls are still logged
        board.lock_emc(3199.0)                                           # other methods are unaffected

    def test_fail_on_call_index_fails_only_that_call(self):
        board = M.FakeBoard(fail_on={"apply_arm": {2: C.ClockError("second")}})
        board.apply_arm("pinned", 612, 3199.0)
        with pytest.raises(C.ClockError):
            board.apply_arm("pinned", 1020, 3199.0)
        assert board.apply_arm("pinned", 1020, 3199.0)["gpu_mhz"] == 1020

    def test_apply_arm_semantics_for_the_three_arms(self):
        board = M.FakeBoard()
        out = board.apply_arm("pinned", 612, 3199.0)
        assert (board.gpu_mhz, board.emc_mhz, board.governor_free) == (612, 3199.0, False)
        assert out["gpu_mhz"] == 612 and out["emc_mhz"] == 3199.0
        board.apply_arm("governor", None, 3199.0)
        assert (board.governor_free, board.emc_mhz) == (True, 3199.0)
        board.apply_arm("stock", None, None)
        assert (board.governor_free, board.emc_mhz) == (True, None)
        with pytest.raises(ValueError):
            board.apply_arm("turbo", None, None)

    def test_power_mode_query_and_ensure(self):
        board = M.FakeBoard(power_mode=("MODE_15W", 1))
        assert board.query_power_mode() == ("MODE_15W", 1)
        board.ensure_power_mode(2)
        assert board.query_power_mode() == (C.POWER_MODE_NAME, C.POWER_MODE_ID)

    def test_jetson_clocks_restore_needs_a_prior_store(self):
        board = M.FakeBoard()
        with pytest.raises(C.ClockError):
            board.restore_jetson_clocks("/tmp/x")
        board.store_jetson_clocks("/tmp/x")
        board.restore_jetson_clocks("/tmp/x")

    def test_snapshot_reports_state_and_close_is_recorded(self):
        board = M.FakeBoard()
        board.apply_arm("pinned", 612, 3199.0)
        snap = board.snapshot()
        assert snap["gpu_mhz"] == 612 and snap["emc_mhz"] == 3199.0
        board.close()
        assert board.closed is True and board.names()[-1] == "close"


class TestStubModels:
    def test_stub_slowdown_is_n_over_c_of_n(self):
        sd = M.StubSlowdown(c={1: 1.0, 2: 0.93})
        assert isinstance(sd, C.SlowdownModel)
        assert sd.predict("a", ("a",), 1020) == 1.0
        assert sd.predict("a", ("a", "b"), 1020) == pytest.approx(2 / 0.93)
        assert sd.predict("a", ("a", "b", "c"), 612) == pytest.approx(3 / 0.93)      # c(n >= 2) = 0.93
        assert sd.sample("a", ("a", "b"), 1020, rng=None) == sd.predict("a", ("a", "b"), 1020)

    def test_stub_slowdown_with_three_levels_uses_the_largest_known_n(self):
        sd = M.StubSlowdown(c={1: 1.0, 2: 0.93, 3: 0.85})
        assert sd.predict("a", ("a", "b", "c"), 1020) == pytest.approx(3 / 0.85)
        assert sd.predict("a", ("a", "b", "c", "d"), 1020) == pytest.approx(4 / 0.85)

    def test_stub_slowdown_json_passes_the_contract(self):
        out = M.StubSlowdown().to_json()
        C.validate_slowdown(out)
        assert out["name"] in ("M0", "M1", "M2", "M3", "M4lin", "M4")

    def test_stub_slowdown_fit_returns_self_and_rejects_heldout_rows(self):
        sd = M.StubSlowdown()
        assert sd.fit([{"split": "train"}], [], None) is sd
        with pytest.raises(C.HeldOutError):
            sd.fit([{"split": "heldout"}], [], None)

    def test_unit_slowdown_is_always_one(self):
        sd = M.UnitSlowdown()
        assert isinstance(sd, C.SlowdownModel)
        assert {sd.predict("a", ("a", "b", "c"), f) for f in (612, 1020)} == {1.0}
        assert sd.sample("a", ("a", "b"), 612, rng=None) == 1.0
        C.validate_slowdown(sd.to_json())

    def test_stub_power_follows_the_k13_formula(self):
        pw = M.StubPower(p_idle={612: 5.1, 1020: 5.6}, dp={612: 4.2, 1020: 7.9}, eta={612: 0.93, 1020: 0.91}, u=0.4)
        assert isinstance(pw, C.PowerModelP)
        assert pw.p((), 1020) == 5.6                                                  # exactly p_idle
        assert pw.p(("a",), 1020) == pytest.approx(5.6 + 7.9 * 0.4 / 0.91)
        assert pw.p(("a", "b", "c"), 612) == pytest.approx(5.1 + 4.2 * min(1.0, 1.2 / 0.93))
        assert pw.p(("a", "b", "c"), 612) == pytest.approx(5.1 + 4.2)                 # saturates at p_idle + dp

    def test_stub_power_reads_per_type_utilisation(self):
        pw = M.StubPower(p_idle=5.0, dp=10.0, eta=1.0, u={"a": 0.25, "b": 0.5})
        assert pw.p(("a", "b"), 1020) == pytest.approx(5.0 + 10.0 * 0.75)

    def test_stub_power_json_passes_the_contract(self):
        out = M.StubPower(p_idle=5.0, dp=3.0, eta=0.9).to_json()
        C.validate_power(out)
        assert set(out["p_idle_w"]) == {"612", "1020"}


class TestMakeTypes:
    def test_default_table_is_a_valid_typetable(self):
        types = M.make_types()
        assert isinstance(types, C.TypeTable)
        C.validate_types(types.to_json())
        assert types.type_ids() == ["bert:5", "vision:24"]
        assert types.f_max_mhz == 1020 and types.alpha == 5.0

    def test_class_rule_splits_at_the_median_of_means(self):
        types = M.make_types({"bert:5": {"tau": 0.03}, "vision:24": {"tau": 0.1}, "yolo:1": {"tau": 0.01}})
        assert [types.get(t).klass for t in ("bert:5", "vision:24", "yolo:1")] == ["short", "long", "short"]

    def test_explicit_klass_wins(self):
        types = M.make_types({"bert:5": {"tau": 0.03, "klass": "long"}, "vision:24": {"tau": 0.1, "klass": "short"}})
        assert (types.get("bert:5").klass, types.get("vision:24").klass) == ("long", "short")

    def test_scalar_and_per_level_tau(self):
        types = M.make_types({"bert:5": {"tau": {612: 0.05, 1020: 0.03}}, "vision:24": {"tau": 0.2}})
        assert types.tau("bert:5", 612) == 0.05 and types.tau("bert:5", 1020, "mean") == 0.03
        assert types.tau("vision:24", 612) == types.tau("vision:24", 1020) == 0.2

    def test_full_stat_dict_and_samples_are_kept(self):
        tau = {1020: {"p50": 0.5, "mean": 0.6, "p95": 0.9, "samples": [0.1, 0.5, 0.9]}}
        types = M.make_types({"bert:5": {"tau": tau}}, f_levels=(1020,), corun_levels=(1020,))
        assert types.get("bert:5").tau_s[1020]["samples"] == [0.1, 0.5, 0.9]
        assert types.tau("bert:5", 1020, "mean") == 0.6

    def test_ckpt_frac_and_head_frac_are_consistent_with_the_contract(self):
        types = M.make_types({"bert:23": {"tau": 0.1}, "llama:15": {"tau": 0.8}})
        for t in types.types:
            target = t.ckpts[-1]
            assert abs(t.ckpt_frac[target] + t.head_frac - 1.0) <= 0.05
        assert types.get("llama:15").head_frac == 0.0

    def test_alpha_none_and_levels_are_passed_through(self):
        types = M.make_types(alpha=None, f_levels=(408, 1020), corun_levels=(1020,))
        assert types.alpha is None and types.corun_levels_mhz == (1020,) and types.f_levels_mhz == (408, 1020)

    def test_rate_for_load_works_on_a_made_table(self):
        types = M.make_types({"bert:5": {"tau": 0.02}, "vision:24": {"tau": 0.1}})
        wl = {"types": ["bert:5", "vision:24"], "probs": [0.9, 0.1]}
        assert C.rate_for_load(0.6, wl, types) == pytest.approx(0.6 / (0.9 * 0.02 + 0.1 * 0.1))

    def test_invalid_request_is_a_contract_error(self):
        with pytest.raises(C.ContractError):
            M.make_types({"bert:99": {"tau": 0.03}})


class TestSimResultAndStubSim:
    def test_make_simresult_is_valid_and_rates_sum_to_one(self):
        res = M.make_simresult("s", {"bert:5": {"complete_rate": 0.9, "truncation_rate": 0.1, "mean_shortfall": 0.02}},
                               energy_per_req_j=0.7)
        assert isinstance(res, C.SimResult)
        C.validate_simresult(res.to_json())
        st = res.per_type["bert:5"]
        assert st.complete_rate == 0.9 and st.truncation_rate == 0.1 and st.mean_shortfall == 0.02
        assert res.energy_per_req_j == 0.7

    def test_make_simresult_refuses_rates_that_do_not_sum_to_one(self):
        with pytest.raises(C.ContractError):
            M.make_simresult("s", {"bert:5": {"complete_rate": 0.9}})

    def spec(self, ks, kl, order="fcfs", f=1020):
        return C.PolicySpec("p", ks, kl, order, None, "static", f, {})

    def test_stub_sim_returns_the_table_entry_and_logs_the_call(self):
        entry = {"energy_per_req_j": 0.5, "per_type": {"bert:5": {"complete_rate": 1.0}}}
        sim = M.StubSim({(1, 1, "fcfs", 1020): entry})
        out = sim("W", self.spec(1, 1), "types", "sd", "pw", "cfg")
        assert out.energy_per_req_j == 0.5
        assert len(sim.calls) == 1 and sim.calls[0].key == (1, 1, "fcfs", 1020)

    def test_stub_sim_accepts_the_spec_as_a_keyword_or_a_dict(self):
        sim = M.StubSim({(2, 1, "stf", 612): {"energy_per_req_j": 0.4}})
        assert sim(spec=self.spec(2, 1, "stf", 612)).energy_per_req_j == 0.4
        assert sim("W", self.spec(2, 1, "stf", 612).to_json()).energy_per_req_j == 0.4

    def test_stub_sim_missing_key_names_the_key(self):
        with pytest.raises(KeyError, match="1000000"):
            M.StubSim({}, )(self.spec(C.K_INF, 1))

    def test_stub_sim_short_key_fallback_and_default(self):
        sim = M.StubSim({(1, 1): {"energy_per_req_j": 0.9}}, default={"energy_per_req_j": 2.0})
        assert sim(self.spec(1, 1, "stf", 612)).energy_per_req_j == 0.9
        assert sim(self.spec(3, 3)).energy_per_req_j == 2.0

    def test_stub_sim_passes_simresult_objects_through(self):
        res = M.make_simresult("fixed", {"bert:5": {}})
        assert M.StubSim({(1, 1): res})(self.spec(1, 1)) is res

    def test_response_maps_make_the_named_player_cheapest_at_its_best_response(self):
        K = (1, 2, 3)
        br_s, br_l = {1: 2, 2: 3, 3: 1}, {1: 1, 2: 2, 3: 3}
        sim = M.StubSim.from_response_maps(br_s, br_l, K)
        for ks in K:
            for kl in K:
                res = sim(self.spec(ks, kl))
                short, long_ = res.per_type["bert:5"], res.per_type["vision:24"]
                assert (short.mean_shortfall == 0.0) == (ks == br_s[kl])
                assert (long_.mean_shortfall == 0.0) == (kl == br_l[ks])
        # a worse response costs strictly more than the best one (so shortfall alone ranks the choices)
        costs = [sim(self.spec(ks, 1)).per_type["bert:5"].mean_shortfall for ks in K]
        assert costs.count(0.0) == 1 and len(set(costs)) == 3

    def test_response_maps_can_encode_a_two_cycle(self):
        K = (1, 2)
        sim = M.StubSim.from_response_maps({1: 2, 2: 1}, {1: 1, 2: 2}, K)        # matching pennies: no fixed point
        fixed = [(ks, kl) for ks in K for kl in K
                 if sim(self.spec(ks, kl)).per_type["bert:5"].mean_shortfall == 0.0
                 and sim(self.spec(ks, kl)).per_type["vision:24"].mean_shortfall == 0.0]
        assert fixed == []


# ===================================================================================================== B. FakeAdapter
from exp2.serve import fake_adapter as FA  # noqa: E402  (kept next to its tests)


def virtual_adapter(family="bert", exits=(5, 23), **kw):
    clock = M.FakeClock()
    ad = FA.FakeAdapter(FA.default_spec(family, list(exits), clock="virtual", **kw), clock=clock)
    ad.load(list(exits), {})
    ad.make_inputs(4, 0, "synthetic")
    return ad, clock


class TestFakeAdapter:
    def test_implements_the_familyadapter_protocol(self):
        ad, _ = virtual_adapter()
        assert isinstance(ad, C.FamilyAdapter) and ad.family == "bert"

    def test_default_spec_is_valid_for_every_family(self):
        for fam, exits in (("bert", [5, 23]), ("vision", [6, 24]), ("llama", [3, 15]), ("yolo", [1, 5])):
            C.validate_fake_spec(FA.default_spec(fam, exits))

    def test_virtual_mode_needs_a_clock(self):
        with pytest.raises(ValueError, match="clock"):
            FA.FakeAdapter(FA.default_spec("bert", [5], clock="virtual"))

    def test_invalid_spec_is_refused(self):
        spec = FA.default_spec("bert", [5])
        spec["types"]["5"]["seg_ms"] = [20.0]                  # exit 5 has two checkpoints
        with pytest.raises(C.ContractError):
            FA.FakeAdapter(spec)

    def test_load_waits_load_s_on_the_virtual_clock_and_rejects_unknown_exits(self):
        clock = M.FakeClock()
        ad = FA.FakeAdapter(FA.default_spec("bert", [5], clock="virtual", load_s=0.5), clock=clock)
        out = ad.load([5], {"preprocess": "per_request"})
        assert clock() == 500_000_000 and out["load_s"] == 0.5
        with pytest.raises(ValueError, match="not in the fake spec"):
            ad.load([23], {})

    def test_checkpoints_equal_the_contract_for_all_families(self):
        for fam, exits in (("bert", [5, 23]), ("vision", [6, 24]), ("llama", [3, 15]), ("yolo", [1, 5])):
            ad, _ = virtual_adapter(fam, exits)
            for e in exits:
                assert ad.checkpoints(e) == C.default_checkpoints(fam, e)

    def test_begin_needs_inputs_and_a_valid_index(self):
        clock = M.FakeClock()
        ad = FA.FakeAdapter(FA.default_spec("bert", [5], clock="virtual"), clock=clock)
        ad.load([5], {})
        with pytest.raises(RuntimeError, match="make_inputs"):
            ad.begin(0, 5)
        ad.make_inputs(2, 0, "synthetic")
        for bad in (-1, 2):
            with pytest.raises(ValueError):
                ad.begin(bad, 5)
        with pytest.raises(ValueError, match="not loaded"):
            ad.begin(0, 23)

    def test_only_the_first_begin_of_an_exit_pays_first_infer_s(self):
        ad, clock = virtual_adapter(exits=(5, 23), first_infer_s=0.3)
        t0 = clock()
        ad.begin(0, 5)
        assert clock() - t0 == 300_000_000
        ad.begin(1, 5)
        assert clock() - t0 == 300_000_000
        ad.begin(0, 23)
        assert clock() - t0 == 600_000_000

    def test_advance_moves_the_clock_by_the_segment_and_keeps_order(self):
        ad, clock = virtual_adapter(exits=(23,), seg_ms=20.0)
        st = ad.begin(0, 23)
        t0 = clock()
        ad.advance(st, 3)
        assert clock() - t0 == 20_000_000 and st.done == [3]
        ad.advance(st, 7)
        assert clock() - t0 == 40_000_000
        with pytest.raises(ValueError):
            ad.advance(st, 3)                                  # going back
        with pytest.raises(ValueError):
            ad.advance(st, 5)                                  # not a checkpoint of bert:23

    def test_advance_can_jump_and_pays_for_every_skipped_segment(self):
        spec = FA.default_spec("bert", [11], clock="virtual")
        spec["types"]["11"]["seg_ms"] = [10.0, 20.0, 40.0]       # checkpoints 3, 7, 11
        clock = M.FakeClock()
        ad = FA.FakeAdapter(spec, clock=clock)
        ad.load([11], {})
        ad.make_inputs(1, 0, "synthetic")
        st = ad.begin(0, 11)
        t0 = clock()
        ad.advance(st, 11)
        assert clock() - t0 == 70_000_000 and st.done == [3, 7, 11]

    def test_finish_costs_head_ms_for_a_reached_checkpoint_and_nothing_for_none(self):
        ad, clock = virtual_adapter(exits=(5,), head_ms=5.0)
        st = ad.begin(0, 5)
        ad.advance(st, 3)
        t0 = clock()
        assert ad.finish(st, None) is None and clock() == t0
        assert ad.finish(st, 3)["reached"] == 3 and clock() - t0 == 5_000_000
        with pytest.raises(ValueError, match="not a finished checkpoint"):
            ad.finish(st, 5)

    def test_llama_has_32_checkpoints_and_no_head(self):
        ad, clock = virtual_adapter("llama", (3,), seg_ms=2.0)
        st = ad.begin(0, 3)
        t0 = clock()
        ad.advance(st, 32)
        assert st.done == list(range(1, 33)) and clock() - t0 == 64_000_000
        t1 = clock()
        ad.finish(st, 32)
        assert clock() == t1

    def test_call_log_records_one_request_in_order(self):
        ad, _ = virtual_adapter(exits=(5,))
        ad.calls.clear()
        st = ad.begin(2, 5)
        ad.advance(st, 3)
        ad.sync()
        ad.advance(st, 5)
        ad.sync()
        ad.finish(st, 5)
        assert ad.calls == [("begin", 2, 5), ("advance", 3), ("sync",), ("advance", 5), ("sync",), ("finish", 5)]

    def test_memory_peak_follows_activations_since_reset(self):
        ad, _ = virtual_adapter(exits=(5, 23))
        ad.reset_peak()
        base = ad.mem_stats()["torch_peak_alloc_mb"]
        ad.begin(0, 5)
        small = ad.mem_stats()["torch_peak_alloc_mb"] - base
        ad.begin(0, 23)
        large = ad.mem_stats()["torch_peak_alloc_mb"] - base
        assert 0 < small < large
        ad.reset_peak()
        assert ad.mem_stats()["torch_peak_alloc_mb"] == base
        assert ad.mem_stats()["torch_reserved_mb"] == 1020.0

    def test_make_inputs_validates_its_arguments(self):
        ad, _ = virtual_adapter()
        for args in ((0, 0, "synthetic"), (3, 0, "jpeg")):
            with pytest.raises(ValueError):
                ad.make_inputs(*args)

    def test_sleep_mode_sleeps_the_spec_durations_and_never_touches_a_clock(self, monkeypatch):
        slept = []
        monkeypatch.setattr(FA.time, "sleep", slept.append)
        ad = FA.FakeAdapter(FA.default_spec("bert", [5], clock="sleep", load_s=0.1, first_infer_s=0.2, seg_ms=20.0,
                                            head_ms=5.0))
        ad.load([5], {})
        ad.make_inputs(1, 0, "synthetic")
        st = ad.begin(0, 5)
        ad.advance(st, 3)
        ad.advance(st, 5)
        ad.finish(st, 5)
        assert slept == pytest.approx([0.1, 0.2, 0.02, 0.02, 0.005])


# ===================================================================================================== C. fixtures
import copy  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import shutil  # noqa: E402
from pathlib import Path  # noqa: E402

from exp2 import stats as S  # noqa: E402

FIX = M.FIXTURES
RUN1 = ("solo", "solo_samples", "corun", "corun_samples", "idle", "ckpt", "warmup", "mem")
SESSIONS = ("session_closed", "session_open")
TOP_FILES = (
    "README.md", "workload.json", "policyspecs.json", "wire_messages.json", "root_helper_messages.json",
    "fake_spec.json", "plan_run1.json", "plan_run2.json", "types_from_run1.json", "slowdown_model.json",
    "power_model.json", "freeze.json", "freeze.json.sha256", "simresult.json", "open_summary.json",
    "open_conditions.csv", "pilot_summary.json")


def jload(rel):
    with open(FIX / rel, encoding="utf-8") as fh:
        return json.load(fh)


def load_session(name):
    """(session.json, {condition_id: manifest}, {condition_id: rows}, power rows, clocks rows)."""
    sdir = FIX / name
    session = jload(name + "/session.json")
    mans = {cid: jload("%s/%s/manifest.json" % (name, cid)) for cid in session["conditions"]}
    rows = {cid: C.read_requests(sdir / cid / "requests.jsonl") for cid in session["conditions"]}
    return (session, mans, rows, C.read_csv(sdir / "telemetry" / "power.csv", "power"),
            C.read_csv(sdir / "telemetry" / "clocks.csv", "clocks"))


def normalized(path):
    """File bytes with CRLF folded to LF, so a Windows checkout with autocrlf compares equal."""
    return Path(path).read_bytes().replace(b"\r\n", b"\n")


def tree_files(root):
    return sorted(p.relative_to(root).as_posix() for p in Path(root).rglob("*") if p.is_file())


class TestInventory:
    def test_every_listed_fixture_exists_and_is_not_empty(self):
        for name in TOP_FILES:
            assert (FIX / name).is_file() and (FIX / name).stat().st_size > 0, name
        for kind in RUN1:
            assert (FIX / "run1_csv" / (kind + ".csv")).stat().st_size > 0, kind

    @pytest.mark.parametrize("name", SESSIONS)
    def test_session_layout_is_k9(self, name):
        session = jload(name + "/session.json")
        want = {"session.json", "telemetry/power.csv", "telemetry/clocks.csv", "telemetry/meta.json"}
        for cid in session["conditions"]:
            want |= {cid + "/manifest.json", cid + "/requests.jsonl", cid + "/freq_switch.csv"}
        assert set(tree_files(FIX / name)) == want

    def test_readme_indexes_every_fixture(self):
        text = (FIX / "README.md").read_text(encoding="utf-8")
        names = [n for n in TOP_FILES if n != "README.md"] + ["session_closed/", "session_open/", "run1_csv/"]
        names += ["run1_csv/%s.csv" % k for k in RUN1] + ["admission_cases.json", "admission_traces.json",
                                                          "truncation_cases.json"]
        missing = [n for n in names if "`%s`" % n not in text and "`%s`" % n.split("/")[-1] not in text]
        assert missing == []

    def test_fixtures_are_not_gitignored_and_use_lf(self):
        for p in FIX.rglob("*"):
            if p.is_file() and p.suffix in (".json", ".csv", ".jsonl", ".sha256", ".md"):
                assert b"\r\n" not in p.read_bytes() or p.name == "README.md", p


# ---------------------------------------------------------------- validators accept every fixture
class TestFixturesValidate:
    def test_types(self):
        C.validate_types(jload("types_from_run1.json"))
        assert C.load_types(FIX / "types_from_run1.json").type_ids() == ["bert:5", "vision:24"]

    def test_workloads_and_rate(self):
        types = C.load_types(FIX / "types_from_run1.json")
        wls = jload("workload.json")
        assert [w["name"] for w in wls] == ["W_hi", "W_lo", "W_burst"]
        for w in wls:
            C.validate_workload(w, types)
            assert w["rate_per_s"] == pytest.approx(C.rate_for_load(w["rho"], w, types), rel=1e-9)
        assert wls[2]["arrival"] == "gamma" and wls[2]["cv_a"] == 2.0

    def test_rate_uses_the_mean_not_the_p50(self):
        types = C.load_types(FIX / "types_from_run1.json")
        w = jload("workload.json")[0]
        by_p50 = w["rho"] / sum(p * types.tau(t, types.f_max_mhz, "p50") for t, p in zip(w["types"], w["probs"]))
        assert w["rate_per_s"] != pytest.approx(by_p50, rel=1e-6)

    def test_policyspecs_registry(self):
        reg = jload("policyspecs.json")
        assert list(reg) == ["default", "default_stock", "serial", "serial_stf", "serial_class", "serial_edgeserving",
                             "corun", "greenllm_lanes", "occupancy_f"]
        for key, spec in reg.items():
            C.validate_policyspec(spec)
            assert spec["name"] == key and C.PolicySpec.from_json(spec).to_json() == spec
        assert (reg["serial"]["k_short"], reg["serial"]["k_long"], reg["serial"]["f_mhz"]) == (1, 1, 1020)
        assert reg["default"]["k_short"] is None and reg["default"]["f_rule"] == "governor"
        assert reg["serial_edgeserving"]["params"] == {"C": 10}
        assert reg["greenllm_lanes"]["lane_cap"] == {"short": 1, "long": 1}
        assert reg["occupancy_f"]["params"] == {"f_base_mhz": 612, "alpha_p": 2}

    def test_wire_messages(self):
        msgs = jload("wire_messages.json")
        assert set(msgs) >= {"ready", "run", "run_no_deadline", "run_miss", "run_llama", "done_complete",
                             "done_truncated", "done_miss", "done_error", "done_llama_truncated", "ping", "pong",
                             "stop"}
        for m in msgs.values():
            C.validate_server_msg(m)
        C.validate_ready(msgs["ready"])

    def test_wire_done_messages_agree_with_truncation_outcome(self):
        msgs = jload("wire_messages.json")
        pairs = {"done_truncated": "run", "done_miss": "run_miss", "done_llama_truncated": "run_llama",
                 "done_complete": "run_no_deadline"}
        for key, run_key in pairs.items():
            done, run = msgs[key], msgs[run_key]
            assert (done["req_id"], done["type_id"]) == (run["req_id"], run["type_id"]), key
            fam, ex = C.parse_type_id(done["type_id"])
            ck = C.default_checkpoints(fam, ex)
            reached, outcome = C.truncation_outcome(ck, [t for _, t in done["ckpt"]], run["deadline_ns"])
            assert (reached, outcome) == (done["reached"], done["outcome"]), key
            assert done["completion"] == pytest.approx(C.completion(fam, ex, reached), abs=1e-3)
            assert done["t_recv_ns"] >= run["t_admit_ns"] or key == "done_complete"

    def test_root_helper_messages(self):
        msgs = jload("root_helper_messages.json")
        for m in msgs.values():
            C.validate_root_msg(m)
        assert msgs["hello"]["fake_root"] is None and msgs["hello_dry"]["fake_root"] is not None

    def test_fake_spec_validates_and_drives_a_fake_adapter(self):
        spec = jload("fake_spec.json")
        C.validate_fake_spec(spec)
        ad = FA.FakeAdapter.from_file(FIX / "fake_spec.json")
        assert ad.load([5, 23], {})["load_s"] == 0.05
        assert ad.checkpoints(23) == C.default_checkpoints("bert", 23)

    def test_models(self):
        sd, pw = jload("slowdown_model.json"), jload("power_model.json")
        C.validate_slowdown(sd)
        C.validate_power(pw)
        assert sd["name"] == "M2" and sd["fit_rows_sha256"] == C.sha256_file(FIX / "run1_csv" / "corun.csv")
        assert set(pw["p_idle_w"]) == {"612", "1020"}

    def test_simresult(self):
        C.validate_simresult(jload("simresult.json"))
        assert C.SimResult.from_json(jload("simresult.json")).to_json() == jload("simresult.json")

    def test_pilot_summary_has_one_gate_per_step(self):
        out = jload("pilot_summary.json")
        C.validate_pilot(out)
        assert sorted(g["step"] for g in out["gates"].values()) == list(range(10))
        assert out["dry"] is True

    def test_plans(self):
        C.validate_plan(jload("plan_run1.json"))
        p2 = jload("plan_run2.json")
        C.validate_plan(p2)
        assert p2["kind"] == "open" and p2["freeze"]["path"] == "freeze.json"

    def test_run1_csv_headers_and_rows(self):
        for kind in RUN1:
            rows = C.read_csv(FIX / "run1_csv" / (kind + ".csv"), kind)
            assert rows, kind

    def test_open_conditions_csv(self):
        rows = C.read_csv(FIX / "open_conditions.csv", "open_conditions")
        assert {r["f_mhz"] for r in rows} == {612, 1020, "governor"}
        for r in rows:
            total = r["complete_rate"] + r["truncation_rate"] + r["miss_rate"] + r["expired_rate"] + r["error_rate"]
            assert total == pytest.approx(1.0)

    def test_open_summary(self):
        C.validate_open_summary(jload("open_summary.json"))

    @pytest.mark.parametrize("name", SESSIONS)
    def test_sessions(self, name):
        session, mans, rows, power, clocks = load_session(name)
        C.validate_session(session)
        C.validate_telemetry_meta(jload(name + "/telemetry/meta.json"))
        t_end = power[-1]["t_mono_ns"]
        assert session["idle"]["pre"]["t0_ns"] >= power[0]["t_mono_ns"]
        assert session["idle"]["resident"]["t1_ns"] <= t_end
        for cid, man in mans.items():
            C.validate_manifest(man)
            assert man["condition_id"] == cid and man["t_mono_end_ns"] <= t_end
            assert man["n_requests"] == len(rows[cid])
            C.read_csv(FIX / name / cid / "freq_switch.csv", "freq_switch")
            for r in rows[cid]:
                assert r["condition_id"] == cid
            assert len({r["req_id"] for r in rows[cid]}) == len(rows[cid])
        assert [r["t_mono_ns"] for r in power] == sorted(r["t_mono_ns"] for r in power)
        for r in power:
            assert r["vdd_in_mw"] == pytest.approx(r["vdd_in_mv"] * r["vdd_in_ma"] / 1000, rel=1e-12)

    def test_closed_session_rows_follow_the_closed_loop_rules(self):
        _, mans, rows, _, _ = load_session("session_closed")
        for cid, rs in rows.items():
            for r in rs:
                assert r["mode"] == "closed" and r["t_deadline_ns"] is None and r["t_arrival_ns"] == r["t_admit_ns"]
                assert r["outcome"] == "complete" and r["policy_spec"] is None
            if mans[cid]["mode"] == "closed":
                assert sum(r["is_warmup"] for r in rs if r["type_id"] == mans[cid]["cell"][0]) == 2

    def test_open_session_has_every_outcome_and_a_governor_arm(self):
        session, mans, rows, _, _ = load_session("session_open")
        outcomes = {r["outcome"] for rs in rows.values() for r in rs}
        assert outcomes == set(C.OUTCOMES)
        assert {m["arm"] for m in mans.values()} == {"pinned", "governor"}
        for cid, rs in rows.items():
            for r in rs:
                assert r["mode"] == "open" and r["t_deadline_ns"] is not None
                if r["outcome"] == "expired_in_queue":
                    assert r["t_done_ns"] == r["t_deadline_ns"] and r["ckpt"] == []
        governor = [cid for cid, m in mans.items() if m["arm"] == "governor"]
        assert governor and all(not (FIX / "session_open" / c / "freq_switch.csv").read_text().count("\n") > 1
                                for c in governor)

    def test_open_rows_agree_with_truncation_outcome(self):
        _, _, rows, _, _ = load_session("session_open")
        types = C.load_types(FIX / "types_from_run1.json")
        n = 0
        for rs in rows.values():
            for r in rs:
                if r["outcome"] in ("complete", "truncated", "miss"):
                    ck = C.default_checkpoints(r["family"], r["exit"])
                    got = C.truncation_outcome(ck, [t for _, t in r["ckpt"]], r["t_deadline_ns"])
                    assert got == (r["reached"], r["outcome"])
                    n += 1
        assert n > 100 and types.alpha == 5.0


# ---------------------------------------------------------------- removing or adding one key is always caught
def _delete(obj, path, key):
    t = obj
    for p in path:
        t = t[p]
    del t[key]


def _dicts(obj, paths):
    for path in paths:
        t = obj
        for p in path:
            t = t[p]
        yield path, t


def _row(name, outcome):
    _, _, rows, _, _ = load_session(name)
    return next(r for rs in rows.values() for r in rs if r["outcome"] == outcome)


def _first_manifest(name, mode, split=None):
    _, mans, _, _, _ = load_session(name)
    return next(m for m in mans.values() if m["mode"] == mode and (split is None or m["split"] == split))


# fixture -> (loader, validator, dict paths whose key set is exact in the contract)
MUTATION_TARGETS = {
    "types_from_run1.json": (lambda: jload("types_from_run1.json"), C.validate_types,
                             [(), ("families", "bert"), ("provenance",), ("types", 0), ("types", 0, "tau_s", "1020"),
                              ("types", 1, "tau_s", "612")]),
    "workload.json": (lambda: jload("workload.json")[0], C.validate_workload, [()]),
    "workload_gamma": (lambda: jload("workload.json")[2], C.validate_workload, [()]),
    "fake_spec.json": (lambda: jload("fake_spec.json"), C.validate_fake_spec, [(), ("types", "5")]),
    "simresult.json": (lambda: jload("simresult.json"), C.validate_simresult,
                       [(), ("per_type", "bert:5"), ("reps", 0)]),
    "pilot_summary.json": (lambda: jload("pilot_summary.json"), C.validate_pilot, [(), ("gates", "inventory")]),
    "slowdown_model.json": (lambda: jload("slowdown_model.json"), C.validate_slowdown, [()]),
    "power_model.json": (lambda: jload("power_model.json"), C.validate_power, [()]),
    "open_summary.json": (lambda: jload("open_summary.json"), C.validate_open_summary,
                          [(), ("checks",), ("counts",), ("mape",), ("regret",), ("per_condition", "W_hi_r060_F"),
                           ("energy_diff", "W_hi_r060_default"), ("poa_detail", "W_hi_r060")]),
    "plan_run1.json": (lambda: jload("plan_run1.json"), C.validate_plan,
                       [(), ("session",), ("conditions", 0), ("conditions", 2)]),
    "plan_run2.json": (lambda: jload("plan_run2.json"), C.validate_plan,
                       [(), ("session",), ("freeze",), ("conditions", 0), ("conditions", 0, "workload"),
                        ("conditions", 0, "policy")]),
    "freeze.json": (lambda: jload("freeze.json"), C.validate_freeze,
                    [(), ("git",), ("inputs",), ("slowdown_model",), ("sim",), ("constraint",), ("conditions", 0),
                     ("conditions", 0, "workload"), ("conditions", 0, "choices", "F"), ("conditions", 0, "grid", 0),
                     ("conditions", 0, "game"), ("conditions", 0, "game", "equilibria", 0)]),
    "session_closed/session.json": (lambda: jload("session_closed/session.json"), C.validate_session,
                                    [(), ("git",), ("host",), ("board",), ("idle",), ("idle", "pre"),
                                     ("servers", "bert")]),
    "session_open/session.json": (lambda: jload("session_open/session.json"), C.validate_session,
                                  [(), ("idle", "resident"), ("servers", "vision")]),
    "telemetry/meta.json": (lambda: jload("session_closed/telemetry/meta.json"), C.validate_telemetry_meta,
                            [(), ("vdd_in",)]),
    "manifest_idle": (lambda: _first_manifest("session_closed", "idle"), C.validate_manifest, [()]),
    "manifest_solo": (lambda: _first_manifest("session_closed", "closed", "solo"), C.validate_manifest, [()]),
    "manifest_pair": (lambda: _first_manifest("session_closed", "closed", "train"), C.validate_manifest, [()]),
    "manifest_open": (lambda: _first_manifest("session_open", "open"), C.validate_manifest,
                      [(), ("policy_spec",), ("workload",)]),
    "row_closed": (lambda: _row("session_closed", "complete"), C.validate_request_row, [()]),
    "row_open_complete": (lambda: _row("session_open", "complete"), C.validate_request_row, [(), ("policy_spec",)]),
    "row_truncated": (lambda: _row("session_open", "truncated"), C.validate_request_row, [()]),
    "row_miss": (lambda: _row("session_open", "miss"), C.validate_request_row, [()]),
    "row_expired": (lambda: _row("session_open", "expired_in_queue"), C.validate_request_row, [()]),
    "row_error": (lambda: _row("session_open", "error"), C.validate_request_row, [()]),
}
for _k in ("ready", "run", "run_miss", "done_complete", "done_truncated", "done_miss", "done_error",
           "done_llama_truncated", "pong"):
    MUTATION_TARGETS["wire:" + _k] = ((lambda k=_k: jload("wire_messages.json")[k]), C.validate_server_msg, [()])
MUTATION_TARGETS["wire:ready.warmup"] = (lambda: jload("wire_messages.json")["ready"], C.validate_server_msg,
                                         [("warmup",), ("versions",), ("mem",)])
for _k in ("hello", "hello_dry", "write_req", "read_req", "run_req"):
    MUTATION_TARGETS["root:" + _k] = ((lambda k=_k: jload("root_helper_messages.json")[k]), C.validate_root_msg, [()])


def _undetected(name, how):
    load, validate, paths = MUTATION_TARGETS[name]
    base = load()
    validate(copy.deepcopy(base))                 # the unmutated fixture passes, so a failure below is the mutation
    missed = []
    for path, d in _dicts(base, paths):
        for key in list(d):
            obj = copy.deepcopy(base)
            if how == "delete":
                _delete(obj, path, key)
            else:
                dict(_dicts(obj, [path]))[path]["zz_extra_key"] = 1
            try:
                validate(obj)
            except C.ContractError:
                pass
            else:
                missed.append("/".join(map(str, path + (key,))) if how == "delete" else "/".join(map(str, path)))
            if how == "extra":
                break                                   # one extra-key probe per dict is enough
    return missed


@pytest.mark.parametrize("name", sorted(MUTATION_TARGETS))
def test_removing_any_key_makes_the_validator_fail(name):
    assert _undetected(name, "delete") == []


@pytest.mark.parametrize("name", sorted(MUTATION_TARGETS))
def test_an_unknown_extra_key_makes_the_validator_fail(name):
    assert _undetected(name, "extra") == []


def test_root_helper_replies_need_their_ok_key():
    msgs = jload("root_helper_messages.json")
    for key in ("write_resp", "read_resp", "run_resp", "stop_resp", "error_resp"):
        broken = {k: v for k, v in msgs[key].items() if k != "ok"}
        with pytest.raises(C.ContractError):
            C.validate_root_msg(broken)
    with pytest.raises(C.ContractError):
        C.validate_root_msg({"ok": False})              # a failure must say why


CSV_FILES = ([("run1_csv/%s.csv" % k, k) for k in RUN1] + [("open_conditions.csv", "open_conditions")]
             + [("session_closed/telemetry/power.csv", "power"), ("session_closed/telemetry/clocks.csv", "clocks"),
                ("session_open/telemetry/clocks.csv", "clocks")])


@pytest.mark.parametrize("rel,kind", CSV_FILES)
def test_csv_header_changes_are_rejected(rel, kind):
    with open(FIX / rel, encoding="utf-8", newline="") as fh:
        header = next(csv.reader(fh))
    C.validate_csv_header(kind, header)
    mandatory = len(C.CLOCKS_CSV_FIXED_COLS) if kind == "clocks" else len(header)   # temp_<zone>_mc columns may vary
    for i in range(mandatory):
        with pytest.raises(C.ContractError):
            C.validate_csv_header(kind, header[:i] + header[i + 1:])             # a column removed
    with pytest.raises(C.ContractError):
        C.validate_csv_header(kind, header + ["extra"])
    with pytest.raises(C.ContractError):
        C.validate_csv_header(kind, [header[0] + "_x"] + header[1:])


def test_request_row_validator_rejects_wrong_outcome_shapes():
    row = _row("session_open", "complete")
    for change in ({"reached": 3}, {"completion": 0.5}, {"err": "boom"}, {"target": 99}, {"outcome": "dropped"}):
        with pytest.raises(C.ContractError):
            C.validate_request_row(dict(row, **change))
    expired = _row("session_open", "expired_in_queue")
    with pytest.raises(C.ContractError):
        C.validate_request_row(dict(expired, t_done_ns=expired["t_done_ns"] + 1))
    with pytest.raises(C.ContractError):
        C.validate_request_row(dict(_row("session_closed", "complete"), t_deadline_ns=5))


# ---------------------------------------------------------------- hashes, pre-registration, cross-file links
class TestHashesAndLinks:
    def test_freeze_verifies_and_sidecar_has_the_sha256sum_format(self):
        freeze = C.verify_freeze_file(FIX / "freeze.json")
        raw = (FIX / "freeze.json").read_bytes()
        side = (FIX / "freeze.json.sha256").read_bytes()
        assert side == ("%s  freeze.json\n" % hashlib.sha256(raw).hexdigest()).encode("ascii")
        assert freeze["content_sha256"] == C.content_sha256(freeze)

    def test_freeze_detects_one_changed_byte_and_a_wrong_sidecar(self, tmp_path):
        for name in ("freeze.json", "freeze.json.sha256"):
            shutil.copy(FIX / name, tmp_path / name)
        raw = bytearray((tmp_path / "freeze.json").read_bytes())
        i = raw.index(b"0.6")
        raw[i + 2] = ord("7")                                               # rho 0.6 -> 0.7 inside the file
        (tmp_path / "freeze.json").write_bytes(bytes(raw))
        with pytest.raises(C.FreezeError):
            C.verify_freeze_file(tmp_path / "freeze.json")
        shutil.copy(FIX / "freeze.json", tmp_path / "freeze.json")
        C.verify_freeze_file(tmp_path / "freeze.json")
        (tmp_path / "freeze.json.sha256").write_text("0" * 64 + "  freeze.json\n")
        with pytest.raises(C.FreezeError):
            C.verify_freeze_file(tmp_path / "freeze.json")

    def test_freeze_inputs_are_the_hashes_of_the_fixture_files(self):
        inp = jload("freeze.json")["inputs"]
        want = {"types_sha256": "types_from_run1.json", "corun_sha256": "run1_csv/corun.csv",
                "solo_sha256": "run1_csv/solo.csv", "run1_plan_sha256": "plan_run1.json",
                "slowdown_sha256": "slowdown_model.json", "power_sha256": "power_model.json"}
        for key, rel in want.items():
            assert inp[key] == C.sha256_file(FIX / rel), key

    def test_types_provenance_hashes_match_run1_csv(self):
        prov = jload("types_from_run1.json")["provenance"]
        assert sorted(prov["sha256"]) == sorted(prov["sources"])
        for name, digest in prov["sha256"].items():
            assert digest == C.sha256_file(FIX / "run1_csv" / name), name

    def test_plan_hashes_and_freeze_link(self):
        p1, p2, fz = jload("plan_run1.json"), jload("plan_run2.json"), jload("freeze.json")
        assert p1["sha256"] == C.plan_sha256(p1) and p2["sha256"] == C.plan_sha256(p2)
        assert p2["freeze"] == {"path": "freeze.json", "content_sha256": fz["content_sha256"]}
        assert p1["freeze"] is None and p2["estimated_wall_s"] == fz["runtime_estimate_s"]
        changed = dict(p2, estimated_wall_s=p2["estimated_wall_s"] + 1)
        with pytest.raises(C.ContractError):
            C.validate_plan(changed)                                          # the plan hash covers its content

    def test_sessions_point_at_their_plan_and_freeze(self):
        s1, s2 = jload("session_closed/session.json"), jload("session_open/session.json")
        fz = jload("freeze.json")
        assert s1["plan_sha256"] == jload("plan_run1.json")["sha256"] and s1["freeze_content_sha256"] is None
        assert s2["plan_sha256"] == jload("plan_run2.json")["sha256"]
        assert s2["freeze_content_sha256"] == fz["content_sha256"]
        assert s2["t_wall_start_ns"] > fz["created_wall_ns"] > s1["t_wall_end_ns"]       # Run 2 starts after the freeze

    def test_plan_conditions_are_the_session_conditions(self):
        for plan, name in (("plan_run1.json", "session_closed"), ("plan_run2.json", "session_open")):
            p = jload(plan)
            session, mans, _, _, _ = load_session(name)
            assert [c["condition_id"] for c in p["conditions"]] == session["conditions"]
            for c in p["conditions"]:
                m = mans[c["condition_id"]]
                assert (m["mode"], m["arm"], m["gpu_mhz"], m["settle_s"]) == (c["mode"], c["arm"], c["gpu_mhz"],
                                                                              p["session"]["settle_s"])
                for key in ("policy_key", "freeze_condition_id", "workload", "alpha", "seed", "repeat", "cell",
                            "split"):
                    if key in c:
                        want = c[key]
                        assert m[key] == (list(want) if key == "cell" else want), (c["condition_id"], key)
                if "policy" in c:
                    assert m["policy_spec"] == c["policy"]

    def test_freeze_workload_is_the_first_fixture_workload_and_predictions_cover_all_keys(self):
        fz = jload("freeze.json")
        cond = fz["conditions"][0]
        assert cond["workload"] == jload("workload.json")[0]
        keys = set(cond["choices"]) | {g["name"] for g in cond["grid"]}
        assert set(cond["predictions"]) == keys
        assert cond["game"]["worst"]["policy_key"] in keys and cond["game"]["optimum"]["policy_key"] in keys

    def test_slowdown_selection_follows_the_rule(self):
        mape = jload("freeze.json")["slowdown_model"]["heldout_mape"]
        order = ["M1", "M2", "M3", "M4lin", "M4"]
        first = next(m for m in order if mape[m] <= 10.0 and mape[m] < mape["M0"])
        assert jload("freeze.json")["slowdown_model"]["selected"] == first == jload("slowdown_model.json")["name"]


# ---------------------------------------------------------------- the "=" fixtures against the sessions they come from
def _rel(a, b):
    return a == pytest.approx(b, rel=1e-9, abs=1e-12)


@pytest.fixture(scope="module")
def closed_world():
    return load_session("session_closed")


@pytest.fixture(scope="module")
def open_world():
    return load_session("session_open")


class TestRun1CsvMatchesSessionClosed:
    """run1_csv is the expected F1 output for session_closed (K11). The CI columns need the bootstrap RNG, so they are
    only bounded here; the other columns are recomputed from the raw rows."""

    @pytest.fixture
    def world(self, closed_world):
        return closed_world

    def test_solo_rows(self, world):
        session, mans, rows, power, clocks = world
        got = C.read_csv(FIX / "run1_csv" / "solo.csv", "solo")
        assert [(r["type_id"], r["f_mhz"], r["repeat"]) for r in got] == [
            ("bert:5", 612, 0), ("bert:5", 1020, 0), ("vision:24", 612, 0), ("vision:24", 1020, 0)]
        for g in got:
            man = next(m for m in mans.values() if m["split"] == "solo" and m["cell"] == [g["type_id"]]
                       and m["gpu_mhz"] == g["f_mhz"])
            nw = sorted((r for r in rows[man["condition_id"]] if not r["is_warmup"]), key=lambda r: r["t_admit_ns"])
            tau = [(r["t_done_ns"] - r["t_admit_ns"]) / 1e9 for r in nw]
            lo, hi = nw[0]["t_admit_ns"], max(r["t_done_ns"] for r in nw)
            assert g["n"] == len(nw) and _rel(g["tau_p50"], S.p_nearest_rank(tau, 0.5))
            assert _rel(g["tau_mean"], sum(tau) / len(tau)) and _rel(g["tau_p95"], S.p_nearest_rank(tau, 0.95))
            assert _rel(g["window_s"], (hi - lo) / 1e9) and _rel(g["ess"], S.ess(tau))
            assert _rel(g["p_board_w"], S.energy_j(power, lo, hi) / g["window_s"])
            load = [c["gpu_load_permille"] / 1000 for c in clocks if lo <= c["t_mono_ns"] <= hi]
            assert _rel(g["u"], sum(load) / len(load))
            assert g["tau_p50_lo"] <= g["tau_p50"] <= g["tau_p50_hi"]

    def test_solo_samples_are_every_measured_tau_in_time_order(self, world):
        _, mans, rows, _, _ = world
        got = C.read_csv(FIX / "run1_csv" / "solo_samples.csv", "solo_samples")
        want = []
        for key in sorted({(m["cell"][0], m["gpu_mhz"], m["repeat"]) for m in mans.values() if m["split"] == "solo"}):
            man = next(m for m in mans.values()
                       if m["split"] == "solo" and (m["cell"][0], m["gpu_mhz"], m["repeat"]) == key)
            nw = sorted((r for r in rows[man["condition_id"]] if not r["is_warmup"]), key=lambda r: r["t_admit_ns"])
            want += [(key, (r["t_done_ns"] - r["t_admit_ns"]) / 1e9) for r in nw]
        assert [((r["type_id"], r["f_mhz"], r["repeat"]), r["tau_s"]) for r in got] == [
            (k, pytest.approx(v, rel=1e-9)) for k, v in want]

    def test_corun_rows_and_samples(self, world):
        session, mans, rows, power, clocks = world
        solo = C.read_csv(FIX / "run1_csv" / "solo.csv", "solo")
        solo_p50 = {(r["type_id"], r["f_mhz"]): r["tau_p50"] for r in solo}
        u = {(r["type_id"], r["f_mhz"]): r["u"] for r in solo}
        got = C.read_csv(FIX / "run1_csv" / "corun.csv", "corun")
        samples = C.read_csv(FIX / "run1_csv" / "corun_samples.csv", "corun_samples")
        assert len(got) == 4
        for g in got:
            man = next(m for m in mans.values() if m["split"] == "train" and m["gpu_mhz"] == g["f_mhz"])
            assert g["cell_id"] == man["condition_id"].rsplit("_f", 1)[0] == "c001"
            assert g["A"] == "|".join(man["cell"]) and g["split"] == man["split"]
            assert g["family"] == C.parse_type_id(g["type_id"])[0]
            per = {t: sorted((r for r in rows[man["condition_id"]] if r["type_id"] == t and not r["is_warmup"]),
                             key=lambda r: r["t_admit_ns"]) for t in man["cell"]}
            lo = max(v[0]["t_admit_ns"] for v in per.values())
            hi = min(max(r["t_done_ns"] for r in v) for v in per.values())
            inside = [r for r in per[g["type_id"]] if r["t_admit_ns"] >= lo and r["t_done_ns"] <= hi]
            s = [((r["t_done_ns"] - r["t_admit_ns"]) / 1e9) / solo_p50[(g["type_id"], g["f_mhz"])] for r in inside]
            assert g["n"] == len(per[g["type_id"]]) and g["n_window"] == len(inside) and 0 < len(inside) <= g["n"]
            assert _rel(g["s_p50"], S.p_nearest_rank(s, 0.5)) and _rel(g["s_mean"], sum(s) / len(s))
            assert _rel(g["s_p95"], S.p_nearest_rank(s, 0.95)) and _rel(g["ess"], S.ess(s))
            assert _rel(g["tau_solo_p50"], solo_p50[(g["type_id"], g["f_mhz"])])
            assert _rel(g["U_A"], sum(u[(t, g["f_mhz"])] for t in man["cell"]))
            assert _rel(g["window_s"], (hi - lo) / 1e9) and _rel(g["energy_j"], S.energy_j(power, lo, hi))
            assert _rel(g["p_board_w"], g["energy_j"] / g["window_s"])
            assert g["s_p50_lo"] <= g["s_p50"] <= g["s_p50_hi"]
            mine = [r["s"] for r in samples if (r["f_mhz"], r["type_id"]) == (g["f_mhz"], g["type_id"])]
            assert mine == [pytest.approx(x, rel=1e-9) for x in s]
        assert any(g["n_window"] < g["n"] for g in got)          # a half-outside sample is dropped somewhere
        assert any(g["n_window"] == g["n"] for g in got)

    def test_idle_ckpt_warmup_mem(self, world):
        session, mans, rows, power, clocks = world
        idle = C.read_csv(FIX / "run1_csv" / "idle.csv", "idle")
        pre = session["idle"]["pre"]
        want = [(1020, "pre", S.energy_j(power, pre["t0_ns"], pre["t1_ns"]) / ((pre["t1_ns"] - pre["t0_ns"]) / 1e9))]
        for f in (612, 1020):
            m = next(m for m in mans.values() if m["mode"] == "idle" and m["gpu_mhz"] == f)
            t0 = m["t_mono_start_ns"] + round(m["settle_s"] * 1e9)
            want.append((f, "resident", S.energy_j(power, t0, m["t_mono_end_ns"]) / ((m["t_mono_end_ns"] - t0) / 1e9)))
        assert [(r["f_mhz"], r["kind"]) for r in idle] == [(f, k) for f, k, _ in want]
        for r, (_, _, w) in zip(idle, want):
            assert _rel(r["p_idle_w"], w)
        p_idle_pre = idle[0]["p_idle_w"]
        warm = C.read_csv(FIX / "run1_csv" / "warmup.csv", "warmup")
        assert [r["family"] for r in warm] == ["bert", "vision"] and {r["cache"] for r in warm} == {"cold"}
        for r in warm:
            sv = session["servers"][r["family"]]
            span = (sv["t_ready_ns"] - sv["t_spawn_ns"]) / 1e9
            e = S.energy_j(power, sv["t_spawn_ns"], sv["t_ready_ns"]) - p_idle_pre * span
            assert r["session_id"] == session["session_id"] and _rel(r["warm_energy_j"], e) and e > 0
            assert _rel(r["t_load_s"], sv["ready"]["warmup"]["load_s"])
            assert _rel(r["first_infer_s_max"], max(sv["ready"]["warmup"]["first_infer_s"].values()))
        mem = C.read_csv(FIX / "run1_csv" / "mem.csv", "mem")
        assert [(r["family"], r["exit"]) for r in mem] == [("bert", 5), ("vision", 24)]
        ck = C.read_csv(FIX / "run1_csv" / "ckpt.csv", "ckpt")
        assert sorted({(r["type_id"], r["f_mhz"]) for r in ck}) == [
            ("bert:5", 612), ("bert:5", 1020), ("vision:24", 612), ("vision:24", 1020)]
        for r in ck:
            last = C.default_checkpoints(*C.parse_type_id(r["type_id"]))[-1]
            if r["ckpt"] == last:
                assert r["frac_p50"] + r["head_frac_p50"] == pytest.approx(1.0, abs=0.05)

    def test_types_from_run1_follow_run1_csv(self):
        types = jload("types_from_run1.json")
        solo = C.read_csv(FIX / "run1_csv" / "solo.csv", "solo")
        samples = C.read_csv(FIX / "run1_csv" / "solo_samples.csv", "solo_samples")
        assert types["f_levels_mhz"] == [612, 1020] and types["corun_levels_mhz"] == [612, 1020]
        assert types["heldout_level_mhz"] is None and types["alpha"] == 5.0 and "synthetic" not in types
        means = {}
        for t in types["types"]:
            for f, st in t["tau_s"].items():
                xs = [r["tau_s"] for r in samples if r["type_id"] == t["type_id"] and r["f_mhz"] == int(f)]
                assert st["samples"] == xs and st["p50"] == S.p_nearest_rank(xs, 0.5)
                assert st["p95"] == S.p_nearest_rank(xs, 0.95) and _rel(st["mean"], sum(xs) / len(xs))
                row = next(r for r in solo if r["type_id"] == t["type_id"] and r["f_mhz"] == int(f))
                assert _rel(st["p50"], row["tau_p50"]) and _rel(t["u"][f], row["u"])
            means[t["type_id"]] = t["tau_s"]["1020"]["mean"]
        median = sorted(means.values())[0] / 2 + sorted(means.values())[1] / 2
        for t in types["types"]:
            assert t["klass"] == ("short" if means[t["type_id"]] <= median else "long")
        assert [t["klass"] for t in types["types"]] == ["short", "long"]

    def test_run1_csv_energy_and_window_are_inside_the_telemetry(self, world):
        _, _, _, power, _ = world
        for r in C.read_csv(FIX / "run1_csv" / "corun.csv", "corun"):
            assert r["energy_j"] > 0 and r["window_s"] > 0 and power[0]["t_mono_ns"] < power[-1]["t_mono_ns"]


KEYS3 = ("default", "default_pinned", "serial", "serial_stf")      # the non-F keys with 3 repeats (R32)


class TestOpenOutputsMatchSessionOpen:
    @pytest.fixture
    def world(self, open_world):
        return open_world

    def test_open_conditions_rows(self, world):
        session, mans, rows, power, clocks = world
        got = C.read_csv(FIX / "open_conditions.csv", "open_conditions")
        assert len(got) == 2 * len(mans)
        for g in got:
            man, rs = mans[g["condition_id"]], rows[g["condition_id"]]
            nw = [r for r in rs if not r["is_warmup"]]
            mine = [r for r in nw if r["type_id"] == g["type_id"]]
            resp = [(r["t_done_ns"] - r["t_arrival_ns"]) / 1e9 for r in mine]
            t0, t1, n = S.request_window(rs)
            assert g["n"] == len(mine) and g["policy_key"] == man["policy_key"] and g["repeat"] == man["repeat"]
            assert g["f_mhz"] == (man["gpu_mhz"] if man["arm"] == "pinned" else "governor")
            for col, outcome in (("complete_rate", "complete"), ("truncation_rate", "truncated"), ("miss_rate", "miss"),
                                 ("expired_rate", "expired_in_queue"), ("error_rate", "error")):
                assert _rel(g[col], sum(r["outcome"] == outcome for r in mine) / len(mine)), col
            assert _rel(g["mean_completion"], sum(r["completion"] for r in mine) / len(mine))
            assert _rel(g["p95_R_s"], S.p_nearest_rank(resp, 0.95))
            assert _rel(g["energy_per_req_j"], S.energy_per_request(rs, power))
            assert _rel(g["energy_sess_per_req_j"], S.energy_j(power, man["t_mono_start_ns"], man["t_mono_end_ns"]) / n)
            assert _rel(g["throughput_rps"], n / ((t1 - t0) / 1e9))
            mem = min(c["mem_available_kb"] for c in clocks
                      if man["t_mono_start_ns"] <= c["t_mono_ns"] <= man["t_mono_end_ns"])
            assert _rel(g["min_mem_available_mb"], mem / 1024)
            assert g["energy_sess_per_req_j"] > g["energy_per_req_j"]           # settle and tail time cost energy too

    def test_open_summary_numbers(self, world):
        _, mans, _, _, _ = world
        summ = jload("open_summary.json")
        fz = jload("freeze.json")
        csv_rows = C.read_csv(FIX / "open_conditions.csv", "open_conditions")
        assert summ["freeze_content_sha256"] == fz["content_sha256"]
        assert summ["checks"] == {"hash_ok": True, "wall_order_ok": True,
                                  "t_wall_start_ns": jload("session_open/session.json")["t_wall_start_ns"],
                                  "freeze_created_wall_ns": fz["created_wall_ns"]}
        energy = {}
        for key in fz["conditions"][0]["predictions"]:
            es = sorted({(r["repeat"], r["energy_per_req_j"]) for r in csv_rows if r["policy_key"] == key})
            energy[key] = [e for _, e in es]
            pc = summ["per_condition"]["W_hi_r060_" + key]
            assert _rel(pc["energy"], sum(energy[key]) / len(energy[key]))
            assert (pc["energy_ci"] is None) == (len(energy[key]) == 1)
            if len(energy[key]) > 1:
                assert [pytest.approx(x, rel=1e-9) for x in S.mean_ci(energy[key])[1:]] == pc["energy_ci"]
            pred = fz["conditions"][0]["predictions"][key]["energy_per_req_j"]
            assert _rel(pc["pred"], pred) and _rel(pc["ape"], abs(pred - pc["energy"]) / pc["energy"] * 100)
            def pooled(t, key=key):
                rs = [r for r in csv_rows if r["policy_key"] == key and r["type_id"] == t]
                return sum(r["complete_rate"] * r["n"] for r in rs) / sum(r["n"] for r in rs)

            complete = min(pooled(t) for t in ("bert:5", "vision:24"))
            assert _rel(pc["complete_rate_min"], complete) and pc["feasible"] == (complete >= C.COMPLETE_TARGET)
        apes = [v["ape"] for v in summ["per_condition"].values()]
        assert _rel(summ["mape"]["energy"], sum(apes) / len(apes)) and _rel(summ["mape"]["max"], max(apes))
        assert _rel(summ["mape"]["median"], sorted(apes)[len(apes) // 2 - 1] / 2 + sorted(apes)[len(apes) // 2] / 2)
        assert set(summ["energy_diff"]) == {"W_hi_r060_" + k for k in KEYS3}   # keys with >= 2 repeats (D11)
        for key in KEYS3:
            d, lo, hi = S.paired_diff_ci(energy["F"], energy[key])               # paired per repeat since C2b
            got = summ["energy_diff"]["W_hi_r060_" + key]
            assert _rel(got["diff_j"], d) and got["n_pairs"] == 3 and got["method"] == "paired_t"
            assert got["ci"] == [pytest.approx(lo, rel=1e-9), pytest.approx(hi, rel=1e-9)]
        for key, pc in ((k, summ["per_condition"]["W_hi_r060_" + k]) for k in energy):
            assert pc["measured_key"] == key and pc["n_repeats"] == len(energy[key])
            assert pc["status"] in C.OPEN_STATUSES and 0.0 <= pc["wilson_lo_min"] <= pc["complete_rate_min"]
            assert pc["feasible"] == (pc["status"] != "infeasible")
        assert summ["counts"]["runs_valid"] == len(mans) and summ["counts"]["runs_invalid"] == 0
        assert summ["poa_detail"]["W_hi_r060"]["worst_key"] == fz["conditions"][0]["game"]["worst"]["policy_key"]
        assert all(0.0 <= r["late_rate"] <= 1.0 for r in csv_rows)
        feas = {k: summ["per_condition"]["W_hi_r060_" + k] for k in ("F", "g0", "g1")}
        ok = {k: v["energy"] for k, v in feas.items() if v["feasible"]}
        best = min(ok, key=ok.get)
        assert summ["regret"]["best_key"] == best
        assert _rel(summ["regret"]["F"], (ok["F"] - ok[best]) / ok[best])
        assert _rel(summ["regret"]["grid_spread"], (max(ok.values()) - min(ok.values())) / min(ok.values()))
        assert _rel(summ["poa_measured"], summ["per_condition"]["W_hi_r060_g1"]["energy"]
                    / summ["per_condition"]["W_hi_r060_F"]["energy"])
        every_ok = {k: v["energy"] for k, v in summ["per_condition"].items() if v["feasible"]}
        assert "W_hi_r060_" + summ["oracle"]["W_hi_r060"]["policy_key"] == min(every_ok, key=every_ok.get)

    def test_open_summary_v3_fields(self, world):
        summ = jload("open_summary.json")
        csv_rows = C.read_csv(FIX / "open_conditions.csv", "open_conditions")
        assert summ["schema"] == "exp2.open_summary/3"
        energy = {}
        for r in csv_rows:
            energy.setdefault(r["policy_key"], {})[r["repeat"]] = r["energy_per_req_j"]
        # the R32 fixture freeze measures every reference key itself (README D12)
        for ref, key in (("default_pinned", "default_pinned"), ("serial_stf", "serial_stf"), ("default", "default"),
                         ("serial", "serial")):
            fam = summ["primary"] if ref in C.PRIMARY_REFS else summ["secondary"]
            reps = sorted(set(energy["F"]) & set(energy[key]))
            m, lo, hi, p = S.paired_t_test([math.log(energy["F"][r]) for r in reps],
                                           [math.log(energy[key][r]) for r in reps])
            got = fam[ref]
            assert got["n_pairs"] == len(reps) == 3
            assert _rel(got["log_ratio_mean"], m) and _rel(got["ci_lo"], lo) and _rel(got["ci_hi"], hi)
            assert _rel(got["p"], p) and got["p_holm"] >= got["p"]
        assert [x["p_holm"] for x in summ["primary"].values()] == pytest.approx(
            S.holm([x["p"] for x in summ["primary"].values()]))
        for pc in summ["per_condition"].values():
            assert 0.0 <= pc["attain_boot_lo_min"] <= 1.0 and pc["energy_per_work_j"] > 0
            assert (pc["status"] == "feasible") == (pc["attain_boot_lo_min"] >= C.COMPLETE_TARGET)
        sa = summ["sim_accuracy"]["W_hi_r060"]
        assert set(sa) == {"abs_err_pp_max", "abs_err_pp_median", "class_agreement", "spearman_energy",
                           "energy_ape_median"}
        assert sa["abs_err_pp_median"] <= sa["abs_err_pp_max"]

    def test_summary_covers_feasible_and_infeasible_keys(self):
        flags = {v["feasible"] for v in jload("open_summary.json")["per_condition"].values()}
        assert flags == {True, False}


# ---------------------------------------------------------------- regeneration: the mocks write the committed bytes
def _same_tree(a, b):
    assert tree_files(a) == tree_files(b)
    for rel in tree_files(a):
        assert normalized(Path(a) / rel) == normalized(Path(b) / rel), rel


class TestRegeneration:
    def test_closed_plan_is_plan_run1(self):
        assert M.closed_plan() == jload("plan_run1.json")

    def test_synthetic_closed_session_reproduces_the_fixture(self, tmp_path):
        out = M.synthetic_closed_session(tmp_path, dirname="session_closed")
        _same_tree(out, FIX / "session_closed")

    def test_synthetic_open_session_reproduces_the_fixture(self, tmp_path):
        types = C.load_types(FIX / "types_from_run1.json")
        out = M.synthetic_open_session(tmp_path, jload("plan_run2.json"), jload("freeze.json"), types,
                                       dirname="session_open")
        _same_tree(out, FIX / "session_open")

    def test_generation_is_deterministic(self, tmp_path):
        a = M.synthetic_closed_session(tmp_path / "a", dirname="s")
        b = M.synthetic_closed_session(tmp_path / "b", dirname="s")
        _same_tree(a, b)

    def test_session_stress_and_errors_change_the_rows(self, tmp_path):
        types = C.load_types(FIX / "types_from_run1.json")
        out = M.synthetic_open_session(tmp_path, jload("plan_run2.json"), jload("freeze.json"), types,
                                       dirname="x", stress={}, errors={})
        rows = [r for p in out.glob("*/requests.jsonl") for r in map(json.loads, p.read_text().splitlines())]
        assert {r["outcome"] for r in rows} == {"complete"}
        assert normalized(out / "W_hi_r060_F_r0" / "requests.jsonl") == normalized(
            FIX / "session_open" / "W_hi_r060_F_r0" / "requests.jsonl")        # F had no stress to begin with

    def test_a_bad_ckpt_frac_length_is_refused(self):
        with pytest.raises(ValueError):
            M.closed_rows("c", "bert:5", "short", t_begin_ns=0, t_stop_ns=10, gpu_mhz=1020, base_tau_s=0.03,
                          ckpt_frac=(0.9,), n_warmup=0, n_in_service=0)


class TestMockSessionBuilders:
    def test_write_session_rows_pass_validation_and_power_identity_holds(self, tmp_path):
        out = M.synthetic_closed_session(tmp_path)
        assert out.name == M.SESSION_CLOSED_ID
        power = C.read_csv(out / "telemetry" / "power.csv", "power")
        assert power[1]["t_mono_ns"] - power[0]["t_mono_ns"] == 100_000_000      # 10 Hz
        assert all(r["vdd_in_mw"] == r["vdd_in_mv"] * r["vdd_in_ma"] / 1000 for r in power)

    def test_make_ready_passes_validate_ready_and_scales_with_exits(self):
        msg = M.make_ready("vision", [6, 24], first_infer_s={6: 1.0, 24: 2.0})
        C.validate_ready(msg)
        assert msg["warmup"]["first_infer_s"] == {"6": 1.0, "24": 2.0}

    def test_open_schedule_follows_the_probabilities_and_is_deterministic(self):
        wl = jload("workload.json")[0]
        a = M.open_schedule(wl, 100, 0)
        assert a == M.open_schedule(wl, 100, 0) and a != M.open_schedule(wl, 100, 1)
        assert sum(it["type_id"] == "vision:24" for it in a) == 10
        gaps = [(y["t_arrival_ns"] - x["t_arrival_ns"]) / 1e9 for x, y in zip(a, a[1:])]
        assert sum(gaps) / len(gaps) == pytest.approx(1 / wl["rate_per_s"], rel=0.05)

    def test_open_schedule_refuses_probabilities_that_are_not_tenths(self):
        wl = dict(jload("workload.json")[0], probs=[0.77, 0.23])
        with pytest.raises(ValueError):
            M.open_schedule(wl, 5, 0)

    def test_open_rows_expire_requests_that_wait_past_their_deadline(self):
        types = C.load_types(FIX / "types_from_run1.json")
        spec = C.PolicySpec("s", 1, 1, "fcfs", None, "static", 1020, {})
        wl = jload("workload.json")[0]
        rows = M.open_rows("c", spec, wl, types, t0_ns=0, n_requests=16, n_warmup=0, seed=11, alpha=5.0,
                           stress={4: 6.0})
        expired = [r for r in rows if r["outcome"] == "expired_in_queue"]
        assert expired and all(r["t_done_ns"] == r["t_deadline_ns"] and r["ckpt"] == [] for r in expired)
        for r in rows:
            C.validate_request_row(r)
        loose = M.open_rows("c", C.PolicySpec("s", None, None, "fcfs", None, "static", 1020, {}), wl, types, t0_ns=0,
                            n_requests=16, n_warmup=0, seed=11, alpha=5.0)
        assert not [r for r in loose if r["outcome"] == "expired_in_queue"]       # k = inf admits at once
        assert all(r["n_in_service_at_admit"] is not None for r in loose)

    def test_open_rows_without_alpha_have_no_deadline_and_always_complete(self):
        types = C.load_types(FIX / "types_from_run1.json")
        spec = C.PolicySpec("s", 1, 1, "fcfs", None, "static", 1020, {})
        rows = M.open_rows("c", spec, jload("workload.json")[0], types, t0_ns=0, n_requests=10, n_warmup=2, seed=0,
                           alpha=None, stress={3: 50.0})
        assert {r["outcome"] for r in rows} == {"complete"} and all(r["t_deadline_ns"] is None for r in rows)
        assert sum(r["is_warmup"] for r in rows) == 2

    def test_timeline_later_segments_override_earlier_ones(self):
        tl = M.Timeline(1, 2, 3)
        tl.add(0, 10, mw=100)
        tl.add(5, 8, mw=lambda t: t * 10, load=7)
        assert (tl.at_mw(2), tl.at_mw(6), tl.at_mw(9), tl.at_mw(50)) == (100, 60, 100, 1)
        assert (tl.at_load(6), tl.at_load(2), tl.at_gpu(6)) == (7, 2, 3)

    def test_golden_loader_reads_the_s0a_files(self):
        assert len(M.load_golden("admission_cases")) >= 20
        trace = M.load_golden("admission_traces")[0]
        assert M.golden_types(trace).type_ids()
        assert len(M.load_golden("truncation_cases")) >= 12
