"""S0a gate: exp2.wire (K5): JSON round trip, atomic port file, recv timeout."""
from __future__ import annotations

import json
import os
import threading
import time

import pytest

from exp2 import contracts as C
from exp2 import wire


@pytest.fixture
def pair():
    """A connected (client, server) pair over loopback."""
    listener = wire.listen()
    port = wire.listener_port(listener)
    box = {}
    t = threading.Thread(target=lambda: box.setdefault("srv", listener.accept()), daemon=True)
    t.start()
    client = wire.connect(port)
    t.join(5)
    server = box["srv"]
    yield client, server, port
    for c in (client, server):
        c.close()
    listener.close()


class TestListenConnect:
    def test_listener_binds_loopback_ephemeral_port(self):
        listener = wire.listen()
        try:
            host, port = listener.address
            assert host == "127.0.0.1" and port > 0 and wire.listener_port(listener) == port
        finally:
            listener.close()

    def test_two_listeners_get_different_ports(self):
        a, b = wire.listen(), wire.listen()
        try:
            assert wire.listener_port(a) != wire.listener_port(b)
        finally:
            a.close()
            b.close()


class TestRoundTrip:
    def test_json_round_trip_both_directions(self, pair):
        client, server, _ = pair
        msg = {"op": "run", "req_id": "r000001", "type_id": "bert:23", "exit": 23, "input_idx": 1,
               "deadline_ns": None, "t_admit_ns": 1712150000000, "nested": {"a": [1, 2.5, None, True]}}
        wire.send(client, msg)
        assert wire.recv(server, timeout_s=5) == msg
        reply = {"op": "done", "text": "héllo ☃ \U0001F600", "big": 2 ** 62}
        wire.send(server, reply)
        assert wire.recv(client, timeout_s=5) == reply

    def test_messages_keep_their_boundaries_and_order(self, pair):
        client, server, _ = pair
        for i in range(20):
            wire.send(client, {"op": "ping", "i": i})
        assert [wire.recv(server, 5)["i"] for i in range(20)] == list(range(20))

    def test_wire_format_is_plain_json_not_pickle(self, pair):
        client, server, _ = pair
        wire.send(client, {"op": "ping"})
        raw = server.recv_bytes()
        assert json.loads(raw.decode("utf-8")) == {"op": "ping"} and raw == b'{"op":"ping"}'

    def test_documented_messages_survive_the_wire(self, pair):
        client, server, _ = pair
        msgs = [
            {"op": "run", "req_id": "r000001", "type_id": "bert:23", "exit": 23, "input_idx": 1,
             "deadline_ns": 1712300000000, "t_admit_ns": 1712150000000},
            {"op": "done", "req_id": "r000001", "type_id": "bert:23", "t_recv_ns": 1712150100000,
             "t_start_ns": 1712150200000, "t_end_ns": 1712320000000, "reached": 7, "target": 23, "completion": 0.3333,
             "outcome": "truncated", "ckpt": [[3, 1712160000000], [7, 1712250000000], [11, 1712310000000]],
             "tokens_done": None, "err": None},
            {"op": "pong", "t_ns": 5}]
        for m in msgs:
            wire.send(client, m)
            got = wire.recv(server, 5)
            assert got == m and C.validate_server_msg(got) is got

    def test_send_rejects_non_dict_and_nan(self, pair):
        client, _, _ = pair
        for bad in ([1], "x", None, 3):
            with pytest.raises(TypeError):
                wire.send(client, bad)
        with pytest.raises(ValueError):
            wire.send(client, {"x": float("nan")})

    def test_recv_rejects_non_object_json(self, pair):
        client, server, _ = pair
        client.send_bytes(b"[1, 2]")
        with pytest.raises(C.ContractError):
            wire.recv(server, 5)
        client.send_bytes(b"not json")
        with pytest.raises(ValueError):
            wire.recv(server, 5)

    def test_recv_after_peer_closed_raises_eof(self, pair):
        client, server, _ = pair
        client.close()
        with pytest.raises((EOFError, OSError)):
            wire.recv(server, 5)


class TestRecvTimeout:
    def test_times_out_when_nothing_arrives(self, pair):
        client, server, _ = pair
        t0 = time.monotonic()
        with pytest.raises(TimeoutError):
            wire.recv(server, timeout_s=0.15)
        assert 0.1 <= time.monotonic() - t0 < 15.0

    def test_zero_timeout_polls_without_blocking(self, pair):
        _, server, _ = pair
        t0 = time.monotonic()
        with pytest.raises(TimeoutError):
            wire.recv(server, timeout_s=0)
        assert time.monotonic() - t0 < 10.0

    def test_message_already_queued_is_returned_even_with_tiny_timeout(self, pair):
        client, server, _ = pair
        wire.send(client, {"op": "ping"})
        time.sleep(0.05)
        assert wire.recv(server, timeout_s=0.5) == {"op": "ping"}

    def test_default_blocks_until_a_message_arrives(self, pair):
        client, server, _ = pair
        threading.Timer(0.1, lambda: wire.send(client, {"op": "ping"})).start()
        assert wire.recv(server) == {"op": "ping"}


class TestPortFile:
    def test_write_then_wait_returns_the_port(self, tmp_path):
        p = tmp_path / "server_bert.port"
        wire.write_port_file(p, 40123)
        assert p.read_text() == "40123\n"
        assert wire.wait_port_file(p, 1.0) == 40123

    def test_write_overwrites_and_leaves_no_tmp_files(self, tmp_path):
        p = tmp_path / "a.port"
        wire.write_port_file(p, 1)
        wire.write_port_file(p, 22222)
        assert wire.wait_port_file(p, 1.0) == 22222
        assert sorted(x.name for x in tmp_path.iterdir()) == ["a.port"]

    def test_publication_is_a_rename_of_a_complete_tmp_file(self, tmp_path, monkeypatch):
        """Deterministic atomicity check: when the file appears (os.replace), it is already complete."""
        seen = []
        real = os.replace

        def spy(src, dst):
            seen.append((os.fspath(src), os.fspath(dst), open(src).read(), os.path.exists(dst)))
            return real(src, dst)

        monkeypatch.setattr(os, "replace", spy)
        p = tmp_path / "x.port"
        wire.write_port_file(p, 54321)
        (src, dst, content, existed), = seen
        assert dst == str(p) and src != dst and content == "54321\n" and existed is False
        assert os.path.dirname(src) == os.path.dirname(dst)        # same directory: the rename cannot cross devices

    def test_reader_never_sees_a_partial_file(self, tmp_path):
        p = tmp_path / "s.port"
        ports = [int("9" * (1 + i % 5)) for i in range(150)]
        stop = threading.Event()
        errors, seen = [], set()

        def writer():
            try:
                for port in ports:
                    wire.write_port_file(p, port)
            except Exception as e:           # pragma: no cover
                errors.append(e)
            finally:
                stop.set()

        t = threading.Thread(target=writer, daemon=True)
        wire.write_port_file(p, 9)
        t.start()
        while not stop.is_set():
            try:
                text = p.read_text()
            except (FileNotFoundError, PermissionError):
                continue
            seen.add(text)
        t.join(5)
        assert not errors
        assert seen <= {f"{x}\n" for x in ports}, seen - {f"{x}\n" for x in ports}

    def test_wait_blocks_until_the_file_appears(self, tmp_path):
        p = tmp_path / "late.port"
        threading.Timer(0.2, lambda: wire.write_port_file(p, 777)).start()
        t0 = time.monotonic()
        assert wire.wait_port_file(p, 10.0) == 777
        assert 0.1 <= time.monotonic() - t0 < 15.0

    def test_wait_times_out(self, tmp_path):
        t0 = time.monotonic()
        with pytest.raises(TimeoutError, match="no port file"):
            wire.wait_port_file(tmp_path / "never.port", 0.2)
        assert 0.15 <= time.monotonic() - t0 < 15.0

    def test_empty_file_is_not_a_port(self, tmp_path):
        p = tmp_path / "e.port"
        p.write_text("")
        with pytest.raises(TimeoutError):
            wire.wait_port_file(p, 0.15)
        p.write_text("not a number\n")
        with pytest.raises(TimeoutError):
            wire.wait_port_file(p, 0.15)

    def test_dead_process_stops_the_wait_early(self, tmp_path):
        class Dead:
            returncode = 3

            def poll(self):
                return 3

        t0 = time.monotonic()
        with pytest.raises(RuntimeError, match="exited with code 3"):
            wire.wait_port_file(tmp_path / "never.port", 30.0, Dead())
        assert time.monotonic() - t0 < 15.0

    def test_live_process_does_not_stop_the_wait(self, tmp_path):
        class Alive:
            returncode = None

            def poll(self):
                return None

        with pytest.raises(TimeoutError):
            wire.wait_port_file(tmp_path / "never.port", 0.15, Alive())

    def test_file_written_just_before_death_is_still_read(self, tmp_path):
        p = tmp_path / "s.port"
        wire.write_port_file(p, 4242)

        class Dead:
            returncode = 0

            def poll(self):
                return 0

        assert wire.wait_port_file(p, 1.0, Dead()) == 4242
