"""exp2.serve.dispatcher: the client side of the per-family servers (BRIEF_v1 C2a; contracts K4, K5, K8).

    d = Dispatcher(conns, types, *, board=None, clock=time.monotonic_ns, n_inputs=32)
        conns   {family: Connection to that family's server}  (multiprocessing Connections from exp2.wire.connect;
                a test double with send_bytes / recv_bytes / poll also works)
        board   anything with set_gpu_level(mhz) -> read-back MHz (exp2.board.clocks.Board); None = no actuation
    d.run_open(cond, schedule, policy, out_dir)  -> summary dict
    d.run_closed(cond, out_dir)                  -> summary dict
Both write <out_dir>/requests.jsonl (K8, one row per request) and <out_dir>/freq_switch.csv.

cond is the plan condition (K15): condition_id (required), mode, arm, gpu_mhz, alpha, deadlines_s, policy_key, seed,
n_warmup, and for closed runs cell (type_ids, one per family) and duration_s.  schedule is arrivals.schedule(...)
output, times relative to the start of the run; a missing t_deadline_ns is filled with
contracts.with_deadlines(cond["alpha"]) or, for a Run 3 condition, contracts.with_family_deadlines(cond["deadlines_s"])
(both set: ValueError); a given one is kept.  While a run is in progress the board is used only by the actuator thread.
policy is an exp2.policies.Policy (a PolicySpec or dict is wrapped) or, for a spec with params.exit_rule, an
exp2.exits.ExitPolicy (anything else: ValueError); it must keep family_exclusive=True because every server is single
threaded.

One thread runs the event loop; the only other thread is the FreqActuator.  The loop waits on all server connections at
once (multiprocessing.connection.wait), so a `done` is read by the thread that acts on it and no event can be lost.
Per event the loop (1) reads every message that is ready, (2) fires the next arrival once it is due; a completion at the
same instant goes first.  Every admission decision comes from policies.AdmissionCore (K4): arrive / done return a
Decision and the loop only carries it out: rows for the dropped, `run` for the admitted, a frequency command.

Time and rows
    t0                    clock() when the run starts; schedule times are offsets from it.
    t_arrival_ns          the SCHEDULED arrival (t0 + offset), also when the loop wakes late; t_arrival_actual_ns is
                          when the loop really handled it.  Response time is t_done_ns - t_arrival_ns, so lateness
                          counts.
    expired in queue      dropped by AdmissionCore (now >= deadline) at the next event, never sent; the row has
                          t_done_ns = t_deadline_ns, completion 0, and no admit/server fields.
    t_admit_ns            the loop's clock when the `run` was sent.  t_done_ns: when the loop read the `done`.
    exit choice (Run 3)   with an ExitPolicy, admit sets type_id/exit/target to the RUN type of Decision.run_type_ids,
                          so the `run` message carries it; every row keeps the arrival type in req_type_id.  A row that
                          never ran keeps type_id == req_type_id.  Rows of any other policy have exactly REQUEST_KEYS.
    f_mhz                 the last COMMANDED level once this event's decision is applied (never a read-back), or
                          "governor" for a governor policy or a governor/stock arm.
    closed loop           all tenants start at t0 (one common instant), then each sends its next request the moment the
                          previous one is done, until t0 + duration_s; requests in flight at the end are awaited.
                          t_arrival = t_admit, no deadline, policy "closed"; is_warmup marks the first n_warmup of
                          each type.

Limited preemption (spec params.preempt_llama_at_token: edgeserving_lp, coexit_lp; PLAN PL3 1.2)
    pause                 after the admits of every Decision, {"op":"pause"} goes out for each of Decision.pause_ids
                          that is in flight, not paused and not pending; it joins policy.pause_pending and its t_req
                          (the loop's clock) is kept.  A failed send is a dead server.  Only llama requests pause:
                          pause/resume never go to another server (RuntimeError, a policy bug).
    paused                the server stopped at a token boundary: pending -> policy.paused[req_id] = tokens_done, the
                          row gets [tokens_done, t_req, t_pause, None] and core.tick(now) decides at once.  A `paused`
                          that is invalid or not pending is only counted (unexpected_msgs).
    resume (R1)           after every applied Decision: a request is paused, every request in flight is paused and no
                          pause is pending -> {"op":"resume"}, policy.paused_ns[req_id] += now - t_pause.
    done                  a done cancels a pending pause (the server ignores the stale one).  The row's pauses are the
                          done's [[tokens_done, t_pause, t_resume], ...] with t_req merged in; a done that disagrees
                          with the `paused` messages is an invalid done.  An error row keeps the dispatcher's record
                          (t_resume None while still paused).
    hang timer            skips paused requests; for the others it runs from max(t_admit, last resume).
    rows                  every row of an LP spec has "pauses" ([] when never paused); other rows never have it.
    invariant             "something paused, nothing running, nothing pending" never holds while the loop waits.

Failure handling (nothing is lost, nothing hangs)
    A server whose connection breaks, whose send fails, or that does not answer within HANG_TIMEOUT_NS is declared dead
    (summary["dead_servers"], kept for later runs of this Dispatcher).  Its in-flight requests, and every later or
    queued request of that family, become outcome "error" rows with the reason in `err`.  A `done` that fails
    contracts.validate_done also becomes an error row; a message nobody waits for is only counted (unexpected_msgs).
    If nothing is in flight, no arrival is left and the policy still holds queued requests, they are written as error
    rows ("never admitted") instead of waiting forever.  After the loop the number of rows must equal the number of
    requests.

Summary returned by run_open / run_closed (JSON-able)
    condition_id, mode, n_requests, n_rows (equal), outcomes {outcome: count}, t0_ns, t_end_ns,
    max_arrival_lag_ns, unexpected_msgs, dead_servers {family: reason},
    freq {commanded [mhz in order], applied, superseded, errors [{t_ns, target_mhz, error}]}.
The caller (run_session) treats a non-empty dead_servers as "server died" and non-empty freq.errors as a board fault.

Never imports exp2.board.telemetry (and nothing from torch).
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from multiprocessing import connection

from exp2 import contracts as C
from exp2 import wire
from exp2.policies import AdmissionCore, Policy
from exp2.serve.freq_actuator import FreqActuator, write_freq_switch_csv

__all__ = ["Dispatcher", "HANG_TIMEOUT_NS"]

HANG_TIMEOUT_NS = 120 * 1_000_000_000      # a request in service this long means the server is gone
SPIN_NS = 1_500_000                         # the last stretch before an arrival is spun, not slept (timer granularity)
POLL_SLEEP_S = 0.0002                       # poll-only test doubles: pause between polls


class _Down(Exception):
    """The server of this family cannot take or answer requests."""


class Dispatcher:
    def __init__(self, conns, types, *, board=None, clock=time.monotonic_ns, n_inputs=32):
        if int(n_inputs) < 1:
            raise ValueError("n_inputs must be >= 1, got %r" % (n_inputs,))
        self.conns = dict(conns)
        self.types = types
        self.board = board
        self.clock = clock
        self.n_inputs = int(n_inputs)
        self._dead = {}                     # family -> reason; survives from one run to the next

    def run_open(self, cond, schedule, policy, out_dir) -> dict:
        return _OpenRun(self, cond, schedule, policy, out_dir).execute()

    def run_closed(self, cond, out_dir) -> dict:
        return _ClosedRun(self, cond, out_dir).execute()

    # ---------------------------------------------------------------------------------------- waiting
    def _poll(self, timeout_ns):
        """Wait up to timeout_ns for server messages.  -> [(family, message | exception, stamp_ns)].

        The stamp is taken right after the wait returns, before the message is decoded.  An exception in the second slot
        means the connection is gone.  This is the only place the loop blocks; tests replace it to run in virtual
        time."""
        live = {}
        for fam, conn in self.conns.items():
            if fam not in self._dead:
                live[conn] = fam
        out = []
        for conn, fam in list(live.items()):
            if getattr(conn, "closed", False):
                out.append((fam, EOFError("connection closed on our side"), self.clock()))
                del live[conn]
        if out:
            return out
        if not live:
            time.sleep(timeout_ns / 1e9)
            return []
        ready = self._wait(list(live), timeout_ns)
        stamp = self.clock()
        for conn in ready:
            try:
                out.append((live[conn], wire.recv(conn), stamp))
            except (EOFError, OSError, ValueError) as exc:        # ValueError covers bad JSON and ContractError
                out.append((live[conn], exc, stamp))
        return out

    @staticmethod
    def _wait(objs, timeout_ns):
        if all(hasattr(c, "fileno") for c in objs):
            ready = connection.wait(objs, max(0, timeout_ns - SPIN_NS) / 1e9)
            end = time.monotonic_ns() + min(timeout_ns, SPIN_NS)
            while not ready and time.monotonic_ns() < end:
                ready = connection.wait(objs, 0)
            return ready
        end = time.monotonic_ns() + timeout_ns                    # test doubles that only have poll()
        while True:
            ready = [c for c in objs if c.poll(0)]
            if ready or time.monotonic_ns() >= end:
                return ready
            time.sleep(POLL_SLEEP_S)


class _Run:
    """State of one condition.  Single threaded: only Dispatcher.run_* callers touch it."""
    mode = None
    lp = False                              # limited preemption (an LP spec of an open run)

    def __init__(self, disp, cond, out_dir):
        if cond.get("mode", self.mode) != self.mode:
            raise ValueError("%s run given a condition with mode %r" % (self.mode, cond.get("mode")))
        self.d = disp
        self.cond = cond
        self.out_dir = os.fspath(out_dir)
        self.cid = cond["condition_id"]
        self.arm = cond.get("arm", "pinned")
        self.seed = int(cond.get("seed", 0))
        self.n_warmup = int(cond.get("n_warmup", 0))
        self.live = {}                      # req_id -> row, for every request that is queued or in service
        self.inflight = set()               # req_ids sent to a server and not answered yet
        self.counts = Counter()
        self.n_rows = 0
        self.commanded = []
        self.unexpected = 0
        self.max_lag = 0
        self.t0 = None
        self.act = None
        self.fh = None
        self._input_next = {}
        self._targets = {}
        self._paused = {}                   # LP: req_id -> tokens_done (the policy's dict on an LP run)
        self._pending = set()               # LP: req_ids with a pause sent and no `paused` yet
        self._paused_ns = {}                # LP: req_id -> total paused ns
        self._t_req = {}                    # LP: req_id -> when its pending pause was sent
        self._since = {}                    # req_id -> last resume (hang timer start; t_admit when absent)

    # ---------------------------------------------------------------------------------------- run
    def execute(self) -> dict:
        d = self.d
        os.makedirs(self.out_dir, exist_ok=True)
        act = FreqActuator(d.board, d.clock) if d.board is not None else None
        self.act = act
        try:
            with open(os.path.join(self.out_dir, "requests.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
                self.fh = fh
                if act is not None:
                    act.start()
                self.t0 = d.clock()
                self._start()
                self._loop()
                t_end = d.clock()
        finally:
            if act is not None:
                act.close()
        write_freq_switch_csv(os.path.join(self.out_dir, "freq_switch.csv"), act.rows if act is not None else [])
        n = self._n_requests()
        if self.live or self.n_rows != n:
            raise RuntimeError("%s: %d rows for %d requests (%d still open)"
                               % (self.cid, self.n_rows, n, len(self.live)))
        return {
            "condition_id": self.cid, "mode": self.mode, "n_requests": n, "n_rows": self.n_rows,
            "outcomes": dict(self.counts), "t0_ns": self.t0, "t_end_ns": t_end,
            "max_arrival_lag_ns": self.max_lag, "unexpected_msgs": self.unexpected,
            "dead_servers": dict(d._dead),
            "freq": {"commanded": list(self.commanded), "applied": len(act.rows) if act else 0,
                     "superseded": act.n_superseded if act else 0, "errors": list(act.errors) if act else []}}

    def _loop(self):
        d = self.d
        while True:
            now = d.clock()
            self._check_hangs(now)
            if self._finished():
                return
            if self._stuck():
                self._drain_stuck(d.clock())
                return
            assert not (self._paused and not self._pending and all(r in self._paused for r in self.inflight)), \
                "%s: %s paused with nothing running and no pause pending (resume rule R1 missed)" % (
                    self.cid, sorted(self._paused))
            due = self._next_arrival()
            wake = [t for t in (due, self._hang_wake()) if t is not None]
            if not wake:
                raise RuntimeError("%s: the event loop has nothing to wait for" % self.cid)
            events = d._poll(max(0, min(wake) - now))
            if events:
                for fam, msg, stamp in events:                    # completions first: they beat an arrival due at once
                    self._on_message(fam, msg, stamp)
                continue
            now = d.clock()
            if due is not None and due <= now:
                self._fire(now)

    # ---------------------------------------------------------------------------------------- hooks
    def _start(self):
        raise NotImplementedError

    def _n_requests(self):
        raise NotImplementedError

    def _finished(self):
        raise NotImplementedError

    def _stuck(self):
        return False

    def _next_arrival(self):
        return None

    def _fire(self, now):
        raise NotImplementedError

    def _released(self, row, now):
        """A request left service.  -> a Decision to carry out, or None."""
        raise NotImplementedError

    # ---------------------------------------------------------------------------------------- rows
    def _target(self, type_id):
        t = self._targets.get(type_id)
        if t is None:
            fam, ex = C.parse_type_id(type_id)
            t = self._targets[type_id] = C.default_checkpoints(fam, ex)[-1]
        return t

    def _new_row(self, req_id, type_id, t_arrival, t_actual, t_deadline, queue_len, is_warmup, policy, policy_spec):
        fam, ex = C.parse_type_id(type_id)
        return {
            "condition_id": self.cid, "req_id": req_id, "type_id": type_id, "family": fam, "exit": ex,
            "klass": self.d.types.get(type_id).klass, "target": self._target(type_id),
            "t_arrival_ns": t_arrival, "t_arrival_actual_ns": t_actual, "t_deadline_ns": t_deadline,
            "t_admit_ns": None, "t_srv_recv_ns": None, "t_start_ns": None, "t_end_ns": None, "t_done_ns": None,
            "server": fam, "policy": policy, "policy_spec": policy_spec, "f_mhz": None,
            "n_in_service_at_admit": None, "queue_len_at_arrival": queue_len, "reached": None, "completion": 0.0,
            "outcome": None, "tokens_done": None, "ckpt": [], "is_warmup": bool(is_warmup), "seed": self.seed,
            "mode": self.mode, "err": None}

    def _emit(self, row):
        try:
            C.validate_request_row(row)
        except C.ContractError as exc:                            # a bug must not cost the request its row
            row.update(outcome="error", err="invalid row (%s): %s" % (row["outcome"], exc), reached=None,
                       completion=0.0, ckpt=[], tokens_done=None)
            C.validate_request_row(row)
        self.fh.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        self.fh.flush()
        self.counts[row["outcome"]] += 1
        self.n_rows += 1

    def _error_now(self, row, reason, now):
        row.update(outcome="error", err=reason, reached=None, completion=0.0, ckpt=[], tokens_done=None, t_done_ns=now)

    # ---------------------------------------------------------------------------------------- sending
    def _f_label(self, now):
        raise NotImplementedError

    def _send(self, row, now):
        fam = row["family"]
        if fam in self.d._dead:
            raise _Down("server %s is down: %s" % (fam, self.d._dead[fam]))
        idx = self._input_next.get(fam, 0)
        self._input_next[fam] = (idx + 1) % self.d.n_inputs
        msg = {"op": "run", "req_id": row["req_id"], "type_id": row["type_id"], "exit": row["exit"],
               "input_idx": idx, "deadline_ns": row["t_deadline_ns"], "t_admit_ns": now}
        try:
            wire.send(self.d.conns[fam], msg)
        except (OSError, EOFError, ValueError) as exc:
            reason = "send to %s server failed: %s: %s" % (fam, type(exc).__name__, exc)
            self.d._dead.setdefault(fam, reason)
            raise _Down(reason) from exc
        self.inflight.add(row["req_id"])

    def _command(self, mhz):
        self.commanded.append(mhz)
        if self.act is not None:
            self.act.request(mhz)

    # ---------------------------------------------------------------------------------------- events
    def _on_message(self, fam, msg, stamp):
        now = max(self.d.clock(), stamp)
        if isinstance(msg, BaseException):
            self._kill(fam, "connection to the %s server lost: %s: %s" % (fam, type(msg).__name__, msg), now)
            return
        rid = msg.get("req_id")
        if self.lp and msg.get("op") == "paused":
            self._on_paused(msg, now)
            return
        if msg.get("op") != "done" or rid not in self.inflight:
            self.unexpected += 1
            return
        row = self.live[rid]
        try:
            C.validate_done(msg)
            if msg["type_id"] != row["type_id"]:
                raise C.ContractError("type_id %r, expected %r" % (msg["type_id"], row["type_id"]))
            pauses = self._merged_pauses(row, msg) if self.lp else None
        except C.ContractError as exc:
            self._error_now(row, "invalid done message: %s" % exc, stamp)
        else:
            row.update(t_srv_recv_ns=msg["t_recv_ns"], t_start_ns=msg["t_start_ns"], t_end_ns=msg["t_end_ns"],
                       reached=msg["reached"], completion=msg["completion"], outcome=msg["outcome"],
                       tokens_done=msg["tokens_done"], ckpt=[list(p) for p in msg["ckpt"]], err=msg["err"],
                       t_done_ns=stamp)
            if pauses is not None:
                row["pauses"] = pauses
        self._apply([self._finish(rid, now)], now)

    def _finish(self, rid, now):
        row = self.live.pop(rid)
        self.inflight.discard(rid)
        self._paused.pop(rid, None)                               # LP: a done (or error) cancels a pending pause
        self._pending.discard(rid)
        self._paused_ns.pop(rid, None)
        self._t_req.pop(rid, None)
        self._since.pop(rid, None)
        self._emit(row)
        return self._released(row, now)

    # ---------------------------------------------------------------------------------------- LP (pause / resume)
    def _on_paused(self, msg, now):
        """The llama server stopped at a token boundary: record it, then decide again (core.tick)."""
        rid = msg.get("req_id")
        try:
            C.validate_paused(msg)
            ok = rid in self._pending and msg["type_id"] == self.live[rid]["type_id"]
        except C.ContractError:
            ok = False
        if not ok:
            self.unexpected += 1
            return
        self._pending.discard(rid)
        self._paused[rid] = msg["tokens_done"]
        self.live[rid]["pauses"].append([msg["tokens_done"], self._t_req.pop(rid), msg["t_pause_ns"], None])
        self._apply([self.core.tick(now)], now)

    @staticmethod
    def _merged_pauses(row, msg):
        """The done's [[tokens_done, t_pause, t_resume], ...] with the t_req of this loop: the row's pauses."""
        srv, mine = msg.get("pauses", []), row["pauses"]
        if [(p[0], p[1]) for p in srv] != [(p[0], p[2]) for p in mine]:
            raise C.ContractError("pauses %s disagree with the paused messages %s" % (srv, mine))
        return [[tok, m[1], t_pause, t_resume] for (tok, t_pause, t_resume), m in zip(srv, mine)]

    def _ctl(self, rid, op, now):
        """Send {"op": op, "req_id": rid} (pause / resume) to the llama server.  -> False when the send failed: the
        server is then dead and its requests are error rows."""
        fam = self.live[rid]["family"]
        if fam != "llama":
            raise RuntimeError("%s: the policy asked to %s %s, a %s request (only llama pauses)" % (self.cid, op, rid,
                                                                                                 fam))
        try:
            wire.send(self.d.conns[fam], {"op": op, "req_id": rid})
        except (OSError, EOFError, ValueError) as exc:
            self._kill(fam, "send to %s server failed: %s: %s" % (fam, type(exc).__name__, exc), now)
            return False
        return True

    def _pause(self, ids, now):
        """Decision.pause_ids: pause each request that is in flight, not paused and not pending."""
        for rid in ids:
            if rid in self.inflight and rid not in self._paused and rid not in self._pending:
                if self._ctl(rid, "pause", now):
                    self._pending.add(rid)
                    self._t_req[rid] = now

    def _resume_idle(self, now):
        """R1: a request is paused, every request in flight is paused and no pause is pending -> resume it."""
        if not self._paused or self._pending or any(r not in self._paused for r in self.inflight):
            return
        for rid in sorted(self._paused):
            del self._paused[rid]
            if self._ctl(rid, "resume", now):
                p = self.live[rid]["pauses"][-1]
                p[3] = now
                self._paused_ns[rid] = self._paused_ns.get(rid, 0) + now - p[2]
                self._since[rid] = now

    def _fail(self, rid, reason, now):
        self._error_now(self.live[rid], reason, now)
        return self._finish(rid, now)

    def _kill(self, fam, reason, now):
        self.d._dead.setdefault(fam, reason)
        decisions = [self._fail(rid, self.d._dead[fam], now)
                     for rid in sorted(self.inflight) if self.live[rid]["family"] == fam]
        self._apply(decisions, now)

    def _check_hangs(self, now):
        for rid in sorted(self.inflight):
            row = self.live.get(rid)
            if row is None or rid in self._paused:                # a paused request waits for us, not the server
                continue
            fam = row["family"]
            if fam not in self.d._dead and now - self._since.get(rid, row["t_admit_ns"]) >= HANG_TIMEOUT_NS:
                self._kill(fam, "no reply from the %s server for %g s (request %s)" % (fam, HANG_TIMEOUT_NS / 1e9, rid),
                           now)

    def _hang_wake(self):
        starts = [self._since.get(r, self.live[r]["t_admit_ns"]) for r in self.inflight if r not in self._paused]
        return min(starts) + HANG_TIMEOUT_NS if starts else None

    def _apply(self, decisions, now):
        raise NotImplementedError

    def _drain_stuck(self, now):
        pass


class _OpenRun(_Run):
    mode = "open"

    def __init__(self, disp, cond, schedule, policy, out_dir):
        super().__init__(disp, cond, out_dir)
        types = disp.types
        if not hasattr(policy, "admit"):
            policy = Policy(policy, types)
        if not getattr(policy, "family_exclusive", True):
            raise ValueError("the dispatcher needs a policy with family_exclusive=True "
                             "(every server is single threaded)")
        self.spec = policy.spec
        self.exit_run = "exit_rule" in policy.spec.params
        if self.exit_run and not hasattr(policy, "admit_exits"):
            raise ValueError("spec %r has params.exit_rule: it needs exits.ExitPolicy, not %s"
                             % (policy.spec.name, type(policy).__name__))
        self.lp = bool(policy.spec.params.get("preempt_llama_at_token"))
        if self.lp:                                               # the policy reads this state; the loop fills it
            self._paused, self._pending, self._paused_ns = policy.paused, policy.pause_pending, policy.paused_ns
            self._paused.clear()
            self._pending.clear()
            self._paused_ns.clear()
        items = sorted((dict(it) for it in schedule), key=lambda it: it["t_arrival_ns"])
        ids = [it["req_id"] for it in items]
        if len(set(ids)) != len(ids):
            dup = sorted(i for i, n in Counter(ids).items() if n > 1)
            raise ValueError("duplicate req_id in the schedule: %s" % dup[:5])
        for it in items:
            fam = types.get(it["type_id"]).family                # ContractError (a ValueError) for an unknown type
            if fam not in disp.conns:
                raise ValueError("schedule has a %s request but there is no connection for family %r"
                                 % (it["type_id"], fam))
        alpha, fam_dl = cond.get("alpha"), cond.get("deadlines_s")
        if alpha is not None and fam_dl is not None:
            raise ValueError("%s: the condition sets both alpha and deadlines_s" % self.cid)
        if (alpha is not None or fam_dl is not None) and any(it.get("t_deadline_ns") is None for it in items):
            filled = C.with_deadlines(items, types, alpha) if fam_dl is None else C.with_family_deadlines(items, fam_dl)
            items = [it if it.get("t_deadline_ns") is not None else f for it, f in zip(items, filled)]
        self.items = items
        self.idx = 0
        f0 = cond.get("gpu_mhz")
        self.core = AdmissionCore(policy, types, f0 if f0 is not None else policy.spec.f_mhz)
        self.policy_name = cond.get("policy_key") or policy.spec.name
        self.policy_json = policy.spec.to_json()
        self.governor = self.arm in ("governor", "stock") or policy.spec.f_rule == "governor"

    def _start(self):
        pass

    def _n_requests(self):
        return len(self.items)

    def _finished(self):
        return self.idx >= len(self.items) and not self.live

    def _stuck(self):
        return self.idx >= len(self.items) and bool(self.live) and not self.inflight

    def _drain_stuck(self, now):
        for rid in list(self.live):
            row = self.live.pop(rid)
            self._error_now(row, "never admitted: nothing in service and no arrival left", now)
            self._emit(row)

    def _next_arrival(self):
        return None if self.idx >= len(self.items) else self.t0 + self.items[self.idx]["t_arrival_ns"]

    def _f_label(self, now):
        return "governor" if self.governor else self.core.view(now).f_current_mhz

    def _fire(self, now):
        it = self.items[self.idx]
        warm = self.idx < self.n_warmup
        self.idx += 1
        t_arr = self.t0 + it["t_arrival_ns"]
        t_dl = None if it.get("t_deadline_ns") is None else self.t0 + it["t_deadline_ns"]
        self.max_lag = max(self.max_lag, now - t_arr)
        ts = self.d.types.get(it["type_id"])
        view = self.core.view(now)
        row = self._new_row(it["req_id"], it["type_id"], t_arr, now, t_dl, len(view.queue), warm, self.policy_name,
                            self.policy_json)
        if self.exit_run:
            row["req_type_id"] = it["type_id"]
        if self.lp:
            row["pauses"] = []
        if ts.family in self.d._dead:
            self._error_now(row, "server %s is down: %s" % (ts.family, self.d._dead[ts.family]), now)
            self._emit(row)
            return
        self.live[row["req_id"]] = row
        req = C.QueuedReq(it["req_id"], it["type_id"], ts.family, ts.klass, t_arr, t_dl, it.get("k_override"))
        self._apply([self.core.arrive(req, now)], now)

    def _released(self, row, now):
        return self.core.done(row["req_id"], now)

    def _apply(self, decisions, now):
        todo = list(decisions)
        while todo:
            dec = todo.pop(0)
            if dec is None:
                continue
            for rid in dec.expired:
                row = self.live.pop(rid)
                row.update(t_done_ns=row["t_deadline_ns"], outcome="expired_in_queue", reached=None, completion=0.0,
                           ckpt=[], err=None)
                self._emit(row)
            runs = dec.run_type_ids or (None,) * len(dec.admitted)
            for rid, n_at, run in zip(dec.admitted, dec.n_in_service_at_admit, runs):
                row = self.live[rid]
                if run is not None and run != row["type_id"]:            # the exit policy runs another exit
                    row.update(type_id=run, exit=C.parse_type_id(run)[1], target=self._target(run))
                row.update(t_admit_ns=now, n_in_service_at_admit=n_at, f_mhz=self._f_label(now))
                try:
                    self._send(row, now)
                except _Down as exc:
                    todo.append(self._fail(rid, str(exc), now))
            if self.lp:
                self._pause(dec.pause_ids, now)                   # after the admits: the view the policy read
            if dec.commanded_f is not None:
                self._command(dec.commanded_f)
        if self.lp:
            self._resume_idle(now)


class _ClosedRun(_Run):
    mode = "closed"

    def __init__(self, disp, cond, out_dir):
        super().__init__(disp, cond, out_dir)
        self.cell = list(cond["cell"])
        fams = [disp.types.get(t).family for t in self.cell]
        if len(set(fams)) != len(fams):
            raise ValueError("closed cell %s has two types of one family; a server runs one request at a time"
                             % self.cell)
        for fam in fams:
            if fam not in disp.conns:
                raise ValueError("closed cell %s needs a connection for family %r" % (self.cell, fam))
        self.duration_ns = int(round(float(cond["duration_s"]) * 1e9))
        self.t_stop = None
        self.n_launched = 0
        self._per_type = Counter()
        self.f_label = "governor" if self.arm in ("governor", "stock") else cond.get("gpu_mhz")

    def _start(self):
        self.t_stop = self.t0 + self.duration_ns
        for tid in self.cell:                                    # the barrier: every tenant starts at this one instant
            self._launch(tid, self.t0)

    def _n_requests(self):
        return self.n_launched

    def _finished(self):
        return not self.inflight

    def _f_label(self, now):
        return self.f_label

    def _launch(self, tid, now):
        self.n_launched += 1
        warm = self._per_type[tid] < self.n_warmup
        self._per_type[tid] += 1
        row = self._new_row("r%06d" % self.n_launched, tid, now, now, None, 0, warm, "closed", None)
        row.update(t_admit_ns=now, n_in_service_at_admit=len(self.inflight), f_mhz=self._f_label(now))
        self.live[row["req_id"]] = row
        try:
            self._send(row, now)
        except _Down as exc:
            self._fail(row["req_id"], str(exc), now)

    def _released(self, row, now):
        if row["family"] not in self.d._dead and now < self.t_stop:
            self._launch(row["type_id"], now)
        return None

    def _fire(self, now):
        raise RuntimeError("a closed run has no scheduled arrivals")

    def _apply(self, decisions, now):
        pass
