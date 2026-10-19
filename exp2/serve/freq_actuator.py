"""exp2.serve.freq_actuator: applies GPU frequency commands off the dispatcher's event loop (BRIEF_v1 C2a).

    act = FreqActuator(board, clock=time.monotonic_ns)      # board: anything with set_gpu_level(mhz) -> read-back MHz
    act.start()
    act.request(612)         # never blocks; a target not yet applied is replaced by the newer one
    act.close()              # applies the newest pending target, waits for the running call, joins the thread

One thread talks to the board.  set_gpu_level takes tens of milliseconds, so while it runs the loop may command
several levels; only the newest is applied afterwards (the others count in n_superseded).  Nothing is dropped
silently: every request is either applied (a row in `rows`) or superseded by a later one, and a board that refuses
(ClockError, or ValueError for a level it does not offer) leaves a row with an empty read-back plus an entry in
`errors`; the thread goes on with the next target.  The caller reads `errors` (the Dispatcher puts them in its summary).

`rows` entries are exactly the freq_switch.csv columns (contracts.FREQ_SWITCH_COLS):
    t_req_ns     when the applied target was requested   t_done_ns    when set_gpu_level returned (or failed)
    target_mhz   the commanded level                     readback_mhz what the board reported, None on failure
The dispatcher's f_current is the last COMMANDED level; the read-back here is only logged, never fed back.
"""
from __future__ import annotations

import csv
import logging
import threading
import time

from exp2 import contracts as C

__all__ = ["FreqActuator", "write_freq_switch_csv"]

log = logging.getLogger(__name__)
JOIN_TIMEOUT_S = 60.0


def write_freq_switch_csv(path, rows) -> None:
    """freq_switch.csv: header plus one line per row dict (None -> empty cell)."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(C.FREQ_SWITCH_COLS)
        for r in rows:
            w.writerow(["" if r[c] is None else r[c] for c in C.FREQ_SWITCH_COLS])


class FreqActuator:
    def __init__(self, board, clock=time.monotonic_ns):
        self.board = board
        self.clock = clock
        self.rows: list = []
        self.errors: list = []
        self.n_requests = 0
        self.n_superseded = 0
        self._cv = threading.Condition()
        self._pending = None            # (target_mhz, t_req_ns) not yet picked up by the thread
        self._closing = False
        self._thread = None

    def start(self) -> None:
        with self._cv:
            if self._thread is not None:
                raise RuntimeError("actuator already started")
            self._thread = threading.Thread(target=self._run, name="freq-actuator", daemon=True)
        self._thread.start()

    def request(self, mhz: int) -> None:
        with self._cv:
            if self._thread is None or self._closing:
                raise RuntimeError("actuator is not running (request before start() or after close())")
            if self._pending is not None:
                self.n_superseded += 1
            self._pending = (int(mhz), self.clock())
            self.n_requests += 1
            self._cv.notify()

    def close(self) -> None:
        """Apply the newest pending target, wait for the running call, stop the thread.  Safe to call twice."""
        with self._cv:
            thread = self._thread
            if thread is None:
                return
            self._closing = True
            self._cv.notify_all()
        thread.join(JOIN_TIMEOUT_S)
        if thread.is_alive():
            raise RuntimeError("freq actuator: the board call did not return within %g s" % JOIN_TIMEOUT_S)

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._pending is None and not self._closing:
                    self._cv.wait()
                if self._pending is None:
                    return
                (target, t_req), self._pending = self._pending, None
            readback, error = None, None
            try:
                readback = int(self.board.set_gpu_level(target))
            except Exception as exc:                     # ClockError, ValueError, a dead helper: record, do not die
                error = "%s: %s" % (type(exc).__name__, exc)
                log.warning("set_gpu_level(%s) failed: %s", target, error)
            t_done = self.clock()
            with self._cv:
                self.rows.append({"t_req_ns": t_req, "t_done_ns": t_done, "target_mhz": target,
                                  "readback_mhz": readback})
                if error is not None:
                    self.errors.append({"t_ns": t_done, "target_mhz": target, "error": error})
