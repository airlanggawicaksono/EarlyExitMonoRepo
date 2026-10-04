"""Phase-level wall-clock and energy timer for cold-start overhead measurement.

Intended use: wrap the model-load, compile, and tokenizer phases that happen
BEFORE BenchmarkProfiler starts, then merge ov.as_dict() into the profiler's
meta= argument so the fields land in hw_results.json.

Energy reading reuses device_energy_mj from shared.hw_profiler (the same
function BenchmarkProfiler uses internally). When that helper returns None
(dev box, Jetson, no pynvml), energy fields are None, never zero.

Duplicate phase names: last-writer wins. The earlier phase is overwritten.
This is intentional and documented here so callers are not surprised.
Document the choice with a ponytail: comment at the relevant line below.
"""

import time
from contextlib import contextmanager
from typing import Dict, Optional


def _try_energy_mj() -> Optional[float]:
    """Return device energy counter in mJ, or None if unavailable.

    Tries to import device_energy_mj from shared.hw_profiler without raising.
    Falls back to None on any import or call failure so this module works on
    a dev box with no pynvml and no torch.
    """
    try:
        from shared.hw_profiler import device_energy_mj  # type: ignore
        val = device_energy_mj()
        return val  # already Optional[float]
    except Exception:
        return None


class OverheadTimer:
    """Records named wall-clock phases and optional energy deltas.

    Usage:
        ov = OverheadTimer()
        with ov.phase("model_load"):
            model = load(...)
        with ov.phase("compile"):
            compile(...)
        d = ov.as_dict()
        # -> {"overhead_model_load_sec": ..., "overhead_compile_sec": ...,
        #     "overhead_model_load_j": ... or None, "overhead_compile_j": ... or None,
        #     "overhead_total_sec": ...}

    Nesting: not supported. If caller nests a phase inside another, both are
    recorded independently by their respective context entries and exits. The
    inner phase records its own duration; the outer phase records wall time that
    includes the inner phase's duration. This is defined behaviour: both records
    appear in as_dict() under their own names.

    Duplicate names: last-writer wins. Earlier record is silently overwritten.
    # ponytail: last-wins keeps the dict flat; accumulation would need a list
    # which breaks JSON round-trip assumptions in the profiler meta channel.
    Callers that sweep multiple exits and call phase("compile") per exit get the
    LAST compile duration. If you need per-exit granularity, include the exit
    index in the phase name.

    Exception safety: a phase that raises records its duration up to the
    exception and re-raises. It will appear in as_dict() with that partial
    duration and a None energy delta (energy close could not be read).
    """

    def __init__(self) -> None:
        self._phases: Dict[str, Dict] = {}  # name -> {sec, j_or_none}

    @contextmanager
    def phase(self, name: str):
        """Context manager for one named phase."""
        e0 = _try_energy_mj()
        t0 = time.perf_counter()
        try:
            yield
        except Exception:
            dur = round(time.perf_counter() - t0, 3)
            # Energy close skipped on exception path: e1 read may itself raise.
            self._phases[name] = {"sec": dur, "j": None}
            raise
        dur = round(time.perf_counter() - t0, 3)
        e1 = _try_energy_mj()
        if e0 is not None and e1 is not None:
            energy_j = round((e1 - e0) / 1000.0, 6)
        else:
            energy_j = None
        self._phases[name] = {"sec": dur, "j": energy_j}

    def as_dict(self) -> Dict:
        """Return a flat, JSON-serialisable dict with overhead_ prefix.

        Keys emitted:
            overhead_<phase>_sec     wall-clock duration in seconds (float)
            overhead_<phase>_j       energy in joules (float) or None
            overhead_total_sec       sum of all phase durations (float)

        The total is the sum of recorded phase durations, not end-to-end wall
        time of the OverheadTimer instance itself, which could include gaps
        between phases. This matches what the brief specifies.
        """
        out: Dict = {}
        total = 0.0
        for name, rec in self._phases.items():
            out[f"overhead_{name}_sec"] = rec["sec"]
            out[f"overhead_{name}_j"] = rec["j"]
            total += rec["sec"]
        out["overhead_total_sec"] = round(total, 3)
        return out
