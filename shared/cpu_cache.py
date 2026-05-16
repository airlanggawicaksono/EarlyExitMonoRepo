"""Linux-only PAPI binding for LLC (last-level cache) + memory bandwidth.

PAPI = C-based perf counter library. Provides hardware-accurate cache miss /
TLB / instruction counts via perf_event_open. Not available on Windows.

Install (Linux): apt install libpapi-dev && pip install python-papi

Falls back gracefully if PAPI unavailable -> returns empty dict.

Usage:
    with CacheCounter() as cc:
        # ... do work ...
        result = cc.read()  # {llc_misses, llc_references, instructions, ...}
"""

from typing import Dict

_PAPI_AVAILABLE = False
_PAPI_EVENTS = []

try:
    from pypapi import events as pe  # type: ignore
    from pypapi import papi_low as pl  # type: ignore
    pl.library_init()
    _PAPI_AVAILABLE = True
    _PAPI_EVENTS = [
        ("llc_misses", pe.PAPI_L3_TCM),
        ("llc_references", pe.PAPI_L3_TCA),
        ("instructions", pe.PAPI_TOT_INS),
        ("cycles", pe.PAPI_TOT_CYC),
    ]
except Exception:
    _PAPI_AVAILABLE = False


class CacheCounter:
    """Context manager for LLC + instruction counts. Skips silently if PAPI absent."""

    def __init__(self):
        self._eventset = None
        self._enabled = _PAPI_AVAILABLE
        self._labels = []

    def __enter__(self):
        if not self._enabled:
            return self
        try:
            self._eventset = pl.create_eventset()
            for label, ev in _PAPI_EVENTS:
                try:
                    pl.add_event(self._eventset, ev)
                    self._labels.append(label)
                except Exception:
                    pass  # event not available on this CPU
            if not self._labels:
                self._enabled = False
                return self
            pl.start(self._eventset)
        except Exception:
            self._enabled = False
        return self

    def __exit__(self, *args):
        if not self._enabled or self._eventset is None:
            return
        try:
            pl.stop(self._eventset)
            pl.cleanup_eventset(self._eventset)
            pl.destroy_eventset(self._eventset)
        except Exception:
            pass

    def read(self) -> Dict[str, int]:
        if not self._enabled or self._eventset is None:
            return {}
        try:
            vals = pl.read(self._eventset)
            return {label: int(v) for label, v in zip(self._labels, vals)}
        except Exception:
            return {}


def is_available() -> bool:
    return _PAPI_AVAILABLE
