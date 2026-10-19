"""FakeAdapter (BRIEF_v1 K6): a FamilyAdapter that waits instead of computing.

It is what `--adapter fake` runs in the server subprocess and what the unit tests drive. No torch, no GPU.

Spec (exp2.fake_spec/1, validated with contracts.validate_fake_spec):
    {"schema": "exp2.fake_spec/1", "family": "bert", "clock": "sleep", "load_s": 0.05, "first_infer_s": 0.02,
     "types": {"5": {"seg_ms": [20.0, 20.0], "head_ms": 5.0}}}
`seg_ms` holds one duration per default checkpoint of the type; `head_ms` is what finish(reached) costs.

Clock modes
    "sleep"    time.sleep for every wait (integration runs).
    "virtual"  advance the clock object passed to the constructor (anything with advance_ns(int), e.g.
               tests.mocks.FakeClock) and never sleep. The server must read time from the same clock.
Every method call is appended to `calls`:  ("load", exits) ("make_inputs", n, seed, mode) ("begin", input_idx, exit)
("advance", ckpt) ("sync",) ("finish", reached) ("mem_stats",) ("reset_peak",) ("checkpoints", exit).

Behaviour the server relies on
    * The first begin() of each exit after load() also waits first_infer_s (the compile/cuDNN warm-up cost).
    * advance(state, ckpt) runs every segment from the next unfinished checkpoint up to `ckpt`, so a jump straight to
      the target (--no-checkpoints) costs the sum of the skipped segments. ckpt must be a checkpoint after the last
      one advanced.
    * finish(state, reached) needs reached to be a checkpoint that finished; None does nothing and costs nothing.
    * mem_stats(): torch_reserved_mb = 1000 + 10 per loaded exit, rss_mb = 2000, torch_peak_alloc_mb = 900 +
      the largest activation since reset_peak(), where an exit's activation is 2 + 0.5 * depth MB.
"""
import json
import time

from exp2 import contracts

BASE_ALLOC_MB = 900.0


def default_spec(family, exits, *, clock="sleep", seg_ms=None, head_ms=5.0, load_s=0.05, first_infer_s=0.02):
    """A valid fake spec for `family` and `exits`: every segment takes seg_ms (20 ms, 2 ms for llama's 32 tokens)."""
    seg = seg_ms if seg_ms is not None else (2.0 if family == "llama" else 20.0)
    types = {}
    for e in exits:
        n = len(contracts.default_checkpoints(family, e))
        types[str(e)] = {"seg_ms": [seg] * n, "head_ms": 0.0 if family == "llama" else head_ms}
    return {"schema": contracts.SCHEMAS["fake_spec"], "family": family, "clock": clock, "load_s": load_s,
            "first_infer_s": first_infer_s, "types": types}


class FakeState:
    """Per-request state: how many checkpoints have finished."""

    __slots__ = ("exit", "input_idx", "ckpts", "n_done")

    def __init__(self, exit, input_idx, ckpts):
        self.exit = exit
        self.input_idx = input_idx
        self.ckpts = ckpts
        self.n_done = 0

    @property
    def done(self):
        return self.ckpts[:self.n_done]


class FakeAdapter:
    def __init__(self, spec, *, clock=None):
        contracts.validate_fake_spec(spec)
        self.spec = spec
        self.family = spec["family"]
        self.mode = spec["clock"]
        if self.mode == "virtual" and clock is None:
            raise ValueError("a 'virtual' fake spec needs a clock object (anything with advance_ns)")
        self._clock = clock
        self.calls = []
        self.cfg = {}
        self._loaded = set()
        self._first_done = set()
        self._n_inputs = None
        self._peak_act = 0.0

    @classmethod
    def from_file(cls, path, *, clock=None):
        with open(path, encoding="utf-8") as fh:
            return cls(json.load(fh), clock=clock)

    # ---- helpers
    def _wait_ms(self, ms):
        ns = round(ms * 1_000_000)
        if ns <= 0:
            return
        if self.mode == "virtual":
            self._clock.advance_ns(ns)
        else:
            time.sleep(ns / 1e9)

    def _type(self, exit):
        if exit not in self._loaded:
            raise ValueError("exit %r was not loaded (loaded: %s)" % (exit, sorted(self._loaded)))
        return self.spec["types"][str(exit)]

    # ---- FamilyAdapter
    def load(self, exits, cfg):
        exits = list(exits)
        if not exits:
            raise ValueError("load() needs at least one exit")
        missing = [e for e in exits if str(e) not in self.spec["types"]]
        if missing:
            have = sorted(self.spec["types"], key=int)
            raise ValueError("exit(s) %s are not in the fake spec (have %s)" % (missing, have))
        self.calls.append(("load", tuple(exits)))
        self.cfg = dict(cfg)
        self._wait_ms(self.spec["load_s"] * 1000.0)
        self._loaded = set(exits)
        self._first_done = set()
        return {"load_s": self.spec["load_s"], "fake": True, "exits": exits}

    def make_inputs(self, n, seed, mode):
        if mode not in ("synthetic", "dataset"):
            raise ValueError("input mode %r not in ('synthetic', 'dataset')" % (mode,))
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError("n must be a positive int, got %r" % (n,))
        self.calls.append(("make_inputs", n, seed, mode))
        self._n_inputs = n

    def checkpoints(self, exit):
        self._type(exit)
        self.calls.append(("checkpoints", exit))
        return contracts.default_checkpoints(self.family, exit)

    def begin(self, input_idx, exit):
        self._type(exit)
        if self._n_inputs is None:
            raise RuntimeError("make_inputs() must run before begin()")
        if not isinstance(input_idx, int) or not 0 <= input_idx < self._n_inputs:
            raise ValueError("input_idx %r outside 0..%d" % (input_idx, self._n_inputs - 1))
        self.calls.append(("begin", input_idx, exit))
        if exit not in self._first_done:
            self._first_done.add(exit)
            self._wait_ms(self.spec["first_infer_s"] * 1000.0)
        self._peak_act = max(self._peak_act, 2.0 + 0.5 * contracts.depth(self.family, exit))
        return FakeState(exit, input_idx, contracts.default_checkpoints(self.family, exit))

    def advance(self, state, ckpt):
        segs = self._type(state.exit)["seg_ms"]
        if ckpt not in state.ckpts or state.ckpts.index(ckpt) < state.n_done:
            raise ValueError("ckpt %r is not a checkpoint after %s in %s" % (ckpt, state.done[-1:], state.ckpts))
        upto = state.ckpts.index(ckpt) + 1
        self.calls.append(("advance", ckpt))
        self._wait_ms(sum(segs[state.n_done:upto]))
        state.n_done = upto

    def sync(self):
        self.calls.append(("sync",))

    def finish(self, state, reached):
        self.calls.append(("finish", reached))
        if reached is None:
            return None
        if reached not in state.done:
            raise ValueError("reached %r is not a finished checkpoint (finished: %s)" % (reached, state.done))
        self._wait_ms(self._type(state.exit)["head_ms"])
        return {"family": self.family, "exit": state.exit, "reached": reached}

    def mem_stats(self):
        self.calls.append(("mem_stats",))
        return {"torch_reserved_mb": 1000.0 + 10.0 * len(self._loaded),
                "torch_peak_alloc_mb": BASE_ALLOC_MB + self._peak_act, "rss_mb": 2000.0}

    def reset_peak(self):
        self.calls.append(("reset_peak",))
        self._peak_act = 0.0
