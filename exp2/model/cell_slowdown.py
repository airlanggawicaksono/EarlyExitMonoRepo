"""CellSlowdown: the measured Run 1 co-run cells as the simulator's ground-truth slowdown (CoExit pilot).

M1 sees only |A| and the family, so it cannot tell which exit runs (1020 MHz next to bert:5: llama:15 measured
about 1.17, llama:3 about 1.61, M1 gives 1.50 for both). The policies keep predicting with the fitted model; the
simulator reads its truth here. Same interface as exp2.model.slowdown: predict(i, A, f_mhz), sample(i, A, f_mhz, rng).

Key = (tuple(sorted(A)), int(f_mhz), type_id), built from corun.csv (A = type ids joined by "|") and
corun_samples.csv (cell_id -> A through corun.csv). Every split is used: this is measured truth, not a fit.
  predict  median of s_p50 over every row (cell, repeat) of the key
  sample   one uniform draw from the pooled per-request samples of the key
A key with no measurement goes to `fallback` (any object with predict / sample); coverage() counts those calls.
A keeps duplicates: two jobs of one type form a set no Run 1 cell measured, so that key goes to the fallback.
"""
import statistics
from pathlib import Path

from exp2 import contracts as C


def _key(i, A, f_mhz):
    return tuple(sorted(A)), int(f_mhz), i


class CellSlowdown:
    def __init__(self, s_p50, samples, fallback):
        """s_p50, samples: {key: [float]}; fallback: predict / sample for every key absent from them."""
        self.median = {k: float(statistics.median(v)) for k, v in s_p50.items()}
        self.samples = {k: tuple(float(x) for x in v) for k, v in samples.items()}
        self.fallback = fallback
        self.fallback_calls = 0

    @classmethod
    def from_run1_dir(cls, run1_dir, fallback):
        d = Path(run1_dir)
        cell_A, s_p50 = {}, {}
        for r in C.read_csv(d / "corun.csv", "corun"):
            A = tuple(r["A"].split("|"))
            if cell_A.setdefault(r["cell_id"], A) != A:
                raise ValueError("corun.csv: cell %s has two sets %s and %s" % (r["cell_id"], cell_A[r["cell_id"]], A))
            s_p50.setdefault(_key(r["type_id"], A, r["f_mhz"]), []).append(r["s_p50"])
        samples = {}
        for r in C.read_csv(d / "corun_samples.csv", "corun_samples"):
            if r["cell_id"] not in cell_A:
                raise ValueError("corun_samples.csv: cell %s is not in corun.csv" % r["cell_id"])
            samples.setdefault(_key(r["type_id"], cell_A[r["cell_id"]], r["f_mhz"]), []).append(r["s"])
        return cls(s_p50, samples, fallback)

    @staticmethod
    def _alone(A):
        if len(A) < 1:
            raise ValueError("A must contain the type itself")
        return len(A) == 1

    def predict(self, i, A, f_mhz):
        if self._alone(A):
            return 1.0
        s = self.median.get(_key(i, A, f_mhz))
        if s is not None:
            return s
        self.fallback_calls += 1
        return self.fallback.predict(i, A, f_mhz)

    def sample(self, i, A, f_mhz, rng):
        if self._alone(A):
            return 1.0
        pool = self.samples.get(_key(i, A, f_mhz))
        if not pool:
            self.fallback_calls += 1
            return self.fallback.sample(i, A, f_mhz, rng)
        n = len(pool)
        return pool[int(rng.integers(n)) if hasattr(rng, "integers") else int(rng.randrange(n))]

    def coverage(self):
        return {"measured_keys": len(self.median), "fallback_calls": self.fallback_calls}
