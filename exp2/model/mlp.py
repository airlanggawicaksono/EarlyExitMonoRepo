"""Small numpy MLP for the M4 slowdown model (slice E1a).

64-32 ReLU network, L2 weight penalty, full-batch Adam, sigmoid output scaled to [lo, hi] (lo = 1, hi = s_max).
Deterministic for a given seed. numpy only.
"""
import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


class MLP:
    def __init__(self, n_in, hidden=(64, 32), l2=1e-3, seed=0, lo=1.0, hi=2.0):
        self.n_in, self.hidden, self.l2, self.seed = int(n_in), tuple(hidden), float(l2), int(seed)
        self.lo, self.hi = float(lo), float(hi)
        rng = np.random.default_rng(self.seed)
        sizes = (self.n_in,) + self.hidden + (1,)
        self.W = [rng.normal(0.0, np.sqrt(2.0 / a), size=(a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]

    def _forward(self, X):
        acts = [X]
        for k, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b
            acts.append(z if k == len(self.W) - 1 else np.maximum(z, 0.0))
        return acts

    def predict(self, X):
        z = self._forward(np.asarray(X, dtype=float))[-1][:, 0]
        return self.lo + (self.hi - self.lo) * _sigmoid(z)

    def fit(self, X, y, epochs=1500, lr=0.01):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        p = min(max((float(y.mean()) - self.lo) / (self.hi - self.lo), 1e-3), 1 - 1e-3)
        self.b[-1][:] = np.log(p / (1.0 - p))  # start at the mean target
        params = self.W + self.b
        m = [np.zeros_like(a) for a in params]
        v = [np.zeros_like(a) for a in params]
        n, depth = len(y), len(self.W)
        for t in range(1, epochs + 1):
            acts = self._forward(X)
            sg = _sigmoid(acts[-1][:, 0])
            out = self.lo + (self.hi - self.lo) * sg
            delta = (2.0 / n * (out - y) * (self.hi - self.lo) * sg * (1.0 - sg))[:, None]
            gW, gb = [None] * depth, [None] * depth
            for k in range(depth - 1, -1, -1):
                gW[k] = acts[k].T @ delta + 2.0 * self.l2 * self.W[k]
                gb[k] = delta.sum(axis=0)
                if k:
                    delta = (delta @ self.W[k].T) * (acts[k] > 0)
            for j, g in enumerate(gW + gb):
                m[j] = 0.9 * m[j] + 0.1 * g
                v[j] = 0.999 * v[j] + 0.001 * g * g
                params[j] -= lr * (m[j] / (1 - 0.9 ** t)) / (np.sqrt(v[j] / (1 - 0.999 ** t)) + 1e-8)
        return self

    def to_dict(self):
        return {"n_in": self.n_in, "hidden": list(self.hidden), "l2": self.l2, "seed": self.seed,
                "lo": self.lo, "hi": self.hi, "W": [w.tolist() for w in self.W], "b": [b.tolist() for b in self.b]}

    @classmethod
    def from_dict(cls, d):
        net = cls(d["n_in"], d["hidden"], d["l2"], d["seed"], d["lo"], d["hi"])
        net.W = [np.array(w, dtype=float) for w in d["W"]]
        net.b = [np.array(b, dtype=float) for b in d["b"]]
        return net
