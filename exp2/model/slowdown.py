"""Slowdown models M0-M4 (slice E1a, contract K12, schema exp2.slowdown/1). numpy + stdlib only.

s_i(A, f): the slowdown of type i when the set A (which includes i) runs together at GPU level f.
  M0     n / c(f), one capacity per level            M1  1 + beta_family(f) (n - 1)
  M2     (1-u_i) + u_i max(1, U_A / eta(f))          M3  M2 * (1 + sum_{l in A\\i} kappa_family(l)(f))
  M4lin  per-family ridge on [type one-hot, f, U_A, n]     M4  per-family 64-32 ReLU MLP on the same features
Selection (D32-7): leave-one-cell-out MdAPE over the training cells; held-out cells are only reported, per cell.
Fit rows are corun.csv rows (dicts, the target is s_p50). A fit that sees a split == 'heldout' row raises HeldOutError.
CLI: python -m exp2.model.slowdown --run1-csv DIR --types T --out M.json
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from exp2 import contracts as C
from exp2.model.mlp import MLP
from exp2.stats import mape

ORDER = ("M1", "M2", "M3", "M4lin", "M4")   # simplest first; M0 is the baseline
TARGET_MAPE = 10.0


# ------------------------------------------------------------------------------------------------ helpers
def _rows(raw, check=True):
    """corun dict rows -> list of (type_id, family, A, f_mhz, s). HeldOutError on a heldout row when check."""
    out = []
    for r in raw:
        if check and r.get("split") == "heldout":
            raise C.HeldOutError("fit received a split == 'heldout' row")
        A = r["A"]
        A = tuple(A.split("|")) if isinstance(A, str) else tuple(A)
        tid = r["type_id"]
        out.append((tid, r.get("family") or tid.split(":")[0], A, int(float(r["f_mhz"])), float(r["s_p50"])))
    return out


def _near(d, f, default=None):
    if not d:
        return default
    return d[f] if f in d else d[min(d, key=lambda k: abs(k - f))]


def _near_fam(d, fam, f, default):
    ks = [k for k in d if k[0] == fam]
    return default if not ks else d[min(ks, key=lambda k: abs(k[1] - f))]


def _pair_json(d):
    return {f"{k[0]}|{k[1]}": v for k, v in d.items()}


def _pair_load(d):
    return {(k.split("|")[0], int(k.split("|")[1])): float(v) for k, v in d.items()}


def _int_json(d):
    return {str(k): v for k, v in d.items()}


def _int_load(d):
    return {int(k): float(v) for k, v in d.items()}


# ------------------------------------------------------------------------------------------------ base
class _Base:
    name = ""

    def __init__(self, seed=0):
        self.seed = seed
        self.types = None
        self.residuals = {}
        self.heldout_mape = {}
        self.heldout_cells = {}
        self.selection = None
        self.fit_rows_sha256 = "0" * 64

    # -- helpers on the type table
    def _u(self, tid, f):
        return _near(self.types.get(tid).u, f)

    def _U(self, A, f):
        return sum(self._u(t, f) for t in A)

    # -- K12
    def fit(self, corun_rows, solo_rows, types):
        rows = _rows(corun_rows)
        self.types = types
        self._fit(rows)
        self.residuals = {}
        for tid, fam, A, f, s in rows:
            if len(A) > 1:
                self.residuals.setdefault(f"{fam}|{len(A)}", []).append(s / self.predict(tid, A, f))
        blob = json.dumps([[r[0], list(r[2]), r[3], r[4]] for r in rows], separators=(",", ":"))
        self.fit_rows_sha256 = hashlib.sha256(blob.encode("ascii")).hexdigest()
        return self

    def predict(self, i, A, f_mhz):
        if len(A) < 1:
            raise ValueError("A must contain the type itself")
        if len(A) == 1:
            return 1.0
        return max(1.0, float(self._pred(i, tuple(A), int(f_mhz))))

    def sample(self, i, A, f_mhz, rng):
        base = self.predict(i, A, f_mhz)
        if len(A) == 1:
            return base
        fam = i.split(":")[0]
        pool = self.residuals.get(f"{fam}|{len(A)}")
        if not pool:
            pool = [x for k in sorted(self.residuals) if k.split("|")[0] == fam for x in self.residuals[k]]
        if not pool:
            return base
        k = int(rng.integers(len(pool))) if hasattr(rng, "integers") else int(rng.randrange(len(pool)))
        return base * pool[k]

    def to_json(self):
        obj = {"schema": C.SCHEMAS["slowdown"], "name": self.name, "params": self._dump(),
               "residuals": {k: list(v) for k, v in self.residuals.items()},
               "heldout_mape": dict(self.heldout_mape), "fit_rows_sha256": self.fit_rows_sha256}
        if self.selection:
            obj["selection"] = self.selection
        return obj


# ------------------------------------------------------------------------------------------------ M0, M1
class M0(_Base):
    name = "M0"

    def _fit(self, rows):
        by = {}
        for _, _, A, f, s in rows:
            if len(A) > 1:
                by.setdefault(f, []).append(len(A) / s)
        self.c = {f: float(np.mean(v)) for f, v in by.items()} or {0: 1.0}

    def _pred(self, i, A, f):
        return len(A) / _near(self.c, f)

    def _dump(self):
        return {"c": _int_json(self.c)}

    def _load(self, p):
        self.c = _int_load(p["c"])


class M1(_Base):
    name = "M1"

    def _fit(self, rows):
        num, den = {}, {}
        for _, fam, A, f, s in rows:
            if len(A) > 1:
                k = (fam, f)
                num[k] = num.get(k, 0.0) + (s - 1.0) * (len(A) - 1)
                den[k] = den.get(k, 0.0) + (len(A) - 1) ** 2
        self.beta = {k: num[k] / den[k] for k in num}

    def _pred(self, i, A, f):
        fam = i.split(":")[0]
        dflt = float(np.mean(list(self.beta.values()))) if self.beta else 0.0
        return 1.0 + _near_fam(self.beta, fam, f, dflt) * (len(A) - 1)

    def _dump(self):
        return {"beta": _pair_json(self.beta)}

    def _load(self, p):
        self.beta = _pair_load(p["beta"])


# ------------------------------------------------------------------------------------------------ M2, M3
class M2(_Base):
    name = "M2"
    _GRID = np.arange(0.20, 1.0005, 0.001)

    def _m2(self, i, A, f):
        u = self._u(i, f)
        return (1.0 - u) + u * max(1.0, self._U(A, f) / _near(self.eta, f, 1.0))

    def _fit(self, rows):
        by = {}
        for tid, _, A, f, s in rows:
            if len(A) > 1:
                by.setdefault(f, []).append((self._u(tid, f), self._U(A, f), s))
        self.eta = {}
        for f, v in by.items():
            u, U, s = (np.array(x) for x in zip(*v))
            pred = (1.0 - u[:, None]) + u[:, None] * np.maximum(1.0, U[:, None] / self._GRID[None, :])
            loss = (((pred - s[:, None]) / s[:, None]) ** 2).sum(axis=0)
            best = np.flatnonzero(loss <= loss.min() + 1e-12)
            self.eta[f] = float(self._GRID[best[-1]])   # flat loss (never oversubscribed): the largest eta

    def _pred(self, i, A, f):
        return self._m2(i, A, f)

    def _dump(self):
        return {"eta": _int_json(self.eta)}

    def _load(self, p):
        self.eta = _int_load(p["eta"])


class M3(M2):
    name = "M3"

    def _fit(self, rows):
        super()._fit(rows)
        acc = {}
        for tid, _, A, f, s in rows:
            if len(A) == 2:
                other = list(A)
                other.remove(tid)
                acc.setdefault((other[0].split(":")[0], f), []).append(s / self._m2(tid, A, f) - 1.0)
        self.kappa = {k: float(np.mean(v)) for k, v in acc.items()}

    def _pred(self, i, A, f):
        others = list(A)
        others.remove(i)
        k = sum(_near_fam(self.kappa, o.split(":")[0], f, 0.0) for o in others)
        return self._m2(i, A, f) * max(0.1, 1.0 + k)

    def _dump(self):
        return {"eta": _int_json(self.eta), "kappa": _pair_json(self.kappa)}

    def _load(self, p):
        super()._load(p)
        self.kappa = _pair_load(p["kappa"])


# ------------------------------------------------------------------------------------------------ M4lin, M4
class _PerFamily(_Base):
    """Features [type one-hot, f / f_max, U_A, |A|]; one regressor per family plus a pooled fallback '*'."""

    def _x(self, i, A, f):
        oh = [1.0 if t == i else 0.0 for t in self.type_ids]
        return np.array(oh + [f / self.f_max, self._U(A, f), float(len(A))])

    def _fit(self, rows):
        self.type_ids, self.f_max = self.types.type_ids(), self.types.f_max_mhz
        rows = [r for r in rows if len(r[2]) > 1]
        groups = {"*": rows}
        for r in rows:
            groups.setdefault(r[1], []).append(r)
        self.state = {}
        for key, rs in groups.items():
            X = np.array([self._x(t, A, f) for t, _, A, f, _ in rs])
            y = np.array([r[4] for r in rs])
            mu, sd = X.mean(axis=0), X.std(axis=0)
            sd[sd == 0.0] = 1.0
            self.state[key] = {"mu": mu, "sd": sd, "reg": self._reg((X - mu) / sd, y)}

    def _pred(self, i, A, f):
        st = self.state.get(i.split(":")[0]) or self.state["*"]
        z = ((self._x(i, A, f) - st["mu"]) / st["sd"])[None, :]
        return float(self._apply(st["reg"], z)[0])

    def _dump(self):
        return {"type_ids": list(self.type_ids), "f_max": self.f_max,
                "state": {k: {"mu": s["mu"].tolist(), "sd": s["sd"].tolist(), "reg": self._reg_dump(s["reg"])}
                          for k, s in self.state.items()}}

    def _load(self, p):
        self.type_ids, self.f_max = list(p["type_ids"]), p["f_max"]
        self.state = {k: {"mu": np.array(s["mu"]), "sd": np.array(s["sd"]), "reg": self._reg_load(s["reg"])}
                      for k, s in p["state"].items()}


class M4lin(_PerFamily):
    name = "M4lin"
    LAM = 1.0

    def _reg(self, Z, y):
        b0 = float(y.mean())
        w = np.linalg.solve(Z.T @ Z + self.LAM * np.eye(Z.shape[1]), Z.T @ (y - b0))
        return {"w": w, "b0": b0}

    def _apply(self, reg, Z):
        return Z @ reg["w"] + reg["b0"]

    def _reg_dump(self, reg):
        return {"w": reg["w"].tolist(), "b0": reg["b0"]}

    def _reg_load(self, d):
        return {"w": np.array(d["w"]), "b0": d["b0"]}


class M4(_PerFamily):
    name = "M4"

    def _reg(self, Z, y):
        hi = max(1.25 * float(y.max()), 1.05)
        return MLP(Z.shape[1], (64, 32), l2=1e-4, seed=self.seed, lo=1.0, hi=hi).fit(Z, y)

    def _apply(self, reg, Z):
        return reg.predict(Z)

    def _reg_dump(self, reg):
        return reg.to_dict()

    def _reg_load(self, d):
        return MLP.from_dict(d)


MODELS = {c.name: c for c in (M0, M1, M2, M3, M4lin, M4)}


def make(name, seed=0):
    return MODELS[name](seed=seed)


def from_json(obj, types):
    """Rebuild a fitted model from its exp2.slowdown/1 dict (needs the type table for u_i(f))."""
    C.validate_slowdown(obj)
    m = MODELS[obj["name"]]()
    m.types = types
    m._load(obj["params"])
    m.residuals = {k: list(v) for k, v in obj["residuals"].items()}
    m.heldout_mape = dict(obj["heldout_mape"])
    m.fit_rows_sha256 = obj["fit_rows_sha256"]
    m.selection = obj.get("selection")
    return m


# ------------------------------------------------------------------------------------------------ selection
def evaluate_heldout(train_rows, heldout_rows, solo_rows, types, seed=0, names=("M0",) + ORDER):
    """Fit every model on train_rows and return {name: MAPE (%) of s_p50 predictions on heldout_rows}."""
    held = _rows(heldout_rows, check=False)
    meas = [r[4] for r in held]
    out = {}
    for name in names:
        m = make(name, seed).fit(train_rows, solo_rows, types)
        out[name] = mape([m.predict(t, A, f) for t, _, A, f, _ in held], meas)
    return out


def select_model(mapes):
    """First of M1 < M2 < M3 < M4lin < M4 with MAPE <= 10 and below M0 -> (name, True); else lowest MAPE, False."""
    for name in ORDER:
        if name in mapes and mapes[name] <= TARGET_MAPE and mapes[name] < mapes["M0"]:
            return name, True
    return min(mapes, key=lambda k: mapes[k]), False


def _cell_of(r):
    return r.get("cell_id") or (r["A"] if isinstance(r["A"], str) else "|".join(r["A"]))


def loco_mdape(train_rows, solo_rows, types, seed=0, names=("M0",) + ORDER):
    """D32-7 leave-one-cell-out CV over the training cells (a cell is left out with all its levels together).
    -> ({name: MdAPE (%) of s_p50 over all left-out rows}, {name: {cell_id: median APE (%) of that cell}})."""
    cells = sorted({_cell_of(r) for r in train_rows})
    errs = {n: [] for n in names}
    per_cell = {n: {} for n in names}
    for cell in cells:
        fit_rows = [r for r in train_rows if _cell_of(r) != cell]
        held = _rows([r for r in train_rows if _cell_of(r) == cell], check=False)
        for n in names:
            m = make(n, seed).fit(fit_rows, solo_rows, types)
            ape = [100.0 * abs(m.predict(t, A, f) - y) / y for t, _, A, f, y in held]
            errs[n] += ape
            per_cell[n][cell] = float(np.median(ape))
    return {n: float(np.median(v)) for n, v in errs.items()}, per_cell


def _heldout_cells(model, held, solo_rows, train, types, seed):
    by = {}
    for r in held:
        by.setdefault(_cell_of(r), []).append(r)
    out = {}
    for c, v in sorted(by.items()):
        out[c] = [(t, f, 100.0 * abs(model.predict(t, A, f) - y) / y) for t, _, A, f, y in _rows(v, check=False)]
    return {c: float(np.median([x[2] for x in v])) for c, v in out.items()}


def fit_select(corun_rows, solo_rows, types, seed=0, names=("M0",) + ORDER):
    """D32-7: select by LOCO MdAPE over the split != 'heldout' cells, refit the choice on all of them. Held-out rows
    (if any) are only reported: model.heldout_mape (pooled MAPE per model) and model.heldout_cells
    {name: {cell_id: median APE (%)}}; no bootstrap over cells. -> (model, target_met)."""
    train = [r for r in corun_rows if r.get("split") != "heldout"]
    held = [r for r in corun_rows if r.get("split") == "heldout"]
    mdape, _ = loco_mdape(train, solo_rows, types, seed, names)
    name, met = select_model(mdape)
    model = make(name, seed).fit(train, solo_rows, types)
    model.selection = {"method": "loco", "loco_mdape": dict(mdape), "selected": name}
    model.heldout_mape, model.heldout_cells = {}, {}
    if held:
        model.heldout_mape = evaluate_heldout(train, held, solo_rows, types, seed, names)
        for n in names:
            m = model if n == name else make(n, seed).fit(train, solo_rows, types)
            model.heldout_cells[n] = _heldout_cells(m, held, solo_rows, train, types, seed)
    return model, met


def main(argv=None):
    ap = argparse.ArgumentParser(prog="exp2.model.slowdown")
    ap.add_argument("--run1-csv", required=True)
    ap.add_argument("--types", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    d = Path(a.run1_csv)
    with open(d / "corun.csv", newline="") as fh:
        corun = list(csv.DictReader(fh))
    with open(d / "solo.csv", newline="") as fh:
        solo = list(csv.DictReader(fh))
    model, met = fit_select(corun, solo, C.load_types(a.types), a.seed)
    obj = C.validate_slowdown(model.to_json())
    out = Path(a.out)
    out.write_text(json.dumps(obj, indent=2) + "\n")
    if model.heldout_cells:
        out.with_name(out.stem + "_heldout_cells.json").write_text(json.dumps(model.heldout_cells, indent=2) + "\n")
    print(f"selected {model.name} target_met={met} loco_mdape={model.selection['loco_mdape']} "
          f"heldout_mape={model.heldout_mape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
