"""Power model K13: p(A, f) = p_idle(f) + dp(f) * min(1, U_A / eta(f)), U_A = sum_i u_i(f)."""
import argparse
import json
import sys

from exp2 import contracts as C

SCHEMA = "exp2.power/1"


class PowerModel:
    def __init__(self, p_idle_w, dp_w, eta, types):
        self.p_idle_w, self.dp_w, self.eta, self.types = dict(p_idle_w), dict(dp_w), dict(eta), types

    def p(self, A, f_mhz):
        f = int(f_mhz)
        if not A:
            return self.p_idle_w[f]
        u_a = sum(self.types.get(t).u[f] for t in A)
        return self.p_idle_w[f] + self.dp_w[f] * min(1.0, u_a / self.eta[f])

    def to_json(self):
        key = lambda d: {str(k): d[k] for k in sorted(d)}  # noqa: E731
        return {"schema": SCHEMA, "p_idle_w": key(self.p_idle_w), "dp_w": key(self.dp_w), "eta": key(self.eta)}


def fit_power(idle_rows, corun_rows, eta, types):
    """p_idle(f) = resident idle power; dp(f) by least squares through the origin on p_board_w - p_idle against
    min(1, U_A / eta) over the co-run rows (one row per cell, level, repeat)."""
    p_idle = {int(r["f_mhz"]): float(r["p_idle_w"]) for r in idle_rows if r["kind"] == "resident"}
    eta = {int(k): float(v) for k, v in eta.items()}
    seen, num, den = set(), {}, {}
    for r in corun_rows:
        f = int(r["f_mhz"])
        key = (r["cell_id"], f, r["repeat"])
        if key in seen or f not in p_idle:
            continue
        seen.add(key)
        x = min(1.0, float(r["U_A"]) / eta[f])
        num[f] = num.get(f, 0.0) + x * (float(r["p_board_w"]) - p_idle[f])
        den[f] = den.get(f, 0.0) + x * x
    dp = {f: num[f] / den[f] for f in num if den[f] > 0.0}
    return PowerModel({f: p_idle[f] for f in dp}, dp, {f: eta[f] for f in dp}, types)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="exp2.model.power")
    ap.add_argument("--run1-csv", required=True)
    ap.add_argument("--types", required=True)
    ap.add_argument("--slowdown", required=True)
    ap.add_argument("--out", required=True)
    try:
        args = ap.parse_args(argv)
    except SystemExit:
        return 2
    types = C.load_types(args.types)
    idle = C.read_csv(args.run1_csv + "/idle.csv", "idle")
    corun = C.read_csv(args.run1_csv + "/corun.csv", "corun")
    with open(args.slowdown, encoding="utf-8") as fh:
        eta = json.load(fh)["params"]["eta"]
    model = fit_power(idle, corun, eta, types)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(model.to_json(), fh, indent=2)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
