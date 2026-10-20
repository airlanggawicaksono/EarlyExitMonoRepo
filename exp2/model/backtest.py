"""Back-test before the freeze (D32-6, schema exp2.backtest/1).

For every held-out co-run group (split == "heldout" in corun.csv; group = cell_id x f_mhz x repeat) predict each
tenant's median service time and the board energy of the measured window, and compare with the measurement.

Prediction method: "direct". The profiling condition keeps exactly one request per tenant in service, back to back,
so the set in service A is constant for the whole window. simulator.simulate() models open arrivals with queueing,
deadlines and admission, and cannot express this closed loop without a dummy policy and workload. For a constant A
the simulation reduces to
    tau_pred(i)  = tau_solo_p50(i, f) * s_model(i, A, f)          (slowdown model, median)
    energy_pred  = P(A, f) * window_s                              (power model, window length of the measurement)
so both are computed directly. Measured tau = s_p50 * tau_solo_p50 of the row. Repeats of one (cell, f, type) are
aggregated by their median (tau) or mean (energy) on both sides; ape = 100 * |pred - meas| / meas.

CLI: python -m exp2.model.backtest --run1-csv DIR --types T --slowdown M --power P --out DIR   (exit 0 or 2)
"""
import argparse
import json
import os
import statistics
import sys

from exp2 import contracts as C
from exp2.model import power as P
from exp2.model import slowdown as S

METHOD = "direct"


def _ape(pred, meas):
    return 100.0 * abs(pred - meas) / meas


def _med(xs):
    return statistics.median(xs) if xs else None


def backtest(corun_rows, types, slowdown, power):
    """corun_rows: dicts of corun.csv; only split == "heldout" rows are used. Returns an exp2.backtest/1 dict."""
    held = [r for r in corun_rows if r.get("split") == "heldout"]
    tau, groups = {}, {}
    for r in held:
        f = int(float(r["f_mhz"]))
        A = tuple(r["A"].split("|")) if isinstance(r["A"], str) else tuple(r["A"])
        tid = r["type_id"]
        solo = float(r["tau_solo_p50"])
        try:
            solo = float(types.tau(tid, f, "p50"))
        except Exception:  # level missing in the type table: use the solo median measured with the row
            pass
        pred = solo * slowdown.predict(tid, A, f)
        meas = float(r["s_p50"]) * float(r["tau_solo_p50"])
        tau.setdefault((r["cell_id"], f, tid), []).append((pred, meas))
        groups[(r["cell_id"], f, r["repeat"])] = (power.p(A, f) * float(r["window_s"]), float(r["energy_j"]))
    cells = []
    for (cid, f, tid), pm in sorted(tau.items()):
        p, m = _med([x[0] for x in pm]), _med([x[1] for x in pm])
        cells.append({"cell_id": cid, "f_mhz": f, "type_id": tid, "pred_tau_p50_s": p, "meas_tau_p50_s": m,
                      "ape": _ape(p, m)})
    en = {}
    for (cid, f, _rep), pm in groups.items():
        en.setdefault((cid, f), []).append(pm)
    energy = []
    for (cid, f), pm in sorted(en.items()):
        p, m = statistics.fmean(x[0] for x in pm), statistics.fmean(x[1] for x in pm)
        energy.append({"cell_id": cid, "f_mhz": f, "pred_j": p, "meas_j": m, "ape": _ape(p, m)})
    out = {"schema": C.SCHEMAS["backtest"], "cells": cells, "energy": energy,
           "summary": {"tau_mdape": _med([c["ape"] for c in cells]), "energy_mdape": _med([e["ape"] for e in energy])}}
    return C.validate_backtest(out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="exp2.model.backtest")
    for a in ("--run1-csv", "--types", "--slowdown", "--power", "--out"):
        ap.add_argument(a, required=True)
    try:
        args = ap.parse_args(argv)
    except SystemExit:
        return 2
    try:
        types = C.load_types(args.types)
        with open(args.slowdown, encoding="utf-8") as fh:
            slow = S.from_json(json.load(fh), types)
        with open(args.power, encoding="utf-8") as fh:
            pw = P.PowerModel.from_json(json.load(fh), types)
        rows = C.read_csv(os.path.join(args.run1_csv, "corun.csv"), "corun")
        res = backtest(rows, types, slow, pw)
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "backtest.json"), "w", encoding="utf-8", newline="\n") as fh:
            json.dump(res, fh, indent=2)
            fh.write("\n")
    except (OSError, ValueError, KeyError) as e:
        print("backtest: %s" % e, file=sys.stderr)
        return 2
    print("backtest method=%s tau_mdape=%s energy_mdape=%s" % (METHOD, res["summary"]["tau_mdape"],
                                                               res["summary"]["energy_mdape"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
