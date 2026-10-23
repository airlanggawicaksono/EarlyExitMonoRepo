"""S2 tests for exp2.analyze.run3 (PLAN PL2 d, e). Fakes in tmp_path only: no board, no model, no simulator.

    trace     to_trace(run3_rows) + coexit_pilot.metrics == the fixture "expected"
    session   2 conditions: contract headers, rows = runs x (families + 1), deadline_s 0.1, screen, compare
    skips     invalid manifest and duplicate condition -> kept 0; a session without session.json is still read
    asserts   one relative deadline per family; usage errors exit 2
"""
import copy
import csv
import json

import pytest

from exp2 import contracts as C
from exp2 import exits as X
from exp2.analyze import run3 as R
from exp2.model import coexit_pilot as P
from exp2.tests import mocks

FX = mocks.load_fixture("run3_rows.json")
SEED = 20261022
SPECS = X.pilot_specs(1020)
WORKLOAD = {"name": "mixA", "types": ["bert:23", "yolo:5"], "probs": [0.5, 0.5], "rho": 0.6, "rate_per_s": 8.5,
            "arrival": "poisson", "cv_a": 1.0}
CLOCK_COLS = list(C.CLOCKS_CSV_FIXED_COLS) + ["temp_tj_thermal_mc"]
SIM_VIOL = {"coexit": [0.1, 0.2, 0.3, 0.4], "edgeserving": [0.5, 0.5, 0.7, 0.7]}   # pilot reps 0..3, scale 1.0


# ------------------------------------------------------------------------------------------------ fakes
def _rows(cid, method, rep, error_req=None):
    rows = copy.deepcopy(FX["rows"])
    for r in rows:
        r.update(condition_id=cid, seed=SEED + rep, policy=method, policy_spec=SPECS[method].to_json())
        if r["req_id"] == error_req:
            r.update(outcome="error", err="fake server error", reached=None, completion=0.0, ckpt=[])
    return rows


def _manifest(cid, method, rep, valid):
    return {"schema": C.SCHEMAS["manifest"], "condition_id": cid, "mode": "open", "arm": "pinned", "gpu_mhz": 1020,
            "emc_mhz": 3199.0, "t_wall_start_ns": 1, "t_mono_start_ns": 0, "t_mono_end_ns": 2_000_000_000,
            "settle_s": 0.0, "valid": valid, "invalid_reason": None if valid else "server died: {}",
            "board_readback": None, "n_requests": 7, "n_warmup": FX["n_warmup"], "seed": SEED + rep,
            "policy_key": method, "policy_spec": SPECS[method].to_json(), "freeze_condition_id": "A_p_rho060",
            "workload": dict(WORKLOAD), "alpha": None, "cell": None, "split": None, "repeat": rep}


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _session(tmp_path, name, conds, session_json=True):
    """conds: (method, rep[, valid[, error_req]]). A 3-row clocks.csv inside the run window [0, 2 s]."""
    sdir = tmp_path / name
    (sdir / "telemetry").mkdir(parents=True)
    with open(sdir / "telemetry" / "clocks.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(CLOCK_COLS)
        for i, t in enumerate((500_000_000, 1_000_000_000, 1_500_000_000)):
            wr.writerow([t, t, 1_020_000_000, 1_020_000_000, 1_020_000_000, 500, 3199.0, 4_096_000 - 1024 * i,
                         1_000_000, 50_000 + 1000 * i])
    if session_json:
        (sdir / "session.json").write_text(json.dumps({"session_id": "sid_" + name}), encoding="utf-8")
    for c in conds:
        method, rep, valid, error_req = tuple(c) + (True, None)[len(c) - 2:]
        cid = "A_p_rho060_%s_r%d" % (method, rep)
        (sdir / cid).mkdir()
        (sdir / cid / "manifest.json").write_text(json.dumps(_manifest(cid, method, rep, valid)), encoding="utf-8")
        if valid:
            _write_jsonl(sdir / cid / "requests.jsonl", _rows(cid, method, rep, error_req))
    return sdir


def _pilot(tmp_path):
    """Fake pilot dir: reps.csv (scale 1.0 and a 0.75 decoy) and summary.csv from coexit_pilot.summarize."""
    d = tmp_path / "pilot"
    d.mkdir()
    rows = []
    for scale in (0.75, 1.0):
        for m, vs in SIM_VIOL.items():
            for rep, v in enumerate(vs):
                for fam in ("bert", "yolo", "all"):
                    row = dict.fromkeys(C.COEXIT_PILOT_COLS)
                    row.update(method=m, mix="A", arrival="poisson", cv_a=1.0, rho=0.6, rate_per_s=8.5,
                               deadline_scale=scale, rep=rep, seed=SEED + rep, family=fam, n=10,
                               deadline_s=None if fam == "all" else 0.1 * scale,
                               slo_violation_ratio=v if scale == 1.0 else 0.9, mean_norm_depth=0.5,
                               sim_runtime_s=0.01)
                    rows.append(row)
    with open(d / "reps.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(C.COEXIT_PILOT_COLS)
        wr.writerows([P._fmt(r[c]) for c in C.COEXIT_PILOT_COLS] for r in rows)
    P.summarize(d / "reps.csv", d / "summary.csv")
    return d


def _read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        return tuple(header), [dict(zip(header, r)) for r in rd]


# ------------------------------------------------------------------------------------------------ (1) trace
def test_to_trace_and_metrics_reproduce_the_fixture():
    trace = R.to_trace(FX["rows"])
    got = P.metrics(trace, FX["n_warmup"], FX["families"])
    for fam, exp in FX["expected"].items():
        for k, v in exp.items():
            assert got[fam][k] == pytest.approx(v, abs=FX["tolerance"]), (fam, k, got[fam][k])
    by = {t["req_id"]: t for t in trace}
    assert (by["b2"]["type_id"], by["b2"]["run_type_id"]) == ("bert:23", "bert:5")
    assert (by["b3"]["type_id"], by["b3"]["run_type_id"]) == ("bert:23", None)


# ------------------------------------------------------------------------------------------------ (2) session
def test_two_conditions_write_contract_files_and_compare(tmp_path):
    s1 = _session(tmp_path, "s1", [("coexit", 0), ("edgeserving", 0, True, "b1")])
    out = tmp_path / "out"
    assert R.main(["--sessions", str(s1), "--out", str(out), "--sim-dir", str(_pilot(tmp_path))]) == 0

    hr, reps = _read(out / "reps.csv")
    hs, summ = _read(out / "summary.csv")
    hu, runs = _read(out / "runs.csv")
    hc, comp = _read(out / "compare.csv")
    assert (hr, hs, hu, hc) == (C.COEXIT_PILOT_COLS, C.COEXIT_SUMMARY_COLS, C.RUN3_RUNS_COLS, C.RUN3_COMPARE_COLS)
    assert len(reps) == 2 * (2 + 1) and len(summ) == 6 and len(comp) == 6 and len(runs) == 2

    assert {(r["family"], r["deadline_s"]) for r in reps} == {("bert", "0.1"), ("yolo", "0.1"), ("all", "")}
    assert {(r["method"], r["mix"], r["arrival"], r["cv_a"], r["rho"], r["deadline_scale"], r["rep"], r["seed"],
             r["sim_runtime_s"]) for r in reps} == {
        (m, "A", "poisson", "1.0", "0.6", "1.0", "0", str(SEED), "2.0") for m in ("coexit", "edgeserving")}
    viol = {(r["method"], r["family"]): float(r["slo_violation_ratio"]) for r in reps}
    assert viol[("coexit", "all")] == pytest.approx(0.5)                # fixture expected
    assert viol[("edgeserving", "bert")] == pytest.approx(0.75)        # the error row b1 is a failure
    assert viol[("edgeserving", "all")] == pytest.approx(4 / 6)

    run = {r["condition_id"]: r for r in runs}
    co, es = run["A_p_rho060_coexit_r0"], run["A_p_rho060_edgeserving_r0"]
    assert (co["session_id"], co["kept"], co["reason"], co["n_rows"], co["n_error"]) == ("sid_s1", "1", "", "7", "0")
    assert es["n_error"] == "1"
    assert (co["screen_ok"], float(co["gpu_lock_frac"]), float(co["emc_lock_frac"])) == ("1", 1.0, 1.0)
    assert (float(co["max_tj_c"]), float(co["min_mem_available_mb"]), float(co["wall_s"])) == (52.0, 3998.0, 2.0)
    assert (float(co["p99_arrival_lag_ms"]), float(co["p99_dispatch_ms"])) == (0.0, 0.0)

    cmp_ = {(r["method"], r["family"]): r for r in comp}
    c = cmp_[("coexit", "all")]
    assert c["board_n_reps"] == "1" and float(c["board_viol_mean"]) == pytest.approx(0.5)
    assert float(c["sim_viol_mean"]) == pytest.approx(0.25)            # scale 0.75 decoy excluded
    assert float(c["sim_viol_same_seeds"]) == pytest.approx(0.1)       # only the board seed 20261022
    assert float(c["sim_d_viol_vs_es_mean"]) == pytest.approx(-0.35)
    assert float(c["board_d_viol_vs_es_mean"]) == pytest.approx(0.5 - 4 / 6)
    assert float(cmp_[("edgeserving", "bert")]["sim_viol_same_seeds"]) == pytest.approx(0.5)
    assert {k: r["same_sign"] for k, r in cmp_.items()} == {
        ("coexit", "bert"): "1", ("coexit", "yolo"): "0", ("coexit", "all"): "1",
        ("edgeserving", "bert"): "1", ("edgeserving", "yolo"): "1", ("edgeserving", "all"): "1"}


def test_invalid_and_duplicate_kept_zero_and_session_without_session_json_read(tmp_path):
    s1 = _session(tmp_path, "s1", [("coexit", 0), ("edgeserving", 0)])
    s2 = _session(tmp_path, "s2", [("coexit", 0), ("greedy", 0, False), ("all_early", 0)], session_json=False)
    out = tmp_path / "out"
    assert R.main(["--sessions", str(s1), str(s2), "--out", str(out)]) == 0
    _, runs = _read(out / "runs.csv")
    got = {(r["session_id"], r["condition_id"]): (r["kept"], r["reason"]) for r in runs}
    assert got == {("sid_s1", "A_p_rho060_coexit_r0"): ("1", ""),
                   ("sid_s1", "A_p_rho060_edgeserving_r0"): ("1", ""),
                   ("s2", "A_p_rho060_coexit_r0"): ("0", "duplicate"),
                   ("s2", "A_p_rho060_greedy_r0"): ("0", "invalid:server died: {}"),
                   ("s2", "A_p_rho060_all_early_r0"): ("1", "")}
    _, reps = _read(out / "reps.csv")
    assert len(reps) == 3 * (2 + 1)
    assert {r["method"] for r in reps} == {"coexit", "edgeserving", "all_early"}
    assert not (out / "compare.csv").exists()


# ------------------------------------------------------------------------------------------------ asserts, CLI
def test_two_relative_deadlines_in_one_family_exit_2(tmp_path, capsys):
    s1 = _session(tmp_path, "s1", [("coexit", 0)])
    path = s1 / "A_p_rho060_coexit_r0" / "requests.jsonl"
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()]
    rows[1]["t_deadline_ns"] += 1                     # b1 complete at 1.08 s, still on time
    _write_jsonl(path, rows)
    assert R.main(["--sessions", str(s1), "--out", str(tmp_path / "out")]) == 2
    assert "deadline" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [[], ["--out", "x"], ["--sessions", "s"]])
def test_usage_errors_exit_2(argv):
    assert R.main(argv) == 2
