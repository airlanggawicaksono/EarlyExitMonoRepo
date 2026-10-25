"""S2 tests for exp2.analyze.run3 (PLAN PL2 d, e). Fakes in tmp_path only: no board, no model, no simulator.

    trace     to_trace(run3_rows) + coexit_pilot.metrics == the fixture "expected"
    session   2 conditions: contract headers, rows = runs x (families + 1), deadline_s 0.1, screen, compare
    skips     invalid manifest and duplicate condition -> kept 0; a session without session.json is still read
    asserts   one relative deadline per family; usage errors exit 2
    lp        (PLAN PL3 S5) old Run 3 outputs byte-identical to HEAD 5a02eaa and no lp.csv; hand-made mix B LP rows ->
              hand-computed lp.csv; old Run 3 sessions + an LP session -> LP methods next to the Run 3 methods
"""
import copy
import csv
import hashlib
import json

import pytest

from exp2 import contracts as C
from exp2 import exits as X
from exp2.analyze import run3 as R
from exp2.model import coexit_pilot as P
from exp2.tests import mocks

FX = mocks.load_fixture("run3_rows.json")
SEED = 20261022
SPECS = X.run3_specs(1020)          # the five pilot methods + the coexit_nogate ablation
WORKLOAD = {"name": "mixA", "types": ["bert:23", "yolo:5"], "probs": [0.5, 0.5], "rho": 0.6, "rate_per_s": 8.5,
            "arrival": "poisson", "cv_a": 1.0}
CLOCK_COLS = list(C.CLOCKS_CSV_FIXED_COLS) + ["temp_tj_thermal_mc"]
SIM_VIOL = {"coexit": [0.1, 0.2, 0.3, 0.4], "edgeserving": [0.5, 0.5, 0.7, 0.7]}   # pilot reps 0..3, scale 1.0
# sha256 of the outputs of test_two_conditions_write_contract_files_and_compare at HEAD 5a02eaa (before lp.csv)
GOLDEN_OLD = {"compare.csv": "43b09464276fe2742110fdf493c197661493ad318a73edaf640703c0386c0abd",
              "reps.csv": "ceeafcb0ac8b1be67b993020225d65ac63e4589be2cf8b5aeec84064e4c0ae69",
              "runs.csv": "ea39e110e87c2ad1da89a204d0fb3ba4c218df8fb3ccf8e5ac299b03fbdcb3f4",
              "summary.csv": "13030490288d9551eefb31ba77bd10b6f3e2f5be11cd26d7a682acd631743318"}
MS = 1_000_000
B_WORKLOAD = {"name": "mixB", "types": ["bert:23", "llama:15", "yolo:5"], "probs": [0.4, 0.2, 0.4], "rho": 0.9,
              "rate_per_s": 2.0, "arrival": "poisson", "cv_a": 1.0}
# llama L1 token times (ms): prefill 60, then 50 per token; paused after token 3 [161, 249] and after token 6 [405, 406];
# the first token after a resume takes 55 and 52 ms; tokens 19..32 take 52. Gaps without a pause: 15 x 50, 14 x 52.
L1_TOKENS = ([60, 110, 160, 304, 354, 404, 458] + [458 + 50 * i for i in range(1, 12)]
             + [1008 + 52 * i for i in range(1, 15)])
L1_PAUSES = [[3, 130, 161, 249], [6, 380, 405, 406]]        # [tokens_done, t_req, t_pause, t_resume] ms


# ------------------------------------------------------------------------------------------------ fakes
def _spec(method):
    """The Run 3 specs, then the board LP specs (exits.board_specs, PLAN PL3)."""
    return SPECS[method] if method in SPECS else X.board_specs(1020)[method]


def _rows(cid, method, rep, error_req=None):
    rows = copy.deepcopy(FX["rows"])
    for r in rows:
        r.update(condition_id=cid, seed=SEED + rep, policy=method, policy_spec=_spec(method).to_json())
        if r["policy_spec"]["params"].get("preempt_llama_at_token"):
            r["pauses"] = []                          # every open row of an LP spec, [] when never paused
        if r["req_id"] == error_req:
            r.update(outcome="error", err="fake server error", reached=None, completion=0.0, ckpt=[])
    return rows


def _manifest(cid, method, rep, valid):
    return {"schema": C.SCHEMAS["manifest"], "condition_id": cid, "mode": "open", "arm": "pinned", "gpu_mhz": 1020,
            "emc_mhz": 3199.0, "t_wall_start_ns": 1, "t_mono_start_ns": 0, "t_mono_end_ns": 2_000_000_000,
            "settle_s": 0.0, "valid": valid, "invalid_reason": None if valid else "server died: {}",
            "board_readback": None, "n_requests": 7, "n_warmup": FX["n_warmup"], "seed": SEED + rep,
            "policy_key": method, "policy_spec": _spec(method).to_json(), "freeze_condition_id": "A_p_rho060",
            "workload": dict(WORKLOAD), "alpha": None, "cell": None, "split": None, "repeat": rep}


def _b_rows(cid, method, rep, pauses, error=False):
    """Mix B rows (ms): llama L1 runs 0..1736 (error: dies paused, t_end null, ckpt []); bert b1 runs 164..244 while
    L1 is paused; yolo y1 (deadline 400) expires in the queue while the second pause is pending."""
    base = dict(condition_id=cid, server=None, policy=method, policy_spec=_spec(method).to_json(), f_mhz=1020,
                queue_len_at_arrival=0, tokens_done=None, is_warmup=False, seed=SEED + rep, mode="open", err=None,
                pauses=[])

    def row(req_id, type_id, klass, arr, dl, **kw):
        fam, ex = C.parse_type_id(type_id)
        return dict(base, req_id=req_id, type_id=type_id, req_type_id=type_id, family=fam, exit=ex, klass=klass,
                    server=fam, target=C.default_checkpoints(fam, ex)[-1], t_arrival_ns=arr * MS,
                    t_arrival_actual_ns=arr * MS, t_deadline_ns=dl * MS, **kw)

    ran = dict(t_admit_ns=0, t_srv_recv_ns=0, t_start_ns=0, n_in_service_at_admit=0,
               pauses=[[k, a * MS, b * MS, None if c is None else c * MS] for k, a, b, c in pauses])
    if error:
        llama = row("L1", "llama:15", "long", 0, 3000, t_end_ns=None, t_done_ns=1000 * MS, reached=None,
                    completion=0.0, outcome="error", ckpt=[], **dict(ran, err="server died"))
    else:
        llama = row("L1", "llama:15", "long", 0, 3000, t_end_ns=1736 * MS, t_done_ns=1738 * MS, reached=32,
                    completion=1.0, outcome="complete", ckpt=[[k + 1, t * MS] for k, t in enumerate(L1_TOKENS)],
                    **dict(ran, tokens_done=32))
    b1 = row("b1", "bert:23", "short", 130, 280, t_admit_ns=163 * MS, t_srv_recv_ns=163 * MS, t_start_ns=164 * MS,
             t_end_ns=244 * MS, t_done_ns=246 * MS, n_in_service_at_admit=1, reached=23, completion=1.0,
             outcome="complete", ckpt=[[c, t * MS] for c, t in ((3, 175), (7, 190), (11, 205), (15, 220), (19, 232),
                                                                (23, 244))])
    y1 = row("y1", "yolo:5", "short", 380, 400, t_admit_ns=None, t_srv_recv_ns=None, t_start_ns=None, t_end_ns=None,
             t_done_ns=400 * MS, n_in_service_at_admit=None, reached=None, completion=0.0,
             outcome="expired_in_queue", ckpt=[])
    return [llama, b1, y1]


def _add_b(sdir, method, rep, pauses, error=False):
    """One mix B condition B_p_rho090_<method>_r<rep> in session dir sdir."""
    cid = "B_p_rho090_%s_r%d" % (method, rep)
    (sdir / cid).mkdir()
    man = dict(_manifest(cid, method, rep, True), n_requests=3, n_warmup=0, freeze_condition_id="B_p_rho090",
               workload=dict(B_WORKLOAD))
    (sdir / cid / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    _write_jsonl(sdir / cid / "requests.jsonl", _b_rows(cid, method, rep, pauses, error))


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


def test_nogate_ablation_is_analyzed_like_a_pilot_method(tmp_path):
    s1 = _session(tmp_path, "s1", [("coexit_nogate", 0), ("edgeserving", 0)])
    out = tmp_path / "out"
    assert R.main(["--sessions", str(s1), "--out", str(out)]) == 0
    _, runs = _read(out / "runs.csv")
    assert {(r["condition_id"], r["method"], r["kept"]) for r in runs} == {
        ("A_p_rho060_coexit_nogate_r0", "coexit_nogate", "1"), ("A_p_rho060_edgeserving_r0", "edgeserving", "1")}
    _, summ = _read(out / "summary.csv")
    ng = {r["family"]: r for r in summ if r["method"] == "coexit_nogate"}
    assert set(ng) == {"bert", "yolo", "all"}
    assert float(ng["all"]["viol_mean"]) == pytest.approx(0.5)                  # fixture expected
    assert float(ng["all"]["d_viol_vs_edgeserving_mean"]) == pytest.approx(0.0)  # same fake rows as edgeserving


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


# ------------------------------------------------------------------------------------------------ (3) lp.csv
def _lp_ids(method, rep):
    return {"session_id": "sid_lp", "condition_id": "B_p_rho090_%s_r%d" % (method, rep), "method": method,
            "mix": "B", "arrival": "poisson", "cv_a": "1.0", "rho": "0.9", "rep": str(rep), "seed": str(SEED + rep)}


def test_old_run3_outputs_byte_identical_and_no_lp_csv(tmp_path):
    s1 = _session(tmp_path, "s1", [("coexit", 0), ("edgeserving", 0, True, "b1")])
    out = tmp_path / "out"
    assert R.main(["--sessions", str(s1), "--out", str(out), "--sim-dir", str(_pilot(tmp_path))]) == 0
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()} == GOLDEN_OLD


def test_lp_csv_hand_numbers(tmp_path):
    s = _session(tmp_path, "lp", [])
    _add_b(s, "edgeserving_lp", 0, L1_PAUSES)
    _add_b(s, "coexit_lp", 0, [])                                      # never paused: b1 co-runs with L1
    _add_b(s, "edgeserving_lp", 1, [L1_PAUSES[0], L1_PAUSES[1][:3] + [None]], error=True)   # dies in pause 2
    out = tmp_path / "out"
    assert R.main(["--sessions", str(s), "--out", str(out)]) == 0
    h, lp = _read(out / "lp.csv")
    assert h == C.RUN3_LP_COLS and len(lp) == 3
    empty = {c: "" for c in C.RUN3_LP_COLS if c.endswith(("_p50", "_p95", "_max"))}     # every ms cell
    exp = {
        # wait 161-130=31, 405-380=25; pause 2 empty (y1 expired at 400); handoff 163-161; gap 249-246; paused 88+1;
        # first token 304-249=55 and 458-406=52 minus the median gap 50; L1 segments miss b1 [164, 244]
        "B_p_rho090_edgeserving_lp_r0": dict(
            _lp_ids("edgeserving_lp", 0), n_llama="1", n_paused_requests="1", n_pauses="2", n_empty_pauses="1",
            paused_ms_p50="89.0", paused_ms_max="89.0", boundary_wait_ms_p50="25.0", boundary_wait_ms_p95="31.0",
            boundary_wait_ms_max="31.0", handoff_ms_p50="2.0", handoff_ms_p95="2.0", resume_gap_ms_p50="3.0",
            resume_gap_ms_p95="3.0", first_token_penalty_ms_p50="2.0", first_token_penalty_ms_p95="5.0",
            n_overlap="0"),
        "B_p_rho090_coexit_lp_r0": dict(_lp_ids("coexit_lp", 0), n_llama="1", n_paused_requests="0", n_pauses="0",
                                        n_empty_pauses="0", n_overlap="1", **empty),
        # the open pause counts in n_pauses and boundary wait only; no ckpt -> no first-token penalty; t_end null
        "B_p_rho090_edgeserving_lp_r1": dict(
            _lp_ids("edgeserving_lp", 1), n_llama="1", n_paused_requests="1", n_pauses="2", n_empty_pauses="0",
            paused_ms_p50="88.0", paused_ms_max="88.0", boundary_wait_ms_p50="25.0", boundary_wait_ms_p95="31.0",
            boundary_wait_ms_max="31.0", handoff_ms_p50="2.0", handoff_ms_p95="2.0", resume_gap_ms_p50="3.0",
            resume_gap_ms_p95="3.0", first_token_penalty_ms_p50="", first_token_penalty_ms_p95="", n_overlap="0")}
    assert {r["condition_id"]: r for r in lp} == {k: {c: v[c] for c in C.RUN3_LP_COLS} for k, v in exp.items()}


def test_s0_lp_fixture_rows_give_the_numbers_of_its_doc(tmp_path):
    """fixtures/lp/rows.json (slice S0): wait 301.05-250.1, handoff 301.3-301.05, gap 322.5-322.3, paused 322.5-301.05,
    penalty (427.5-322.5)-100 ms; r000003 died paused (open pause: n_pauses and boundary wait 201.05-150 only)."""
    rows = mocks.load_fixture("lp/rows.json")["rows"]
    s, cid = _session(tmp_path, "lp", []), rows[0]["condition_id"]
    (s / cid).mkdir()
    wl = dict(B_WORKLOAD, types=["bert:23", "llama:15"], probs=[0.5, 0.5])
    (s / cid / "manifest.json").write_text(json.dumps(dict(
        _manifest(cid, "edgeserving_lp", 0, True), n_requests=3, n_warmup=0, freeze_condition_id="B_p_rho090",
        workload=wl)), encoding="utf-8")
    _write_jsonl(s / cid / "requests.jsonl", rows)
    out = tmp_path / "out"
    assert R.main(["--sessions", str(s), "--out", str(out)]) == 0
    _, (lp,) = _read(out / "lp.csv")
    assert {k: lp[k] for k in ("n_llama", "n_paused_requests", "n_pauses", "n_empty_pauses", "n_overlap")} == {
        "n_llama": "2", "n_paused_requests": "2", "n_pauses": "2", "n_empty_pauses": "0", "n_overlap": "0"}
    exp = {"paused_ms_p50": 21.45, "paused_ms_max": 21.45, "boundary_wait_ms_p50": 50.95,
           "boundary_wait_ms_p95": 51.05, "boundary_wait_ms_max": 51.05, "handoff_ms_p50": 0.25, "handoff_ms_p95": 0.25,
           "resume_gap_ms_p50": 0.2, "resume_gap_ms_p95": 0.2, "first_token_penalty_ms_p50": 5.0,
           "first_token_penalty_ms_p95": 5.0}
    assert {k: float(lp[k]) for k in exp if k in C.RUN3_LP_COLS} == pytest.approx(
        {k: v for k, v in exp.items() if k in C.RUN3_LP_COLS}, abs=1e-9)


def test_old_run3_sessions_plus_lp_session_show_lp_methods(tmp_path):
    s1 = _session(tmp_path, "s1", [("coexit", 0), ("edgeserving", 0)])
    s_lp = _session(tmp_path, "lp", [("edgeserving_lp", 0), ("coexit_lp", 0)])    # mix A: the LP placebo
    out = tmp_path / "out"
    assert R.main(["--sessions", str(s1), str(s_lp), "--out", str(out), "--sim-dir", str(_pilot(tmp_path))]) == 0
    _, runs = _read(out / "runs.csv")
    assert {(r["session_id"], r["method"], r["kept"]) for r in runs} == {
        ("sid_s1", "coexit", "1"), ("sid_s1", "edgeserving", "1"), ("sid_lp", "edgeserving_lp", "1"),
        ("sid_lp", "coexit_lp", "1")}
    _, summ = _read(out / "summary.csv")
    al = {r["method"]: r for r in summ if r["family"] == "all"}
    assert set(al) == {"coexit", "edgeserving", "edgeserving_lp", "coexit_lp"}
    for m in ("edgeserving_lp", "coexit_lp"):               # same fake rows as edgeserving, paired by seed
        assert float(al[m]["viol_mean"]) == pytest.approx(0.5)
        assert float(al[m]["d_viol_vs_edgeserving_mean"]) == pytest.approx(0.0)
    _, comp = _read(out / "compare.csv")
    cl = {r["method"]: r for r in comp if r["family"] == "all"}
    assert set(cl) == set(al) and (cl["coexit_lp"]["board_viol_mean"], cl["coexit_lp"]["sim_viol_mean"]) == ("0.5", "")
    _, lp = _read(out / "lp.csv")
    assert {(r["session_id"], r["method"], r["n_llama"], r["n_pauses"], r["n_overlap"], r["handoff_ms_p50"])
            for r in lp} == {("sid_lp", "edgeserving_lp", "0", "0", "0", ""), ("sid_lp", "coexit_lp", "0", "0", "0", "")}
