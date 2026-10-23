"""Unit tests of exp2.analyze.fig_run3_req on tiny synthetic Run 3 sessions in tmp_path (no board data, no model).

    r_over_d   expired -> R = D (1.0); truncated and late rows use t_done
    cdf_data   measured rows only, reps 0-2 only, one cell, keyed by method
    window     first measured LLaMA request that starts on arrival in both runs; time since the first arrival
    timeline   bars t_start..t_end per family, floor flag, miss markers at t_done (expired: at the deadline)
    main       --no-titles writes both figures, width <= 7 in, text >= 8 pt at textwidth, paper names
"""
import json
import re
from pathlib import Path

import numpy as np
import pytest
from matplotlib.lines import Line2D
from matplotlib.text import Text

from exp2 import contracts as C
from exp2 import exits as X
from exp2.analyze import fig_run3_req as F
from exp2.analyze import figures

TEXTWIDTH_IN = 372 / 72.27   # \textwidth of the paper: 31pc (sn-jnl.cls, one column)
SPECS = X.run3_specs(1020)
DL = {"bert": 0.1, "yolo": 0.1, "llama": 3.35}
MIX = {"A": {"name": "mixA", "types": ["bert:23", "yolo:5"], "probs": [0.5, 0.5]},
       "B": {"name": "mixB", "types": ["llama:15", "bert:23", "yolo:5"], "probs": [0.2, 0.4, 0.4]}}


# ------------------------------------------------------------------------------------------------ fakes
def _row(rid, req_type, arr, run_type=None, start=None, end=None, outcome="complete", done=None, warm=False):
    """A contract-valid open request row; times in seconds. expired_in_queue: never ran, t_done = deadline."""
    fam = C.parse_type_id(req_type)[0]
    run_type = run_type or req_type
    ex = C.parse_type_id(run_type)[1]
    target = C.default_checkpoints(fam, ex)[-1]
    reached = {"complete": target, "truncated": C.default_checkpoints(fam, ex)[0]}.get(outcome)
    dl = arr + DL[fam]
    ran = outcome != "expired_in_queue"
    done = done if ran else dl
    ns = lambda s: None if s is None else round(s * 1e9)   # noqa: E731
    return {"condition_id": "x", "req_id": rid, "type_id": run_type if ran else req_type, "family": fam,
            "exit": ex if ran else C.parse_type_id(req_type)[1],
            "klass": "long" if fam == "llama" else "short",
            "target": target if ran else C.default_checkpoints(fam, C.parse_type_id(req_type)[1])[-1],
            "t_arrival_ns": ns(arr), "t_arrival_actual_ns": ns(arr), "t_deadline_ns": ns(dl),
            "t_admit_ns": ns(start) if ran else None, "t_srv_recv_ns": ns(start) if ran else None,
            "t_start_ns": ns(start) if ran else None, "t_end_ns": ns(end) if ran else None, "t_done_ns": ns(done),
            "server": fam, "policy": "p", "policy_spec": None, "f_mhz": 1020,
            "n_in_service_at_admit": 0 if ran else None, "queue_len_at_arrival": 0, "reached": reached,
            "completion": 0.0 if reached is None else C.completion(fam, ex, reached), "outcome": outcome,
            "tokens_done": None, "ckpt": [], "is_warmup": warm, "seed": 1, "mode": "open", "err": None,
            "req_type_id": req_type}


def _write_run(sdir, mix, rho, method, rep, rows, t0_s=0.0):
    cid = "%s_p_rho%03d_%s_r%d" % (mix, round(rho * 100), method, rep)
    spec = SPECS[method].to_json()
    wl = dict(MIX[mix], rho=rho, rate_per_s=2.0, arrival="poisson", cv_a=1.0)
    man = {"schema": C.SCHEMAS["manifest"], "condition_id": cid, "mode": "open", "arm": "pinned", "gpu_mhz": 1020,
           "emc_mhz": 3199.0, "t_wall_start_ns": 1, "t_mono_start_ns": round(t0_s * 1e9),
           "t_mono_end_ns": round((t0_s + 100) * 1e9), "settle_s": 0.0, "valid": True, "invalid_reason": None,
           "board_readback": None, "n_requests": len(rows), "n_warmup": sum(r["is_warmup"] for r in rows),
           "seed": 20261022 + rep, "policy_key": method, "policy_spec": spec, "freeze_condition_id": "f",
           "workload": wl, "alpha": None, "cell": None, "split": None, "repeat": rep}
    d = Path(sdir) / cid
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    lines = [json.dumps(dict(r, condition_id=cid, policy=method, policy_spec=spec)) for r in rows]
    (d / "requests.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cdf_rows(late):
    """warm-up (R/D 0.1), on time (0.5), expired (1.0), and a late or truncated one (R/D `late`)."""
    return [_row("w", "bert:23", 0.0, start=0.0, end=0.009, done=0.01, warm=True),
            _row("a", "bert:23", 1.0, start=1.0, end=1.049, done=1.05),
            _row("b", "yolo:5", 2.0, outcome="expired_in_queue"),
            _row("c", "bert:23", 3.0, start=3.0, end=3.0 + 0.1 * late - 0.001, done=3.0 + 0.1 * late,
                 outcome="truncated")]


def _timeline_session(tmp_path):
    """Mix B rho 0.9 r0. LLaMA r1 arrives at 10 s (first arrival at 5 s) and starts on arrival in both runs.
    CoExit runs BERT at its floor exit while LLaMA runs; EdgeServing lets the BERT request expire."""
    sdir = tmp_path / "s"
    llama_late = _row("r0", "llama:15", 6.0, start=7.0, end=8.8, done=8.81)   # starts 1 s after arrival: not picked
    common = [_row("w", "bert:23", 5.0, start=5.0, end=5.05, done=5.06, warm=True), llama_late,
              _row("r1", "llama:15", 10.0, start=10.001, end=11.8, done=11.81)]
    coexit = common + [_row("r2", "bert:23", 10.5, run_type="bert:5", start=10.5, end=10.54, done=10.55),
                       _row("r3", "yolo:5", 11.0, start=11.0, end=11.12, done=11.13)]           # late: a miss
    edge = common + [_row("r2", "bert:23", 10.5, outcome="expired_in_queue"),
                     _row("r3", "yolo:5", 11.0, outcome="expired_in_queue")]
    _write_run(sdir, "B", 0.9, "coexit", 0, coexit, t0_s=4.0)
    _write_run(sdir, "B", 0.9, "edgeserving", 0, edge, t0_s=4.1)
    return sdir


# ------------------------------------------------------------------------------------------------ tests
def test_r_over_d_counts_expired_at_the_deadline_and_truncated_at_t_done():
    exp = _row("e", "bert:23", 1.0, outcome="expired_in_queue")
    tr = _row("t", "bert:23", 1.0, start=1.0, end=1.11, done=1.12, outcome="truncated")
    ok = _row("o", "llama:15", 0.0, start=0.0, end=1.6, done=1.675)
    assert F.r_over_d(exp) == pytest.approx(1.0)
    assert F.r_over_d(tr) == pytest.approx(1.2)
    assert F.r_over_d(ok) == pytest.approx(0.5)
    x, y = F.ecdf([1.0, 0.5, 2.0, 1.0])
    assert list(x) == [0.5, 1.0, 1.0, 2.0] and list(y) == [0.25, 0.5, 0.75, 1.0]


def test_cdf_data_pools_measured_rows_of_reps_0_to_2_in_one_cell(tmp_path):
    s1, s2 = tmp_path / "s1", tmp_path / "s2"
    _write_run(s1, "A", 0.9, "coexit", 0, _cdf_rows(1.2))
    _write_run(s1, "A", 0.9, "coexit", 1, _cdf_rows(1.4))
    _write_run(s2, "A", 0.9, "coexit", 3, _cdf_rows(9.0))       # rep 3: not pooled
    _write_run(s1, "A", 0.6, "coexit", 0, _cdf_rows(9.0))       # other load: not in the cell
    _write_run(s2, "A", 0.9, "edgeserving", 2, _cdf_rows(1.1))
    runs = F.load_runs([s1, s2])
    assert len(runs) == 5
    got = F.cdf_data(runs, ("A", "poisson", 0.9))
    assert set(got) == {"coexit", "edgeserving"}
    np.testing.assert_allclose(got["coexit"], [0.5, 0.5, 1.0, 1.0, 1.2, 1.4])
    np.testing.assert_allclose(got["edgeserving"], [0.5, 1.0, 1.1])


def test_window_and_timeline_bars_floor_and_misses(tmp_path):
    runs = F.load_runs([_timeline_session(tmp_path)])
    by = {r["method"]: r for r in runs}
    t0 = F.pick_window(by["coexit"], by["edgeserving"])
    assert t0 == pytest.approx(5.0 - F.LEAD_S)                    # LLaMA r1 at 10 s, first arrival at 5 s
    co = F.timeline(by["coexit"], t0, t0 + F.WINDOW_S)
    es = F.timeline(by["edgeserving"], t0, t0 + F.WINDOW_S)
    assert [(round(a, 3), round(w, 3), fl) for a, w, fl in co["llama"]["bars"]] == [(5.001, 1.799, False)]
    assert [(round(a, 3), round(w, 3), fl) for a, w, fl in co["bert"]["bars"]] == [(5.5, 0.04, True)]
    assert co["bert"]["miss"] == [] and [round(t, 3) for t in co["yolo"]["miss"]] == [6.13]
    assert es["bert"]["bars"] == [] and es["yolo"]["bars"] == []
    assert [round(t, 3) for t in es["bert"]["miss"]] == [5.6] and [round(t, 3) for t in es["yolo"]["miss"]] == [6.1]
    assert co["llama"]["bars"][0][0] < co["bert"]["bars"][0][0] < co["llama"]["bars"][0][0] + co["llama"]["bars"][0][1]


def _pdf_width_in(path):
    m = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", Path(path).read_bytes())
    return float(m.group(1)) / 72


def test_main_writes_both_figures_for_the_paper(tmp_path, monkeypatch):
    sdir = _timeline_session(tmp_path)
    for mix in ("A", "B"):
        for m in ("coexit", "coexit_nogate", "greedy"):
            _write_run(sdir, mix, 0.9, m, 1, _cdf_rows(1.3))
    _write_run(sdir, "A", 0.9, "edgeserving", 1, _cdf_rows(1.3))
    seen, save = {}, figures._save

    def spy(fig, o, name):
        fig.canvas.draw()
        seen[name] = {"texts": [t.get_text() for t in fig.findobj(Text)],
                      "lines": {ln.get_label(): ln.get_linestyle() for ln in fig.findobj(Line2D)},
                      "font": min(t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip())}
        return save(fig, o, name)

    monkeypatch.setattr(figures, "_save", spy)
    out = tmp_path / "figs"
    assert F.main(["--sessions", str(sdir), "--out", str(out), "--no-titles"]) == 0
    names = ("fig_run3_cdf", "fig_run3_timeline")
    assert sorted(p.name for p in out.iterdir()) == sorted(f"{n}.{e}" for n in names for e in ("pdf", "png"))
    for n in names:
        width = _pdf_width_in(out / f"{n}.pdf")
        assert width <= 7.0, (n, width)
        assert seen[n]["font"] * TEXTWIDTH_IN / width >= 8.0, (n, seen[n]["font"], width)
    lines = seen["fig_run3_cdf"]["lines"]
    assert lines["CoExit"] == "-" and lines["CoExit-NoGate"] == "--", lines
    assert {"EdgeServing", "Deadline-Greedy"} <= set(lines)
    texts = " ".join(seen["fig_run3_timeline"]["texts"])
    assert all(k in texts for k in ("EdgeServing", "CoExit", "BERT", "YOLO", "LLaMA")), texts
    assert not any(k in " ".join(seen[n]["texts"]) for n in names for k in ("coexit_nogate", "edgeserving"))
