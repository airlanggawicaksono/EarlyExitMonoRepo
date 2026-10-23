"""Unit tests of exp2.analyze.fig_run3 on a tiny synthetic summary.csv / compare.csv and two simulator pilot
summary.csv files (no board data, no model)."""
import csv
import re
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.text import Text

from exp2 import contracts
from exp2.analyze import figures, fig_run3

TEXTWIDTH_IN = 372 / 72.27   # \textwidth of the paper: 31pc (sn-jnl.cls, one column)
METHODS = ("all_final", "all_early", "edgeserving", "greedy", "coexit", "coexit_nogate")
CELLS = (("A", "poisson", "1.0", "0.6"), ("A", "poisson", "1.0", "0.9"), ("B", "poisson", "1.0", "0.6"),
         ("B", "poisson", "1.0", "0.9"), ("B", "gamma", "2.0", "0.9"))
NAMES = ("All-Final", "All-Early", "EdgeServing", "Deadline-Greedy", "CoExit", "CoExit-NoGate")
FIGS = ("fig_run3_viol", "fig_run3_depth", "fig_run3_simboard", "fig_run3_family", "fig_run3_ablation",
        "fig_sim_deadline")
SIM_METHODS = ("all_final", "all_early", "edgeserving", "greedy", "coexit")


def _w(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


def _summary_row(method, cell, family, scale, v):
    mix, arrival, cv, rho = cell
    r = dict.fromkeys(contracts.COEXIT_SUMMARY_COLS, "0.0")
    r.update(method=method, mix=mix, arrival=arrival, cv_a=cv, rho=rho, deadline_scale=scale, family=family,
             n_reps="3", viol_mean=v, viol_ci_lo=v - 0.02, viol_ci_hi=v + 0.02, depth_mean=0.5, depth_ci_lo=0.45,
             depth_ci_hi=0.55)
    return r


def _compare_row(method, cell, family, d):
    mix, arrival, cv, rho = cell
    r = dict.fromkeys(contracts.RUN3_COMPARE_COLS, "0.0")
    r.update(method=method, mix=mix, arrival=arrival, cv_a=cv, rho=rho, family=family, board_d_viol_vs_es_mean=d,
             board_d_viol_vs_es_lo=d - 0.03, board_d_viol_vs_es_hi=d + 0.03, sim_d_viol_vs_es_mean=d + 0.01,
             sim_d_viol_vs_es_lo=d - 0.01, sim_d_viol_vs_es_hi=d + 0.03)
    if method == "coexit_nogate":   # the real compare.csv has no simulator value for the ablation
        r.update(sim_d_viol_vs_es_mean="", sim_d_viol_vs_es_lo="", sim_d_viol_vs_es_hi="")
    return r


def _analysis(tmp_path):
    d = tmp_path / "analysis_run3_all"
    d.mkdir()
    rows = []
    for k, m in enumerate(METHODS):
        for cell in CELLS:
            rows.append(_summary_row(m, cell, "all", "1.0", 0.1 * k + 0.05))
            for fam in ("bert", "yolo") + (("llama",) if cell[0] == "B" else ()):   # per-family rows: fig_run3_family
                rows.append(_summary_row(m, cell, fam, "1.0", 0.99))
            rows.append(_summary_row(m, cell, "all", "0.75", 0.98))      # other deadline scale: not plotted
    _w(d / "summary.csv", contracts.COEXIT_SUMMARY_COLS, rows)
    _w(d / "compare.csv", contracts.RUN3_COMPARE_COLS,
       [_compare_row(m, c, f, -0.1) for m in METHODS for c in CELLS for f in ("all", "yolo")])
    return d


def _pilot(tmp_path, name, drop=()):
    """Simulator pilot summary.csv: five methods, Poisson rho 0.6/0.9/1.2 and Gamma 0.9 at deadline scales 0.75/1.0/1.5.
    coexit violation ratio = 0.3 + scale / 10 (family all); cells in `drop` (mix, arrival, rho) are left out."""
    d = tmp_path / name
    d.mkdir()
    rows = []
    for k, m in enumerate(SIM_METHODS):
        for cell in CELLS + (("A", "poisson", "1.0", "1.2"), ("B", "poisson", "1.0", "1.2")):
            if (cell[0], cell[1], float(cell[3])) in drop:
                continue
            for scale in ("0.75", "1.0", "1.5"):
                v = 0.3 + float(scale) / 10 if m == "coexit" else 0.05 * k
                rows.append(_summary_row(m, cell, "all", scale, v))
                rows.append(_summary_row(m, cell, "bert", scale, 0.97))   # per-family row: not plotted
    _w(d / "summary.csv", contracts.COEXIT_SUMMARY_COLS, rows)
    return d


def _pdf_width_in(path):
    m = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", Path(path).read_bytes())
    return float(m.group(1)) / 72


def test_writes_the_six_figures_with_paper_names(tmp_path, monkeypatch):
    d, out = _analysis(tmp_path), tmp_path / "figs"
    m1, oracle = _pilot(tmp_path, "pilot_m1"), _pilot(tmp_path, "pilot_oracle")
    seen = {}
    save = figures._save

    def spy(fig, o, name):   # texts, labelled lines and the smallest drawn font of every figure, before it closes
        fig.canvas.draw()
        seen[name] = {"texts": [t.get_text() for t in fig.findobj(Text)],
                      "lines": {ln.get_label(): ln.get_linestyle() for ln in fig.findobj(Line2D)},
                      "font": min(t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip())}
        return save(fig, o, name)

    monkeypatch.setattr(figures, "_save", spy)
    assert fig_run3.main(["--analysis", str(d), "--out", str(out), "--pilot-m1", str(m1), "--pilot-oracle", str(oracle),
                          "--no-titles"]) == 0
    names = FIGS
    for n in names:
        for ext in ("pdf", "png"):
            assert (out / f"{n}.{ext}").stat().st_size > 0, (n, ext)
        width = _pdf_width_in(out / f"{n}.pdf")
        assert width <= 7.0, (n, width)
        assert seen[n]["font"] * TEXTWIDTH_IN / width >= 8.0, (n, seen[n]["font"], width)
    assert sorted(p.name for p in out.iterdir()) == sorted(f"{n}.{e}" for n in names for e in ("pdf", "png"))
    for n in ("fig_run3_viol", "fig_run3_depth", "fig_run3_family"):
        lines = seen[n]["lines"]
        for name in NAMES:
            assert name in lines, (n, name)
        assert lines["CoExit"] == "-" and lines["CoExit-NoGate"] == "--", lines
        assert not any(k in t for t in seen[n]["texts"] for k in ("coexit_nogate", "all_final", "greedy"))
    assert {"CoExit", "Deadline-Greedy"} <= set(seen["fig_run3_simboard"]["lines"])
    assert "CoExit-NoGate" not in seen["fig_run3_simboard"]["lines"]
    panels = {"Mix A, BERT", "Mix A, YOLO", "Mix B, BERT", "Mix B, YOLO", "Mix B, LLaMA"}
    assert panels <= set(seen["fig_run3_family"]["texts"])
    abl = seen["fig_run3_ablation"]["lines"]
    assert {"EdgeServing", "Deadline-Greedy", "CoExit", "CoExit-NoGate"} <= set(abl), abl
    assert abl["CoExit (sim.)"] == ":" and abl["CoExit-Oracle (sim.)"] == ":", abl
    assert "All-Final" not in abl and "All-Early" not in abl
    dl = seen["fig_sim_deadline"]["lines"]
    assert set(NAMES[:5]) <= set(dl) and "CoExit-NoGate" not in dl, dl
    assert "Deadline scale" in seen["fig_sim_deadline"]["texts"]


def test_pilot_cells_and_missing_series(tmp_path, capsys):
    d, out = _analysis(tmp_path), tmp_path / "figs"
    m1 = _pilot(tmp_path, "pilot_m1")
    cells = fig_run3.summary_cells(m1)
    assert abs(cells[("all", "A", "poisson", 0.9, 0.75)]["coexit"]["viol_mean"] - 0.375) < 1e-12
    assert "coexit_nogate" not in cells[("all", "B", "gamma", 0.9, 1.0)]
    oracle = _pilot(tmp_path, "pilot_oracle", drop={("B", "gamma", 0.9)})   # the oracle misses one board cell
    made, skipped = fig_run3.run({"analysis": str(d), "pilot_m1": str(m1), "pilot_oracle": str(oracle),
                                  "titles": False}, out)
    assert "fig_run3_ablation" in made and not skipped, skipped
    assert "CoExit-Oracle (sim.) skipped at B gamma 0.9" in capsys.readouterr().out
    made, skipped = fig_run3.run({"analysis": str(d), "titles": False}, tmp_path / "figs2")   # no pilot folder given
    assert set(made) == set(FIGS[:4])
    assert set(skipped) == {"fig_run3_ablation", "fig_sim_deadline"}
    assert all("--pilot-m1 not given" in v for v in skipped.values()), skipped


def test_tables_keep_family_all_at_deadline_scale_one(tmp_path):
    d = _analysis(tmp_path)
    tab = fig_run3.summary_table(d)
    assert set(tab) == {("A", "poisson", 0.6), ("A", "poisson", 0.9), ("B", "poisson", 0.6), ("B", "poisson", 0.9),
                        ("B", "gamma", 0.9)}
    assert abs(tab[("B", "gamma", 0.9)]["coexit"]["viol_mean"] - 0.45) < 1e-12   # k = 4: 0.1 * 4 + 0.05
    assert abs(tab[("A", "poisson", 0.6)]["all_final"]["viol_ci_hi"] - 0.07) < 1e-12
    pts = fig_run3.simboard_points(d)
    assert set(pts) == {"greedy", "coexit"}
    assert len(pts["coexit"]) == len(CELLS)
    p = pts["coexit"][0]
    assert abs(p["board"] - (-0.1)) < 1e-12 and abs(p["sim"] - (-0.09)) < 1e-12


def test_missing_input_is_skipped(tmp_path):
    out = tmp_path / "figs"
    nowhere = str(tmp_path / "nowhere")
    ctx = {"analysis": nowhere, "pilot_m1": nowhere, "pilot_oracle": nowhere, "titles": False}
    made, skipped = fig_run3.run(ctx, out)
    assert made == {}
    assert set(skipped) == set(FIGS)
    assert all("missing" in v for v in skipped.values()), skipped
    assert not out.exists() or not any(out.iterdir())
