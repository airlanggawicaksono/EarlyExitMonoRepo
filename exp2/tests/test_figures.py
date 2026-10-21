"""Smoke test of exp2.analyze.figures on a tiny synthetic input: every figure writes its PDF and PNG."""
import csv
import json
import re
from pathlib import Path

from matplotlib.text import Text

from exp2.analyze import figures

TEXTWIDTH_IN = 372 / 72.27   # \textwidth of the paper: 31pc (sn-jnl.cls, one column)


def _w(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _pdf_width_in(path):
    """Width of the saved PDF page (MediaBox, in 1/72 in)."""
    m = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", Path(path).read_bytes())
    return float(m.group(1)) / 72


def test_figures_smoke(tmp_path, monkeypatch):
    r1, fz, r2 = tmp_path / "run1_csv", tmp_path / "frozen", tmp_path / "run2"
    c1, c2 = tmp_path / "s_run1c_bert_cold", tmp_path / "s_run1c_bert_warm"
    for d in (r1, fz, r2, c1 / "telemetry", c2 / "telemetry"):
        d.mkdir(parents=True)
    types = ["bert:5", "bert:23", "llama:3", "llama:15", "yolo:1", "yolo:5"]
    levels = (612, 816, 1020)
    sets = ["bert:5|llama:3", "bert:23|yolo:1", "bert:5|llama:15", "llama:3|yolo:5"]
    cor = []
    for k, A in enumerate(sets):
        for f in levels:
            for rep in range(2):
                for t in A.split("|"):
                    cor.append([f"c{k:03d}", "train", f, rep, t, t.split(":")[0], A, 10, 10, 1.6 + 0.1 * k - f / 5000,
                                0, 0, 0, 0, 0, 20, 0.03, 1.5, 10, 20, 280, 12 + f / 200])
    _w(r1 / "corun.csv", ["cell_id", "split", "f_mhz", "repeat", "type_id", "family", "A", "n", "n_window", "s_p50",
                          "s_mean", "s_p95", "s_p50_lo", "s_p50_hi", "ess", "tau_solo_p50", "U_A", "window_s",
                          "energy_j", "p_board_w", "x", "y"], cor)
    _w(r1 / "solo.csv", ["type_id", "f_mhz", "repeat", "tau_p50", "p_board_w"],
       [[t, f, rep, 0.1 * (1 + i) * 1000 / f, 9 + f / 150] for i, t in enumerate(types) for f in (408, 612, 714, 816, 1020)
        for rep in range(2)])
    _w(r1 / "idle.csv", ["f_mhz", "kind", "p_idle_w"], [[f, "resident", 5.5] for f in (408, 612, 714, 816, 1020)])
    json.dump({"p_idle_w": {"612": 5.5, "816": 5.7, "1020": 6.1}}, open(fz / "power.json", "w"))
    json.dump({"selection": {"loco_mdape": {}}}, open(fz / "slowdown.json", "w"))
    per = {t: {"complete_rate": 0.9} for t in types}
    preds = {k: {"per_type": per, "energy_per_req_j": 2.0} for k in ("F", "default", "g2", "g3")}
    grid = [{"name": "g2", "k_short": 1, "k_long": 2, "order": "stf", "f_mhz": 612},
            {"name": "g3", "k_short": None, "k_long": 1, "order": "stf", "f_mhz": 1020}]
    json.dump({"conditions": [{"condition_id": "A_r060", "predictions": preds, "grid": grid}]}, open(fz / "freeze.json", "w"))
    hdr = ["condition_id", "policy_key", "workload", "rho", "arrival", "type_id", "n", "complete_rate", "energy_per_req_j"]
    rows = [[f"A_r060_{p}_r0", p, "A", "0.6", "poisson", t, 10, 0.95, 2.1] for p in ("F", "default", "g2", "g3") for t in types]
    # fig_admit: serial, default_pinned and F in the four Poisson conditions. B_r060 F has two repeats of different
    # size: llama:3 pooled = (10 x 0.9 + 30 x 0.5) / 40 = 0.6, not the mean of the rates (0.7).
    for c, (wl, rho) in {"A_r030": ("A", "0.3"), "A_r060": ("A", "0.6"), "B_r030": ("B", "0.3"),
                         "B_r060": ("B", "0.6")}.items():
        for p in ("serial", "default_pinned", "F"):
            if (c, p) != ("A_r060", "F"):
                rows += [[f"{c}_{p}_r0", p, wl, rho, "poisson", t, 10, 0.9 if t == "llama:3" else 0.97, 2.0]
                         for t in types]
    rows += [["B_r060_F_r1", "F", "B", "0.6", "poisson", t, 30, 0.5 if t == "llama:3" else 0.97, 2.0] for t in types]
    # fig_run2_energy_attain: a named baseline with one run (hollow marker) and an ablation (grey, with the grid).
    rows += [[f"A_r060_{p}_r0", p, "A", "0.6", "poisson", t, 10, 0.8, 1.9] for p in ("probe_margin", "F_M3") for t in types]
    _w(r2 / "open_conditions.csv", hdr, rows)
    for c in (c1, c2):
        json.dump({"servers": {"bert": {"t_spawn_ns": 0, "t_ready_ns": 5 * 10**9,
                                        "ready": {"warmup": {"first_infer_s": {"5": 1.0, "23": 0.1}}}}}},
                  open(c / "session.json", "w"))
        _w(c / "telemetry" / "power.csv", ["t_mono_ns", "t_wall_ns", "vdd_in_mv", "vdd_in_ma", "vdd_in_mw"],
           [[i * 10**9, 0, 0, 0, 10000.0] for i in range(-1, 8)])
    texts, fonts = [], {}
    save = figures._save
    paper = tmp_path / "paper"

    def spy(fig, out, name):   # every visible text of every figure, before it is closed
        texts.extend(t.get_text() for t in fig.findobj(Text))
        if Path(out) == paper:   # paper version: smallest font of every drawn, non-empty text
            fig.canvas.draw()
            fonts[name] = min(t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip())
        return save(fig, out, name)

    monkeypatch.setattr(figures, "_save", spy)
    out = tmp_path / "out"
    ctx = {"run1_csv": str(r1), "frozen": str(fz), "run1c": [str(c1), str(c2)], "run2": str(r2)}
    made, skipped = figures.run(ctx, out)
    assert not skipped, skipped
    assert "fig_admit" in made
    assert set(made) == set(figures.FIGS)
    for files in made.values():
        for f in files:
            assert (out / f).stat().st_size > 0
    assert abs(figures.load_run1c([str(c1), str(c2)])[("bert", "cold")]["energy_j"] - 50.0) < 1e-6   # 10 W * 5 s
    tab = figures.admit_table(str(r2))
    assert set(tab) == {"A_r030", "A_r060", "B_r030", "B_r060"}
    assert set(tab["B_r060"]) == {"serial", "default_pinned", "F"}
    assert abs(tab["B_r060"]["F"]["llama:3"] - 0.6) < 1e-12        # pooled over the repeats, weighted by n
    assert abs(tab["A_r060"]["F"]["bert:5"] - 0.95) < 1e-12
    # Names of addendum 1, sections A and B: CoSwarm for F, the baseline names, a labelled 95% target.
    for name in ("CoSwarm (this work)", "Unrestricted co-running", "Serial", "target"):
        assert any(name in s for s in texts), name
    for old in ("F (this work)", "Default at 1020 MHz", "Frequency from solo profile"):
        assert not any(old in s for s in texts), old
    # Paper version of the Run 2 figure: at most 7 in wide, every font at least 8 pt at \textwidth, every method named.
    made_p, skipped_p = figures.run({**ctx, "titles": False}, paper, only=["fig_run2_energy_attain"])
    assert not skipped_p and "fig_run2_energy_attain" in made_p
    width = _pdf_width_in(paper / "fig_run2_energy_attain.pdf")
    assert width <= 7.0, width
    assert fonts["fig_run2_energy_attain"] * TEXTWIDTH_IN / width >= 8.0, (fonts, width)
    for name in ("Probe-margin frequency", "One run each:", "Other grid policies and ablations", "Misses the target"):
        assert any(name in s.replace("\n", " ") for s in texts), name
