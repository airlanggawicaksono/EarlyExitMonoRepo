"""Smoke test of exp2.analyze.figures on a tiny synthetic input: every figure writes its PDF and PNG."""
import csv
import json

from exp2.analyze import figures


def _w(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def test_figures_smoke(tmp_path):
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
    _w(r2 / "open_conditions.csv", hdr, rows)
    for c in (c1, c2):
        json.dump({"servers": {"bert": {"t_spawn_ns": 0, "t_ready_ns": 5 * 10**9,
                                        "ready": {"warmup": {"first_infer_s": {"5": 1.0, "23": 0.1}}}}}},
                  open(c / "session.json", "w"))
        _w(c / "telemetry" / "power.csv", ["t_mono_ns", "t_wall_ns", "vdd_in_mv", "vdd_in_ma", "vdd_in_mw"],
           [[i * 10**9, 0, 0, 0, 10000.0] for i in range(-1, 8)])
    out = tmp_path / "out"
    ctx = {"run1_csv": str(r1), "frozen": str(fz), "run1c": [str(c1), str(c2)], "run2": str(r2)}
    made, skipped = figures.run(ctx, out)
    assert not skipped, skipped
    assert set(made) == set(figures.FIGS)
    for files in made.values():
        for f in files:
            assert (out / f).stat().st_size > 0
    assert abs(figures.load_run1c([str(c1), str(c2)])[("bert", "cold")]["energy_j"] - 50.0) < 1e-6   # 10 W * 5 s
