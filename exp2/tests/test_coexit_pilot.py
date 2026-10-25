"""Tests for exp2.model.coexit_pilot (PLAN PL1 f, slice S3). Fakes only: no simulator, no Jetson, no GPU.

    metrics       the exit_trace_example.json fixture (contracts "Pilot metrics")
    CLI           parsing, required --deadlines, bad values, --out inside the repo
    runner        injected simulate_fn / policy_factory: header, row count, CRN, resume, summary, meta, smoke
    cells wiring  --sim-slowdown cells with a stub exp2.model.cell_slowdown module
    oracle        --policy-slowdown cells: the policy predicts with the simulator truth (same object)
    review        --methods with exits.ABLATIONS (B3 greedy_gate), --policy-slowdown one (B4), default unchanged,
                  --methods with exits.SIM_ONLY (B1 edgeserving_lp)
"""
import csv
import hashlib
import json
import sys
import types as pytypes
from types import SimpleNamespace

import pytest

from exp2 import contracts as C
from exp2 import exits
from exp2.model import coexit_pilot as P
from exp2.tests import mocks

SEED = 7
PILOT_TYPES = {"bert:5": {"tau": 0.02}, "bert:23": {"tau": 0.08}, "yolo:1": {"tau": 0.04},
               "yolo:5": {"tau": 0.06}, "llama:3": {"tau": 0.6}, "llama:15": {"tau": 1.6}}
DEADLINES = {"bert": 0.5, "yolo": 0.5, "llama": 5.0}
TINY = ["--mixes", "A", "--rhos", "0.5,1.0", "--gamma-mix", "B", "--gamma-rhos", "0.9", "--gamma-cv", "2.0",
        "--methods", "all_early,edgeserving,coexit", "--reps", "2", "--n-requests", "20", "--n-warmup", "5",
        "--seed", str(SEED)]
TINY_SIMS = 3 * 3 * 2                    # (2 poisson A + 1 gamma B cells) x 3 methods x 2 reps
TINY_ROWS = 2 * 3 * 2 * 3 + 1 * 3 * 2 * 4   # A: bert, yolo, all; B: llama, bert, yolo, all
TINY_GROUPS = 2 * 3 * 3 + 1 * 3 * 4


# ------------------------------------------------------------------------------------------------ fakes
class FakeSim:
    """simulate() stand-in. Every request completes 1 ms after arrival; all_early runs the floor exit;
    edgeserving finishes every second request 1 ms after its deadline (late)."""

    def __init__(self):
        self.calls = []

    def __call__(self, workload, spec, types, slowdown, power, cfg, policy=None):
        self.calls.append(SimpleNamespace(method=spec.name, spec=spec, workload=workload, slowdown=slowdown,
                                          power=power, cfg=cfg, policy=policy))
        rows = []
        for k, it in enumerate(cfg.schedule):
            fam, _ = C.parse_type_id(it["type_id"])
            run = C.type_id(fam, C.EXIT_MENU[fam][0]) if spec.name == "all_early" else it["type_id"]
            late = spec.name == "edgeserving" and k % 2 == 1
            done = it["t_deadline_ns"] + 1_000_000 if late else it["t_arrival_ns"] + 1_000_000
            rows.append({"req_id": it["req_id"], "type_id": it["type_id"], "run_type_id": run, "family": fam,
                         "t_arrival_ns": it["t_arrival_ns"], "t_deadline_ns": it["t_deadline_ns"],
                         "t_admit_ns": it["t_arrival_ns"], "t_done_ns": done, "outcome": "complete",
                         "reached": C.parse_type_id(run)[1]})
        return SimpleNamespace(trace=tuple(rows), runtime_s=0.0)


class FakePolicies:
    def __init__(self):
        self.made = []

    def __call__(self, spec, types, slowdown):
        obj = SimpleNamespace(spec=spec, types=types, slowdown=slowdown)
        self.made.append(obj)
        return obj


@pytest.fixture(autouse=True)
def no_git(monkeypatch):
    monkeypatch.setattr(P, "_git", lambda: {"commit": "f" * 40, "dirty": False})


@pytest.fixture
def inputs():
    types = mocks.make_types(PILOT_TYPES)
    return P.Inputs(types, mocks.StubSlowdown(), mocks.StubPower(types=types))


def _argv(tmp_path, *extra, out=None, deadlines="bert=0.5,yolo=0.5,llama=5"):
    for name in ("types", "slowdown", "power"):
        p = tmp_path / (name + ".json")
        if not p.exists():
            p.write_text('{"name": "%s"}\n' % name, encoding="utf-8")
    out = tmp_path / "out" if out is None else out
    head = ["--types", str(tmp_path / "types.json"), "--slowdown", str(tmp_path / "slowdown.json"),
            "--power", str(tmp_path / "power.json")]
    if deadlines is not None:
        head += ["--deadlines", deadlines]
    return head + ["--out", str(out)] + list(extra)


def _run(tmp_path, inputs, *extra, sim=None, pol=None, **kw):
    sim = FakeSim() if sim is None else sim
    pol = FakePolicies() if pol is None else pol
    rc = P.main(_argv(tmp_path, *extra, **kw), inputs=inputs, simulate_fn=sim, policy_factory=pol)
    return rc, sim, pol


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        return header, [dict(zip(header, r)) for r in rd]


# ------------------------------------------------------------------------------------------------ (1) metrics
def test_metrics_reproduce_the_fixture():
    fx = mocks.load_fixture("exit_trace_example.json")
    got = P.metrics(fx["rows"], fx["n_warmup"], ["bert", "yolo"])
    assert set(got) == {"bert", "yolo", "all"}
    for fam, exp in fx["expected"].items():
        for k, v in exp.items():
            assert got[fam][k] == pytest.approx(v, abs=fx["tolerance"]), (fam, k, got[fam][k])


def test_metrics_use_arrival_order_for_warmup():
    fx = mocks.load_fixture("exit_trace_example.json")
    assert P.metrics(list(reversed(fx["rows"])), fx["n_warmup"], ["bert", "yolo"]) == \
        P.metrics(fx["rows"], fx["n_warmup"], ["bert", "yolo"])


def test_metrics_family_without_requests_has_n_zero_and_no_rates():
    fx = mocks.load_fixture("exit_trace_example.json")
    got = P.metrics(fx["rows"], fx["n_warmup"], ["llama", "bert", "yolo"])
    assert got["llama"]["n"] == 0 and got["llama"]["n_success"] == 0
    assert got["llama"]["slo_violation_ratio"] is None and got["llama"]["share_floor"] is None


def test_metrics_need_run_type_id_in_every_row():
    fx = mocks.load_fixture("exit_trace_example.json")
    rows = [dict(r) for r in fx["rows"]]
    del rows[3]["run_type_id"]
    with pytest.raises(C.ContractError, match="run_type_id"):
        P.metrics(rows, fx["n_warmup"], ["bert", "yolo"])


# ------------------------------------------------------------------------------------------------ (2) CLI
def test_cli_parses_lists_deadlines_and_defaults(tmp_path):
    a = P.parse_args(_argv(tmp_path, "--rhos", "0.3,1.5", "--methods", "greedy,coexit"))
    assert a.deadlines == {"bert": 0.5, "yolo": 0.5, "llama": 5.0}
    assert a.rhos == [0.3, 1.5] and a.methods == ["greedy", "coexit"]
    assert (a.reps, a.n_requests, a.n_warmup, a.seed, a.f_mhz) == (10, 550, 50, 20261022, 1020)
    assert a.mixes == ["A", "B"] and a.deadline_scales == [1.0] and a.gamma_mix == "B"
    assert a.gamma_rhos == [0.9, 1.2] and a.gamma_cv == 2.0 and a.sim_slowdown == "m1"
    assert a.policy_slowdown == "m1"
    assert not a.smoke and not a.summarize_only and not a.no_sample_slowdown


def test_cli_requires_deadlines(tmp_path, inputs, capsys):
    rc, sim, _ = _run(tmp_path, inputs, deadlines=None)
    assert rc == 2 and sim.calls == []
    assert "--deadlines" in capsys.readouterr().err


@pytest.mark.parametrize("extra,deadlines", [
    ((), "bert=-1,yolo=0.5,llama=5"),
    ((), "bert=0.5,yolo=0.5,llama=5,foo=1"),
    ((), "bert,yolo=0.5,llama=5"),
    ((), "bert=0.5,yolo=0.5"),                                 # mix B needs llama
    (("--methods", "all_final,nope"), "bert=0.5,yolo=0.5,llama=5"),
    (("--mixes", "A,Z"), "bert=0.5,yolo=0.5,llama=5"),
    (("--n-requests", "10", "--n-warmup", "10"), "bert=0.5,yolo=0.5,llama=5"),
    (("--rhos", "0.3,-1"), "bert=0.5,yolo=0.5,llama=5"),
    (("--reps", "0"), "bert=0.5,yolo=0.5,llama=5"),
])
def test_cli_rejects_bad_values_with_exit_2(tmp_path, inputs, extra, deadlines):
    rc, sim, _ = _run(tmp_path, inputs, *extra, deadlines=deadlines)
    assert rc == 2 and sim.calls == []
    assert not (tmp_path / "out").exists()


def test_cli_refuses_out_dir_inside_the_repo(tmp_path, inputs, capsys):
    inside = P.REPO_ROOT / "exp2" / "_coexit_pilot_out_must_not_exist"
    rc, sim, _ = _run(tmp_path, inputs, *TINY, out=inside)
    assert rc == 2 and sim.calls == []
    assert not inside.exists()
    assert "inside the repo" in capsys.readouterr().err


# ------------------------------------------------------------------------------------------------ (3) runner
def test_runner_writes_reps_with_exact_header_and_row_count(tmp_path, inputs):
    rc, sim, _ = _run(tmp_path, inputs, *TINY)
    assert rc == 0
    header, rows = _read_csv(tmp_path / "out" / "reps.csv")
    assert tuple(header) == C.COEXIT_PILOT_COLS
    assert len(rows) == TINY_ROWS and len(sim.calls) == TINY_SIMS
    early = [r for r in rows if r["method"] == "all_early" and r["family"] == "all"]
    assert early and {r["share_floor"] for r in early} == {"1.0"}
    assert {r["share_floor"] for r in rows if r["method"] == "coexit"} == {"0.0"}
    assert {r["slo_violation_ratio"] for r in rows if r["method"] == "coexit"} == {"0.0"}
    assert {r["deadline_s"] for r in rows if r["family"] == "all"} == {""}
    assert {r["deadline_s"] for r in rows if r["family"] == "llama"} == {"5.0"}
    assert {(r["arrival"], r["cv_a"]) for r in rows if r["mix"] == "B"} == {("gamma", "2.0")}
    assert {(r["arrival"], r["cv_a"]) for r in rows if r["mix"] == "A"} == {("poisson", "1.0")}
    for r in rows:
        assert int(r["n"]) >= 0 and r["sim_runtime_s"] != ""


def test_runner_uses_common_random_numbers_across_methods(tmp_path, inputs):
    rc, sim, pol = _run(tmp_path, inputs, *TINY)
    assert rc == 0
    cells = {}
    for c in sim.calls:
        key = (c.workload["name"], c.workload["arrival"], c.workload["rho"], c.cfg.seed)
        cells.setdefault(key, []).append(c)
    assert len(cells) == 3 * 2
    for (_, _, _, seed), calls in cells.items():
        assert [c.method for c in calls] == ["all_early", "edgeserving", "coexit"]
        assert all(c.cfg.schedule == calls[0].cfg.schedule for c in calls)
        assert seed in (SEED, SEED + 1)
        assert all(c.cfg.n_reps == 1 and c.cfg.n_warmup == 5 and len(c.cfg.schedule) == 20 for c in calls)
        assert all(c.cfg.family_exclusive and c.cfg.sample_slowdown for c in calls)
    a0 = cells[("mixA", "poisson", 0.5, SEED)][0].cfg.schedule
    a1 = cells[("mixA", "poisson", 0.5, SEED + 1)][0].cfg.schedule
    assert [it["t_arrival_ns"] for it in a0] != [it["t_arrival_ns"] for it in a1]
    for c in sim.calls:
        wl = c.workload
        assert wl["rate_per_s"] == pytest.approx(C.rate_for_load(wl["rho"], wl, inputs.types))
        for it in c.cfg.schedule:
            fam = C.parse_type_id(it["type_id"])[0]
            assert it["t_deadline_ns"] - it["t_arrival_ns"] == round(DEADLINES[fam] * 1e9)
        assert c.slowdown is inputs.m1 and c.power is inputs.power
        assert c.policy.spec is c.spec and c.policy.slowdown is inputs.m1
        assert c.spec.f_mhz == 1020 and c.spec.params["exit_rule"]
    assert len(pol.made) == TINY_SIMS


def test_runner_resumes_without_duplicating_rows(tmp_path, inputs):
    assert _run(tmp_path, inputs, *TINY)[0] == 0
    reps = tmp_path / "out" / "reps.csv"
    rc, sim2, _ = _run(tmp_path, inputs, *TINY)
    assert rc == 0 and sim2.calls == []
    assert len(_read_csv(reps)[1]) == TINY_ROWS
    # crash in the middle of the last rep: two of its four rows lost, plus a torn line without a newline
    lines = reps.read_text(encoding="utf-8").splitlines(keepends=True)
    reps.write_text("".join(lines[:-2]) + "coexit,B,gam", encoding="utf-8")
    rc, sim3, _ = _run(tmp_path, inputs, *TINY)
    assert rc == 0 and len(sim3.calls) == 1
    assert (sim3.calls[0].method, sim3.calls[0].cfg.seed, sim3.calls[0].workload["arrival"]) == \
        ("coexit", SEED + 1, "gamma")
    header, rows = _read_csv(reps)
    assert tuple(header) == C.COEXIT_PILOT_COLS and len(rows) == TINY_ROWS
    keys = [tuple(r[c] for c in P.KEY_COLS) + (r["family"],) for r in rows]
    assert len(set(keys)) == len(keys)


def test_runner_refuses_to_resume_with_another_config(tmp_path, inputs, capsys):
    assert _run(tmp_path, inputs, *TINY)[0] == 0
    rc, sim, _ = _run(tmp_path, inputs, *TINY, "--n-requests", "30")
    assert rc == 2 and sim.calls == []
    assert "n_requests" in capsys.readouterr().err


def test_runner_writes_summary_and_meta(tmp_path, inputs):
    assert _run(tmp_path, inputs, *TINY)[0] == 0
    out = tmp_path / "out"
    header, rows = _read_csv(out / "summary.csv")
    assert tuple(header) == C.COEXIT_SUMMARY_COLS and len(rows) == TINY_GROUPS
    assert {r["n_reps"] for r in rows} == {"2"}
    es = [r for r in rows if r["method"] == "edgeserving"]
    assert {float(r["d_viol_vs_edgeserving_mean"]) for r in es} == {0.0}
    co_all = [r for r in rows if r["method"] == "coexit" and r["family"] == "all"]
    for r in co_all:
        assert float(r["d_viol_vs_edgeserving_mean"]) < 0.0
        assert float(r["d_viol_vs_edgeserving_lo"]) <= float(r["d_viol_vs_edgeserving_hi"])
        assert float(r["viol_mean"]) == 0.0 and r["viol_ci_lo"] != "" and r["depth_ci_hi"] != ""
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    for k in ("args", "git", "inputs_sha256", "python", "start", "end", "wall_s"):
        assert k in meta, k
    for name in ("types", "slowdown", "power"):
        assert meta["inputs_sha256"][name] == C.sha256_file(tmp_path / (name + ".json"))
    assert meta["git"]["commit"] == "f" * 40 and meta["end"] is not None and meta["wall_s"] >= 0
    assert meta["args"]["deadlines"] == DEADLINES and meta["n_sims_run"] == TINY_SIMS


def test_summarize_only_rebuilds_summary_without_simulating(tmp_path):
    assert _run(tmp_path, P.Inputs(mocks.make_types(PILOT_TYPES), mocks.StubSlowdown(), mocks.StubPower()),
                *TINY)[0] == 0
    summary = tmp_path / "out" / "summary.csv"
    first = summary.read_text(encoding="utf-8")
    summary.unlink()
    sim = FakeSim()
    rc = P.main(_argv(tmp_path, *TINY, "--summarize-only"), simulate_fn=sim)   # inputs are not loaded
    assert rc == 0 and sim.calls == []
    assert summary.read_text(encoding="utf-8") == first


def test_smoke_runs_a_small_grid_in_its_own_dir_and_projects_the_total(tmp_path, inputs, capsys):
    rc, sim, _ = _run(tmp_path, inputs, "--mixes", "A", "--rhos", "0.3,0.6,0.9", "--gamma-mix", "",
                      "--methods", "all_final,coexit", "--seed", str(SEED), "--smoke")
    assert rc == 0
    assert (tmp_path / "out" / "smoke" / "reps.csv").exists()
    assert not (tmp_path / "out" / "reps.csv").exists()
    assert len(sim.calls) == 2 * 2 * 2           # first and last rho x 2 reps x 2 methods
    assert {c.workload["rho"] for c in sim.calls} == {0.3, 0.9}
    assert {c.cfg.seed for c in sim.calls} == {SEED, SEED + 1}
    assert all(len(c.cfg.schedule) == 60 and c.cfg.n_warmup == 10 for c in sim.calls)
    assert "projected" in capsys.readouterr().out
    rc, sim2, _ = _run(tmp_path, inputs, "--mixes", "A", "--rhos", "0.3,0.6,0.9", "--gamma-mix", "",
                       "--methods", "all_final,coexit", "--seed", str(SEED), "--smoke")
    assert rc == 0 and len(sim2.calls) == 8        # a smoke run always measures again


def test_floor_slower_than_deadline_warns_but_runs(tmp_path, inputs, capsys):
    rc, sim, _ = _run(tmp_path, inputs, *TINY, deadlines="bert=0.01,yolo=0.5,llama=5")
    assert rc == 0 and len(sim.calls) == TINY_SIMS
    err = capsys.readouterr().err
    assert "warning" in err and "bert:5" in err
    meta = json.loads((tmp_path / "out" / "meta.json").read_text(encoding="utf-8"))
    assert any("bert:5" in w for w in meta["warnings"])


def test_no_sample_slowdown_reaches_the_sim_config(tmp_path, inputs):
    rc, sim, _ = _run(tmp_path, inputs, *TINY, "--no-sample-slowdown")
    assert rc == 0 and not any(c.cfg.sample_slowdown for c in sim.calls)


# ------------------------------------------------------------------------------------------------ (4) cells
@pytest.fixture
def stub_cells(tmp_path, monkeypatch):
    """A stub exp2.model.cell_slowdown module; -> (list of made CellSlowdown stubs, run1 dir)."""
    made = []

    class StubCell:
        def __init__(self, run1_dir, fallback):
            self.run1_dir, self.fallback = run1_dir, fallback

        @classmethod
        def from_run1_dir(cls, run1_dir, fallback=None):
            made.append(cls(run1_dir, fallback))
            return made[-1]

    mod = pytypes.ModuleType("exp2.model.cell_slowdown")
    mod.CellSlowdown = StubCell
    monkeypatch.setitem(sys.modules, "exp2.model.cell_slowdown", mod)
    run1 = tmp_path / "run1_csv"
    run1.mkdir()
    (run1 / "corun.csv").write_text("cell_id\n", encoding="utf-8")
    return made, run1


def test_sim_slowdown_cells_is_the_simulator_truth_and_the_policy_keeps_m1(tmp_path, inputs, stub_cells):
    made, run1 = stub_cells
    rc, sim, pol = _run(tmp_path, inputs, *TINY, "--sim-slowdown", "cells", "--run1-dir", str(run1))
    assert rc == 0 and len(made) == 1
    assert made[0].run1_dir == str(run1) and made[0].fallback is inputs.m1
    assert all(c.slowdown is made[0] for c in sim.calls)
    assert all(p.slowdown is inputs.m1 for p in pol.made)
    meta = json.loads((tmp_path / "out" / "meta.json").read_text(encoding="utf-8"))
    assert meta["args"]["sim_slowdown"] == "cells"
    assert meta["config"]["sim_slowdown"] == "cells" and meta["config"]["policy_slowdown"] == "m1"
    assert meta["inputs_sha256"]["run1_dir/corun.csv"] == C.sha256_file(run1 / "corun.csv")


def test_sim_slowdown_cells_without_the_module_exits_2(tmp_path, inputs, monkeypatch):
    monkeypatch.setitem(sys.modules, "exp2.model.cell_slowdown", None)     # import -> ImportError
    rc, sim, _ = _run(tmp_path, inputs, *TINY, "--sim-slowdown", "cells", "--run1-dir", str(tmp_path))
    assert rc == 2 and sim.calls == []


# ------------------------------------------------------------------------------------------------ (5) oracle
def test_default_run_records_policy_slowdown_m1_in_the_config(tmp_path, inputs):
    rc, _, pol = _run(tmp_path, inputs, *TINY)
    assert rc == 0 and all(p.slowdown is inputs.m1 for p in pol.made)
    config = json.loads((tmp_path / "out" / "meta.json").read_text(encoding="utf-8"))["config"]
    assert config["sim_slowdown"] == "m1" and config["policy_slowdown"] == "m1"


def test_policy_slowdown_cells_predicts_with_the_simulator_truth(tmp_path, inputs, stub_cells):
    made, run1 = stub_cells
    rc, sim, pol = _run(tmp_path, inputs, *TINY, "--sim-slowdown", "cells", "--policy-slowdown", "cells",
                        "--run1-dir", str(run1))
    assert rc == 0 and len(made) == 1 and len(pol.made) == TINY_SIMS
    assert all(p.slowdown is made[0] for p in pol.made)          # the same object, not a copy
    assert all(c.slowdown is made[0] and c.policy.slowdown is made[0] for c in sim.calls)
    meta = json.loads((tmp_path / "out" / "meta.json").read_text(encoding="utf-8"))
    assert meta["args"]["policy_slowdown"] == "cells"
    assert meta["config"]["sim_slowdown"] == "cells" and meta["config"]["policy_slowdown"] == "cells"


def test_policy_slowdown_m1_and_cells_differ_only_in_the_policy_model(tmp_path, inputs, stub_cells):
    made, run1 = stub_cells
    runs = {}
    for ps in ("m1", "cells"):
        out = tmp_path / ("out_" + ps)
        rc, sim, pol = _run(tmp_path, inputs, *TINY, "--sim-slowdown", "cells", "--policy-slowdown", ps,
                            "--run1-dir", str(run1), out=out)
        assert rc == 0
        runs[ps] = (sim, pol, json.loads((out / "meta.json").read_text(encoding="utf-8"))["config"])
    (s1, p1, c1), (s2, p2, c2) = runs["m1"], runs["cells"]
    assert [(c.method, c.workload, c.cfg) for c in s1.calls] == [(c.method, c.workload, c.cfg) for c in s2.calls]
    assert all(c.slowdown is made[0] for c in s1.calls) and all(c.slowdown is made[1] for c in s2.calls)
    assert {id(p.slowdown) for p in p1.made} == {id(inputs.m1)} and {id(p.slowdown) for p in p2.made} == {id(made[1])}
    assert {k for k in set(c1) | set(c2) if c1.get(k) != c2.get(k)} == {"policy_slowdown"}


def test_policy_slowdown_cells_needs_sim_slowdown_cells(tmp_path, inputs, capsys):
    rc, sim, _ = _run(tmp_path, inputs, *TINY, "--policy-slowdown", "cells")
    assert rc == 2 and sim.calls == []
    assert not (tmp_path / "out").exists()
    assert "--sim-slowdown cells" in capsys.readouterr().err


def test_resume_refuses_another_policy_slowdown(tmp_path, inputs, stub_cells, capsys):
    _, run1 = stub_cells
    cells = ("--sim-slowdown", "cells", "--run1-dir", str(run1))
    assert _run(tmp_path, inputs, *TINY, *cells)[0] == 0
    rc, sim, _ = _run(tmp_path, inputs, *TINY, *cells, "--policy-slowdown", "cells")
    assert rc == 2 and sim.calls == []
    assert "policy_slowdown" in capsys.readouterr().err


# ------------------------------------------------------------------------------------------------ (6) review B3, B4
DEFAULT_GRID = ["--mixes", "A", "--rhos", "0.5", "--gamma-mix", "B", "--gamma-rhos", "0.9", "--reps", "2",
                "--n-requests", "20", "--n-warmup", "5", "--seed", str(SEED)]
# sha256 of reps.csv and summary.csv of DEFAULT_GRID (default --methods, FakeSim, FakePolicies, clock frozen at 0),
# computed on the pilot at 194d234, before --methods took exits.ABLATIONS (agent-SIM-b234 STATE.md).
GOLDEN = {"reps.csv": "3449719b87100d7fa54fdf984dbd589fc5cbe66b16314a14b038ee34f62aeb38",
          "summary.csv": "abdeef0a0813324a76c45b3bdd4386620b4dead34090e24fa31a203b6d022787"}


def test_default_methods_keep_the_pilot_specs_and_byte_identical_outputs(tmp_path, inputs, monkeypatch):
    monkeypatch.setattr(P, "time", SimpleNamespace(perf_counter=lambda: 0.0))     # sim_runtime_s = 0.0
    assert P.parse_args(_argv(tmp_path)).methods == list(exits.METHODS)
    rc, sim, pol = _run(tmp_path, inputs, *DEFAULT_GRID)
    assert rc == 0 and len(sim.calls) == 2 * 2 * len(exits.METHODS)
    assert [c.method for c in sim.calls[:5]] == list(exits.METHODS)
    pilot = exits.pilot_specs(1020)
    assert all(c.spec.to_json() == pilot[c.method].to_json() for c in sim.calls)
    for name, sha in GOLDEN.items():
        assert hashlib.sha256((tmp_path / "out" / name).read_bytes()).hexdigest() == sha, name


def test_methods_accept_the_ablations_with_the_run3_specs(tmp_path, inputs):
    sim = FakeSim()
    rc = P.main(_argv(tmp_path, *DEFAULT_GRID, "--methods", "greedy,greedy_gate,coexit_nogate,edgeserving"),
                inputs=inputs, simulate_fn=sim)                                  # the real exits.ExitPolicy
    assert rc == 0 and len(sim.calls) == 2 * 2 * 4
    run3 = exits.run3_specs(1020)
    for c in sim.calls:
        assert c.spec.to_json() == run3[c.method].to_json()
        assert isinstance(c.policy, exits.ExitPolicy) and c.policy.spec is c.spec
    gate = next(c.policy for c in sim.calls if c.method == "greedy_gate")
    assert gate.rule == "greedy" and gate.spec.params["cap_margin"] == C.COEXIT_CAP_MARGIN
    rows = _read_csv(tmp_path / "out" / "reps.csv")[1]
    assert {r["method"] for r in rows} == {"greedy", "greedy_gate", "coexit_nogate", "edgeserving"}
    summary = _read_csv(tmp_path / "out" / "summary.csv")[1]
    assert {r["method"] for r in summary} == {"greedy", "greedy_gate", "coexit_nogate", "edgeserving"}


def test_methods_accept_edgeserving_lp_with_the_sim_specs(tmp_path, inputs):
    # review experiment B1: EdgeServing-LP runs in the pilot next to the methods it is compared with
    sim = FakeSim()
    rc = P.main(_argv(tmp_path, *DEFAULT_GRID, "--methods", "edgeserving,edgeserving_lp,greedy,coexit"),
                inputs=inputs, simulate_fn=sim)                                  # the real exits.ExitPolicy
    assert rc == 0 and len(sim.calls) == 2 * 2 * 4
    specs = exits.sim_specs(1020)
    for c in sim.calls:
        assert c.spec.to_json() == specs[c.method].to_json()
        assert c.policy.preempt == (c.method == "edgeserving_lp")
    lp = next(c.policy for c in sim.calls if c.method == "edgeserving_lp")
    assert lp.rule == "edgeserving" and lp.spec.params["preempt_llama_at_token"] is True
    summary = _read_csv(tmp_path / "out" / "summary.csv")[1]
    assert {r["method"] for r in summary} == {"edgeserving", "edgeserving_lp", "greedy", "coexit"}


@pytest.mark.parametrize("truth", ["m1", "cells"])
def test_policy_slowdown_one_predicts_no_slowdown(tmp_path, inputs, stub_cells, truth):
    made, run1 = stub_cells
    rc, sim, pol = _run(tmp_path, inputs, *TINY, "--sim-slowdown", truth, "--policy-slowdown", "one",
                        "--run1-dir", str(run1))
    assert rc == 0 and len(pol.made) == TINY_SIMS
    want_truth = inputs.m1 if truth == "m1" else made[0]
    assert all(c.slowdown is want_truth for c in sim.calls)          # the simulator truth is untouched
    for p in pol.made:
        assert p.slowdown is not inputs.m1 and p.slowdown is not want_truth
        for i, A, f in (("bert:23", ("bert:23",), 1020), ("bert:23", ("bert:23", "llama:15"), 1020),
                        ("yolo:1", ("bert:5", "llama:3", "yolo:1"), 612)):
            assert p.slowdown.predict(i, A, f) == 1.0
    meta = json.loads((tmp_path / "out" / "meta.json").read_text(encoding="utf-8"))
    assert meta["args"]["policy_slowdown"] == "one" and meta["config"]["policy_slowdown"] == "one"
    assert meta["config"]["sim_slowdown"] == truth and meta["policy_slowdown"] == "one"


def test_policy_slowdown_one_works_with_the_real_exit_policy(tmp_path, inputs):
    sim = FakeSim()
    rc = P.main(_argv(tmp_path, *TINY, "--policy-slowdown", "one"), inputs=inputs, simulate_fn=sim)
    assert rc == 0 and all(isinstance(c.policy, exits.ExitPolicy) for c in sim.calls)
    assert {c.policy.slowdown.predict("bert:23", ("bert:23", "yolo:5"), 1020) for c in sim.calls} == {1.0}


def test_resume_refuses_another_policy_slowdown_one(tmp_path, inputs, capsys):
    assert _run(tmp_path, inputs, *TINY)[0] == 0
    rc, sim, _ = _run(tmp_path, inputs, *TINY, "--policy-slowdown", "one")
    assert rc == 2 and sim.calls == []
    assert "policy_slowdown" in capsys.readouterr().err
