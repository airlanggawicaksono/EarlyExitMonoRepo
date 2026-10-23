"""S4 tests for exp2.model.plan_run3 (PLAN PL2 c, e S4) on the frozen Run 2 freeze and types.

Expected values are written by hand from the plan text, except the pilot functions the plan must reproduce
(coexit_pilot._workload, coexit_pilot._schedule, exits.pilot_specs), which are called as the reference.
"""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from exp2 import arrivals
from exp2 import contracts as C
from exp2 import exits as X
from exp2.model import coexit_pilot as CP
from exp2.model import plan_run2
from exp2.model import plan_run3 as P
from exp2.tests import mocks

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "exp2" / "frozen" / "run2"
FREEZE, TYPES = FROZEN / "freeze.json", FROZEN / "types.json"
SEED = 20261022
METHODS = ("all_final", "all_early", "edgeserving", "greedy", "coexit")
DEADLINES = {"bert": 0.1, "yolo": 0.1, "llama": 3.35}
CELL = {"A_p_rho060": ("A", "poisson", 1.0, 0.6), "A_p_rho090": ("A", "poisson", 1.0, 0.9),
        "A_p_rho120": ("A", "poisson", 1.0, 1.2), "B_p_rho060": ("B", "poisson", 1.0, 0.6),
        "B_p_rho090": ("B", "poisson", 1.0, 0.9), "B_p_rho120": ("B", "poisson", 1.0, 1.2),
        "B_g20_rho090": ("B", "gamma", 2.0, 0.9)}
FAMS = {"A": ["bert", "yolo"], "B": ["bert", "llama", "yolo"]}

EXPECTED_A_P_RHO060_COEXIT_R0 = {
    "condition_id": "A_p_rho060_coexit_r0", "mode": "open", "arm": "pinned", "gpu_mhz": 1020,
    "policy_key": "coexit", "freeze_condition_id": "run3_A_p_rho060",
    "policy": {"name": "coexit", "k_short": None, "k_long": None, "order": "fcfs", "lane_cap": None,
               "f_rule": "static", "f_mhz": 1020,
               "params": {"exit_rule": "coexit", "tau_stat": "p95", "cap_margin": 0.053}},
    "workload": {"name": "mixA", "types": ["bert:23", "yolo:5"], "probs": [0.5, 0.5], "rho": 0.6,
                 "rate_per_s": 8.55710216805127, "arrival": "poisson", "cv_a": 1.0},
    "alpha": None, "deadlines_s": {"bert": 0.1, "yolo": 0.1},
    "n_requests": 550, "n_warmup": 50, "seed": 20261022, "repeat": 0}


@pytest.fixture(scope="module")
def types():
    return C.load_types(TYPES)


@pytest.fixture(scope="module")
def freeze():
    return C.verify_freeze_file(FREEZE)


@pytest.fixture(scope="module")
def plan(freeze, types):
    return P.plan_run3(freeze, types)


def _cell(c):
    return c["freeze_condition_id"][len("run3_"):]


def _ids(plan):
    return [c["condition_id"] for c in plan["conditions"]]


def _run(tmp_path, *extra, name="plan.json"):
    out = tmp_path / name
    rc = P.main(["--freeze", str(FREEZE), "--types", str(TYPES), "--out", str(out), *extra])
    return rc, out


def _manifest(c, valid=True):
    """manifest.json of a plan condition, as run_session writes it."""
    return {"schema": C.SCHEMAS["manifest"], "condition_id": c["condition_id"], "mode": "open", "arm": c["arm"],
            "gpu_mhz": c["gpu_mhz"], "emc_mhz": 3199.0, "t_wall_start_ns": 1, "t_mono_start_ns": 0,
            "t_mono_end_ns": 1, "settle_s": 5.0, "valid": valid,
            "invalid_reason": None if valid else "ClockError: lock lost", "board_readback": None,
            "n_requests": c["n_requests"], "n_warmup": c["n_warmup"], "seed": c["seed"],
            "policy_key": c["policy_key"], "policy_spec": c["policy"],
            "freeze_condition_id": c["freeze_condition_id"], "workload": c["workload"], "alpha": c["alpha"],
            "cell": None, "split": None, "repeat": c["repeat"]}


def _write_manifest(session, cid, obj):
    d = session / cid
    d.mkdir(parents=True)
    text = obj if isinstance(obj, str) else json.dumps(C.validate_manifest(obj))
    (d / "manifest.json").write_text(text, encoding="utf-8")


# ---------------------------------------------------------------- full plan
def test_full_plan_has_105_valid_conditions(plan, freeze):
    C.validate_plan(plan)
    assert plan["kind"] == "open" and plan["sha256"] == C.plan_sha256(plan)
    assert plan["freeze"]["content_sha256"] == freeze["content_sha256"]
    assert plan["order_seed"] == SEED + C.ORDER_SEED_OFFSET
    ids = _ids(plan)
    assert len(ids) == len(set(ids)) == 105
    assert set(ids) == {f"{cell}_{m}_r{r}" for cell in CELL for m in METHODS for r in range(3)}


def test_blocks_run_by_repeat_and_follow_a_pcg64_reference(plan):
    assert [c["repeat"] for c in plan["conditions"]] == [0] * 35 + [1] * 35 + [2] * 35
    rng = np.random.Generator(np.random.PCG64(SEED + C.ORDER_SEED_OFFSET))
    want = []
    for rep in range(3):
        block = [f"{cell}_{m}_r{rep}" for cell in CELL for m in METHODS]
        want += [block[i] for i in rng.permutation(len(block))]
    assert _ids(plan) == want


def test_methods_of_a_cell_and_repeat_share_seed_workload_and_deadlines(plan):
    groups = {}
    for c in plan["conditions"]:
        groups.setdefault((_cell(c), c["repeat"]), []).append(c)
    assert len(groups) == 21
    for (cell, rep), cs in groups.items():
        assert sorted(c["policy_key"] for c in cs) == sorted(METHODS)
        assert {c["seed"] for c in cs} == {SEED + rep}
        assert all(c["workload"] == cs[0]["workload"] and c["deadlines_s"] == cs[0]["deadlines_s"] for c in cs)


def test_workload_is_the_pilot_workload(plan, types):
    for c in plan["conditions"]:
        assert c["workload"] == CP._workload(*CELL[_cell(c)], types)
        C.validate_workload(c["workload"], types)


@pytest.mark.parametrize("cell", ["A_p_rho060", "B_g20_rho090"])
def test_deadlines_reproduce_the_pilot_schedule(plan, cell):
    c = next(x for x in plan["conditions"] if x["condition_id"] == f"{cell}_coexit_r1")
    mix = CELL[cell][0]
    assert c["alpha"] is None
    assert c["deadlines_s"] == {f: DEADLINES[f] for f in FAMS[mix]}
    got = C.with_family_deadlines(arrivals.schedule(c["workload"], c["n_requests"], c["seed"]), c["deadlines_s"])
    want = CP._schedule(c["workload"], c["n_requests"], SEED + 1, DEADLINES, 1.0)
    assert [(i["type_id"], i["t_arrival_ns"], i["t_deadline_ns"]) for i in got] == \
        [(i["type_id"], i["t_arrival_ns"], i["t_deadline_ns"]) for i in want]


def test_session_loads_the_exit_menu_of_all_three_families(plan):
    s = plan["session"]
    assert s["families"] == ["bert", "llama", "yolo"]
    assert s["exits"] == {"bert": [5, 23], "llama": [3, 15], "yolo": [1, 5]}
    assert s["exits"] == {f: list(C.EXIT_MENU[f]) for f in s["families"]}
    assert {k: v for k, v in s.items() if k not in ("families", "exits")} == plan_run2.SESSION


def test_specs_are_the_pilot_specs_pinned_at_1020(plan):
    specs = X.pilot_specs(1020)
    for c in plan["conditions"]:
        assert c["policy"] == specs[c["policy_key"]].to_json()
        assert (c["arm"], c["gpu_mhz"], c["n_requests"], c["n_warmup"]) == ("pinned", 1020, 550, 50)


def test_one_condition_equals_the_hand_written_dict(plan):
    c = next(x for x in plan["conditions"] if x["condition_id"] == "A_p_rho060_coexit_r0")
    assert c == EXPECTED_A_P_RHO060_COEXIT_R0


def test_estimated_wall(plan):
    s = plan["session"]
    want = 2 * s["idle_window_s"] + sum(s["settle_s"] + c["n_requests"] / c["workload"]["rate_per_s"]
                                        for c in plan["conditions"])
    assert plan["estimated_wall_s"] == pytest.approx(want, abs=0.1)
    assert 17000 <= plan["estimated_wall_s"] <= 21000


def test_plan_is_deterministic(freeze, types, plan):
    again = P.plan_run3(freeze, types)
    assert again == plan and again["sha256"] == plan["sha256"]


def test_default_plan_is_unchanged_by_the_nogate_ablation(plan):
    # Values computed at eb0447f, before ABLATIONS existed (agent-E1-nogate STATE.md).
    assert plan["sha256"] == "a3234a15e0cc81e09b85aa376871c80af3d574ac9a1ed106545ee68a8ff52764"
    text = json.dumps(plan, indent=2, allow_nan=False) + "\n"
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == \
        "9f201244b2033937adc9b3417eb87eb7ee18f142dfc1f3dd7be0b1e0a37f9d97"


def test_nogate_conditions_mirror_the_coexit_conditions(freeze, types, plan):
    ng = P.plan_run3(freeze, types, methods=("coexit_nogate",))
    assert len(ng["conditions"]) == 21
    co = {(_cell(c), c["repeat"]): c for c in plan["conditions"] if c["policy_key"] == "coexit"}
    for c in ng["conditions"]:
        ref = co[(_cell(c), c["repeat"])]
        assert c["condition_id"] == f"{_cell(c)}_coexit_nogate_r{c['repeat']}" and c["policy_key"] == "coexit_nogate"
        assert c["policy"] == dict(ref["policy"], name="coexit_nogate", params=dict(ref["policy"]["params"],
                                                                                     cap_margin=-1.0))
        same = ("seed", "workload", "deadlines_s", "freeze_condition_id", "n_requests", "n_warmup", "alpha", "arm",
                "gpu_mhz", "mode")
        assert {k: c[k] for k in same} == {k: ref[k] for k in same}


def test_cli_methods_accepts_the_ablation_in_a_fixed_order(tmp_path):
    rc1, out1 = _run(tmp_path, "--methods", "coexit_nogate", name="ng.json")
    assert rc1 == 0
    assert len(json.loads(out1.read_text(encoding="utf-8"))["conditions"]) == 21
    rc2, out2 = _run(tmp_path, "--methods", "coexit_nogate,coexit", name="a.json")
    rc3, out3 = _run(tmp_path, "--methods", "coexit,coexit_nogate", name="b.json")
    assert rc2 == rc3 == 0 and out2.read_bytes() == out3.read_bytes()
    assert {c["policy_key"] for c in json.loads(out2.read_text(encoding="utf-8"))["conditions"]} == \
        {"coexit", "coexit_nogate"}


# ---------------------------------------------------------------- CLI
def test_cli_writes_the_validated_plan(tmp_path, plan):
    rc, out = _run(tmp_path)
    assert rc == 0
    got = C.validate_plan(json.loads(out.read_text(encoding="utf-8")))
    assert got["conditions"] == plan["conditions"] and got["session"] == plan["session"]
    assert got["estimated_wall_s"] == plan["estimated_wall_s"]
    assert (out.parent / got["freeze"]["path"]).resolve() == FREEZE.resolve()
    rc2, out2 = _run(tmp_path, name="again.json")
    assert rc2 == 0 and out2.read_bytes() == out.read_bytes()


def test_smoke_gives_15_short_conditions(tmp_path):
    rc, out = _run(tmp_path, "--smoke")
    assert rc == 0
    p = C.validate_plan(json.loads(out.read_text(encoding="utf-8")))
    cs = p["conditions"]
    assert len(cs) == 15
    assert {c["condition_id"] for c in cs} == {f"smoke_{cell}_{m}_r0" for cell in ("A_p_rho120", "B_p_rho120",
                                                                                 "B_g20_rho090") for m in METHODS}
    assert {(c["n_requests"], c["n_warmup"], c["repeat"], c["seed"]) for c in cs} == {(60, 10, 0, SEED)}


def test_reps_and_methods_options(tmp_path):
    rc, out = _run(tmp_path, "--reps", "5", "--methods", "coexit,edgeserving")
    assert rc == 0
    cs = json.loads(out.read_text(encoding="utf-8"))["conditions"]
    assert len(cs) == 7 * 2 * 5
    assert [c["repeat"] for c in cs] == [r for r in range(5) for _ in range(14)]
    assert {c["policy_key"] for c in cs} == {"coexit", "edgeserving"}


def test_skip_done_drops_valid_manifests_and_keeps_the_order(tmp_path, plan):
    sess = tmp_path / "sess1"
    by_id = {c["condition_id"]: c for c in plan["conditions"]}
    done = ["B_g20_rho090_coexit_r2", "A_p_rho060_all_final_r0"]
    for cid in done:
        _write_manifest(sess, cid, _manifest(by_id[cid]))
    _write_manifest(sess, "A_p_rho090_greedy_r1", _manifest(by_id["A_p_rho090_greedy_r1"], valid=False))
    _write_manifest(sess, "B_p_rho060_greedy_r0", "{not json")
    (sess / "A_p_rho120_coexit_r0").mkdir()                       # started, no manifest yet
    rc, out = _run(tmp_path, "--skip-done", str(sess))
    assert rc == 0
    p = C.validate_plan(json.loads(out.read_text(encoding="utf-8")))
    assert _ids(p) == [cid for cid in _ids(plan) if cid not in done]
    assert len(p["conditions"]) == 103
    gone = sum(5.0 + by_id[cid]["n_requests"] / by_id[cid]["workload"]["rate_per_s"] for cid in done)
    assert p["estimated_wall_s"] == pytest.approx(plan["estimated_wall_s"] - gone, abs=0.2)


def test_skip_done_with_nothing_left_writes_no_file(tmp_path, freeze, types, capsys):
    smoke = P.plan_run3(freeze, types, smoke=True)
    s1, s2 = tmp_path / "s1", tmp_path / "s2"
    s2.mkdir()
    for i, c in enumerate(smoke["conditions"]):
        _write_manifest(s1 if i % 2 else s2, c["condition_id"], _manifest(c))
    rc, out = _run(tmp_path, "--smoke", "--skip-done", str(s1), str(s2))
    assert rc == 0 and not out.exists()
    assert "nothing left" in capsys.readouterr().out


def test_corrupted_freeze_gives_rc_3(tmp_path):
    shutil.copy(FREEZE, tmp_path / "freeze.json")
    shutil.copy(str(FREEZE) + ".sha256", tmp_path / "freeze.json.sha256")
    p = tmp_path / "freeze.json"
    raw = p.read_bytes()
    p.write_bytes(raw[:100] + (b"1" if raw[100:101] != b"1" else b"2") + raw[101:])
    out = tmp_path / "plan.json"
    assert P.main(["--freeze", str(p), "--types", str(TYPES), "--out", str(out)]) == 3
    assert not out.exists()


def test_types_of_another_freeze_give_rc_3(tmp_path):
    out = tmp_path / "plan.json"
    other = mocks.fixture_path("types_from_run1.json")
    assert P.main(["--freeze", str(FREEZE), "--types", str(other), "--out", str(out)]) == 3
    assert not out.exists()


@pytest.mark.parametrize("extra", [
    ["--methods", "coexit,nope"], ["--methods", "coexit,coexit"], ["--methods", "coexit_nogate,coexit_nogate"],
    ["--methods", ""], ["--reps", "0"],
    ["--skip-done", "does/not/exist"], ["--bogus"]])
def test_usage_errors_give_rc_2(tmp_path, extra):
    rc, out = _run(tmp_path, *[x.replace("does/not/exist", str(tmp_path / "nope")) for x in extra])
    assert rc == 2 and not out.exists()
