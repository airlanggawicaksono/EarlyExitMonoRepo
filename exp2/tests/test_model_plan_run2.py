import json
import shutil

from exp2 import contracts as C
from exp2.model import freeze as F
from exp2.model import plan_run2 as P
from exp2.tests import mocks


def _fx(name):
    with open(mocks.fixture_path(name), encoding="utf-8") as fh:
        return json.load(fh)


def test_plan_equals_fixture():
    freeze = _fx("freeze.json")
    plan = P.plan_run2(freeze)
    assert plan == _fx("plan_run2.json")               # incl. freeze {path, content_sha256}, estimated_wall_s, sha256
    assert plan["freeze"]["content_sha256"] == freeze["content_sha256"]
    assert plan["estimated_wall_s"] == freeze["runtime_estimate_s"]
    assert plan["sha256"] == C.plan_sha256(plan)


def test_repeats_and_dedupe():
    plan = P.plan_run2(_fx("freeze.json"))
    reps = {}
    for c in plan["conditions"]:
        reps[c["policy_key"]] = reps.get(c["policy_key"], 0) + 1
    assert reps == {"F": 3, "F_sens": 1, "default": 3, "default_pinned": 3, "serial": 3, "serial_stf": 3, "g0": 1,
                    "g1": 1}
    freeze = _fx("freeze.json")
    cond = freeze["conditions"][0]
    cond["grid"][0] = {**cond["choices"]["serial"], "name": "gdup"}     # same policy as `serial`, other name
    freeze = C.with_content_sha256(freeze)
    keys = [c["policy_key"] for c in P.plan_run2(freeze)["conditions"]]
    assert "gdup" not in keys


def test_cli_roundtrip(tmp_path):
    for n in ("freeze.json", "freeze.json.sha256"):
        shutil.copy(mocks.fixture_path(n), tmp_path / n)
    out = tmp_path / "plan_run2.json"
    assert P.main(["--freeze", str(tmp_path / "freeze.json"), "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == _fx("plan_run2.json")


def test_cli_rejects_tampered_freeze(tmp_path):
    shutil.copy(mocks.fixture_path("freeze.json"), tmp_path / "freeze.json")
    shutil.copy(mocks.fixture_path("freeze.json.sha256"), tmp_path / "freeze.json.sha256")
    p = tmp_path / "freeze.json"
    p.write_bytes(p.read_bytes().replace(b"0.97", b"0.98", 1))
    assert P.main(["--freeze", str(p), "--out", str(tmp_path / "plan.json")]) == 3
    assert not (tmp_path / "plan.json").exists()


# ------------------------------------------------------------------------------------------ D11: blocks and seeds
THREE = ("F", "default", "default_pinned", "serial", "serial_stf")


def _plan():
    return P.plan_run2(_fx("freeze.json"))


def test_order_seed_is_sim_seed_plus_offset():
    freeze = _fx("freeze.json")
    plan = P.plan_run2(freeze)
    assert plan["order_seed"] == freeze["sim"]["seed"] + C.ORDER_SEED_OFFSET
    C.validate_plan(plan)


def test_arrival_seed_is_paired_by_repeat():
    freeze = _fx("freeze.json")
    for c in _plan()["conditions"]:
        assert c["seed"] == freeze["sim"]["seed"] + C.ARRIVAL_SEED_OFFSET + c["repeat"]


def test_plan_is_deterministic_and_order_seed_changes_the_order():
    a, b = _plan(), _plan()
    assert a == b
    freeze = _fx("freeze.json")
    freeze["sim"]["seed"] = 5
    freeze = C.with_content_sha256(freeze)
    c = P.plan_run2(freeze)
    ids = lambda p: [x["condition_id"] for x in p["conditions"]]
    assert sorted(ids(c)) == sorted(ids(a)) and ids(c) != ids(a)
    assert c["order_seed"] == 5 + C.ORDER_SEED_OFFSET


def test_every_run_appears_once_and_repeats_match_measured_runs():
    freeze = _fx("freeze.json")
    plan = P.plan_run2(freeze)
    ids = [c["condition_id"] for c in plan["conditions"]]
    assert len(ids) == len(set(ids)) == sum(F.n_runs(fc) for fc in freeze["conditions"]) == 18
    want = {f"W_hi_r060_{k}_r{r}" for k, n in (("F", 3), ("default", 3), ("default_pinned", 3), ("serial", 3),
                                                ("serial_stf", 3), ("F_sens", 1), ("g0", 1), ("g1", 1))
            for r in range(n)}
    assert set(ids) == want


def test_blocks_hold_repeat_r_of_every_three_repeat_key_and_singles_are_dealt_round_robin():
    plan = _plan()
    conds = plan["conditions"]
    n_blocks = C.N_BLOCKS
    sizes = [0] * n_blocks
    for c in conds:
        if c["policy_key"] in THREE:
            sizes[c["repeat"]] += 1
    assert sizes == [5, 5, 5]
    # 3 single runs dealt to blocks 0, 1, 2: each block has 5 + 1 = 6 runs, so block b is conds[6b:6b+6]
    for b in range(n_blocks):
        block = conds[6 * b:6 * b + 6]
        reps = [c["repeat"] for c in block if c["policy_key"] in THREE]
        assert reps == [b] * 5
        assert sum(c["policy_key"] in ("F_sens", "g0", "g1") for c in block) == 1


def test_block_order_matches_a_pcg64_reference():
    import numpy as np
    plan = _plan()
    freeze = _fx("freeze.json")
    rng = np.random.Generator(np.random.PCG64(freeze["sim"]["seed"] + C.ORDER_SEED_OFFSET))
    singles = ["F_sens", "g0", "g1"]
    dealt = [singles[i] for i in rng.permutation(3)]
    blocks = [[(k, b) for k in THREE] + [(dealt[b], 0)] for b in range(3)]
    want = []
    for b in range(3):
        for i in rng.permutation(len(blocks[b])):
            k, r = blocks[b][i]
            want.append(f"W_hi_r060_{k}_r{r if k in THREE else 0}")
    assert [c["condition_id"] for c in plan["conditions"]] == want


def test_alias_grid_point_is_not_planned(tmp_path):
    freeze = _fx("freeze.json")
    cond = freeze["conditions"][0]
    cond["grid"][0] = {**cond["choices"]["serial"], "name": "gdup"}
    freeze = C.with_content_sha256(freeze)
    keys = {c["policy_key"] for c in P.plan_run2(freeze)["conditions"]}
    assert "gdup" not in keys and "g1" in keys


# ------------------------------------------------------------------------------------------ D32-5: confirm plan
def _summary(oracle_key="g0", n_repeats=1):
    return {"oracle": {"W_hi_r060": {"policy_key": oracle_key, "energy_per_req_j": 0.3}},
            "per_condition": {f"W_hi_r060_{oracle_key}": {"n_repeats": n_repeats}}}


def test_confirm_plan_remeasures_the_best_key_with_repeats_1_and_2():
    freeze = _fx("freeze.json")
    plan = P.plan_confirm(freeze, _summary("g0"))
    C.validate_plan(plan)
    assert plan["kind"] == "open" and plan["freeze"]["content_sha256"] == freeze["content_sha256"]
    got = sorted((c["policy_key"], c["repeat"], c["confirm_of"], c["seed"]) for c in plan["conditions"])
    base = freeze["sim"]["seed"] + C.ARRIVAL_SEED_OFFSET
    assert got == [("g0", 1, "g0", base + 1), ("g0", 2, "g0", base + 2)]
    assert [c["condition_id"] for c in plan["conditions"]] == ["W_hi_r060_g0_r1", "W_hi_r060_g0_r2"]
    assert P.plan_confirm(freeze, _summary("g0")) == plan                      # deterministic


def test_confirm_plan_is_none_when_f_is_best_or_the_best_key_has_three_repeats():
    freeze = _fx("freeze.json")
    assert P.plan_confirm(freeze, _summary("F")) is None
    assert P.plan_confirm(freeze, _summary("serial", n_repeats=3)) is None
    assert P.plan_confirm(freeze, {"oracle": {}, "per_condition": {}}) is None


def test_confirm_plan_resolves_an_alias_to_its_measured_key():
    freeze = _fx("freeze.json")
    cond = freeze["conditions"][0]
    cond["grid"][0] = {**cond["choices"]["F_sens"], "name": "gdup"}        # alias of F_sens (1 repeat)
    freeze = C.with_content_sha256(freeze)
    plan = P.plan_confirm(freeze, _summary("gdup"))
    assert {c["policy_key"] for c in plan["conditions"]} == {"F_sens"}


def test_confirm_cli_writes_the_plan_or_says_nothing_to_do(tmp_path, capsys):
    for n in ("freeze.json", "freeze.json.sha256"):
        shutil.copy(mocks.fixture_path(n), tmp_path / n)
    summ = tmp_path / "summary.json"
    out = tmp_path / "confirm.json"
    summ.write_text(json.dumps(_summary("g0")), encoding="utf-8")
    assert P.main(["--freeze", str(tmp_path / "freeze.json"), "--confirm-from", str(summ), "--out", str(out)]) == 0
    assert len(C.validate_plan(json.loads(out.read_text(encoding="utf-8")))["conditions"]) == 2
    out2 = tmp_path / "none.json"
    summ.write_text(json.dumps(_summary("F")), encoding="utf-8")
    assert P.main(["--freeze", str(tmp_path / "freeze.json"), "--confirm-from", str(summ), "--out", str(out2)]) == 0
    assert not out2.exists() and "no confirm run needed" in capsys.readouterr().out
    assert P.main(["--freeze", str(tmp_path / "freeze.json"), "--confirm-from", str(tmp_path / "nope.json"),
                   "--out", str(out2)]) == 2
