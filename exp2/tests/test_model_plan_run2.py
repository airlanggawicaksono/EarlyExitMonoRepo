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
    assert reps == {"F": 3, "F_sens": 1, "default": 3, "serial": 3, "g0": 1, "g1": 1}
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
    assert len(ids) == len(set(ids)) == sum(F.n_runs(fc) for fc in freeze["conditions"]) == 12
    want = {f"W_hi_r060_{k}_r{r}" for k, n in (("F", 3), ("default", 3), ("serial", 3), ("F_sens", 1), ("g0", 1),
                                                ("g1", 1)) for r in range(n)}
    assert set(ids) == want


def test_blocks_hold_repeat_r_of_every_three_repeat_key_and_singles_are_dealt_round_robin():
    plan = _plan()
    conds = plan["conditions"]
    n_blocks = C.N_BLOCKS
    sizes = [0] * n_blocks
    for c in conds:
        if c["policy_key"] in ("F", "default", "serial"):
            sizes[c["repeat"]] += 1
    assert sizes == [3, 3, 3]
    # 3 single runs dealt to blocks 0, 1, 2: each block has 3 + 1 = 4 runs, so block b is conds[4b:4b+4]
    for b in range(n_blocks):
        block = conds[4 * b:4 * b + 4]
        reps = [c["repeat"] for c in block if c["policy_key"] in ("F", "default", "serial")]
        assert reps == [b, b, b]
        assert sum(c["policy_key"] in ("F_sens", "g0", "g1") for c in block) == 1


def test_block_order_matches_a_pcg64_reference():
    import numpy as np
    plan = _plan()
    freeze = _fx("freeze.json")
    rng = np.random.Generator(np.random.PCG64(freeze["sim"]["seed"] + C.ORDER_SEED_OFFSET))
    singles = ["F_sens", "g0", "g1"]
    dealt = [singles[i] for i in rng.permutation(3)]
    blocks = [[(k, b) for k in ("F", "default", "serial")] + [(dealt[b], 0)] for b in range(3)]
    want = []
    for b in range(3):
        for i in rng.permutation(len(blocks[b])):
            k, r = blocks[b][i]
            want.append(f"W_hi_r060_{k}_r{r if k in ('F', 'default', 'serial') else 0}")
    assert [c["condition_id"] for c in plan["conditions"]] == want


def test_alias_grid_point_is_not_planned(tmp_path):
    freeze = _fx("freeze.json")
    cond = freeze["conditions"][0]
    cond["grid"][0] = {**cond["choices"]["serial"], "name": "gdup"}
    freeze = C.with_content_sha256(freeze)
    keys = {c["policy_key"] for c in P.plan_run2(freeze)["conditions"]}
    assert "gdup" not in keys and "g1" in keys
