import json
import shutil

import pytest

from exp2 import contracts as C
from exp2.model import freeze as F
from exp2.tests import mocks

FX = mocks.fixture_path("freeze.json")


def _load():
    with open(FX, encoding="utf-8") as fh:
        return json.load(fh)


def _rebuild(obj):
    keep = ("git", "inputs", "slowdown_model", "sim", "constraint", "f_grid_mhz", "conditions")
    return F.build_freeze(**{k: obj[k] for k in keep}, created_wall_ns=obj["created_wall_ns"])


def test_build_matches_fixture_hash_and_runtime():
    obj = _load()
    out = _rebuild(obj)
    assert out == obj                                  # validator passes, content_sha256 and runtime recomputed
    assert out["content_sha256"] == C.content_sha256(out)


def test_write_sidecar_format_and_verify(tmp_path):
    p = tmp_path / "freeze.json"
    digest = F.write_freeze(_load(), str(p))
    assert (tmp_path / "freeze.json.sha256").read_bytes() == f"{digest}  freeze.json\n".encode("ascii")
    assert C.verify_freeze_file(p)["schema"] == "exp2.freeze/1"
    assert F.main(["verify", "--freeze", str(p)]) == 0


def test_one_byte_changed_is_freeze_error(tmp_path):
    p = tmp_path / "freeze.json"
    F.write_freeze(_load(), str(p))
    raw = bytearray(p.read_bytes())
    i = raw.index(b"0.97")
    raw[i + 3] = ord("8")
    p.write_bytes(bytes(raw))
    with pytest.raises(C.FreezeError):
        C.verify_freeze_file(p)
    assert F.main(["verify", "--freeze", str(p)]) == 3


def test_missing_sidecar_is_freeze_error(tmp_path):
    shutil.copy(FX, tmp_path / "freeze.json")
    assert F.main(["verify", "--freeze", str(tmp_path / "freeze.json")]) == 3


def test_make_grid_size_and_inf():
    g = F.make_grid([612, 1020])
    assert len(g) == 3 * 3 * 2 * 2
    assert {(s["k_short"], s["k_long"]) for s in g} == {(a, b) for a in (1, 2, None) for b in (1, 2, None)}
    assert all(C.validate_policyspec(s) for s in g)


def test_evaluate_condition_with_stubs():
    obj = _load()
    cfg = {"condition_id": "W_hi_r060", "workload": obj["conditions"][0]["workload"], "alpha": 5.0}
    ch = obj["conditions"][0]["choices"]
    spec = {k: C.PolicySpec.from_json(v) for k, v in ch.items()}
    sim = mocks.StubSim({}, default={"energy_per_req_j": 0.5})
    grid = F.make_grid([612, 1020])
    cond = F.evaluate_condition(
        cfg, f_fn=lambda c, kind: spec["F"] if kind == "complete_rate" else spec["F_sens"],
        rules_fn=lambda c: {"default": spec["default"], "serial": spec["serial"]},
        game_fn=lambda c: obj["conditions"][0]["game"], sim_fn=lambda s, c: sim(s), grid=grid)
    assert list(cond["choices"]) == ["F", "F_sens", "default", "serial"]
    assert set(cond["predictions"]) == {k for k, _ in F.policy_list(cond)}
    assert "g0" in cond["predictions"] and len(cond["predictions"]) <= 4 + len(grid)
    freeze = _rebuild({**obj, "conditions": [cond]})
    assert freeze["conditions"][0]["compute_s"].keys() == {"F", "game"}


# ---------------------------------------------------------------------------------- D11: measured runs and aliases
def _cond(choices, grid=()):
    return {"choices": {k: dict(v) for k, v in choices.items()}, "grid": [dict(g) for g in grid]}


def _spec(name, ks=1, kl=1, order="fcfs", rule="static", f=1020):
    return C.PolicySpec(name, ks, kl, order, None, rule, f, {}).to_json()


def test_measured_runs_repeats_come_from_the_contract():
    choices = {k: _spec(k, i + 1) for i, k in enumerate(("F", "F_sens", "default", "serial", "corun"))}
    runs = F.measured_runs(_cond(choices))
    assert [(k, n) for k, _, n in runs] == [("F", 3), ("F_sens", 1), ("default", 3), ("serial", 3), ("corun", 3)]
    assert F.n_runs(_cond(choices)) == 13
    assert runs[0][1] == choices["F"]


def test_choice_equal_to_an_earlier_choice_is_an_alias_with_max_repeats():
    choices = {"F": _spec("F", 2), "F_sens": _spec("F_sens", 2), "serial": _spec("serial", 1),
               "serial_stf": _spec("serial_stf", 1, order="stf"), "x": _spec("x", 1)}
    cond = _cond(choices)
    assert [(k, n) for k, _, n in F.measured_runs(cond)] == [("F", 3), ("serial", 3), ("serial_stf", 3)]
    assert F.measured_key_map(cond) == {"F": "F", "F_sens": "F", "serial": "serial", "serial_stf": "serial_stf",
                                        "x": "serial"}


def test_alias_takes_the_max_repeats_over_its_aliases():
    # F_sens (1 repeat) first, F (3 repeats) equal to it: the measured key F_sens runs 3 times
    cond = _cond({"F_sens": _spec("F_sens", 2), "F": _spec("F", 2)})
    assert [(k, n) for k, _, n in F.measured_runs(cond)] == [("F_sens", 3)]
    assert F.measured_key_map(cond) == {"F_sens": "F_sens", "F": "F_sens"}


def test_grid_point_equal_to_a_choice_is_that_choices_alias():
    cond = _cond({"F": _spec("F", 2), "serial": _spec("serial", 1)},
                 [_spec("g0", 1), _spec("g1", 3), _spec("g2", 2), _spec("g3", 3)])
    runs = F.measured_runs(cond)
    assert [(k, n) for k, _, n in runs] == [("F", 3), ("serial", 3), ("g1", 1)]
    assert F.measured_key_map(cond) == {"F": "F", "serial": "serial", "g0": "serial", "g1": "g1", "g2": "F",
                                        "g3": "g1"}


def test_grid_alias_of_a_one_repeat_choice_stays_one_repeat():
    cond = _cond({"F_sens": _spec("F_sens", 2)}, [_spec("g0", 2)])
    assert [(k, n) for k, _, n in F.measured_runs(cond)] == [("F_sens", 1)]


def test_policy_list_is_measured_runs_without_repeats():
    obj = _load()
    cond = obj["conditions"][0]
    assert F.policy_list(cond) == [(k, s) for k, s, _ in F.measured_runs(cond)]
    assert set(F.measured_key_map(cond)) == set(cond["choices"]) | {g["name"] for g in cond["grid"]}


def test_estimate_counts_repeats_of_the_new_table():
    obj = _load()
    cond = obj["conditions"][0]
    assert F.n_runs(cond) == 12            # F, default, serial x3; F_sens, g0, g1 x1
    sim = obj["sim"]
    want = 60.0 + 12 * (5.0 + sim["n_requests"] / cond["workload"]["rate_per_s"])
    assert F.estimate_runtime_s(obj["conditions"], sim) == pytest.approx(round(want, 1))
