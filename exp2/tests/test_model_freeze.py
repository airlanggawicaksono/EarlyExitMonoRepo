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
