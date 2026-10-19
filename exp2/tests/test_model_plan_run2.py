import json
import shutil

from exp2 import contracts as C
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
    assert reps == {"F": 3, "F_sens": 1, "default": 3, "serial": 1, "g0": 1, "g1": 1}
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
