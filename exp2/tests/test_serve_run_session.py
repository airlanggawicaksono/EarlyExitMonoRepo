"""Smoke tests for exp2.serve.run_session (C2b): --adapter fake, dry board, no model."""
import json
import shutil
from pathlib import Path

from exp2 import contracts
from exp2.board.dryroot import make_dry_root
from exp2.serve import run_session
from exp2.serve.fake_adapter import default_spec

FIX = Path(__file__).resolve().parent / "fixtures"


def _closed_plan():
    def cond(cid, mode, f, **kw):
        c = {"condition_id": cid, "mode": mode, "arm": "pinned", "gpu_mhz": f}
        c.update(kw)
        return c
    plan = {"schema": contracts.SCHEMAS["plan"], "kind": "closed",
            "session": {"families": ["bert", "vision"], "exits": {"bert": [5], "vision": [24]}, "emc_mhz": 3199.0,
                        "settle_s": 0.05, "idle_window_s": 0.2, "n_inputs": 4, "preprocess": "per_request"},
            "freeze": None, "estimated_wall_s": 5.0,
            "conditions": [cond("idle_f1020", "idle", 1020, duration_s=0.2),
                           cond("solo_bert-5_f1020_r0", "closed", 1020, cell=["bert:5"], duration_s=0.3,
                                n_warmup=1, split="solo", repeat=0),
                           cond("pair_f612_r0", "closed", 612, cell=["bert:5", "vision:24"], duration_s=0.3,
                                n_warmup=1, split="train", repeat=0)]}
    return contracts.with_plan_sha256(plan)


def test_closed_session_smoke(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(_closed_plan()))
    specs = tmp_path / "specs"
    specs.mkdir()
    for fam, ex in (("bert", [5]), ("vision", [24])):
        (specs / ("%s.json" % fam)).write_text(json.dumps(default_spec(fam, ex, seg_ms=2.0, load_s=0.0)))
    root = make_dry_root(tmp_path / "root")
    out = tmp_path / "sess"
    rc = run_session.main(["--plan", str(plan), "--types", str(FIX / "types_from_run1.json"), "--out", str(out),
                           "--adapter", "fake", "--fake-spec-dir", str(specs), "--dry-run", "--fake-root", str(root)])
    assert rc == 0
    sess = contracts.validate_session(json.loads((out / "session.json").read_text()))
    assert sess["freeze_content_sha256"] is None and sess["t_wall_start_ns"] > 0
    assert sess["board"]["dry"] and sess["conditions"] == [c["condition_id"] for c in _closed_plan()["conditions"]]
    for cid in sess["conditions"]:
        m = contracts.validate_manifest(json.loads((out / cid / "manifest.json").read_text()))
        assert m["valid"], m["invalid_reason"]
    assert (out / "pair_f612_r0" / "requests.jsonl").read_text().strip()
    assert (out / "telemetry" / "power.csv").exists() and (out / "telemetry" / "meta.json").exists()


def test_corrupt_freeze_exit3_no_ports(tmp_path):
    plan = json.loads((FIX / "plan_run2.json").read_text())
    for name in ("freeze.json", "freeze.json.sha256"):
        shutil.copy(FIX / name, tmp_path / name)
    (tmp_path / "freeze.json").write_text((FIX / "freeze.json").read_text() + " ")
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    out = tmp_path / "sess"
    rc = run_session.main(["--plan", str(tmp_path / "plan.json"), "--types", str(FIX / "types_from_run1.json"),
                           "--out", str(out), "--adapter", "fake", "--fake-spec-dir", str(tmp_path)])
    assert rc == 3
    assert not list(out.rglob("*.port"))


def _open_setup(tmp_path, types_name="types_from_run1.json"):
    for name in ("freeze.json", "freeze.json.sha256"):
        shutil.copy(FIX / name, tmp_path / name)
    shutil.copy(FIX / "plan_run2.json", tmp_path / "plan.json")
    types = tmp_path / "types.json"
    shutil.copy(FIX / types_name, types)
    return tmp_path / "plan.json", types


def _no_spawn(monkeypatch):
    """Session.run replaced by a marker: reaching it means the guards passed and nothing was spawned."""
    calls = []
    monkeypatch.setattr(run_session.Session, "run", lambda self: calls.append(1) or 0)
    return calls


def test_types_hash_mismatch_exit3_before_spawn(tmp_path, monkeypatch):
    plan, types = _open_setup(tmp_path)
    types.write_text(types.read_text() + " ")                    # same content, different bytes -> different sha256
    calls = _no_spawn(monkeypatch)
    rc = run_session.main(["--plan", str(plan), "--types", str(types), "--out", str(tmp_path / "sess"),
                           "--adapter", "fake", "--fake-spec-dir", str(tmp_path)])
    assert rc == 3 and calls == []
    assert not (tmp_path / "sess").exists() or not list((tmp_path / "sess").rglob("*.port"))


def test_types_hash_match_passes_guard(tmp_path, monkeypatch):
    plan, types = _open_setup(tmp_path)
    calls = _no_spawn(monkeypatch)
    rc = run_session.main(["--plan", str(plan), "--types", str(types), "--out", str(tmp_path / "sess"),
                           "--adapter", "fake", "--fake-spec-dir", str(tmp_path)])
    assert rc == 0 and calls == [1]


def test_closed_plan_ignores_types_hash(tmp_path, monkeypatch):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(_closed_plan()))
    types = tmp_path / "types.json"
    types.write_text((FIX / "types_from_run1.json").read_text() + " ")
    calls = _no_spawn(monkeypatch)
    rc = run_session.main(["--plan", str(plan), "--types", str(types), "--out", str(tmp_path / "sess"),
                           "--adapter", "fake", "--fake-spec-dir", str(tmp_path)])
    assert rc == 0 and calls == [1]


def test_check_types_hash_unit(tmp_path):
    types = tmp_path / "t.json"
    types.write_text("{}")
    good = contracts.sha256_file(types)
    run_session.check_types_hash({"inputs": {"types_sha256": good}}, types)
    try:
        run_session.check_types_hash({"inputs": {"types_sha256": "0" * 64}}, types)
    except contracts.FreezeError as exc:
        assert "types" in str(exc)
    else:
        raise AssertionError("FreezeError expected")
