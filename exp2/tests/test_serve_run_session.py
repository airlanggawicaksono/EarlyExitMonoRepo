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
