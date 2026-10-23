"""S3 tests for Run 3 in exp2.serve.run_session (PLAN PL2 a, e): exit policies, the --slowdown guards, launch.sh.

No server, no board, no model: Session.run is replaced by a marker, so reaching it means every guard passed.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from exp2 import contracts as C
from exp2 import exits as X
from exp2.model import slowdown as SD
from exp2.policies import Policy
from exp2.serve import run_session

FIX = Path(__file__).resolve().parent / "fixtures"
ROOT = Path(__file__).resolve().parents[2]
LAUNCH = ROOT / "exp2" / "board" / "scripts" / "launch.sh"
STUB_SD = object()           # ExitPolicy only stores the slowdown model; nothing here calls predict


def _types():
    return C.load_types(FIX / "types_from_run1.json")


def _run3_plan(methods=("coexit",)):
    """Open Run 3 conditions (one per method) on the Run 2 plan fixture, freeze reference kept, re-hashed."""
    p = json.loads((FIX / "plan_run2.json").read_text(encoding="utf-8"))
    p["session"]["families"] = ["bert", "yolo"]
    p["session"]["exits"] = {"bert": [5, 23], "yolo": [1, 5]}
    base = next(dict(x) for x in p["conditions"] if x["mode"] == "open")
    specs = X.pilot_specs(1020)
    conds = []
    for m in methods:
        c = dict(base)
        c.update(condition_id="A_p_rho060_%s_r0" % m, policy_key=m, policy=specs[m].to_json(),
                 workload={"name": "mixA", "types": ["bert:23", "yolo:5"], "probs": [0.5, 0.5], "rho": 0.6,
                           "rate_per_s": 8.56, "arrival": "poisson", "cv_a": 1.0},
                 alpha=None, deadlines_s={"bert": 0.1, "yolo": 0.1})
        conds.append(c)
    p["conditions"] = conds
    p["sha256"] = C.plan_sha256(p)
    return p


def _setup(tmp_path, plan):
    """plan.json + fixture freeze + types + slowdown in tmp_path; the fixture freeze hashes match both files."""
    for name in ("freeze.json", "freeze.json.sha256"):
        shutil.copy(FIX / name, tmp_path / name)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    types = tmp_path / "types.json"
    shutil.copy(FIX / "types_from_run1.json", types)
    sd = tmp_path / "slowdown.json"
    shutil.copy(FIX / "slowdown_model.json", sd)
    return plan_path, types, sd


def _no_spawn(monkeypatch):
    """Session.run replaced by a marker that keeps the Session: reaching it means nothing was refused."""
    seen = []
    monkeypatch.setattr(run_session.Session, "run", lambda self: seen.append(self) or 0)
    return seen


def _main(tmp_path, plan_path, types, *extra):
    return run_session.main(["--plan", str(plan_path), "--types", str(types), "--out", str(tmp_path / "sess"),
                             "--adapter", "fake", "--fake-spec-dir", str(tmp_path), *extra])


def _no_ports(tmp_path):
    out = tmp_path / "sess"
    return not out.exists() or not list(out.rglob("*.port"))


# ---------------------------------------------------------------- make_policy
@pytest.mark.parametrize("method", X.METHODS)
def test_make_policy_picks_exit_policy_for_an_exit_rule_spec(method):
    spec = X.pilot_specs(1020)[method].to_json()
    pol = run_session.make_policy(spec, _types(), STUB_SD)
    assert isinstance(pol, X.ExitPolicy)
    assert pol.rule == spec["params"]["exit_rule"] and pol.slowdown is STUB_SD


def test_make_policy_keeps_policy_for_every_run2_spec():
    plan = json.loads((FIX / "plan_run2.json").read_text(encoding="utf-8"))
    specs = [c["policy"] for c in plan["conditions"] if c["mode"] == "open"]
    assert specs
    for spec in specs:
        pol = run_session.make_policy(spec, _types(), STUB_SD)
        assert type(pol) is Policy, spec["name"]


# ---------------------------------------------------------------- --slowdown guards (before any spawn)
@pytest.mark.parametrize("method", ["greedy", "coexit"])
def test_greedy_or_coexit_without_slowdown_exit2_before_spawn(tmp_path, monkeypatch, method):
    plan_path, types, _ = _setup(tmp_path, _run3_plan(("all_final", method)))
    seen = _no_spawn(monkeypatch)
    assert _main(tmp_path, plan_path, types) == 2
    assert seen == [] and _no_ports(tmp_path)


def test_exit_rules_without_a_model_run_without_slowdown(tmp_path, monkeypatch):
    plan_path, types, _ = _setup(tmp_path, _run3_plan(("all_final", "all_early", "edgeserving")))
    seen = _no_spawn(monkeypatch)
    assert _main(tmp_path, plan_path, types) == 0
    assert len(seen) == 1 and seen[0].slowdown is None


def test_wrong_slowdown_hash_exit3_before_spawn(tmp_path, monkeypatch):
    plan_path, types, sd = _setup(tmp_path, _run3_plan())
    sd.write_text(sd.read_text(encoding="utf-8") + " ", encoding="utf-8")   # same JSON, different sha256
    seen = _no_spawn(monkeypatch)
    assert _main(tmp_path, plan_path, types, "--slowdown", str(sd)) == 3
    assert seen == [] and _no_ports(tmp_path)


def test_missing_slowdown_file_exit2_before_spawn(tmp_path, monkeypatch, capsys):
    plan_path, types, _ = _setup(tmp_path, _run3_plan())
    seen = _no_spawn(monkeypatch)
    assert _main(tmp_path, plan_path, types, "--slowdown", str(tmp_path / "nope.json")) == 2
    err = capsys.readouterr().err
    assert seen == [] and "unrecognized" not in err and "nope.json" in err


def test_matching_slowdown_reaches_the_session(tmp_path, monkeypatch):
    plan_path, types, sd = _setup(tmp_path, _run3_plan(("greedy", "coexit")))
    seen = _no_spawn(monkeypatch)
    assert _main(tmp_path, plan_path, types, "--slowdown", str(sd)) == 0
    model = seen[0].slowdown
    assert isinstance(model, SD.M2) and model.eta == {612: 0.93, 1020: 0.91}


def test_run2_plan_still_passes_without_slowdown(tmp_path, monkeypatch):
    plan_path, types, _ = _setup(tmp_path, json.loads((FIX / "plan_run2.json").read_text(encoding="utf-8")))
    seen = _no_spawn(monkeypatch)
    assert _main(tmp_path, plan_path, types) == 0
    assert seen[0].slowdown is None


# ---------------------------------------------------------------- run_condition hands the policy over
class _Board:
    def apply_arm(self, arm, f, emc):
        return {"gpu_mhz": f, "emc_mhz": emc, "governor_free": arm == "governor"}


class _Disp:
    def __init__(self):
        self.policies = []

    def run_open(self, cond, sched, policy, cdir):
        self.policies.append(policy)
        return {"n_requests": len(sched), "dead_servers": {}, "freq": {"errors": []}}


def test_run_condition_gives_the_dispatcher_the_exit_policy(tmp_path):
    plan = _run3_plan(("coexit",))
    run2 = json.loads((FIX / "plan_run2.json").read_text(encoding="utf-8"))
    plan["conditions"].append(next(c for c in run2["conditions"] if c["mode"] == "open"))
    plan["session"]["settle_s"] = 0.0
    args = run_session.build_parser().parse_args(["--plan", "p", "--types", "t", "--out", str(tmp_path / "s")])
    sess = run_session.Session(args, plan, None, _types(), slowdown=STUB_SD)
    sess.board, disp = _Board(), _Disp()
    for cond in plan["conditions"]:
        sess.run_condition(disp, cond)
    assert sess.n_invalid == 0
    ex, base = disp.policies
    assert isinstance(ex, X.ExitPolicy) and ex.rule == "coexit" and ex.slowdown is STUB_SD
    assert type(base) is Policy


# ---------------------------------------------------------------- launch.sh
def test_launch_sh_passes_extra_args_to_run_session():
    text = LAUNCH.read_text(encoding="utf-8")
    call = text[text.index("-m exp2.serve.run_session"):]
    call = call[:call.index(">>")]
    assert '"${@:6}"' in call


def test_launch_sh_syntax():
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    proc = subprocess.run([bash, "-n"], input=LAUNCH.read_bytes(), capture_output=True, timeout=30)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
