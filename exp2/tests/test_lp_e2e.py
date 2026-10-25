"""S7 (board LP plan, PLAN PL3 section 3): local end-to-end proof of limited preemption with FAKE models.

Owner 2026-10-26: "jangan lewat simulator ya, ini tidak valid dude ... kita harus bisa dulu di lokal gitu aja deh
mendingan".

Real pieces: two `python -m exp2.serve.server --adapter fake` subprocesses (llama, bert), real sockets (exp2.wire), the
real Dispatcher, AdmissionCore and exits.ExitPolicy built as run_session.make_policy builds it, then exp2.analyze.run3.
Fake pieces: the adapters sleep instead of computing (llama 20 ms per token, bert 5 ms per checkpoint). No torch.

    t (ms)  0        100       200       300                      ~800
    llama   L1 ----||  b1 ||-----||  b2 ||-----||  b3 ||----------- done (32 tokens)
    bert             b1 arrives, L1 pauses at the next token, b1 runs alone, then L1 resumes (ES-LP)

The session folder is the run_session layout: <sess>/<condition_id>/{requests.jsonl, manifest.json}. The dispatcher
writes requests.jsonl; this test writes manifest.json with the keys of run_session.Session.run_condition.
coexit_lp is skipped while exits.ExitPolicy raises NotImplementedError for it (slice S3).
"""
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from exp2 import contracts as C
from exp2 import exits as X
from exp2 import wire
from exp2.analyze import run3 as R
from exp2.serve import dispatcher as dispatcher_mod
from exp2.serve import run_session
from exp2.serve.dispatcher import Dispatcher
from exp2.serve.fake_adapter import default_spec
from exp2.tests import mocks

ROOT = Path(__file__).resolve().parents[2]
MS = 1_000_000
NS = 1_000_000_000
F_MHZ = 1020
SEED = 20261022
N_INPUTS = 4
LLAMA_SEG_MS = 20.0                                  # per token: a pause lands between tokens
BERT_SEG_MS = 5.0                                    # per checkpoint: bert:23 has 6 -> ~31 ms with the head
SERVERS = (("llama", [15], LLAMA_SEG_MS), ("bert", [5, 23], BERT_SEG_MS))
LLAMA_DL_MS, BERT_DL_MS = 10_000, 2_000              # generous: every request runs its target, nothing is late
BERT_AT_MS = (100, 200, 300)                         # arrive while L1 runs (L1 alone takes 32 x 20 = 640 ms)
TYPES = {"bert:5": {"tau": 0.012, "klass": "short"}, "bert:23": {"tau": 0.031, "klass": "short"},
         "llama:3": {"tau": 0.2, "klass": "long"}, "llama:15": {"tau": 0.7, "klass": "long"}}
WORKLOAD = {"name": "mixB", "types": ["bert:23", "llama:15"], "probs": [0.75, 0.25], "rho": 0.9, "rate_per_s": 4.0,
            "arrival": "poisson", "cv_a": 1.0}
BUDGET_S = 30.0


@pytest.fixture(scope="module")
def types():
    return mocks.make_types(TYPES)


@pytest.fixture(scope="module")
def servers(tmp_path_factory):
    """Spawn both fake servers at once (their warm-ups overlap), connect, read `ready`. Teardown: stop, then kill
    whatever is still alive. -> {"conns", "t0" (monotonic s at spawn), "dir"}."""
    d = tmp_path_factory.mktemp("lp_e2e_servers")
    t0 = time.monotonic()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TORCHINDUCTOR_CACHE_DIR=str(d / "ind"))
    procs, conns = {}, {}
    try:
        for fam, exits, seg in SERVERS:
            spec = d / ("%s.json" % fam)
            spec.write_text(json.dumps(default_spec(fam, exits, seg_ms=seg, head_ms=1.0, load_s=0.0,
                                                    first_infer_s=0.0)), encoding="utf-8")
            cmd = [sys.executable, "-m", "exp2.serve.server", "--family", fam, "--exits", ",".join(map(str, exits)),
                   "--adapter", "fake", "--fake-spec", str(spec), "--out", str(d), "--warmup-iters", "5",
                   "--n-inputs", str(N_INPUTS)]
            with open(d / ("server_%s.log" % fam), "wb") as log:
                procs[fam] = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL, stdout=log,
                                              stderr=subprocess.STDOUT)
        for fam, proc in procs.items():
            port = wire.wait_port_file(d / ("server_%s.port" % fam), 60.0, proc)
            conns[fam] = wire.connect(port)
            ready = C.validate_ready(wire.recv(conns[fam], timeout_s=60.0))
            assert ready["family"] == fam and ready["versions"]["torch"] is None
        yield {"conns": conns, "t0": t0, "dir": d}
    finally:
        for conn in conns.values():
            try:
                wire.send(conn, {"op": "stop"})
            except (OSError, ValueError):
                pass
        for proc in procs.values():
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=15)
        for conn in conns.values():
            conn.close()


def _item(rid, tid, t_ms, dl_ms):
    return {"req_id": rid, "type_id": tid, "t_arrival_ns": t_ms * MS, "t_deadline_ns": (t_ms + dl_ms) * MS,
            "k_override": None}


def _schedule():
    """1 LLaMA at t0, then 3 BERT while it runs. Deadlines are offsets from t0, as arrivals.schedule gives them."""
    return [_item("L1", "llama:15", 0, LLAMA_DL_MS)] + [
        _item("b%d" % (i + 1), "bert:23", t, BERT_DL_MS) for i, t in enumerate(BERT_AT_MS)]


def _manifest(cid, method, spec, summary):
    """manifest.json with the keys run_session.Session.run_condition writes (validated by C.validate_manifest)."""
    return C.validate_manifest({
        "schema": C.SCHEMAS["manifest"], "condition_id": cid, "mode": "open", "arm": "pinned", "gpu_mhz": F_MHZ,
        "emc_mhz": None, "t_wall_start_ns": time.time_ns(), "t_mono_start_ns": summary["t0_ns"],
        "t_mono_end_ns": summary["t_end_ns"], "settle_s": 0.0, "valid": True, "invalid_reason": None,
        "board_readback": None, "n_requests": summary["n_requests"], "n_warmup": 0, "seed": SEED,
        "policy_key": method, "policy_spec": spec, "freeze_condition_id": "B_p_rho090", "workload": dict(WORKLOAD),
        "alpha": None, "cell": None, "split": None, "repeat": 0})


def _busy(row):
    """[(start, end)] of a row: [t_start_ns, t_end_ns] minus its pause intervals (every server stamps with
    time.monotonic_ns, one clock for every process)."""
    a, out = row["t_start_ns"], []
    for _, _, t_pause, t_resume in row["pauses"]:
        out.append((a, t_pause))
        a = t_resume
    return out + [(a, row["t_end_ns"])]


def _logs(d):
    return "\n".join("%s:\n%s" % (p.name, p.read_text(errors="replace")[-2000:]) for p in sorted(d.glob("*.log")))


@pytest.mark.parametrize("method", X.LP_METHODS)
def test_lp_end_to_end_with_fake_servers(method, types, tmp_path, request, monkeypatch):
    spec = X.board_specs(F_MHZ)[method].to_json()
    try:
        policy = run_session.make_policy(spec, types, mocks.StubSlowdown())
    except NotImplementedError as exc:
        pytest.skip("%s is not implemented yet (PLAN PL3 slice S3, exits.ExitPolicy raises): %s" % (method, exc))
    srv = request.getfixturevalue("servers")             # spawned only when a case really runs
    monkeypatch.setattr(dispatcher_mod, "HANG_TIMEOUT_NS", 20 * NS)   # a stuck server fails fast, not after 120 s
    cid = "B_p_rho090_%s_r0" % method
    sess = tmp_path / "session"
    cond = {"condition_id": cid, "mode": "open", "arm": "pinned", "gpu_mhz": F_MHZ, "seed": SEED, "n_warmup": 0,
            "policy_key": method}

    summary = Dispatcher(srv["conns"], types, n_inputs=N_INPUTS).run_open(cond, _schedule(), policy, sess / cid)

    logs = _logs(srv["dir"])
    assert summary["dead_servers"] == {} and summary["unexpected_msgs"] == 0, (summary, logs)
    rows = C.read_requests(sess / cid / "requests.jsonl")           # validate_request_row on every row
    by = {r["req_id"]: r for r in rows}
    assert sorted(by) == ["L1", "b1", "b2", "b3"]
    assert not [r for r in rows if r["outcome"] == "error"], (rows, logs)
    pauses = [p for r in rows for p in r["pauses"]]
    assert all(p[3] is not None for p in pauses), pauses       # every pause closed
    assert all(r["pauses"] == [] for r in rows if r["family"] != "llama")

    L1 = by["L1"]
    if method == "edgeserving_lp":
        assert summary["outcomes"] == {"complete": 4}, (summary, logs)
        assert L1["tokens_done"] == 32 and len(L1["ckpt"]) == 32
        assert 1 <= len(L1["pauses"]) <= len(BERT_AT_MS), L1["pauses"]      # one pause per waiting bert at most
        segs = sorted(s for r in rows for s in _busy(r))
        overlap = [(s1, s2) for s1, s2 in zip(segs, segs[1:]) if s2[0] < s1[1]]
        assert not overlap, overlap                               # serial once the pause intervals are removed
        assert L1["n_in_service_at_admit"] == 0
        for b in (by["b1"], by["b2"], by["b3"]):                  # every bert ran alone inside a pause of L1
            assert b["n_in_service_at_admit"] == 1, b
            assert any(tp <= b["t_start_ns"] and b["t_end_ns"] <= tr for _, _, tp, tr in L1["pauses"]), (b, L1)

    (sess / cid / "manifest.json").write_text(json.dumps(_manifest(cid, method, spec, summary)), encoding="utf-8")
    out = tmp_path / "analysis"
    assert R.main(["--sessions", str(sess), "--out", str(out)]) == 0
    with open(out / "lp.csv", newline="", encoding="utf-8") as fh:
        lp = list(csv.DictReader(fh))
    assert len(lp) == 1 and lp[0]["condition_id"] == cid and lp[0]["n_pauses"] == str(len(pauses))
    if method == "edgeserving_lp":
        assert lp[0]["n_overlap"] == "0" and lp[0]["n_empty_pauses"] == "0", lp[0]

    wall = time.monotonic() - srv["t0"]
    print("%s: %d pauses [tokens_done, boundary wait ms, paused ms] %s; wall %.1f s since spawn" % (
        method, len(pauses), [[p[0], round((p[2] - p[1]) / MS, 2), round((p[3] - p[2]) / MS, 2)] for p in pauses],
        wall))
    print("lp.csv %s" % json.dumps(lp[0]))
    assert "Traceback" not in logs, logs
    assert wall < BUDGET_S, wall
