"""S0a gate: exp2.contracts (BRIEF_v1 sec. 3.3, 3.4 K1-K20, 3.6, 3.7).

Every rule here is written so that a plausible wrong implementation fails it. The golden files under
exp2/tests/fixtures/ (admission_cases, admission_traces, truncation_cases) are checked for shape and for the
parts that belong to contracts (truncation_outcome, completion, validate_policyspec, DispatchView parsing); the
admission semantics themselves are replayed by exp2/tests/test_policies.py.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from exp2 import contracts as C

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "exp2" / "tests" / "fixtures"
EXAMPLE_TYPES = REPO / "exp2" / "configs" / "example_types.json"
HEX_A, HEX_B = "a" * 64, "b" * 64
MS = 1_000_000


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def types():
    return C.load_types(EXAMPLE_TYPES)


@pytest.fixture
def types_json():
    return load_json(EXAMPLE_TYPES)


# ------------------------------------------------------------------------------------------------ valid objects
W_HI = {"name": "W_hi", "types": ["bert:5", "llama:15"], "probs": [0.9, 0.1], "rho": 0.6, "rate_per_s": 5.636,
        "arrival": "poisson", "cv_a": 1.0}
POLICIES = {
    "default": {"name": "default", "k_short": None, "k_long": None, "order": "fcfs", "lane_cap": None,
                "f_rule": "governor", "f_mhz": None, "params": {}},
    "serial": {"name": "serial", "k_short": 1, "k_long": 1, "order": "fcfs", "lane_cap": None, "f_rule": "static",
               "f_mhz": 1020, "params": {}},
    "serial_stf": {"name": "serial_stf", "k_short": 1, "k_long": 1, "order": "stf", "lane_cap": None,
                   "f_rule": "static", "f_mhz": 1020, "params": {}},
    "serial_class": {"name": "serial_class", "k_short": 1, "k_long": 1, "order": "class", "lane_cap": None,
                     "f_rule": "static", "f_mhz": 1020, "params": {}},
    "serial_edgeserving": {"name": "serial_edgeserving", "k_short": 1, "k_long": 1, "order": "edgeserving",
                           "lane_cap": None, "f_rule": "static", "f_mhz": 1020, "params": {"C": 10}},
    "corun": {"name": "corun", "k_short": None, "k_long": None, "order": "fcfs", "lane_cap": None,
              "f_rule": "static", "f_mhz": 1020, "params": {}},
    "greenllm_lanes": {"name": "greenllm_lanes", "k_short": 2, "k_long": 2, "order": "fcfs",
                       "lane_cap": {"short": 1, "long": 1}, "f_rule": "greenllm", "f_mhz": None, "params": {}},
    "occupancy_f": {"name": "occupancy_f", "k_short": None, "k_long": None, "order": "fcfs", "lane_cap": None,
                    "f_rule": "occupancy", "f_mhz": None, "params": {"f_base_mhz": 612, "alpha_p": 2}},
}
POLICY_F = {"name": "F", "k_short": 2, "k_long": 1, "order": "stf", "lane_cap": None, "f_rule": "static",
            "f_mhz": 612, "params": {}}


def request_row():
    """The K8 example."""
    return {
        "condition_id": "W_hi_r060_serial_r0", "req_id": "r000001", "type_id": "bert:23", "family": "bert",
        "exit": 23, "klass": "long", "target": 23, "t_arrival_ns": 1000, "t_arrival_actual_ns": 1350,
        "t_deadline_ns": 550001000, "t_admit_ns": 1400, "t_srv_recv_ns": 90000, "t_start_ns": 95000,
        "t_end_ns": 101095000, "t_done_ns": 101300000, "server": "bert", "policy": "serial",
        "policy_spec": copy.deepcopy(POLICIES["serial"]), "f_mhz": 1020, "n_in_service_at_admit": 0,
        "queue_len_at_arrival": 0,
        "reached": 23, "completion": 1.0, "outcome": "complete", "tokens_done": None,
        "ckpt": [[3, 20000000], [7, 40000000]], "is_warmup": False, "seed": 11, "mode": "open", "err": None}


def expired_row():
    r = request_row()
    r.update({"t_admit_ns": None, "t_srv_recv_ns": None, "t_start_ns": None, "t_end_ns": None,
              "t_done_ns": r["t_deadline_ns"], "n_in_service_at_admit": None, "reached": None, "completion": 0.0,
              "outcome": "expired_in_queue", "ckpt": []})
    return r


def closed_row():
    r = request_row()
    r.update({"mode": "closed", "t_deadline_ns": None, "t_admit_ns": r["t_arrival_ns"], "policy_spec": None})
    return r


def ready_msg():
    return {"op": "ready", "family": "bert", "pid": 4242, "exits_warmed": [5, 23],
            "warmup": {"load_s": 11.8, "first_infer_s": {"5": 3.1, "23": 4.0}, "n_warm": {"5": 20, "23": 20},
                       "cv_last5": {"5": 0.012, "23": 0.009}, "act_mb": {"5": 38.0, "23": 41.0},
                       "inductor_cache_files_at_start": 0},
            "input_source": "per_request",
            "versions": {"python": "3.10.12", "torch": "2.8.0", "transformers": "4.x", "cuda": "12.6", "compile": "on"},
            "mem": {"torch_reserved_mb": 1402.0, "rss_mb": 2110.0}, "t_ready_ns": 1712000000000}


def run_msg():
    return {"op": "run", "req_id": "r000001", "type_id": "bert:23", "exit": 23, "input_idx": 1,
            "deadline_ns": 1712300000000, "t_admit_ns": 1712150000000}


def done_msg():
    return {"op": "done", "req_id": "r000001", "type_id": "bert:23", "t_recv_ns": 1712150100000,
            "t_start_ns": 1712150200000, "t_end_ns": 1712320000000, "reached": 7, "target": 23, "completion": 0.3333,
            "outcome": "truncated", "ckpt": [[3, 1712160000000], [7, 1712250000000], [11, 1712310000000]],
            "tokens_done": None, "err": None}


def session_obj(kind="open"):
    return {
        "schema": "exp2.session/1", "session_id": "s20261020T101500", "plan_sha256": HEX_A, "plan_kind": kind,
        "freeze_content_sha256": HEX_B if kind == "open" else None, "t_wall_start_ns": 1760951700000000000,
        "t_mono_start_ns": 5000, "t_wall_end_ns": 1760960000000000000,
        "git": {"commit": "abc123", "dirty": False, "diff_sha256": None},
        "host": {"python": "3.10.12", "numpy": "1.26.4", "platform": "linux-aarch64", "boot_id": "id"},
        "board": {"dry": False, "snapshot_start": {}, "snapshot_end": {}, "jetson_clocks_store": "/tmp/exp2_jc_store"},
        "idle": {"pre": {"t0_ns": 1, "t1_ns": 30000000001, "gpu_mhz": 1020},
                 "resident": {"t0_ns": 2, "t1_ns": 3, "gpu_mhz": 1020}},
        "servers": {"bert": {"pid": 4242, "t_spawn_ns": 10, "t_ready_ns": 20, "t_spawn_wall_ns": 30, "ready": {}},
                    "vision": {"pid": 4243, "t_spawn_ns": 25, "t_ready_ns": 40, "t_spawn_wall_ns": 35,
                               "ready": ready_msg() | {"family": "vision", "exits_warmed": [6, 24],
                                                       "warmup": ready_msg()["warmup"] | {
                                                           "first_infer_s": {"6": 1.0, "24": 2.0},
                                                           "n_warm": {"6": 20, "24": 20},
                                                           "cv_last5": {"6": 0.01, "24": 0.01},
                                                           "act_mb": {"6": 5.0, "24": 6.0}}}}},
        "conditions": ["W_hi_r060_serial_r0"], "exit_code": 0}


def manifest_obj(mode="open"):
    base = {"schema": "exp2.manifest/1", "condition_id": "W_hi_r060_F_r0", "mode": mode, "arm": "pinned",
            "gpu_mhz": 612, "emc_mhz": 3199.0, "t_wall_start_ns": 1760951700000000000, "t_mono_start_ns": 5000,
            "t_mono_end_ns": 9000, "settle_s": 5.0, "valid": True, "invalid_reason": None,
            "board_readback": {"gpu_mhz": 612}, "n_requests": 1000, "n_warmup": 100, "seed": 11, "policy_key": "F",
            "policy_spec": copy.deepcopy(POLICY_F), "freeze_condition_id": "W_hi_r060",
            "workload": copy.deepcopy(W_HI), "alpha": 5.0,
            "cell": None, "split": None, "repeat": 0}
    if mode == "closed":
        base.update({"condition_id": "solo_bert-5_f612_r0", "policy_key": None, "policy_spec": None,
                     "freeze_condition_id": None, "workload": None, "alpha": None, "cell": ["bert:5"],
                     "split": "solo", "n_requests": None, "seed": None})
    if mode == "idle":
        base.update({"condition_id": "idle_f612", "policy_key": None, "policy_spec": None, "freeze_condition_id": None,
                     "workload": None, "alpha": None, "n_requests": None, "n_warmup": None, "seed": None,
                     "repeat": None})
    return base


def plan_obj(kind="open"):
    session = {"families": ["bert", "vision", "llama", "yolo"],
               "exits": {"bert": [5, 23], "vision": [6, 24], "llama": [3, 15], "yolo": [1, 5]}, "emc_mhz": 3199.0,
               "settle_s": 5.0, "idle_window_s": 30.0, "n_inputs": 32, "preprocess": "per_request"}
    conds = [
        {"condition_id": "idle_f1020", "mode": "idle", "arm": "pinned", "gpu_mhz": 1020, "duration_s": 30.0},
        {"condition_id": "solo_bert-5_f1020_r0", "mode": "closed", "arm": "pinned", "gpu_mhz": 1020,
         "cell": ["bert:5"], "duration_s": 30.0, "n_warmup": 5, "split": "solo", "repeat": 0}]
    if kind == "open":
        conds.append({"condition_id": "W_hi_r060_F_r0", "mode": "open", "arm": "pinned", "gpu_mhz": 612,
                      "policy_key": "F", "freeze_condition_id": "W_hi_r060", "policy": copy.deepcopy(POLICY_F),
                      "workload": copy.deepcopy(W_HI),
                      "alpha": 5.0, "n_requests": 1000, "n_warmup": 100, "seed": 11, "repeat": 0})
        conds.append({"condition_id": "W_hi_r060_default_r0", "mode": "open", "arm": "governor", "gpu_mhz": None,
                      "policy_key": "default", "freeze_condition_id": "W_hi_r060",
                      "policy": copy.deepcopy(POLICIES["default"]), "workload": copy.deepcopy(W_HI), "alpha": 5.0,
                      "n_requests": 1000, "n_warmup": 100, "seed": 11, "repeat": 0})
    plan = {"schema": "exp2.plan/1", "kind": kind, "session": session,
            "freeze": {"path": "freeze.json", "content_sha256": HEX_A} if kind == "open" else None,
            "conditions": conds, "estimated_wall_s": 21600.0}
    return C.with_plan_sha256(plan)


def freeze_obj():
    cond = {"condition_id": "W_hi_r060", "workload": copy.deepcopy(W_HI), "rho": 0.6, "arrival": "poisson", "cv_a": 1.0,
            "alpha": 5.0, "choices": {"F": copy.deepcopy(POLICY_F), "default": copy.deepcopy(POLICIES["default"]),
                        "serial": copy.deepcopy(POLICIES["serial"])},
            "grid": [copy.deepcopy(POLICY_F), copy.deepcopy(POLICIES["serial"])],
            "predictions": {"F": {"per_type": {}, "energy_per_req_j": 0.61}},
            "game": {"player_cost": "shortfall_then_rt",
                     "equilibria": [{"start": [1, 1], "k_short": 3, "k_long": 1000000, "converged": True,
                                     "iters": 3, "cycle": None}],
                     "worst": {"k_short": 3, "k_long": 1000000}, "optimum": {}, "poa_E": 1.12, "poa_R": 1.31},
            "compute_s": {"F": 41.0, "game": 120.0}}
    obj = {"schema": "exp2.freeze/1", "created_utc": "2026-10-21T09:00:00Z", "created_wall_ns": 1761037200000000000,
           "git": {"commit": "abc123", "dirty": False, "diff_sha256": None},
           "inputs": {k: HEX_A for k in ("types_sha256", "corun_sha256", "solo_sha256", "run1_plan_sha256",
                                         "slowdown_sha256", "power_sha256")},
           "slowdown_model": {"selected": "M2", "heldout_mape": {"M0": 9.9, "M1": 7.0, "M2": 4.1, "M3": 4.0,
                                                                 "M4lin": 5.2, "M4": 4.8}, "target_met": True},
           "sim": {"n_requests": 20000, "n_warmup": 500, "n_reps": 5, "seed": 0},
           "constraint": {"headline": "complete_rate>=0.95 per type", "sensitivity": "mean_completion>=0.95 per type"},
           "f_grid_mhz": [612, 1020], "conditions": [cond], "runtime_estimate_s": 64800.0}
    return C.with_content_sha256(obj)


def simresult_obj():
    st = {"n": 18000, "mean_R": 0.061, "p50_R": 0.045, "p95_R": 0.150, "p99_R": 0.21, "complete_rate": 0.97,
          "truncation_rate": 0.02, "miss_rate": 0.005, "expired_rate": 0.005, "mean_completion": 0.985,
          "mean_shortfall": 0.015}
    return {"schema": "exp2.simresult/1", "spec_name": "serial", "per_type": {"bert:5": st},
            "energy_per_req_j": 0.61, "energy_ci": [0.60, 0.62], "mean_power_w": 9.8, "throughput": 15.9,
            "mean_in_system": 1.2, "utilization": 0.58, "unstable": False, "tagged": None,
            "reps": [{"seed": 0, "energy_per_req_j": 0.61, "n_slope": 0.0001, "per_type": {}, "tagged": None}],
            "trace": None, "runtime_s": 12.4}


def slowdown_obj():
    return {"schema": "exp2.slowdown/1", "name": "M2", "params": {"eta": {"612": 0.93, "1020": 0.91}},
            "residuals": {"bert|2": [0.98, 1.01, 1.03]}, "heldout_mape": {"M0": 9.9, "M2": 4.1},
            "fit_rows_sha256": HEX_A}


def power_obj():
    return {"schema": "exp2.power/1", "p_idle_w": {"612": 5.1, "1020": 5.6}, "dp_w": {"612": 4.2, "1020": 7.9},
            "eta": {"612": 0.93, "1020": 0.91}}


def open_summary_obj():
    return {"schema": "exp2.open_summary/2", "freeze_content_sha256": HEX_A,
            "checks": {"hash_ok": True, "wall_order_ok": True, "t_wall_start_ns": 2, "freeze_created_wall_ns": 1},
            "counts": {"runs_valid": 12, "runs_invalid": 0, "runs_duplicate": 0, "runs_screen_flagged": 1},
            "per_condition": {"W_hi_r060_F": {"energy": 0.6, "energy_ci": [0.5, 0.7], "complete_rate_min": 0.96,
                                              "feasible": True, "pred": 0.61, "ape": 1.6, "status": "feasible",
                                              "wilson_lo_min": 0.95, "n_repeats": 3, "measured_key": "F"}},
            "mape": {"completion": 1.0, "energy": 2.0, "median": 1.5, "max": 3.0},
            "regret": {"F": 0.0, "grid_spread": 0.2, "best_key": "F"}, "oracle": {},
            "energy_diff": {"W_hi_r060_default": {"diff_j": 0.1, "ci": [0.05, 0.15], "n_pairs": 3,
                                                  "method": "paired_t"}},
            "poa_detail": {"W_hi_r060": {"worst_key": "g1", "optimum_key": "F", "ratio": 1.2,
                                         "complete_min_worst": 0.9, "complete_min_optimum": 0.97}},
            "poa_measured": None}


def pilot_obj():
    return {"schema": "exp2.pilot/1", "dry": True,
            "gates": {"g4_gpu_hold": {"value": 0.995, "threshold": 0.99, "pass": True, "step": 4}}}


def fake_spec_obj():
    return {"schema": "exp2.fake_spec/1", "family": "bert", "clock": "sleep", "load_s": 0.05, "first_infer_s": 0.02,
            "types": {"5": {"seg_ms": [20.0, 20.0], "head_ms": 5.0},
                      "23": {"seg_ms": [20, 20, 20, 20, 20, 20], "head_ms": 5.0}}}


def telemetry_meta_obj():
    return {"schema": "exp2.telemetry_meta/1",
            "vdd_in": {"hwmon": "/sys/class/hwmon/hwmon3", "channel": 2, "label": "VDD_IN", "update_interval_ms": 1,
                       "samples": 4},
            "target_hz": 20, "achieved_hz": 19.8, "clocks_target_hz": 1, "devfreq": "/sys/class/devfreq/17000000.gpu",
            "emc_source": "tegrastats", "thermal_zones": ["cpu-thermal"], "boot_id": "id", "t_mono_start_ns": 1,
            "t_wall_start_ns": 2, "n_rows": 40}


def root_msgs():
    return [{"op": "hello", "pid": 1234, "uid": 0, "fake_root": None}, {"op": "write", "path": "/p", "value": "1"},
            {"op": "write", "path": "/p", "value": 1020000000}, {"op": "read", "path": "/p"},
            {"op": "run", "argv": ["nvpmodel", "-q"], "timeout_s": 30}, {"op": "stop"},
            {"ok": True, "readback": "1"}, {"ok": True, "value": "x"}, {"ok": True, "rc": 0, "out": ""},
            {"ok": True}, {"ok": False, "err": "boom"}]


# validator, factory
OBJECT_VALIDATORS = [
    ("types", C.validate_types, lambda: load_json(EXAMPLE_TYPES)),
    ("workload", C.validate_workload, lambda: copy.deepcopy(W_HI)),
    ("policyspec", C.validate_policyspec, lambda: copy.deepcopy(POLICIES["serial"])),
    ("request_row", C.validate_request_row, request_row),
    ("ready", C.validate_ready, ready_msg),
    ("run", C.validate_run, run_msg),
    ("done", C.validate_done, done_msg),
    ("session", C.validate_session, session_obj),
    ("manifest", C.validate_manifest, manifest_obj),
    ("plan", C.validate_plan, plan_obj),
    ("freeze", C.validate_freeze, freeze_obj),
    ("simresult", C.validate_simresult, simresult_obj),
    ("slowdown", C.validate_slowdown, slowdown_obj),
    ("power", C.validate_power, power_obj),
    ("open_summary", C.validate_open_summary, open_summary_obj),
    ("pilot", C.validate_pilot, pilot_obj),
    ("fake_spec", C.validate_fake_spec, fake_spec_obj),
    ("telemetry_meta", C.validate_telemetry_meta, telemetry_meta_obj),
]
IDS = [n for n, _, _ in OBJECT_VALIDATORS]
SCHEMA_VALIDATORS = [x for x in OBJECT_VALIDATORS if "schema" in x[2]()]


# ================================================================================================ constants
class TestConstants:
    def test_values_frozen_by_the_brief(self):
        assert C.FAMILIES == ("bert", "vision", "llama", "yolo")
        assert C.EXIT_RANGE == {"bert": (0, 23), "vision": (1, 24), "llama": (0, 15), "yolo": (0, 5)}
        assert C.YOLO_EXIT_MAX_DEPTH == {0: 7, 1: 8, 2: 9, 3: 15, 4: 18, 5: 21}
        assert (C.LLAMA_TOKENS, C.LLAMA_PROMPT_TOKENS, C.CKPT_STRIDE) == (32, 128, 4)
        assert C.K_INF == 1_000_000
        assert C.CLASSES == ("short", "long") and C.ORDERS == ("fcfs", "stf", "class", "edgeserving")
        assert C.F_RULES == ("static", "governor", "greenllm", "occupancy")
        assert C.ARMS == ("pinned", "governor", "stock")
        assert C.OUTCOMES == ("complete", "truncated", "miss", "expired_in_queue", "error")
        assert C.EMC_LOCKABLE_MHZ == (204.0, 665.6, 2133.0, 3199.0)
        assert (C.POWER_MODE_ID, C.POWER_MODE_NAME) == (2, "MAXN_SUPER")
        assert C.SUDO_PW_ENV == "JETSON_SUDO_PW"
        assert (C.COMPLETE_TARGET, C.TIE_SHORTFALL) == (0.95, 0.005)
        assert (C.UNSTABLE_SLOPE_FRAC, C.EDGESERVING_C) == (0.005, 10.0)

    def test_schemas_cover_every_documented_id(self):
        assert set(C.SCHEMAS.values()) == {
            "exp2.types/1", "exp2.plan/1", "exp2.freeze/1", "exp2.session/1", "exp2.manifest/1", "exp2.simresult/1",
            "exp2.slowdown/1", "exp2.power/1", "exp2.open_summary/2", "exp2.pilot/1", "exp2.fake_spec/1",
            "exp2.telemetry_meta/1"}
        assert C.SCHEMA_IDS == frozenset(C.SCHEMAS.values())

    def test_error_hierarchy(self):
        assert issubclass(C.ContractError, ValueError) and issubclass(C.HeldOutError, ValueError)
        assert issubclass(C.ClockError, RuntimeError) and issubclass(C.FreezeError, RuntimeError)
        assert not issubclass(C.ContractError, RuntimeError)

    def test_contracts_wire_dryroot_import_no_heavy_modules(self):
        banned = {"numpy", "scipy", "pandas", "sklearn", "torch", "transformers", "cv2", "shared"}
        for rel in ("exp2/contracts.py", "exp2/wire.py", "exp2/board/dryroot.py"):
            tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
            names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names |= {a.name.split(".")[0] for a in node.names}
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names.add(node.module.split(".")[0])
            assert not names & banned, (rel, names & banned)


# ================================================================================================ type ids, depth
class TestTypeIds:
    def test_round_trip(self):
        assert C.type_id("bert", 5) == "bert:5"
        assert C.parse_type_id("vision:24") == ("vision", 24)
        for fam in C.FAMILIES:
            lo, hi = C.EXIT_RANGE[fam]
            for ex in (lo, hi):
                assert C.parse_type_id(C.type_id(fam, ex)) == (fam, ex)

    @pytest.mark.parametrize("bad", ["bert", "bert:", "bert:x", "gpt:1", "bert:24", "vision:0", "yolo:6", "llama:16",
                                     "bert:-1", "", " bert:5", "bert:5 ", "bert:5\n", None, 5])
    def test_parse_rejects(self, bad):
        with pytest.raises(C.ContractError):
            C.parse_type_id(bad)

    @pytest.mark.parametrize("fam,ex", [("bert", 24), ("vision", 0), ("yolo", 6), ("llama", 16), ("gpt", 1),
                                        ("bert", True), ("bert", 1.0)])
    def test_type_id_rejects(self, fam, ex):
        with pytest.raises(C.ContractError):
            C.type_id(fam, ex)

    def test_depth(self):
        assert C.depth("bert", 23) == 24 and C.depth("bert", 0) == 1
        assert C.depth("llama", 15) == 16
        assert C.depth("vision", 24) == 24 and C.depth("vision", 1) == 1
        assert [C.depth("yolo", e) for e in range(6)] == [8, 9, 10, 16, 19, 22]

    def test_default_checkpoints(self):
        assert C.default_checkpoints("bert", 5) == [3, 5]
        assert C.default_checkpoints("bert", 23) == [3, 7, 11, 15, 19, 23]
        assert C.default_checkpoints("bert", 3) == [3] and C.default_checkpoints("bert", 0) == [0]
        assert C.default_checkpoints("vision", 24) == [4, 8, 12, 16, 20, 24]
        assert C.default_checkpoints("vision", 6) == [4, 6] and C.default_checkpoints("vision", 1) == [1]
        assert C.default_checkpoints("yolo", 5) == [0, 1, 2, 3, 4, 5] and C.default_checkpoints("yolo", 0) == [0]
        assert C.default_checkpoints("llama", 3) == list(range(1, 33)) == C.default_checkpoints("llama", 15)

    def test_every_type_ends_at_its_target(self):
        for fam, (lo, hi) in C.EXIT_RANGE.items():
            for ex in range(lo, hi + 1):
                ck = C.default_checkpoints(fam, ex)
                assert ck == sorted(set(ck)) and (ck[-1] == 32 if fam == "llama" else ck[-1] == ex)


class TestCompletion:
    @pytest.mark.parametrize("fam,ex,reached,want", [("bert", 23, 7, 8 / 24), ("yolo", 5, 2, 10 / 22),
                                                     ("llama", 15, 20, 0.625), ("bert", 23, None, 0.0),
                                                     ("bert", 23, 23, 1.0), ("vision", 24, 12, 0.5),
                                                     ("llama", 3, 32, 1.0), ("yolo", 0, 0, 1.0)])
    def test_values(self, fam, ex, reached, want):
        assert C.completion(fam, ex, reached) == pytest.approx(want, abs=1e-12)

    @pytest.mark.parametrize("fam,ex,reached", [("bert", 5, 6), ("bert", 5, -1), ("vision", 6, 0), ("llama", 15, 0),
                                                ("llama", 15, 33), ("yolo", 2, 3), ("bert", 5, 1.5)])
    def test_out_of_range_rejected(self, fam, ex, reached):
        with pytest.raises(C.ContractError):
            C.completion(fam, ex, reached)


# ================================================================================================ truncation (K7)
TRUNC_CASES = load_json(FIX / "truncation_cases.json") if (FIX / "truncation_cases.json").exists() else []


class TestTruncation:
    def test_at_least_12_golden_cases_and_every_kind(self):
        assert len(TRUNC_CASES) >= 12
        outcomes = {c["expect"].get("outcome") for c in TRUNC_CASES}
        assert {"complete", "truncated", "miss"} <= outcomes
        assert any(c["expect"].get("error") for c in TRUNC_CASES)
        assert any(c["deadline_ns"] is None for c in TRUNC_CASES)
        assert {c["family"] for c in TRUNC_CASES} == set(C.FAMILIES)

    @pytest.mark.parametrize("case", TRUNC_CASES, ids=[c["id"] for c in TRUNC_CASES])
    def test_golden_case(self, case):
        want = case["expect"]
        if want.get("error"):
            with pytest.raises(ValueError):
                C.truncation_outcome(case["ckpts"], case["t_ckpt_ns"], case["deadline_ns"])
            return
        reached, outcome = C.truncation_outcome(case["ckpts"], case["t_ckpt_ns"], case["deadline_ns"])
        assert (reached, outcome) == (want["reached"], want["outcome"])
        assert C.completion(case["family"], case["exit"], reached) == pytest.approx(want["completion"], abs=1e-4)
        assert case["ckpts"][-1] == C.default_checkpoints(case["family"], case["exit"])[-1]

    def test_time_equal_to_deadline_is_on_time(self):
        assert C.truncation_outcome([3, 7], [10, 20], 20) == (7, "complete")
        assert C.truncation_outcome([3, 7], [10, 21], 20) == (3, "truncated")

    def test_first_late_decides_even_if_later_ones_would_be_on_time(self):
        with pytest.raises(ValueError):
            C.truncation_outcome([3, 7, 11], [10, 40, 15], 30)

    def test_empty_ckpts(self):
        with pytest.raises(ValueError):
            C.truncation_outcome([], [], 5)

    def test_accepts_tuples(self):
        assert C.truncation_outcome((3, 7), (10, 20), None) == (7, "complete")


# ================================================================================================ shared functions
class TestSharedFunctions:
    def test_rate_for_load_uses_mean_not_p50(self, types):
        w = C.Workload.from_json(W_HI)
        rate = C.rate_for_load(0.6, w, types)
        assert rate == pytest.approx(0.6 / (0.9 * 0.0294 + 0.1 * 0.80), rel=1e-12)
        assert rate == pytest.approx(5.636, abs=1e-3)
        assert rate != pytest.approx(0.6 / (0.9 * 0.0290 + 0.1 * 0.79), rel=1e-3)      # the p50 variant
        assert C.rate_for_load(0.6, W_HI, types) == rate                              # json dict accepted

    def test_rate_for_load_scales_linearly_and_rejects_bad_rho(self, types):
        assert C.rate_for_load(1.2, W_HI, types) == pytest.approx(2 * C.rate_for_load(0.6, W_HI, types))
        for bad in (0, -1, None, float("nan")):
            with pytest.raises(C.ContractError):
                C.rate_for_load(bad, W_HI, types)

    def test_deadline_s_uses_alpha_times_p50(self, types):
        assert C.deadline_s(types, "bert:5") == pytest.approx(5.0 * 0.0290)
        assert C.deadline_s(types, "llama:15") == pytest.approx(5.0 * 0.79)

    def test_deadline_s_none_without_alpha(self, types_json):
        types_json["alpha"] = None
        assert C.deadline_s(C.TypeTable.from_json(types_json), "bert:5") is None

    def test_with_deadlines_matches_k2_example(self, types):
        items = [{"req_id": "r000001", "type_id": "bert:5", "t_arrival_ns": 134999872, "t_deadline_ns": None,
                  "k_override": None}]
        out = C.with_deadlines(items, types, 5.0)
        assert out[0]["t_deadline_ns"] == 279999872 and out[0]["t_deadline_ns"] != 134999872 + 147000000
        assert items[0]["t_deadline_ns"] is None                                      # input untouched
        assert out[0]["req_id"] == "r000001" and out[0]["k_override"] is None

    def test_with_deadlines_alpha_none_and_per_type(self, types):
        items = [{"req_id": "a", "type_id": "llama:15", "t_arrival_ns": 0, "t_deadline_ns": 7, "k_override": None},
                 {"req_id": "b", "type_id": "yolo:1", "t_arrival_ns": 10, "t_deadline_ns": None, "k_override": 2}]
        assert [x["t_deadline_ns"] for x in C.with_deadlines(items, types, None)] == [None, None]
        out = C.with_deadlines(items, types, 2.0)
        assert out[0]["t_deadline_ns"] == round(2.0 * 0.79 * 1e9)
        assert out[1]["t_deadline_ns"] == 10 + round(2.0 * 0.0148 * 1e9)

    def test_k_of(self):
        spec = C.PolicySpec("x", None, 3)
        assert C.k_of(spec, "short") == C.K_INF and C.k_of(spec, "long") == 3
        assert C.k_of({"k_short": None, "k_long": 2}, "short") == C.K_INF
        assert C.k_of({"k_short": 1, "k_long": 2}, "long") == 2
        with pytest.raises(C.ContractError):
            C.k_of(spec, "medium")

    def test_k_limit_prefers_override(self):
        spec = C.PolicySpec("x", 1, 1)
        base = dict(req_id="r", type_id="bert:5", family="bert", klass="short", t_arrival_ns=0)
        assert C.k_limit(spec, C.QueuedReq(**base)) == 1
        assert C.k_limit(spec, C.QueuedReq(**base, k_override=4)) == 4


# ================================================================================================ hashing and freeze
class TestHashing:
    def test_content_sha256_formula(self):
        obj = {"b": [1, 2], "a": "é", "content_sha256": "ignored"}
        text = json.dumps({"b": [1, 2], "a": "é"}, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                          allow_nan=False)
        assert C.content_sha256(obj) == hashlib.sha256(text.encode("ascii")).hexdigest()
        assert "\\u00e9" in text

    def test_stable_under_key_order_and_changes_with_a_value(self):
        a = {"x": 1, "y": {"p": [1, 2], "q": 2.5}}
        b = {"y": {"q": 2.5, "p": [1, 2]}, "x": 1}
        assert C.content_sha256(a) == C.content_sha256(b)
        for mutate in (lambda o: o.__setitem__("x", 2), lambda o: o["y"]["p"].__setitem__(1, 3),
                       lambda o: o["y"].__setitem__("q", 2.50001), lambda o: o.__setitem__("z", None)):
            c = copy.deepcopy(a)
            mutate(c)
            assert C.content_sha256(c) != C.content_sha256(a)

    def test_ignores_only_its_own_field(self):
        a = {"x": 1}
        assert C.content_sha256(a) == C.content_sha256({"x": 1, "content_sha256": HEX_A})
        assert C.content_sha256({"x": 1, "sha256": HEX_A}) != C.content_sha256(a)

    def test_nan_is_refused(self):
        with pytest.raises(ValueError):
            C.content_sha256({"x": float("nan")})

    def test_plan_sha256_excludes_sha256_field_only(self):
        p = plan_obj("closed")
        assert p["sha256"] == C.plan_sha256(p)
        q = copy.deepcopy(p)
        q["estimated_wall_s"] += 1
        assert C.plan_sha256(q) != p["sha256"]
        assert C.plan_sha256(dict(p, sha256=HEX_B)) == p["sha256"]

    def test_with_helpers_do_not_mutate(self):
        o = {"x": 1}
        out = C.with_content_sha256(o)
        assert "content_sha256" not in o and out["content_sha256"] == C.content_sha256(o)
        p = {"x": 1}
        assert "sha256" not in p and C.with_plan_sha256(p)["sha256"] == C.plan_sha256(p)


class TestFreezeFile:
    @staticmethod
    def write(tmp_path, obj=None):
        obj = obj or freeze_obj()
        path = tmp_path / "freeze.json"
        path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        C.write_sidecar(path)
        return path

    def test_sidecar_format_is_sha256sum_compatible(self, tmp_path):
        path = self.write(tmp_path)
        side = (tmp_path / "freeze.json.sha256").read_bytes()
        assert side == f"{hashlib.sha256(path.read_bytes()).hexdigest()}  freeze.json\n".encode("ascii")

    def test_good_file_verifies_and_returns_content(self, tmp_path):
        obj = freeze_obj()
        assert C.verify_freeze_file(self.write(tmp_path, obj)) == obj

    def test_one_changed_byte_is_rejected(self, tmp_path):
        path = self.write(tmp_path)
        raw = path.read_bytes()
        i = raw.index(b"64800.0")
        path.write_bytes(raw[:i] + b"64801.0" + raw[i + 7:])          # sidecar still holds the old hash
        with pytest.raises(C.FreezeError, match="sidecar"):
            C.verify_freeze_file(path)

    def test_changed_content_with_refreshed_sidecar_fails_on_content_hash(self, tmp_path):
        path = self.write(tmp_path)
        raw = path.read_bytes()
        i = raw.index(b"64800.0")
        path.write_bytes(raw[:i] + b"64801.0" + raw[i + 7:])
        C.write_sidecar(path)                                          # file bytes and sidecar now agree
        with pytest.raises(C.FreezeError, match="content_sha256"):
            C.verify_freeze_file(path)

    def test_mismatching_sidecar_is_rejected(self, tmp_path):
        path = self.write(tmp_path)
        side = tmp_path / "freeze.json.sha256"
        side.write_text(f"{HEX_B}  freeze.json\n")
        with pytest.raises(C.FreezeError):
            C.verify_freeze_file(path)

    @pytest.mark.parametrize("text", ["", "not a hash\n", "abc  freeze.json\n", "A" * 64 + "  freeze.json\n"])
    def test_malformed_sidecar_is_rejected(self, tmp_path, text):
        path = self.write(tmp_path)
        (tmp_path / "freeze.json.sha256").write_text(text)
        with pytest.raises(C.FreezeError):
            C.verify_freeze_file(path)

    def test_missing_sidecar_or_file_is_rejected(self, tmp_path):
        path = self.write(tmp_path)
        (tmp_path / "freeze.json.sha256").unlink()
        with pytest.raises(C.FreezeError):
            C.verify_freeze_file(path)
        with pytest.raises(C.FreezeError):
            C.verify_freeze_file(tmp_path / "absent.json")

    def test_not_json_is_a_freeze_error_not_a_decode_error(self, tmp_path):
        path = tmp_path / "freeze.json"
        path.write_text("{oops")
        C.write_sidecar(path)
        with pytest.raises(C.FreezeError):
            C.verify_freeze_file(path)

    def test_hash_consistent_but_schema_violation_is_freeze_error(self, tmp_path):
        bad = freeze_obj()
        del bad["content_sha256"]
        del bad["sim"]
        path = self.write(tmp_path, C.with_content_sha256(bad))
        with pytest.raises(C.FreezeError, match="contract"):
            C.verify_freeze_file(path)

    def test_file_without_content_hash_is_rejected(self, tmp_path):
        obj = freeze_obj()
        del obj["content_sha256"]
        with pytest.raises(C.FreezeError):
            C.verify_freeze_file(self.write(tmp_path, obj))


# ================================================================================================ PolicySpec, TypeTable
class TestPolicySpec:
    def test_json_round_trip_and_inf_is_null(self):
        spec = C.PolicySpec.from_json(POLICIES["default"])
        assert (spec.k_short, spec.k_long) == (C.K_INF, C.K_INF)
        assert spec.to_json() == POLICIES["default"]
        for name, d in POLICIES.items():
            assert C.PolicySpec.from_json(d).to_json() == d
            assert C.PolicySpec.from_json(C.PolicySpec.from_json(d).to_json()) == C.PolicySpec.from_json(d)

    def test_none_and_k_inf_are_the_same_spec(self):
        assert C.PolicySpec("a", None, 1) == C.PolicySpec("a", C.K_INF, 1)
        assert C.PolicySpec("a", C.K_INF, 1).to_json()["k_short"] is None

    def test_defaults(self):
        s = C.PolicySpec("d", 1, 1)
        assert (s.order, s.lane_cap, s.f_rule, s.f_mhz, s.params) == ("fcfs", None, "governor", None, {})

    def test_frozen(self):
        with pytest.raises(Exception):
            C.PolicySpec("d", 1, 1).name = "e"

    def test_from_json_requires_exactly_the_keys(self):
        d = copy.deepcopy(POLICIES["serial"])
        del d["order"]
        with pytest.raises(C.ContractError):
            C.PolicySpec.from_json(d)
        with pytest.raises(C.ContractError):
            C.PolicySpec.from_json(dict(POLICIES["serial"], extra=1))

    def test_from_json_does_not_enforce_cross_field_rules(self):
        bad = dict(POLICIES["serial_edgeserving"], k_short=2)
        spec = C.PolicySpec.from_json(bad)                  # Policy() must call validate_policyspec
        with pytest.raises(C.ContractError):
            C.validate_policyspec(spec)

    @pytest.mark.parametrize("name", sorted(POLICIES))
    def test_registry_specs_are_valid(self, name):
        assert C.validate_policyspec(POLICIES[name]) is POLICIES[name]
        C.validate_policyspec(C.PolicySpec.from_json(POLICIES[name]))               # object form

    @pytest.mark.parametrize("patch", [
        {"order": "edgeserving", "k_short": 2},                      # edgeserving needs 1/1
        {"order": "edgeserving", "k_long": None},
        {"order": "edgeserving", "k_short": None, "k_long": None},
        {"f_rule": "static", "f_mhz": None},                        # static needs f_mhz
        {"f_rule": "governor", "f_mhz": 1020},                      # governor needs null f_mhz
        {"f_rule": "greenllm", "f_mhz": None, "lane_cap": None},   # greenllm needs lane_cap
        {"f_rule": "occupancy", "f_mhz": None, "params": {"alpha_p": 2}},          # occupancy needs f_base_mhz
        {"f_rule": "occupancy", "f_mhz": None, "params": {"f_base_mhz": 612}},     # ... and alpha_p
        {"order": "lifo"}, {"f_rule": "magic"}, {"k_short": 0}, {"k_long": -1}, {"k_short": 1.5},
        {"lane_cap": {"short": 1}}, {"lane_cap": {"short": 0, "long": 1}}, {"lane_cap": [1, 1]},
        {"name": ""}, {"params": None},
    ])
    def test_rule_violations(self, patch):
        d = dict(copy.deepcopy(POLICIES["serial"]), **patch)
        with pytest.raises(C.ContractError):
            C.validate_policyspec(d)

    def test_missing_and_extra_keys(self):
        for k in POLICIES["serial"]:
            d = {x: v for x, v in POLICIES["serial"].items() if x != k}
            with pytest.raises(C.ContractError):
                C.validate_policyspec(d)
        with pytest.raises(C.ContractError):
            C.validate_policyspec(dict(POLICIES["serial"], bonus=1))


class TestTypeTable:
    def test_example_loads_and_is_labelled_synthetic(self, types):
        assert types.synthetic is True and types.alpha == 5.0
        assert types.f_levels_mhz == (408, 612, 714, 816, 1020) and types.corun_levels_mhz == (612, 1020)
        assert (types.heldout_level_mhz, types.f_max_mhz) == (714, 1020)
        assert types.type_ids() == ["bert:5", "bert:23", "vision:6", "vision:24", "llama:3", "llama:15", "yolo:1",
                                    "yolo:5"]

    def test_example_matches_k1_numbers(self, types):
        t = types.get("bert:5")
        assert t.ckpts == (3, 5) and t.klass == "short" and t.head_frac == 0.10
        assert types.tau("bert:5", 1020, "p50") == 0.0290 and types.tau("bert:5", 1020, "mean") == 0.0294
        assert types.tau("bert:5", 612, "p95") == 0.0450 and types.tau("bert:5", 612) == 0.0412
        assert t.ckpt_frac == {3: 0.62, 5: 0.90} and (t.u[612], t.u[1020]) == (0.71, 0.66)
        assert types.families["bert"]["resident_mb"] == 1380.0 and t.act_mb == 38.0

    def test_class_rule_short_iff_mean_at_fmax_le_median(self, types):
        means = {t.type_id: types.tau(t.type_id, 1020, "mean") for t in types.types}
        vals = sorted(means.values())
        median = (vals[3] + vals[4]) / 2
        for t in types.types:
            assert (t.klass == "short") == (means[t.type_id] <= median), t.type_id
        assert sum(t.klass == "short" for t in types.types) == 4

    def test_every_family_has_two_exits_for_run1(self, types):
        by = {}
        for t in types.types:
            by.setdefault(t.family, []).append(t.exit)
        assert by == {"bert": [5, 23], "vision": [6, 24], "llama": [3, 15], "yolo": [1, 5]}

    def test_tau_ordering_is_physical(self, types):
        for t in types.types:
            assert t.tau_s[408]["p50"] > t.tau_s[612]["p50"] > t.tau_s[1020]["p50"]
            for f in t.tau_s:
                assert t.tau_s[f]["p50"] <= t.tau_s[f]["p95"]

    def test_json_round_trip(self, types, types_json):
        again = C.TypeTable.from_json(types.to_json())
        assert again == types and again.to_json() == types.to_json()
        assert json.loads(json.dumps(types.to_json())) == types_json

    def test_get_and_tau_errors(self, types):
        with pytest.raises(C.ContractError):
            types.get("bert:6")
        with pytest.raises(C.ContractError):
            types.tau("bert:5", 999)
        with pytest.raises(C.ContractError):
            types.tau("bert:5", 1020, "p99")
        assert types.tau("bert:5", 1020.0) == 0.0290

    def test_load_types_reads_a_path(self):
        assert C.load_types(str(EXAMPLE_TYPES)).f_max_mhz == 1020

    def test_non_synthetic_table_omits_the_flag(self, types_json):
        del types_json["synthetic"]
        t = C.TypeTable.from_json(types_json)
        assert t.synthetic is False and "synthetic" not in t.to_json()

    @pytest.mark.parametrize("mutate,why", [
        (lambda o: o.__setitem__("schema", "exp2.types/2"), "schema"),
        (lambda o: o.__setitem__("alpha", 0), "alpha"),
        (lambda o: o.__setitem__("alpha", "5"), "alpha type"),
        (lambda o: o.__setitem__("f_levels_mhz", [1020, 612]), "levels order"),
        (lambda o: o.__setitem__("corun_levels_mhz", [612, 999]), "corun not subset"),
        (lambda o: o.__setitem__("f_max_mhz", 816), "f_max"),
        (lambda o: o.__setitem__("corun_levels_mhz", [612]), "f_max not corun"),
        (lambda o: o.__setitem__("heldout_level_mhz", 612), "heldout is corun"),
        (lambda o: o.__setitem__("heldout_level_mhz", 700), "heldout not a level"),
        (lambda o: o["families"].pop("llama"), "family missing"),
        (lambda o: o["families"].__setitem__("gpt", {"resident_mb": 1}), "family unknown"),
        (lambda o: o["types"][0].__setitem__("type_id", "bert:6"), "type_id vs exit"),
        (lambda o: o["types"][1].__setitem__("type_id", "bert:5"), "duplicate id"),
        (lambda o: o["types"][0].__setitem__("klass", "medium"), "klass"),
        (lambda o: o["types"][0].__setitem__("ckpts", [3, 4, 5]), "ckpts"),
        (lambda o: o["types"][0]["tau_s"].pop("612"), "tau missing corun level"),
        (lambda o: o["types"][0]["tau_s"].__setitem__("999", o["types"][0]["tau_s"]["612"]), "tau unknown level"),
        (lambda o: o["types"][0]["tau_s"]["612"].pop("mean"), "tau stat missing"),
        (lambda o: o["types"][0]["tau_s"]["612"].__setitem__("p50", 0.5), "p50 > p95"),
        (lambda o: o["types"][0]["tau_s"]["612"].__setitem__("samples", []), "samples empty"),
        (lambda o: o["types"][0]["ckpt_frac"].pop("3"), "ckpt_frac keys"),
        (lambda o: o["types"][0]["ckpt_frac"].__setitem__("5", 0.5), "ckpt_frac+head far from 1"),
        (lambda o: o["types"][0]["ckpt_frac"].__setitem__("3", 0.95), "ckpt_frac decreasing"),
        (lambda o: o["types"][0].__setitem__("head_frac", 0.3), "head_frac vs ckpt_frac"),
        (lambda o: o["types"][4].__setitem__("head_frac", 0.05), "llama head must be 0"),
        (lambda o: o["types"][0]["u"].pop("1020"), "u missing level"),
        (lambda o: o["types"][0]["u"].__setitem__("612", 1.5), "u > 1"),
        (lambda o: o["types"][0].__setitem__("act_mb", -1), "act_mb"),
        (lambda o: o["types"][0].pop("u"), "type key missing"),
        (lambda o: o["types"][0].__setitem__("extra", 1), "type extra key"),
        (lambda o: o["provenance"].pop("sources"), "provenance"),
        (lambda o: o.__setitem__("types", []), "no types"),
        (lambda o: o.__setitem__("extra", 1), "extra key"),
    ])
    def test_validate_types_rejects(self, types_json, mutate, why):
        mutate(types_json)
        with pytest.raises(C.ContractError):
            C.validate_types(types_json)
        with pytest.raises(C.ContractError):
            C.TypeTable.from_json(types_json)

    def test_ckpt_frac_within_0_05_boundary(self, types_json):
        t = types_json["types"][0]
        t["ckpt_frac"]["5"] = 0.94999                  # 0.94999 + 0.10 = 1.04999: inside
        C.validate_types(types_json)
        t["ckpt_frac"]["5"] = 0.9502                   # 1.0502: outside
        with pytest.raises(C.ContractError):
            C.validate_types(types_json)

    def test_alpha_may_be_null(self, types_json):
        types_json["alpha"] = None
        assert C.TypeTable.from_json(types_json).alpha is None


class TestWorkload:
    def test_round_trip_and_rate_check(self, types):
        w = C.Workload.from_json(W_HI)
        assert w.to_json() == W_HI and w.types == ("bert:5", "llama:15")
        C.validate_workload(W_HI, types)

    def test_rate_must_match_rate_for_load_when_types_given(self, types):
        with pytest.raises(C.ContractError, match="rate_for_load"):
            C.validate_workload(dict(W_HI, rate_per_s=5.0), types)
        C.validate_workload(dict(W_HI, rate_per_s=5.0))                 # without types the rate is only positive

    @pytest.mark.parametrize("patch", [
        {"probs": [0.5, 0.4]}, {"probs": [1.0]}, {"types": ["bert:5", "bert:5"]}, {"types": ["bert:5", "gpt:1"]},
        {"rho": 0}, {"rate_per_s": -1}, {"arrival": "uniform"}, {"cv_a": 2.0}, {"cv_a": 0}, {"name": ""},
        {"probs": [1.2, -0.2]},
    ])
    def test_rejects(self, patch):
        with pytest.raises(C.ContractError):
            C.validate_workload(dict(W_HI, **patch))

    def test_gamma_allows_any_positive_cv(self):
        C.validate_workload(dict(W_HI, arrival="gamma", cv_a=2.0))


# ================================================================================================ dataclasses
class TestDataclasses:
    def test_dispatch_view_freezes_lists_and_round_trips(self):
        case = load_json(FIX / "admission_cases.json")[3]
        view = C.DispatchView.from_json(case["view"])
        assert isinstance(view.queue, tuple) and isinstance(view.in_service, tuple)
        assert all(isinstance(q, C.QueuedReq) for q in view.queue)
        assert view.to_json() == case["view"]
        assert C.DispatchView(1, [], [], None).queue == ()
        with pytest.raises(Exception):
            view.now_ns = 5

    def test_sim_config_defaults_match_the_brief(self):
        c = C.SimConfig()
        assert (c.n_requests, c.n_warmup, c.n_reps, c.seed) == (20000, 500, 5, 0)
        assert (c.family_exclusive, c.sample_slowdown, c.alpha, c.switch_lag_s) == (True, True, 5.0, 0.0)
        assert c.tag is None and c.schedule is None

    def test_sim_config_replay_forces_one_rep(self):
        c = C.SimConfig(n_reps=5, schedule=[{"req_id": "r1"}])
        assert c.n_reps == 1 and isinstance(c.schedule, tuple)
        assert C.SimConfig(n_reps=3).n_reps == 3

    def test_tag_spec(self):
        t = C.TagSpec("short", 3)
        assert (t.klass, t.k_dev, t.every) == ("short", 3, 50)

    def test_sim_result_round_trip(self):
        obj = simresult_obj()
        res = C.SimResult.from_json(obj)
        assert isinstance(res.per_type["bert:5"], C.TypeStats) and res.energy_ci == (0.60, 0.62)
        assert res.to_json() == obj
        assert C.validate_simresult(res.to_json())

    def test_type_stats_keys(self):
        st = C.TypeStats.from_json(simresult_obj()["per_type"]["bert:5"])
        assert st.to_json() == simresult_obj()["per_type"]["bert:5"]


# ================================================================================================ object validators
class TestValidatorsAcceptTheDocumentedExamples:
    @pytest.mark.parametrize("name,validator,factory", OBJECT_VALIDATORS, ids=IDS)
    def test_valid_object_passes_and_is_returned(self, name, validator, factory):
        obj = factory()
        assert validator(obj) is obj

    @pytest.mark.parametrize("name,validator,factory", OBJECT_VALIDATORS, ids=IDS)
    def test_every_missing_key_is_rejected(self, name, validator, factory):
        obj = factory()
        optional = {"synthetic"}
        for key in list(obj):
            if key in optional:
                continue
            bad = {k: v for k, v in obj.items() if k != key}
            with pytest.raises(C.ContractError, match="missing"):
                validator(bad)

    @pytest.mark.parametrize("name,validator,factory", OBJECT_VALIDATORS, ids=IDS)
    def test_extra_key_is_rejected(self, name, validator, factory):
        with pytest.raises(C.ContractError, match="unexpected"):
            validator(dict(factory(), surprise=1))

    @pytest.mark.parametrize("name,validator,factory", OBJECT_VALIDATORS, ids=IDS)
    def test_non_dict_is_rejected(self, name, validator, factory):
        for bad in (None, [], "x", 3):
            with pytest.raises(C.ContractError):
                validator(bad)

    @pytest.mark.parametrize("name,validator,factory", SCHEMA_VALIDATORS, ids=[x[0] for x in SCHEMA_VALIDATORS])
    def test_wrong_schema_id_is_rejected(self, name, validator, factory):
        with pytest.raises(C.ContractError, match="schema"):
            validator(dict(factory(), schema="exp2.other/1"))

    def test_validate_schema_dispatches(self):
        for name, validator, factory in OBJECT_VALIDATORS:
            obj = factory()
            if "schema" in obj:
                assert C.validate_schema(obj) is obj
        with pytest.raises(C.ContractError):
            C.validate_schema({"schema": "exp2.nope/1"})
        with pytest.raises(C.ContractError):
            C.validate_schema({})
        assert set(C.VALIDATORS) == set(C.SCHEMAS.values())


class TestRequestRow:
    def test_keys_are_exactly_request_keys_in_order(self):
        assert tuple(request_row()) == C.REQUEST_KEYS and len(C.REQUEST_KEYS) == 30

    def test_expired_and_closed_variants_are_valid(self):
        C.validate_request_row(expired_row())
        C.validate_request_row(closed_row())

    def test_error_and_warmup_rows_are_valid(self):
        r = request_row()
        r.update({"outcome": "error", "err": "ConnectionResetError", "reached": None, "completion": 0.0, "ckpt": [],
                  "t_end_ns": None, "t_done_ns": 5, "is_warmup": True})
        C.validate_request_row(r)

    def test_governor_f_mhz_string(self):
        C.validate_request_row(dict(request_row(), f_mhz="governor"))
        with pytest.raises(C.ContractError):
            C.validate_request_row(dict(request_row(), f_mhz="fast"))

    def test_truncated_and_miss_rows(self):
        C.validate_request_row(dict(request_row(), outcome="truncated", reached=7, completion=8 / 24))
        C.validate_request_row(dict(request_row(), outcome="miss", reached=None, completion=0.0))

    @pytest.mark.parametrize("patch", [
        {"outcome": "truncated"},                                           # reached == target
        {"outcome": "complete", "reached": 7, "completion": 8 / 24},
        {"outcome": "complete", "completion": 0.9},
        {"outcome": "miss"},                                                # reached not null
        {"outcome": "bogus"}, {"klass": "medium"}, {"mode": "batch"}, {"family": "vision"}, {"exit": 22},
        {"target": 22}, {"completion": 1.5}, {"completion": -0.1}, {"reached": 7},   # completion disagrees with reached
        {"is_warmup": 0}, {"seed": "11"}, {"t_done_ns": None}, {"t_arrival_ns": 1.5}, {"type_id": "bert"},
        {"err": "boom"},                                                    # err on a complete row
        {"queue_len_at_arrival": -1}, {"ckpt": [[3, 5], [3, 6]]}, {"ckpt": [[3]]}, {"ckpt": [[7, 9], [11, 8]]},
        {"policy_spec": {"name": "x"}}, {"f_mhz": 0},
    ])
    def test_rejects(self, patch):
        with pytest.raises(C.ContractError):
            C.validate_request_row(dict(request_row(), **patch))

    def test_error_outcome_requires_err_text(self):
        r = dict(request_row(), outcome="error", reached=None, completion=0.0)
        with pytest.raises(C.ContractError, match="err"):
            C.validate_request_row(r)

    @pytest.mark.parametrize("patch", [
        {"t_admit_ns": 5}, {"t_start_ns": 5}, {"n_in_service_at_admit": 0}, {"ckpt": [[3, 5]]},
        {"t_done_ns": 7}, {"completion": 0.1}, {"reached": 3},
    ])
    def test_expired_rule_violations(self, patch):
        with pytest.raises(C.ContractError):
            C.validate_request_row(dict(expired_row(), **patch))

    def test_expired_needs_a_deadline(self):
        with pytest.raises(C.ContractError):
            C.validate_request_row(dict(expired_row(), t_deadline_ns=None, t_done_ns=5))

    def test_closed_loop_rules(self):
        with pytest.raises(C.ContractError):
            C.validate_request_row(dict(closed_row(), t_deadline_ns=100))
        with pytest.raises(C.ContractError):
            C.validate_request_row(dict(closed_row(), t_admit_ns=closed_row()["t_arrival_ns"] + 1))

    def test_llama_rows_use_token_units(self):
        r = request_row()
        r.update({"type_id": "llama:15", "family": "llama", "exit": 15, "target": 32, "reached": 20,
                  "completion": 0.625, "outcome": "truncated", "tokens_done": 20, "server": "llama"})
        C.validate_request_row(r)


class TestServerMessages:
    def test_ready_rules(self):
        for patch in ({"op": "run"}, {"family": "gpt"}, {"exits_warmed": []}, {"exits_warmed": [99]},
                      {"input_source": "magic"}, {"t_ready_ns": "1"}, {"pid": -1}):
            with pytest.raises(C.ContractError):
                C.validate_ready(dict(ready_msg(), **patch))

    def test_ready_warmup_keys_must_match_exits(self):
        m = ready_msg()
        m["warmup"]["cv_last5"] = {"5": 0.01}
        with pytest.raises(C.ContractError, match="exits_warmed"):
            C.validate_ready(m)

    def test_ready_versions_and_compile(self):
        m = ready_msg()
        m["versions"]["compile"] = "maybe"
        with pytest.raises(C.ContractError):
            C.validate_ready(m)
        m = ready_msg()
        m["versions"].update({"torch": None, "transformers": None, "cuda": None, "compile": "off"})
        C.validate_ready(m)                                           # fake adapter: no torch
        del m["versions"]["cuda"]
        with pytest.raises(C.ContractError):
            C.validate_ready(m)

    def test_run_rules(self):
        for patch in ({"op": "done"}, {"exit": 5}, {"type_id": "bert"}, {"input_idx": -1}, {"deadline_ns": "x"}):
            with pytest.raises(C.ContractError):
                C.validate_run(dict(run_msg(), **patch))
        C.validate_run(dict(run_msg(), deadline_ns=None))

    def test_done_rules(self):
        for patch in ({"op": "run"}, {"outcome": "expired_in_queue"}, {"target": 22}, {"completion": 0.9},
                      {"outcome": "complete"}, {"reached": 23, "completion": 1.0}, {"err": "x"},
                      {"t_start_ns": 1, "t_recv_ns": 5}, {"ckpt": [[3, 9], [3, 10]]}):
            with pytest.raises(C.ContractError):
                C.validate_done(dict(done_msg(), **patch))

    def test_done_miss_complete_error(self):
        C.validate_done(dict(done_msg(), outcome="miss", reached=None, completion=0.0, ckpt=[[3, 1712320000000]]))
        C.validate_done(dict(done_msg(), outcome="complete", reached=23, completion=1.0))
        C.validate_done(dict(done_msg(), outcome="error", reached=None, completion=0.0, err="CUDA OOM", ckpt=[]))

    def test_validate_server_msg_dispatch(self):
        for m in (ready_msg(), run_msg(), done_msg(), {"op": "ping"}, {"op": "pong", "t_ns": 5}, {"op": "stop"}):
            assert C.validate_server_msg(m) is m
        for bad in ({"op": "bogus"}, {}, {"op": "pong"}, {"op": "ping", "x": 1}, {"op": "done"}):
            with pytest.raises(C.ContractError):
                C.validate_server_msg(bad)


class TestRootMessages:
    @pytest.mark.parametrize("msg", root_msgs(), ids=lambda m: json.dumps(m)[:40])
    def test_valid(self, msg):
        assert C.validate_root_msg(msg) is msg

    @pytest.mark.parametrize("msg", [
        {}, {"op": "bogus"}, {"op": "write", "path": "/p"}, {"op": "write", "path": "/p", "value": None},
        {"op": "write", "path": "/p", "value": True}, {"op": "read", "path": ""}, {"op": "run", "argv": [],
                                                                                      "timeout_s": 5},
        {"op": "run", "argv": ["x"], "timeout_s": 0}, {"op": "hello", "pid": 1}, {"op": "stop", "x": 1},
        {"ok": "yes"}, {"ok": False}, {"ok": False, "err": ""}, {"ok": True, "readback": 5},
        {"ok": True, "rc": "0", "out": ""}, {"ok": True, "rc": 0}, {"ok": True, "unknown": 1}, {"readback": "1"},
    ])
    def test_invalid(self, msg):
        with pytest.raises(C.ContractError):
            C.validate_root_msg(msg)


class TestSession:
    def test_closed_session_has_null_freeze_hash(self):
        C.validate_session(session_obj("closed"))
        with pytest.raises(C.ContractError):
            C.validate_session(dict(session_obj("closed"), freeze_content_sha256=HEX_A))
        with pytest.raises(C.ContractError):
            C.validate_session(dict(session_obj("open"), freeze_content_sha256=None))

    def test_servers_must_start_one_at_a_time(self):
        s = session_obj()
        s["servers"]["vision"]["t_spawn_ns"] = 15            # bert is ready only at 20
        with pytest.raises(C.ContractError, match="one at a time"):
            C.validate_session(s)

    def test_server_ready_not_before_spawn(self):
        s = session_obj()
        s["servers"]["bert"]["t_ready_ns"] = 5
        with pytest.raises(C.ContractError):
            C.validate_session(s)

    def test_spawn_at_exactly_previous_ready_is_allowed(self):
        s = session_obj()
        s["servers"]["vision"]["t_spawn_ns"] = 20
        C.validate_session(s)

    def test_inline_ready_is_validated_when_present(self):
        s = session_obj()
        s["servers"]["vision"]["ready"]["pid"] = -5
        with pytest.raises(C.ContractError):
            C.validate_session(s)

    @pytest.mark.parametrize("patch", [{"plan_kind": "weird"}, {"plan_sha256": "xyz"},
                                       {"plan_sha256": HEX_A + "\n"}, {"exit_code": "0"},
                                       {"conditions": ["a", "a"]}, {"t_wall_start_ns": 0},
                                       {"idle": {"pre": {"t0_ns": 5, "t1_ns": 5, "gpu_mhz": 1020},
                                                 "resident": {"t0_ns": 2, "t1_ns": 3, "gpu_mhz": 1020}}}])
    def test_rejects(self, patch):
        with pytest.raises(C.ContractError):
            C.validate_session(dict(session_obj(), **patch))

    def test_unfinished_session_may_lack_end_time(self):
        C.validate_session(dict(session_obj(), t_wall_end_ns=None))


class TestManifest:
    def test_closed_and_idle_variants(self):
        C.validate_manifest(manifest_obj("closed"))
        C.validate_manifest(manifest_obj("idle"))

    def test_valid_false_needs_a_reason_and_vice_versa(self):
        C.validate_manifest(dict(manifest_obj(), valid=False, invalid_reason="ClockError: readback"))
        with pytest.raises(C.ContractError):
            C.validate_manifest(dict(manifest_obj(), valid=False))
        with pytest.raises(C.ContractError):
            C.validate_manifest(dict(manifest_obj(), invalid_reason="why"))

    def test_governor_arm_has_null_gpu(self):
        C.validate_manifest(dict(manifest_obj(), arm="governor", gpu_mhz=None))
        with pytest.raises(C.ContractError):
            C.validate_manifest(dict(manifest_obj(), arm="governor", gpu_mhz=612))
        with pytest.raises(C.ContractError):
            C.validate_manifest(dict(manifest_obj(), arm="pinned", gpu_mhz=None))

    @pytest.mark.parametrize("patch", [{"mode": "weird"}, {"arm": "turbo"}, {"policy_spec": None},
                                       {"workload": None}, {"n_requests": None}, {"t_mono_end_ns": 1},
                                       {"settle_s": -1}, {"valid": "yes"}])
    def test_open_rejects(self, patch):
        with pytest.raises(C.ContractError):
            C.validate_manifest(dict(manifest_obj(), **patch))

    def test_closed_needs_cell_and_split(self):
        with pytest.raises(C.ContractError):
            C.validate_manifest(dict(manifest_obj("closed"), cell=None))
        with pytest.raises(C.ContractError):
            C.validate_manifest(dict(manifest_obj("closed"), split="train2"))


class TestConditionIds:
    """A condition id names a directory under the session folder: it must be safe on Windows and POSIX."""

    SAFE = ["W_hi_r060_serial_r0", "idle_f1020", "c001_f612_r0", "solo_bert-5_f612_r0", "a.b", "0"]
    UNSAFE = ["solo_bert:5_f1020_r0", "a b", "a/b", "a\\b", "", ".", "..", ".hidden", "-x", "\u00e9", "a*b", "a\n"]

    @pytest.mark.parametrize("cid", SAFE)
    def test_safe_ids_pass_everywhere(self, cid):
        C.validate_request_row(dict(request_row(), condition_id=cid))
        C.validate_manifest(dict(manifest_obj(), condition_id=cid))
        C.validate_session(dict(session_obj(), conditions=[cid]))
        p = plan_obj("closed")
        p["conditions"][0]["condition_id"] = cid
        p.pop("sha256")
        C.validate_plan(C.with_plan_sha256(p))

    @pytest.mark.parametrize("cid", UNSAFE)
    def test_unsafe_ids_are_rejected_everywhere(self, cid):
        with pytest.raises(C.ContractError):
            C.validate_request_row(dict(request_row(), condition_id=cid))
        with pytest.raises(C.ContractError):
            C.validate_manifest(dict(manifest_obj(), condition_id=cid))
        with pytest.raises(C.ContractError):
            C.validate_session(dict(session_obj(), conditions=[cid]))
        p = plan_obj("closed")
        p["conditions"][0]["condition_id"] = cid
        p.pop("sha256")
        with pytest.raises(C.ContractError):
            C.validate_plan(C.with_plan_sha256(p))


class TestPlan:
    def test_closed_plan(self):
        p = plan_obj("closed")
        assert p["freeze"] is None
        C.validate_plan(p)

    def test_hash_is_enforced(self):
        p = plan_obj()
        p["estimated_wall_s"] = 1.0
        with pytest.raises(C.ContractError, match="sha256"):
            C.validate_plan(p)

    def _bad(self, mutate):
        p = plan_obj()
        mutate(p)
        p.pop("sha256")
        return C.with_plan_sha256(p)

    @pytest.mark.parametrize("mutate", [
        lambda p: p.__setitem__("freeze", None),
        lambda p: p["freeze"].__setitem__("content_sha256", "short"),
        lambda p: p["freeze"].pop("path"),
        lambda p: p["session"]["exits"].pop("yolo"),
        lambda p: p["session"].__setitem__("emc_mhz", 1000.0),
        lambda p: p["session"].__setitem__("preprocess", "lazy"),
        lambda p: p["session"].__setitem__("n_inputs", 0),
        lambda p: p["conditions"].append(dict(p["conditions"][0])),                     # duplicate id
        lambda p: p["conditions"][0].__setitem__("gpu_mhz", None),                    # pinned needs a level
        lambda p: p["conditions"][-1].__setitem__("gpu_mhz", 612),                    # governor needs null
        lambda p: p["conditions"][1].__setitem__("cell", ["bert:7"]),                 # exit not in session
        lambda p: p["conditions"][1].__setitem__("split", "dev"),
        lambda p: p["conditions"][2].__setitem__("policy", {"name": "x"}),
        lambda p: p["conditions"][2]["workload"].__setitem__("probs", [0.5, 0.4]),
        lambda p: p["conditions"][2]["workload"].__setitem__("types", ["bert:7", "llama:15"]),
        lambda p: p["conditions"][2].pop("seed"),
        lambda p: p["conditions"][2].__setitem__("mode", "weird"),
        lambda p: p["conditions"][0].pop("duration_s"),
        lambda p: p.__setitem__("conditions", []),
    ])
    def test_rejects(self, mutate):
        with pytest.raises(C.ContractError):
            C.validate_plan(self._bad(mutate))

    def test_closed_plan_cannot_hold_open_conditions(self):
        p = plan_obj("open")
        p["kind"] = "closed"
        p["freeze"] = None
        p.pop("sha256")
        with pytest.raises(C.ContractError, match="kind"):
            C.validate_plan(C.with_plan_sha256(p))

    def test_stock_arm_needs_null_gpu(self):
        p = plan_obj()
        p["conditions"][0].update({"arm": "stock", "gpu_mhz": None})
        p.pop("sha256")
        C.validate_plan(C.with_plan_sha256(p))


class TestFreeze:
    def test_hash_is_enforced(self):
        f = freeze_obj()
        f["runtime_estimate_s"] = 1.0
        with pytest.raises(C.ContractError, match="content_sha256"):
            C.validate_freeze(f)

    def _bad(self, mutate):
        f = freeze_obj()
        f.pop("content_sha256")
        mutate(f)
        return C.with_content_sha256(f)

    @pytest.mark.parametrize("mutate", [
        lambda f: f.__setitem__("created_utc", "2026-10-21 09:00:00"),
        lambda f: f.__setitem__("created_wall_ns", 0),
        lambda f: f["inputs"].__setitem__("corun_sha256", "nope"),
        lambda f: f["slowdown_model"].__setitem__("selected", "M9"),
        lambda f: f["slowdown_model"].__setitem__("target_met", "yes"),
        lambda f: f["sim"].pop("seed"),
        lambda f: f.__setitem__("f_grid_mhz", [1020, 612]),
        lambda f: f["conditions"][0]["choices"].__setitem__("F", {}),
        lambda f: f["conditions"][0].__setitem__("choices", {}),
        lambda f: f["conditions"][0]["grid"].append({"name": "x"}),
        lambda f: f["conditions"][0]["predictions"].__setitem__("F", {"per_type": {}}),
        lambda f: f["conditions"][0]["game"].pop("poa_E"),
        lambda f: f["conditions"][0]["game"]["equilibria"][0].pop("cycle"),
        lambda f: f["conditions"][0].__setitem__("rho", 0.9),                           # disagrees with workload
        lambda f: f["conditions"][0]["workload"].__setitem__("rho", -1),
        lambda f: f["conditions"].append(dict(f["conditions"][0])),                     # duplicate condition id
        lambda f: f["conditions"][0]["compute_s"].__setitem__("F", -1),
        lambda f: f["git"].__setitem__("dirty", "no"),
    ])
    def test_rejects(self, mutate):
        with pytest.raises(C.ContractError):
            C.validate_freeze(self._bad(mutate))

    def test_poa_may_be_null(self):
        f = freeze_obj()
        f.pop("content_sha256")
        f["conditions"][0]["game"]["poa_R"] = None
        C.validate_freeze(C.with_content_sha256(f))


class TestSimResult:
    @pytest.mark.parametrize("mutate", [
        lambda o: o["per_type"]["bert:5"].__setitem__("complete_rate", 0.5),         # rates stop summing to 1
        lambda o: o["per_type"]["bert:5"].__setitem__("miss_rate", -0.005),
        lambda o: o["per_type"]["bert:5"].pop("p99_R"),
        lambda o: o["per_type"]["bert:5"].__setitem__("n", -1),
        lambda o: o["per_type"].__setitem__("bert:99", o["per_type"]["bert:5"]),
        lambda o: o.__setitem__("energy_ci", [0.7, 0.6]),
        lambda o: o.__setitem__("energy_ci", [0.6]),
        lambda o: o.__setitem__("unstable", 0),
        lambda o: o.__setitem__("utilization", float("nan")),
        lambda o: o["reps"][0].pop("n_slope"),
        lambda o: o.__setitem__("tagged", {"klass": "short"}),
    ])
    def test_rejects(self, mutate):
        o = simresult_obj()
        mutate(o)
        with pytest.raises(C.ContractError):
            C.validate_simresult(o)

    def test_tagged_shape(self):
        o = simresult_obj()
        o["tagged"] = dict(o["per_type"]["bert:5"], klass="short", k_dev=3)
        C.validate_simresult(o)

    def test_zero_requests_type_needs_no_rate_sum(self):
        o = simresult_obj()
        st = o["per_type"]["bert:5"]
        st.update({"n": 0, "complete_rate": 0.0, "truncation_rate": 0.0, "miss_rate": 0.0, "expired_rate": 0.0})
        C.validate_simresult(o)


class TestOtherSchemas:
    def test_slowdown_rules(self):
        for mutate in (lambda o: o.__setitem__("name", "M7"), lambda o: o["residuals"].__setitem__("bert", [1.0]),
                       lambda o: o["residuals"]["bert|2"].append(-1.0), lambda o: o.__setitem__("fit_rows_sha256", "x"),
                       lambda o: o["heldout_mape"].__setitem__("M9", 1.0)):
            o = slowdown_obj()
            mutate(o)
            with pytest.raises(C.ContractError):
                C.validate_slowdown(o)

    def test_power_levels_must_agree(self):
        o = power_obj()
        del o["eta"]["612"]
        with pytest.raises(C.ContractError):
            C.validate_power(o)
        o = power_obj()
        o["p_idle_w"]["1020"] = 0
        with pytest.raises(C.ContractError):
            C.validate_power(o)

    def test_fake_spec_segment_count_matches_checkpoints(self):
        o = fake_spec_obj()
        o["types"]["5"]["seg_ms"] = [20.0]
        with pytest.raises(C.ContractError, match="one per checkpoint"):
            C.validate_fake_spec(o)
        o = fake_spec_obj()
        o["clock"] = "wall"
        with pytest.raises(C.ContractError):
            C.validate_fake_spec(o)
        o = fake_spec_obj()
        o["family"] = "llama"
        o["types"] = {"3": {"seg_ms": [1.0] * 32, "head_ms": 0.0}}
        C.validate_fake_spec(o)

    def test_pilot_gate_shape(self):
        o = pilot_obj()
        o["gates"]["g"] = {"value": 1, "threshold": 1}
        with pytest.raises(C.ContractError):
            C.validate_pilot(o)
        o = pilot_obj()
        o["gates"]["g4_gpu_hold"]["pass"] = "yes"
        with pytest.raises(C.ContractError):
            C.validate_pilot(o)
        C.validate_pilot({"schema": "exp2.pilot/1", "gates": {}, "dry": False})

    def test_telemetry_meta_label_is_vdd_in(self):
        o = telemetry_meta_obj()
        o["vdd_in"]["label"] = "VDD_SOC"
        with pytest.raises(C.ContractError):
            C.validate_telemetry_meta(o)
        o = telemetry_meta_obj()
        o.update({"emc_source": None, "devfreq": None, "boot_id": None})
        C.validate_telemetry_meta(o)

    def test_open_summary_inner_keys(self):
        o = open_summary_obj()
        del o["mape"]["max"]
        with pytest.raises(C.ContractError):
            C.validate_open_summary(o)
        o = open_summary_obj()
        o["energy_diff"]["W_hi_r060_default"].pop("ci")
        with pytest.raises(C.ContractError):
            C.validate_open_summary(o)
        o = open_summary_obj()
        o["per_condition"]["c"] = {"energy": 1}
        with pytest.raises(C.ContractError):
            C.validate_open_summary(o)

    @pytest.mark.parametrize("edit", [
        lambda o: o.pop("counts"),
        lambda o: o.pop("poa_detail"),
        lambda o: o["counts"].pop("runs_screen_flagged"),
        lambda o: o["counts"].update(runs_valid=-1),
        lambda o: o["per_condition"]["W_hi_r060_F"].update(status="maybe"),
        lambda o: o["per_condition"]["W_hi_r060_F"].pop("measured_key"),
        lambda o: o["per_condition"]["W_hi_r060_F"].update(wilson_lo_min=1.5),
        lambda o: o["per_condition"]["W_hi_r060_F"].update(n_repeats=0),
        lambda o: o["energy_diff"]["W_hi_r060_default"].update(method="welch"),
        lambda o: o["energy_diff"]["W_hi_r060_default"].pop("n_pairs"),
        lambda o: o["poa_detail"]["W_hi_r060"].pop("ratio"),
        lambda o: o["poa_detail"]["W_hi_r060"].update(worst_key=""),
    ])
    def test_open_summary_v2_rejects(self, edit):
        o = open_summary_obj()
        edit(o)
        with pytest.raises(C.ContractError):
            C.validate_open_summary(o)

    def test_open_summary_v2_nulls_allowed(self):
        o = open_summary_obj()
        o["poa_detail"]["W_hi_r060"].update(ratio=None, complete_min_worst=None, complete_min_optimum=None)
        o["per_condition"]["W_hi_r060_F"].update(energy_ci=None, n_repeats=1)
        C.validate_open_summary(o)

    def test_open_summary_v1_is_rejected(self):
        o = open_summary_obj()
        o["schema"] = "exp2.open_summary/1"
        with pytest.raises(C.ContractError):
            C.validate_open_summary(o)


# ================================================================================================ csv and jsonl
class TestCsv:
    def test_column_tuples(self):
        assert C.POWER_CSV_COLS == ("t_mono_ns", "t_wall_ns", "vdd_in_mv", "vdd_in_ma", "vdd_in_mw")
        assert C.FREQ_SWITCH_COLS == ("t_req_ns", "t_done_ns", "target_mhz", "readback_mhz")
        assert C.CLOCKS_CSV_FIXED_COLS == ("t_mono_ns", "t_wall_ns", "gpu_cur_hz", "gpu_min_hz", "gpu_max_hz",
                                           "gpu_load_permille", "emc_mhz", "mem_available_kb", "cpu0_cur_khz")
        assert set(C.RUN1_CSV_COLS) == {"solo", "solo_samples", "corun", "corun_samples", "idle", "ckpt", "warmup",
                                        "mem"}
        assert C.RUN1_CSV_COLS["solo"] == ("type_id", "f_mhz", "repeat", "n", "tau_p50", "tau_mean", "tau_p95",
                                           "tau_p50_lo", "tau_p50_hi", "ess", "u", "p_board_w", "window_s")
        assert len(C.RUN1_CSV_COLS["corun"]) == 20 and C.RUN1_CSV_COLS["corun"][6] == "A"
        assert C.RUN1_CSV_COLS["idle"] == ("f_mhz", "kind", "p_idle_w")
        assert len(C.OPEN_CONDITIONS_COLS) == 24 and C.OPEN_CONDITIONS_COLS[0] == "condition_id"
        assert C.OPEN_CONDITIONS_COLS[-3:] == ("late_rate", "overshoot_p95_s", "overshoot_max_s")
        assert set(C.CSV_KINDS) == set(C.RUN1_CSV_COLS) | {"power", "clocks", "freq_switch", "open_conditions",
                                                           "screen"}

    @pytest.mark.parametrize("kind", sorted(C.RUN1_CSV_COLS) + ["power", "freq_switch", "open_conditions"])
    def test_header_must_match_exactly(self, kind):
        cols = {**C.RUN1_CSV_COLS, "power": C.POWER_CSV_COLS, "freq_switch": C.FREQ_SWITCH_COLS,
                "open_conditions": C.OPEN_CONDITIONS_COLS}[kind]
        C.validate_csv_header(kind, list(cols))
        for bad in (list(cols[:-1]), list(cols) + ["extra"], list(reversed(cols)), []):
            with pytest.raises(C.ContractError):
                C.validate_csv_header(kind, bad)

    def test_clocks_header(self):
        fixed = list(C.CLOCKS_CSV_FIXED_COLS)
        C.validate_csv_header("clocks", fixed)
        C.validate_csv_header("clocks", fixed + ["temp_cpu_thermal_mc", "temp_gpu_thermal_mc"])
        for tail in (["temp_gpu_thermal_mc", "temp_cpu_thermal_mc"], ["temp_cpu_thermal_mc", "temp_cpu_thermal_mc"],
                     ["temp_CPU_mc"], ["cpu_temp"], ["temp_x"]):
            with pytest.raises(C.ContractError):
                C.validate_csv_header("clocks", fixed + tail)
        with pytest.raises(C.ContractError):
            C.validate_csv_header("clocks", fixed[1:])

    def test_unknown_kind(self):
        with pytest.raises(C.ContractError):
            C.validate_csv_header("nope", [])

    def test_read_corun_row_is_typed(self, tmp_path):
        p = tmp_path / "corun.csv"
        p.write_text(",".join(C.RUN1_CSV_COLS["corun"]) + "\n"
                     "c017,train,1020,0,bert:5,bert,bert:5|vision:24,400,388,2.21,2.25,2.60,2.18,2.24,61.0,0.0290,"
                     "1.55,24.1,301.2,12.5\n", encoding="utf-8")
        (row,) = C.read_csv(p, "corun")
        assert row["cell_id"] == "c017" and row["A"] == "bert:5|vision:24" and row["split"] == "train"
        assert row["f_mhz"] == 1020 and isinstance(row["f_mhz"], int) and row["n"] == 400 and row["repeat"] == 0
        assert row["s_p50"] == 2.21 and row["ess"] == 61.0 and row["tau_solo_p50"] == 0.0290
        assert list(row) == list(C.RUN1_CSV_COLS["corun"])

    def test_read_other_kinds(self, tmp_path):
        p = tmp_path / "idle.csv"
        p.write_text("f_mhz,kind,p_idle_w\n1020,resident,5.62\n612,pre,\n")
        assert C.read_csv(p, "idle") == [{"f_mhz": 1020, "kind": "resident", "p_idle_w": 5.62},
                                         {"f_mhz": 612, "kind": "pre", "p_idle_w": None}]
        p = tmp_path / "power.csv"
        p.write_text(",".join(C.POWER_CSV_COLS) + "\n5000000,1760951700000000000,5072,1203,6101.616\n")
        assert C.read_csv(p, "power")[0]["vdd_in_mw"] == 6101.616
        p = tmp_path / "clocks.csv"
        p.write_text(",".join(C.CLOCKS_CSV_FIXED_COLS + ("temp_cpu_thermal_mc",)) +
                     "\n5,6,1020000000,306000000,1020000000,273,,5234180,1497600,45000\n")
        row = C.read_csv(p, "clocks")[0]
        assert row["emc_mhz"] is None and row["temp_cpu_thermal_mc"] == 45000 and row["gpu_cur_hz"] == 1020000000
        p = tmp_path / "oc.csv"
        row = {c: "1" for c in C.OPEN_CONDITIONS_COLS}
        row.update({"condition_id": "c", "policy_key": "F", "workload": "W", "arrival": "poisson",
                    "type_id": "bert:5", "f_mhz": "governor"})
        p.write_text(",".join(C.OPEN_CONDITIONS_COLS) + "\n" + ",".join(row[c] for c in C.OPEN_CONDITIONS_COLS) + "\n")
        out = C.read_csv(p, "open_conditions")[0]
        assert out["f_mhz"] == "governor" and out["type_id"] == "bert:5" and out["n"] == 1

    def test_integral_float_text_in_int_column_is_accepted(self, tmp_path):
        p = tmp_path / "s.csv"
        p.write_text("type_id,f_mhz,repeat,tau_s\nbert:5,1020.0,0,0.0291\n")
        assert C.read_csv(p, "solo_samples")[0]["f_mhz"] == 1020

    @pytest.mark.parametrize("body", ["bert:5,fast,0,0.0291\n", "bert:5,1020.5,0,0.0291\n", "bert:5,1020,0\n",
                                      "bert:5,1020,0,abc\n"])
    def test_bad_cells_name_the_file_and_line(self, tmp_path, body):
        p = tmp_path / "s.csv"
        p.write_text("type_id,f_mhz,repeat,tau_s\n" + body)
        with pytest.raises(C.ContractError, match=r"s\.csv:2"):
            C.read_csv(p, "solo_samples")

    def test_empty_file_and_wrong_header(self, tmp_path):
        p = tmp_path / "e.csv"
        p.write_text("")
        with pytest.raises(C.ContractError):
            C.read_csv(p, "idle")
        p.write_text("f_mhz,kind\n")
        with pytest.raises(C.ContractError):
            C.read_csv(p, "idle")

    def test_blank_lines_are_skipped(self, tmp_path):
        p = tmp_path / "i.csv"
        p.write_text("f_mhz,kind,p_idle_w\n1020,pre,5.0\n\n612,pre,5.1\n")
        assert [r["f_mhz"] for r in C.read_csv(p, "idle")] == [1020, 612]


class TestRequestsJsonl:
    def test_reads_valid_rows_and_skips_blank_lines(self, tmp_path):
        p = tmp_path / "requests.jsonl"
        p.write_text(json.dumps(request_row()) + "\n\n" + json.dumps(expired_row()) + "\n")
        rows = C.read_requests(p)
        assert [r["outcome"] for r in rows] == ["complete", "expired_in_queue"]

    def test_invalid_row_is_reported_with_its_line(self, tmp_path):
        p = tmp_path / "requests.jsonl"
        bad = request_row()
        bad["surprise"] = 1
        p.write_text(json.dumps(request_row()) + "\n" + json.dumps(bad) + "\n")
        with pytest.raises(C.ContractError, match=r"requests\.jsonl:2.*unexpected"):
            C.read_requests(p)

    def test_invalid_json_is_a_contract_error(self, tmp_path):
        p = tmp_path / "requests.jsonl"
        p.write_text("{nope\n")
        with pytest.raises(C.ContractError, match=r":1"):
            C.read_requests(p)


# ================================================================================================ protocols
class TestProtocols:
    def test_protocols_are_runtime_checkable(self):
        class Adapter:
            family = "bert"

            def load(self, exits, cfg): ...
            def make_inputs(self, n, seed, mode): ...
            def checkpoints(self, exit): ...
            def begin(self, input_idx, exit): ...
            def advance(self, state, ckpt): ...
            def sync(self): ...
            def finish(self, state, reached): ...
            def mem_stats(self): ...
            def reset_peak(self): ...

        assert isinstance(Adapter(), C.FamilyAdapter)
        assert not isinstance(object(), C.FamilyAdapter)

        class Power:
            def p(self, A, f_mhz): ...
            def to_json(self): ...

        assert isinstance(Power(), C.PowerModelP)

    def test_policy_and_helper_protocol_names(self):
        for name in ("Policy", "FamilyAdapter", "SlowdownModel", "PowerModelP", "BoardControl", "RootHelperClient"):
            assert hasattr(C, name)


# =============================================================================================== golden fixtures (sha
ADMISSION = load_json(FIX / "admission_cases.json") if (FIX / "admission_cases.json").exists() else []
TRACES = load_json(FIX / "admission_traces.json") if (FIX / "admission_traces.json").exists() else []


class TestGoldenFixtureShape:
    def test_counts_and_unique_ids(self):
        assert len(ADMISSION) >= 20 and len(TRACES) >= 6 and len(TRUNC_CASES) >= 12
        for items in (ADMISSION, TRACES, TRUNC_CASES):
            ids = [x["id"] for x in items]
            assert len(ids) == len(set(ids))

    def test_brief_trace_t1_is_present_verbatim(self):
        t1 = next(t for t in TRACES if t["id"] == "T1_serial_expiry")
        assert t1["service_ns"] == {"bert:5": 10000000, "vision:24": 40000000}
        assert t1["expect"]["r2"] == {"t_admit_ns": 40000000, "outcome": "admitted"}
        assert t1["expect"]["r3"] == {"t_admit_ns": None, "outcome": "expired_in_queue", "t_done_ns": 38000000}
        assert t1["expect_commanded_f"] == []

    def test_brief_k20_example_is_present(self):
        c = next(c for c in ADMISSION if c["id"].startswith("k04_head_family_busy"))
        assert c["expect_admit"] == ["r3"] and c["expect_desired_f"] is None
        assert [q["req_id"] for q in c["view"]["queue"]] == ["r2", "r3", "r4"]

    def test_the_22_listed_scenarios_are_covered(self):
        ids = " ".join(c["id"] for c in ADMISSION)
        for needle in ("empty_queue", "k1_n0", "k1_n1", "nonblocking", "same_family", "four_families", "per_class",
                       "n_grows", "stf_orders", "stf_tie", "class_short", "lane_cap", "override_raises",
                       "override_lowers", "edgeserving_three", "edgeserving_tie", "k_2_1_is_contracterror",
                       "desired_f_static", "greenllm_feasible", "infeasible", "occupancy_n3", "family_exclusive_false"):
            assert needle in ids, needle
        assert sum("edgeserving" in c["id"] for c in ADMISSION) >= 4
        assert sum("greenllm" in c["id"] for c in ADMISSION) >= 5

    @pytest.mark.parametrize("case", ADMISSION, ids=[c["id"] for c in ADMISSION])
    def test_admission_case_is_well_formed(self, case, types):
        assert case["types_ref"] == "exp2/configs/example_types.json"
        assert (REPO / case["types_ref"]).exists()
        view = C.DispatchView.from_json(case["view"])
        assert view.to_json() == case["view"]
        for r in list(view.queue) + list(view.in_service):
            t = types.get(r.type_id)
            assert (r.family, r.klass) == (t.family, t.klass)
        assert [q.t_arrival_ns for q in view.queue] == sorted(q.t_arrival_ns for q in view.queue)
        everyone = list(view.queue) + list(view.in_service)
        assert len({r.req_id for r in everyone}) == len(everyone)
        if case.get("expect_error"):
            assert "expect_admit" not in case
            with pytest.raises(getattr(C, case["expect_error"])):
                C.validate_policyspec(case["spec"])
            return
        C.validate_policyspec(case["spec"])
        assert set(case["expect_admit"]) <= {q.req_id for q in view.queue}
        assert len(set(case["expect_admit"])) == len(case["expect_admit"])
        assert case["expect_desired_f"] in (None, *types.corun_levels_mhz)
        if case["spec"]["order"] == "edgeserving":
            assert all(q.t_deadline_ns is not None and q.t_deadline_ns > view.now_ns for q in view.queue)

    @pytest.mark.parametrize("trace", TRACES, ids=[t["id"] for t in TRACES])
    def test_trace_is_well_formed(self, trace, types):
        C.validate_policyspec(trace["spec"])
        assert isinstance(trace["f_initial_mhz"], int) and trace["f_initial_mhz"] in types.corun_levels_mhz
        ids = [a["req_id"] for a in trace["arrivals"]]
        assert len(set(ids)) == len(ids) and set(trace["expect"]) <= set(ids)
        times = [a["t_arrival_ns"] for a in trace["arrivals"]]
        assert times == sorted(times) and len(set(times)) == len(times)
        for a in trace["arrivals"]:
            assert set(a) == {"req_id", "type_id", "t_arrival_ns", "t_deadline_ns", "k_override"}
            assert a["type_id"] in trace["service_ns"] and types.get(a["type_id"])
        assert all(isinstance(f, int) and f in types.corun_levels_mhz for f in trace["expect_commanded_f"])
        for rid, want in trace["expect"].items():
            assert want["outcome"] in ("admitted", "expired_in_queue")
            if want["outcome"] == "expired_in_queue":
                dl = next(a["t_deadline_ns"] for a in trace["arrivals"] if a["req_id"] == rid)
                assert want["t_admit_ns"] is None and want["t_done_ns"] == dl
            else:
                arrival = next(a["t_arrival_ns"] for a in trace["arrivals"] if a["req_id"] == rid)
                assert want["t_admit_ns"] >= arrival

    def test_t5_deadline_equals_a_done_event_time(self):
        t5 = next(t for t in TRACES if t["id"].startswith("T5"))
        r1 = next(a for a in t5["arrivals"] if a["req_id"] == "r1")
        r2 = next(a for a in t5["arrivals"] if a["req_id"] == "r2")
        assert r1["t_arrival_ns"] + t5["service_ns"][r1["type_id"]] == r2["t_deadline_ns"]
        assert t5["expect"]["r2"]["outcome"] == "expired_in_queue"

    def test_example_types_ref_is_valid(self):
        C.validate_types(load_json(EXAMPLE_TYPES))


# ================================================================================================ R30 contract (C1, C2a, AM1)
class TestR30Constants:
    def test_k_set(self):
        assert C.K_SET == (1, 2, C.K_INF)

    def test_run2_plan_constants(self):
        assert C.REPEATS == {"F": 3, "default": 3, "default_pinned": 3, "corun": 3, "serial": 3, "serial_stf": 3,
                             "solo_profile_freq": 3}
        assert C.N_BLOCKS == 3 and C.ORDER_SEED_OFFSET == 23 and C.ARRIVAL_SEED_OFFSET == 11

    def test_screen_constants(self):
        assert (C.SCREEN_GPU_LOCK_MIN, C.SCREEN_EMC_LOCK_MIN, C.SCREEN_LOCK_TOL_MHZ) == (0.99, 0.99, 1.0)
        assert C.SCREEN_TJ_COL == "temp_tj_thermal_mc"

    def test_screen_columns_and_kind(self):
        assert C.SCREEN_CSV_COLS == (
            "session_id", "condition_id", "mode", "arm", "gpu_mhz", "kept", "reason", "n_clock_rows",
            "gpu_lock_frac", "emc_lock_frac", "max_tj_c", "min_mem_available_mb", "max_swap_used_mb",
            "pgmajfault_delta", "screen_ok")
        assert "screen" in C.CSV_KINDS
        C.validate_csv_header("screen", list(C.SCREEN_CSV_COLS))
        with pytest.raises(C.ContractError):
            C.validate_csv_header("screen", list(C.SCREEN_CSV_COLS[:-1]))

    def test_screen_csv_is_typed(self, tmp_path):
        p = tmp_path / "screen.csv"
        p.write_text(",".join(C.SCREEN_CSV_COLS) + "\n"
                     "s1,solo_bert-5_f612_r0,closed,pinned,612,1,,29,1.0,0.99,61.5,5200.5,0,12,1\n"
                     "s1,corun_c1,closed,governor,,0,few_requests:llama:15=3,,,,,,,,\n", encoding="utf-8")
        a, b = C.read_csv(p, "screen")
        assert a["gpu_mhz"] == 612 and isinstance(a["kept"], int) and a["gpu_lock_frac"] == 1.0
        assert a["reason"] is None and a["screen_ok"] == 1 and a["pgmajfault_delta"] == 12
        assert a["max_swap_used_mb"] == 0.0 and a["mode"] == "closed" and a["arm"] == "pinned"
        assert b["kept"] == 0 and b["reason"] == "few_requests:llama:15=3" and b["gpu_mhz"] is None
        assert b["screen_ok"] is None

    def test_docstring_tables(self):
        doc = C.__doc__
        for needle in ("D11", "order_seed", "K13", "PowerModel.from_json", "measured_key_map", "measured_runs",
                       "K_SET", "run1c.sh", "freeze_build", "types_sha256", "few_requests"):
            assert needle in doc, needle


class TestClocksOptionalBlock:
    FIXED = list(C.CLOCKS_CSV_FIXED_COLS)
    OPT = list(C.CLOCKS_CSV_OPTIONAL_COLS)
    TEMPS = ["temp_cpu_thermal_mc", "temp_gpu_thermal_mc"]

    def test_names(self):
        assert C.CLOCKS_CSV_OPTIONAL_COLS == ("swap_total_kb", "swap_free_kb", "pgmajfault")

    @pytest.mark.parametrize("header", [FIXED, FIXED + TEMPS, FIXED + OPT, FIXED + OPT + TEMPS])
    def test_accepts(self, header):
        C.validate_csv_header("clocks", header)

    @pytest.mark.parametrize("tail", [
        OPT[:2], OPT[1:], [OPT[0], OPT[2], OPT[1]], OPT[::-1],                       # partial or reordered block
        OPT + OPT, TEMPS + OPT, ["temp_cpu_thermal_mc"] + OPT + ["temp_gpu_thermal_mc"],   # block misplaced
        OPT + TEMPS[::-1], OPT + ["swap_total_kb"]])
    def test_rejects(self, tail):
        with pytest.raises(C.ContractError):
            C.validate_csv_header("clocks", self.FIXED + tail)

    def test_read_new_file_is_typed(self, tmp_path):
        p = tmp_path / "clocks.csv"
        p.write_text(",".join(self.FIXED + self.OPT + ["temp_cpu_thermal_mc"]) +
                     "\n5,6,1020000000,306000000,1020000000,273,3199,5234180,1497600,3000000,2900000,77,45000\n")
        row = C.read_csv(p, "clocks")[0]
        assert row["swap_total_kb"] == 3000000 and row["swap_free_kb"] == 2900000 and row["pgmajfault"] == 77
        assert all(isinstance(row[c], int) for c in self.OPT) and row["temp_cpu_thermal_mc"] == 45000

    def test_empty_optional_cells_are_none(self, tmp_path):
        p = tmp_path / "clocks.csv"
        p.write_text(",".join(self.FIXED + self.OPT) + "\n5,6,1,2,3,4,3199,5,6,,,\n")
        row = C.read_csv(p, "clocks")[0]
        assert row["swap_total_kb"] is None and row["pgmajfault"] is None

    def test_old_file_without_block_still_reads(self, tmp_path):
        p = tmp_path / "clocks.csv"
        p.write_text(",".join(self.FIXED) + "\n5,6,1,2,3,4,3199,5,6\n")
        assert "swap_total_kb" not in C.read_csv(p, "clocks")[0]


class TestOrderSeed:
    def _with(self, **kw):
        p = plan_obj("open")
        p.pop("sha256")
        p.update(kw)
        return C.with_plan_sha256(p)

    def test_plan_without_order_seed_is_valid(self):
        assert "order_seed" not in plan_obj("open")
        C.validate_plan(plan_obj("open"))

    @pytest.mark.parametrize("seed", [0, 23, 10 ** 9])
    def test_open_plan_accepts_int(self, seed):
        C.validate_plan(self._with(order_seed=seed))

    @pytest.mark.parametrize("seed", [-1, 1.5, "23", None, True])
    def test_rejects_bad_value(self, seed):
        with pytest.raises(C.ContractError):
            C.validate_plan(self._with(order_seed=seed))

    def test_closed_plan_cannot_carry_it(self):
        p = plan_obj("closed")
        p.pop("sha256")
        p["order_seed"] = 5
        with pytest.raises(C.ContractError, match="order_seed"):
            C.validate_plan(C.with_plan_sha256(p))

    def test_order_seed_is_hashed(self):
        a = self._with(order_seed=1)
        b = self._with(order_seed=2)
        assert a["sha256"] != b["sha256"]


class TestRun1bPlanHash:
    def _with(self, value):
        f = freeze_obj()
        f.pop("content_sha256")
        f["inputs"]["run1b_plan_sha256"] = value
        return C.with_content_sha256(f)

    def test_old_freeze_without_key_is_valid(self):
        assert "run1b_plan_sha256" not in freeze_obj()["inputs"]
        C.validate_freeze(freeze_obj())

    def test_accepts_hex(self):
        C.validate_freeze(self._with(HEX_B))

    @pytest.mark.parametrize("v", ["nope", HEX_B.upper(), None, 5, HEX_B[:-1]])
    def test_rejects_bad_hex(self, v):
        with pytest.raises(C.ContractError):
            C.validate_freeze(self._with(v))

    def test_other_unknown_input_still_rejected(self):
        f = freeze_obj()
        f.pop("content_sha256")
        f["inputs"]["run2_plan_sha256"] = HEX_A
        with pytest.raises(C.ContractError):
            C.validate_freeze(C.with_content_sha256(f))


class TestFewRequests:
    def test_constant(self):
        assert C.MIN_MEASURED_REQ_PER_TENANT == 10

    def test_enough_requests(self):
        assert C.few_requests_reason({"bert:5": 10, "llama:15": 400}) is None
        assert C.few_requests_reason({}) is None

    def test_boundary_nine_is_few(self):
        assert C.few_requests_reason({"bert:5": 50, "llama:15": 9}) == "few_requests:llama:15=9"

    def test_zero_is_few(self):
        assert C.few_requests_reason({"yolo:1": 0}) == "few_requests:yolo:1=0"

    def test_first_failing_in_sorted_order(self):
        assert C.few_requests_reason({"llama:15": 1, "bert:5": 2}) == "few_requests:bert:5=2"

    def test_never_raises(self):
        C.few_requests_reason({"bert:5": 0, "llama:15": 0})
