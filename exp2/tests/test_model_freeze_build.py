"""exp2.model.freeze_build on the fixture inputs, tiny sizes (N=40, W=8, R=3, one worker)."""
import json
import shutil

import pytest

from exp2 import contracts as C
from exp2.model import freeze as FZ
from exp2.model import freeze_build as FB
from exp2.tests import mocks

GIT = {"commit": "0123456789abcdef0123456789abcdef01234567", "dirty": False, "diff_sha256": None}


@pytest.fixture(autouse=True)
def _key_map_stand_in(monkeypatch):
    """freeze.measured_key_map is added by slice G5. Until it is merged use an equivalent local stand-in:
    {every choice key and grid name: first key with the same policy (ignoring the name)}."""
    if hasattr(FZ, "measured_key_map"):
        return

    def stand_in(cond):
        first, out = {}, {}
        for k, s in list(cond["choices"].items()) + [(g["name"], g) for g in cond["grid"]]:
            out[k] = first.setdefault(FB._sig(s), k)
        return out
    monkeypatch.setattr(FZ, "measured_key_map", stand_in, raising=False)


def _fx(rel):
    return mocks.fixture_path(rel)


def _argv(tmp_path, **over):
    conds = tmp_path / "conditions.json"
    if not conds.exists():
        full = json.load(open(C.__file__.replace("contracts.py", "configs/run2_conditions.example.json"),
                              encoding="utf-8"))
        conds.write_text(json.dumps(full[:1]))
    a = {"run1-csv": _fx("run1_csv"), "types": _fx("types_from_run1.json"),
         "slowdown": _fx("slowdown_model.json"), "power": _fx("power_model.json"), "conditions": str(conds),
         "run1-plan": _fx("plan_run1.json"), "n-requests": 40, "n-warmup": 8, "n-reps": 3, "seed": 0,
         "workers": 1, "out": str(tmp_path / "out")}
    a.update(over)
    out = []
    for k, v in a.items():
        out += ["--" + k, str(v)]
    return out


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fb")
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = FB.main(_argv(tmp), git=lambda: dict(GIT))
    return rc, buf.getvalue(), tmp, json.load(open(tmp / "out" / "freeze.json", encoding="utf-8"))


def test_exit_zero_and_freeze_verifies(built):
    rc, out, tmp, obj = built
    assert rc == 0
    assert C.verify_freeze_file(str(tmp / "out" / "freeze.json"))["content_sha256"] == obj["content_sha256"]
    assert (tmp / "out" / "freeze.json.sha256").exists()
    assert "runtime_estimate_s=" in out
    assert out.splitlines()[0].startswith("W_hi_r060 F k=")


def test_sim_inputs_and_git(built):
    _, _, _, obj = built
    assert obj["sim"] == {"n_requests": 40, "n_warmup": 8, "n_reps": 3, "seed": 0}
    assert obj["git"] == GIT
    assert obj["f_grid_mhz"] == [612, 1020]
    assert obj["slowdown_model"]["selected"] == "M2" and obj["slowdown_model"]["target_met"] is True
    assert obj["inputs"]["types_sha256"] == C.sha256_file(_fx("types_from_run1.json"))
    assert obj["inputs"]["run1_plan_sha256"] == C.sha256_file(_fx("plan_run1.json"))
    assert "run1b_plan_sha256" not in obj["inputs"]


def test_choices_in_contract_order_without_default_stock(built):
    cond = built[3]["conditions"][0]
    assert list(cond["choices"]) == list(FB.CHOICE_ORDER)
    assert "default_stock" not in cond["choices"]
    pinned, corun = cond["choices"]["default_pinned"], cond["choices"]["corun"]
    assert pinned["name"] == "default_pinned"
    assert {k: v for k, v in pinned.items() if k != "name"} == {k: v for k, v in corun.items() if k != "name"}
    assert corun["f_mhz"] == 1020 and corun["f_rule"] == "static"


def test_searches_use_k_set(built):
    cond = built[3]["conditions"][0]
    allowed = {1, 2, None}                      # JSON null = K_INF; k=3 is not searched
    for key in ("F", "F_sens", "F_M1", "F_M3", "conc_only"):
        s = cond["choices"][key]
        assert s["k_short"] in allowed and s["k_long"] in allowed, key


def test_predictions_cover_choices_and_grid(built):
    cond = built[3]["conditions"][0]
    for key in list(cond["choices"]) + [g["name"] for g in cond["grid"]]:
        p = cond["predictions"][key]
        assert set(p["per_type"]) == {"bert:5", "vision:24"} and p["energy_per_req_j"] > 0
    assert cond["compute_s"]["total"] >= cond["compute_s"]["F"] >= 0


def test_game_worst_and_optimum_name_a_measured_key(built):
    cond = built[3]["conditions"][0]
    gm = cond["game"]
    assert gm["player_cost"] == "shortfall_then_rt" and len(gm["equilibria"]) == len(C.K_SET) ** 2
    keys = set(cond["choices"]) | {g["name"] for g in cond["grid"]}
    assert gm["optimum"]["order"] == "fcfs" and gm["optimum"]["f_mhz"] == 1020
    assert gm["optimum"]["policy_key"] in keys
    if gm["worst"]:
        assert gm["worst"]["policy_key"] in keys and gm["worst"]["f_mhz"] == 1020


def test_run1b_plan_hash_recorded(tmp_path):
    extra = tmp_path / "plan1b.json"
    shutil.copy(_fx("plan_run1.json"), extra)
    assert FB.main(_argv(tmp_path) + ["--run1b-plan", str(extra)], git=lambda: dict(GIT)) == 0
    obj = json.load(open(tmp_path / "out" / "freeze.json", encoding="utf-8"))
    assert obj["inputs"]["run1b_plan_sha256"] == C.sha256_file(str(extra))


def test_fewer_than_three_reps_exits_2(tmp_path, capsys):
    assert FB.main(_argv(tmp_path, **{"n-reps": 2}), git=lambda: dict(GIT)) == 2
    assert "n-reps" in capsys.readouterr().err
    assert not (tmp_path / "out" / "freeze.json").exists()


def test_missing_input_and_missing_flag_exit_2(tmp_path, capsys):
    assert FB.main(_argv(tmp_path, types=str(tmp_path / "nope.json")), git=lambda: dict(GIT)) == 2
    assert FB.main(["--types", "x"]) == 2


def test_empty_conditions_exits_2(tmp_path):
    (tmp_path / "conditions.json").write_text("[]")
    assert FB.main(_argv(tmp_path), git=lambda: dict(GIT)) == 2


def test_capacity_one_slowdown_is_set_size():
    m = FB.CapacityOne()
    assert m.predict("a", ("a",), 612) == 1.0
    assert m.sample("a", ("a", "b", "c"), 612, None) == 3.0


def test_sims_memoise_specs_that_differ_only_by_name(monkeypatch):
    calls = []
    monkeypatch.setattr(FB, "simulate", lambda *a, **k: calls.append(a[1].name) or object())
    sims = FB.Sims(None, None, None, None)
    f = sims.fn("main", object())
    a = C.PolicySpec("a", 1, 1, "fcfs", None, "static", 612, {})
    b = C.PolicySpec("b", 1, 1, "fcfs", None, "static", 612, {})
    c = C.PolicySpec("c", 2, 1, "fcfs", None, "static", 612, {})
    assert f(a) is f(b)
    f(c)
    sims.fn("other", object())(a)
    assert calls == ["a", "c", "a"]


def test_probe_margins_from_train_pair_rows_at_fmax():
    types = C.load_types(_fx("types_from_run1.json"))
    rows = [
        {"type_id": "bert:5", "f_mhz": 1020, "split": "train", "A": "bert:5|vision:24", "s_p50": 2.0,
         "tau_solo_p50": 0.03},
        {"type_id": "bert:5", "f_mhz": 1020, "split": "train", "A": "bert:5|vision:24", "s_p50": 1.5,
         "tau_solo_p50": 0.03},
        {"type_id": "bert:5", "f_mhz": 1020, "split": "heldout", "A": "bert:5|vision:24", "s_p50": 9.0,
         "tau_solo_p50": 0.03},
        {"type_id": "bert:5", "f_mhz": 612, "split": "train", "A": "bert:5|vision:24", "s_p50": 9.0,
         "tau_solo_p50": 0.03},
        {"type_id": "vision:24", "f_mhz": 1020, "split": "train", "A": "a|b|c", "s_p50": 9.0, "tau_solo_p50": 1.0},
    ]
    m = FB.probe_margins(rows, types, ["bert:5", "vision:24"])
    assert m["bert:5"] == pytest.approx(0.03)
    assert m["vision:24"] == 0.0
