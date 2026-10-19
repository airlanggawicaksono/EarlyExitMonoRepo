import json
import os
from collections import Counter

from exp2 import contracts as C
from exp2.model.plan_run1 import make_cells, plan_run1, split_cells

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
EXITS = {"bert": [5, 23], "vision": [6, 24], "llama": [3, 15], "yolo": [1, 5]}


def _fams(cell):
    return [C.parse_type_id(t)[0] for t in cell]


def test_cell_counts_and_distinct_families():
    cells = make_cells(EXITS)
    assert Counter(len(c) for c in cells) == {2: 24, 3: 32, 4: 16}
    assert all(len(set(_fams(c))) == len(c) for c in cells)


def test_heldout_stratified_and_seeded():
    cells = make_cells(EXITS)
    s = split_cells(cells, 0.3, 0)
    held = Counter(len(c) for c, v in s.items() if v == "heldout")
    assert held == {2: 7, 3: 10, 4: 5}
    assert s == split_cells(cells, 0.3, 0)
    assert s != split_cells(cells, 0.3, 1)


def test_full_plan_structure():
    plan = plan_run1(EXITS, [408, 612, 816, 1020], [612, 1020], 714)
    C.validate_plan(plan)
    conds = plan["conditions"]
    assert sorted(c["gpu_mhz"] for c in conds if c["mode"] == "idle") == [408, 612, 714, 816, 1020]
    solo = [c for c in conds if c.get("split", "").startswith("solo")]
    assert {c["gpu_mhz"] for c in solo if c["split"] == "solo_heldout_level"} == {714}
    assert {c["gpu_mhz"] for c in solo if c["split"] == "solo"} == {408, 612, 816, 1020}
    corun = [c for c in conds if len(c.get("cell", ())) > 1]
    assert {c["gpu_mhz"] for c in corun} == {612, 1020}
    assert {c["repeat"] for c in corun} == {0, 1, 2}
    assert len(corun) == 72 * 2 * 3


def test_matches_fixture_plan():
    durations = {"idle": 2.0, "bert:5": 0.8, "vision:24": 1.0, "corun": 1.5}

    def dur(kind, cell):
        return durations[cell[0] if kind == "solo" else kind]

    got = plan_run1({"bert": [5], "vision": [24]}, [1020, 612], [1020, 612], None, repeats=1, settle_s=1.0,
                    idle_window_s=2.5, n_warmup=2, duration_fn=dur)
    with open(os.path.join(FIX, "plan_run1.json"), encoding="utf-8") as fh:
        want = json.load(fh)
    assert got == want
    assert got["sha256"] == C.plan_sha256(got)
    assert got["estimated_wall_s"] == 23.6
