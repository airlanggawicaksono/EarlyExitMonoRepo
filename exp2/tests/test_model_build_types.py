import json
import os

import pytest

from exp2 import contracts as C
from exp2.model.build_types import build_types, split_classes

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


def _close(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            _close(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _close(x, y)
    elif isinstance(a, float):
        assert a == pytest.approx(b, rel=1e-9)
    else:
        assert a == b


def test_matches_fixture_and_validates():
    got = build_types(os.path.join(FIX, "run1_csv"), alpha=5.0)
    with open(os.path.join(FIX, "types_from_run1.json"), encoding="utf-8") as fh:
        want = json.load(fh)
    # the fixture still carries the pre-D32-8 rule text; only that string is allowed to differ
    assert got["provenance"]["klass_rule"].startswith("largest-gap")
    want["provenance"]["klass_rule"] = got["provenance"]["klass_rule"]
    _close(got, want)
    C.validate_types(got)


def test_class_is_largest_gap_split():
    got = build_types(os.path.join(FIX, "run1_csv"))
    assert {t["type_id"]: t["klass"] for t in got["types"]} == {"bert:5": "short", "vision:24": "long"}


def test_run1_values_split_encoders_short_llama_long():
    p50 = {"bert:5": 0.022, "yolo:1": 0.039, "yolo:5": 0.063, "bert:23": 0.077, "llama:3": 0.640, "llama:15": 1.729}
    klass, gap = split_classes(p50)
    assert {k for k, v in klass.items() if v == "short"} == {"bert:5", "yolo:1", "yolo:5", "bert:23"}
    assert {k for k, v in klass.items() if v == "long"} == {"llama:3", "llama:15"}
    assert gap == pytest.approx(0.640 / 0.077)


def test_tie_takes_first_largest_gap_and_single_type():
    klass, gap = split_classes({"a": 1.0, "b": 2.0, "c": 4.0})
    assert klass == {"a": "short", "b": "long", "c": "long"} and gap == 2.0
    assert split_classes({"a": 1.0}) == ({"a": "short"}, None)
