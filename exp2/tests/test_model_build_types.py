import json
import os

import pytest

from exp2 import contracts as C
from exp2.model.build_types import build_types

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
    _close(got, want)
    C.validate_types(got)


def test_class_is_median_split():
    got = build_types(os.path.join(FIX, "run1_csv"))
    assert {t["type_id"]: t["klass"] for t in got["types"]} == {"bert:5": "short", "vision:24": "long"}
