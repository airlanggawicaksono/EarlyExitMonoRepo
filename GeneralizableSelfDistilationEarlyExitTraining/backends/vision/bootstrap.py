"""Inject REPO_ROOT so `shared.has_valid_result` is importable. Idempotent."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (REPO_ROOT,):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
