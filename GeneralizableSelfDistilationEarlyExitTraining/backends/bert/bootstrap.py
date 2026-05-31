"""Inject repo + AnyTimeBert reference paths so modeling_elasticbert / load_data
/ shared can import. Idempotent."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
_BERT = REPO_ROOT / "AnyTimeBert"

_PATHS = [
    REPO_ROOT,
    _BERT / "reference",
    _BERT / "reference" / "finetune-static",
]

for _p in _PATHS:
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
