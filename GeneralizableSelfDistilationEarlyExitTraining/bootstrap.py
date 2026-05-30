"""Inject repo + ElasticBERT reference paths onto sys.path.

Import this FIRST in any module that pulls repo-level deps
(modeling_elasticbert, load_data, shared). Idempotent.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent
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
