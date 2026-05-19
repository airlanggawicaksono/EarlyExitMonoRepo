"""Skip-if-exists helper for benchmark continuation.

has_valid_result(path) -> True if file exists AND is valid JSON dict AND has no "error" key.
Errored runs are NOT skipped — they re-run on next invocation.
"""

import json
from pathlib import Path
from typing import Union


def has_valid_result(path: Union[str, Path]) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    return "error" not in data
