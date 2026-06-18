"""Skip-if-exists helper for benchmark continuation.

has_valid_result(path) -> True if file exists AND is valid JSON dict AND has no "error" key.
Errored runs are NOT skipped — they re-run on next invocation.
"""

import json
import time
from pathlib import Path
from typing import Union


def has_valid_result(path: Union[str, Path]) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    # The file EXISTS, so the stage is done. Reading it off a Drive/FUSE mount can
    # throw a transient IOError or return a partial buffer -> json.loads raises.
    # Retry before giving up; a one-off read glitch must NOT re-run a complete
    # stage. Only a file that stays unparseable (truly corrupt) returns False.
    data = None
    for attempt in range(3):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            break
        except Exception:
            if attempt < 2:
                time.sleep(0.2)
    if not isinstance(data, dict):
        return False
    return "error" not in data
