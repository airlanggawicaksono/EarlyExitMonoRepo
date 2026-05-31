"""Inject AnyTimeYolo source paths so EarlyExitModel + yolov9 import. Idempotent."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
_YOLO_SRC = REPO_ROOT / "AnyTimeYolo" / "src"

for _p in (REPO_ROOT, _YOLO_SRC):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
