"""Inject AnyTimeYolo source paths so EarlyExitModel + yolov9 import. Idempotent.

Layout:
    AnyTimeYolo/src/early_exit/...      <- our EarlyExitModel  (needs src/ on path)
    AnyTimeYolo/model/yolov9/utils/...  <- vendored yolov9     (needs yolov9/ on path)
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
_YOLO_SRC = REPO_ROOT / "AnyTimeYolo" / "src"
_YOLOV9   = REPO_ROOT / "AnyTimeYolo" / "model" / "yolov9"

for _p in (REPO_ROOT, _YOLO_SRC, _YOLOV9):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
