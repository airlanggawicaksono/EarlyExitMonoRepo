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

# If a prior import (BERT bootstrap loads `models.modeling_elasticbert`) cached
# `models` as a namespace pkg rooted at AnyTimeBert/.../models, extend its
# __path__ so yolov9's `models.yolo` resolves too. Without this, putting
# yolov9 on sys.path AFTER `models` is cached has no effect — Python won't
# re-resolve the package.
_yolov9_models = str(_YOLOV9 / "models")
_existing = sys.modules.get("models")
if _existing is not None and hasattr(_existing, "__path__"):
    _paths = list(_existing.__path__)
    if _yolov9_models not in _paths:
        _existing.__path__ = [_yolov9_models] + _paths
