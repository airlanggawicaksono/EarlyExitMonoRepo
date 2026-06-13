"""Generalizable self-distillation early-exit training.

Layout:
    plan.py          shared protocol (Stage, MODE_BUILDERS) — joint/pairwise/segd
    backends/bert/   ElasticBERT classification (CE + logit KD)
    backends/yolo/   YOLOv9 gelan-s-ee detection (TAL + cls-KD + box-LD)
    backends/...     (vision / llama — same protocol, own losses)

Back-compat shim: older notebooks expect `Cfg, train` at the top level; we
re-export from backends.bert so existing imports keep working. New code should
prefer the explicit backend path:

    from GeneralizableSelfDistilationEarlyExitTraining.backends.bert  import Cfg, train
    from GeneralizableSelfDistilationEarlyExitTraining.backends.yolo  import YoloCfg, train
"""

from .plan import MODE_BUILDERS
from .backends.bert import Cfg, train
from .runner import GridItem, run_grid

__all__ = ["Cfg", "train", "MODE_BUILDERS", "GridItem", "run_grid"]
