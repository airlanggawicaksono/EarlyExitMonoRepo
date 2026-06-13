"""Training config. IO-free pure dataclass — no disk reads/writes here.

All knobs in one place. Resolve relative paths in cli/train, not at import.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from .bootstrap import REPO_ROOT

_DATA_DEFAULT = REPO_ROOT / "AnyTimeBert" / "glue_data"
_OUT_DEFAULT = REPO_ROOT / "logs" / "selfdistill"


@dataclass
class Cfg:
    # ---- what -------------------------------------------------------------
    task: str = "SST-2"
    mode: str = "segd"                       # pairwise | segd
    model_id: str = "OpenMOSS-Team/elasticbert-large"     # 24-layer, 335M params, ~1.3GB fp32
    n_exits: int = 24                                     # match large's hidden layer count

    # ---- distillation -----------------------------------------------------
    temperature: float = 2.0
    alpha_kd: float = 0.9                      # KD weight; (1-alpha) on CE for students
    use_true_labels: bool = True              # add CE(student, label) term
    lambda_feat: float = 0.1                  # BYOT feature-hint L2 weight (joint only)

    # ---- LoRA (pairwise/segd only) -------------------------------------
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_targets: Tuple[str, ...] = ("query", "value")

    # ---- optimisation -----------------------------------------------------
    epochs: int = 3
    batch_size: int = 32
    lr: float = 5e-4
    max_seq_length: int = 128
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_train_samples: Optional[int] = None   # cap train set (dry-run smoke)
    save_every_steps: int = 500               # mid-stage ckpt to _resume/ for crash recovery

    # ---- io / runtime -----------------------------------------------------
    data_dir: Path = field(default=_DATA_DEFAULT)
    out_root: Path = field(default=_OUT_DEFAULT)
    seed: int = 42
    device: str = "cuda"

    # ---- derived ----------------------------------------------------------
    @property
    def deepest(self) -> int:
        return self.n_exits - 1

    @property
    def run_dir(self) -> Path:
        """Per (mode, task) output root holding all stage subdirs."""
        return self.out_root / self.mode / self.task
