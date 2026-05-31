"""Decoder-LM self-distill config. Dry-run defaults target GPT-2 (small, open,
no gate). Swap `model_id` to LLaMA / Mistral for real training; the wrapper
auto-locates `decoder + final_norm + lm_head` for either family.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from .bootstrap import REPO_ROOT

_OUT = REPO_ROOT / "logs" / "selfdistill_llama"


@dataclass
class Cfg:
    dataset: str = "wikitext"                              # HF datasets name
    dataset_config: str = "wikitext-2-raw-v1"              # subconfig (or "")
    mode: str = "joint"                                    # joint | pairwise | cascade
    model_id: str = "gpt2"                                 # gpt2 dry-run; swap for meta-llama/Llama-3.2-1B
    n_exits: int = 4                                       # evenly spaced over decoder blocks
    seq_len: int = 256

    # distill
    temperature: float = 2.0
    alpha_kd: float = 0.9
    use_true_labels: bool = True

    # lora — defaults for GPT-2 (c_attn combined QKV); LLaMA uses ("q_proj","v_proj")
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_targets: Tuple[str, ...] = ("c_attn",)

    # optim
    epochs: int = 1
    batch_size: int = 4
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    max_train_samples: Optional[int] = None
    seed: int = 42

    out_root: Path = field(default=_OUT)
    device: str = "cuda"

    @property
    def deepest(self) -> int:
        return self.n_exits - 1

    @property
    def run_dir(self) -> Path:
        return self.out_root / self.mode / self.dataset_config
