"""Decoder-LM self-distill config. Default = LLaMA-3.2-1B (gated; needs HF
license accept + login). Wrapper auto-locates `decoder + final_norm + lm_head`
so swapping to Mistral / GPT-2 / Qwen works without code edits.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from .bootstrap import REPO_ROOT

_OUT = REPO_ROOT / "logs" / "selfdistill_llama"


@dataclass
class Cfg:
    dataset: str = "allenai/c4"                            # canonical LLaMA pretrain
    dataset_config: str = "en"                             # english subset (~750GB; stream it)
    streaming: bool = True                                  # iter dataset, no full download
    dataset_path: Optional[Path] = None                    # pre-tokenized HF arrow on disk; overrides streaming
    mode: str = "joint"                                    # joint | pairwise | cascade
    model_id: str = "meta-llama/Llama-3.2-1B"              # gated; needs HF login + license accept
    torch_dtype: str = "bfloat16"                          # load backbone in bf16 (~2.5GB) to fit ≤4GB inference budget
    n_exits: int = 4                                       # evenly spaced over decoder blocks
    seq_len: int = 256

    # distill
    temperature: float = 2.0
    alpha_kd: float = 0.9
    use_true_labels: bool = True

    # lora — LLaMA family targets (q_proj/v_proj). GPT-2 needs ("c_attn",).
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_targets: Tuple[str, ...] = ("q_proj", "v_proj")

    # optim
    epochs: int = 1
    batch_size: int = 4
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    max_train_samples: Optional[int] = None
    save_every_steps: int = 1000              # mid-stage ckpt to _resume/ for crash recovery
    seed: int = 42

    out_root: Path = field(default=_OUT)
    device: str = "cuda"

    @property
    def deepest(self) -> int:
        return self.n_exits - 1

    @property
    def run_dir(self) -> Path:
        return self.out_root / self.mode / self.dataset_config
