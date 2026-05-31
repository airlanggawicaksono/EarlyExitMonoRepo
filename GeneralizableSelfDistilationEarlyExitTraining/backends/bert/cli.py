"""CLI entry. argparse only -> Cfg -> train. No logic lives here.

Run from repo root:
    python -m GeneralizableSelfDistilationEarlyExitTraining.cli \
        --task SST-2 --mode pairwise --epochs 3
"""

import argparse
from pathlib import Path

from .config import Cfg
from ...plan import MODE_BUILDERS
from .train import train


def _parse(argv=None) -> Cfg:
    p = argparse.ArgumentParser(description="Self-distillation early-exit training (ElasticBERT).")
    p.add_argument("--task", default="SST-2")
    p.add_argument("--mode", default="joint", choices=list(MODE_BUILDERS))
    p.add_argument("--model-id", default="OpenMOSS-Team/elasticbert-base")
    p.add_argument("--n-exits", type=int, default=12)

    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--alpha-kd", type=float, default=0.9)
    p.add_argument("--no-true-labels", action="store_true")

    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-root", type=Path, default=None)
    a = p.parse_args(argv)

    kw = dict(
        task=a.task, mode=a.mode, model_id=a.model_id, n_exits=a.n_exits,
        temperature=a.temperature, alpha_kd=a.alpha_kd, use_true_labels=not a.no_true_labels,
        lora_r=a.lora_r, lora_alpha=a.lora_alpha,
        epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, device=a.device,
        max_train_samples=a.max_train_samples,
    )
    if a.out_root is not None:
        kw["out_root"] = a.out_root
    return Cfg(**kw)


def main(argv=None):
    cfg = _parse(argv)
    out = train(cfg)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
