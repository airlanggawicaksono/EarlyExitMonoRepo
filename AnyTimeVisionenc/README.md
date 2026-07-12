# AnyTimeVisionenc

ViT-large early-exit benchmarks (per-layer exits, 24 blocks) on CIFAR-10/100.

## Train

Training lives in the self-distill grid — see
`GeneralizableSelfDistilationEarlyExitTraining/backends/vision` (modes
pairwise/segd), driven by the root `train_colab.ipynb`. Checkpoints push to
`{HF_USER}/selfdistill-vision-{dataset}-{mode}` (per-exit LoRA adapter +
classifier head).

## Benchmark

All knobs in `benchmark_config/vision.py` (repo root):

```python
from benchmark_config import vision
vision.run_all(skip_quality=False)
```

- `src/benchmark_trained_vit.py` — trained selfdistill ckpts (adapters + heads)
- `src/benchmark_pretrained_vit.py` — pretrained ViT baseline
- `src/prepare_data.py` — dataset prep helpers

Outputs `logs/benchmark/vision/{dataset}/{mode}/exit_<k>/{hw,quality}_results.json`.

## See also

Root [README](../README.md) for monorepo workflow.
