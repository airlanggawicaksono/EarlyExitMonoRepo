# AnyTimeLLaMa

LLaMA-3.2-1B multi-exit benchmarks — 16 per-layer exits on a frozen backbone.

## Train

Training lives in the self-distill grid — see
`GeneralizableSelfDistilationEarlyExitTraining/backends/llama` (modes
pairwise/segd on C4), driven by the root `train_colab.ipynb`. Checkpoints push
to `{HF_USER}/selfdistill-llama-c4-{mode}` (per-exit LoRA adapter + lm-head
projection).

## Benchmark

All knobs in `benchmark_config/llama.py` (repo root):

```python
from benchmark_config import llama
llama.run_all(skip_quality=False)
```

- `src/benchmark_trained.py` — trained selfdistill ckpts
- `src/benchmark.py` — pretrained baseline (base lm_head broadcast to every exit)
- `src/ee/` — inference/eval helpers (generators, hub IO, wrapper)
- `src/prepare_data.py` — C4 prep

Quality datasets: cnn_dailymail (ROUGE-L), gsm8k (exact match), arc_challenge /
hellaswag / mmlu (acc_norm). Outputs
`logs/benchmark/llama/{dataset}/{mode}/exit_<k>/{hw,quality}_results.json`.

`.env` at repo root: `HF_TOKEN` + `HF_USER`.

## See also

Root [README](../README.md) for monorepo workflow.
