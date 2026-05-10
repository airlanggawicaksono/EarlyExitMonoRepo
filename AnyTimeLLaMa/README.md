# AnyTimeLLaMa

LLaMa-3-8B early-exit profiling. Trains lightweight exit heads (RMSNorm + Linear) on a frozen LLaMA-3-8B backbone at layers 8/16/24.

**Base model**: `meta-llama/Meta-Llama-3-8B` (gated — accept license at HF first).

## Two training modes

| Mode | When | File | GPU |
|------|------|------|-----|
| **Colab** | Primary — 8B needs ≥A100/L4 | `scripts/train_colab.ipynb` | A100 (Colab Pro) |
| **Local** | Tiny smoke / smaller backbone | `train.py` | ≥40GB VRAM for 8B |

8B base = ~16 GB BF16 weights + grads + optimizer. Won't fit on consumer GPUs. Use Colab.

## Layout

```
AnyTimeLLaMa/
├── config.py                       # central knobs (TODO refactor)
├── train.py                        # def train(...) — local CLI/function (TODO)
├── benchmark.py                    # def profile_hw / evaluate_quality (TODO)
├── ee/
│   ├── exit_head.py                # exit classifier heads
│   ├── inference.py                # EarlyExitGenerator (KV cache, force_exit_layer)
│   ├── benchmark.py                # legacy bench (kept)
│   ├── callbacks.py                # TrainingMetricsCallback (HW per-step)
│   ├── loss.py
│   ├── evaluate.py
│   └── hub.py
├── scripts/
│   └── train_colab.ipynb           # Colab notebook (primary train)
├── finetune.py                     # HF Trainer entry
├── finetune_ee.py                  # EE training entry (used by Colab)
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

`.env` at repo root: `HF_TOKEN` (gated Llama-3) + `HF_USER`.

## 1. Train

### Colab (recommended)

Open `scripts/train_colab.ipynb` in Colab Pro:

1. Mount Drive
2. Set `HF_TOKEN` in **Colab Secrets** (key icon, sidebar)
3. Edit Train Config cell — `EXIT_LAYERS`, `MAX_TRAIN_SAMPLES`, `EPOCHS`, etc.
4. Run all cells → trains exit heads → pushes to HF

Auto-pushes to: `wicaksonolxn/llama3-8b-ee-heads`

### Local (smoke / smaller model)

```python
from AnyTimeLLaMa.train import train      # TODO: build this wrapper

train(
    base_model="meta-llama/Llama-3.2-1B",   # smaller than 8B for local fit
    exit_layers=[4, 8, 12],
    max_train_samples=1000,
    epochs=1,
)
```

8B local OOM unless ≥40 GB VRAM.

## 2. Benchmark

```python
from AnyTimeLLaMa.benchmark import profile_hw, evaluate_quality   # TODO

profile_hw(
    base_model_id="meta-llama/Meta-Llama-3-8B",
    exit_heads_id="wicaksonolxn/llama3-8b-ee-heads",
    exit_layers=[8, 16, 24],
    confidence_threshold=0.9,
    n_samples=100,
    out_dir="logs/benchmark/llama/dynamic",
)

evaluate_quality(
    base_model_id="meta-llama/Meta-Llama-3-8B",
    exit_heads_id="wicaksonolxn/llama3-8b-ee-heads",
    n_samples=100,
    out_dir="logs/benchmark/llama/dynamic",
)
```

Or via root `benchmark.ipynb`:
```python
from benchmark_config import llama as cfg
cfg.run_all()
```

## 3. Outputs

```
AnyTimeLLaMa/logs/benchmark/llama/{run_name}/
├── hw_results.json        # TTFT, per-token latency, J/token, VRAM, GPU%
└── quality_results.json   # ROUGE-2, ROUGE-L, perplexity per exit
```

## Configuration

| Knob | Default |
|------|---------|
| `BASE_MODEL` | `meta-llama/Meta-Llama-3-8B` |
| `EXIT_LAYERS` | `[8, 16, 24]` |
| `EXIT_WEIGHTS` | `[0.4, 0.6, 0.8]` |
| `CONFIDENCE_THRESHOLD` | `0.9` |
| `SEQ_LEN` | `2048` |
| `BENCHMARK_DATASET` | `cnn_dailymail` |
| `MAX_NEW_TOKENS` | `128` |

Sweep in `benchmark_config/llama.py`:
- `EXIT_LAYERS = [8, 16, 24]`
- `CONFIDENCE_THRESHOLDS = [0.5, 0.7, 0.9]`

## Troubleshooting

| Error | Fix |
|-------|-----|
| `Repo not accessible` (Llama-3-8B) | accept license at https://huggingface.co/meta-llama/Meta-Llama-3-8B |
| OOM 8B local | switch to Llama-3.2-1B / 3B variant or use Colab |
| Colab token missing | add `HF_TOKEN` in Colab Secrets, restart runtime |
| C4 download stalls | use local C4 cache cell in notebook |
