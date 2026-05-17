# AnyTime Monorepo

Early-exit profiling for 4 model families. Single repo, one `.git`, shared HW profiler, common config pattern.

```
spd/
├── shared/                       # HW profiler, CSV, averager, .env loader
├── benchmark_config/             # per-model bench sweeps (one file each)
├── benchmark.ipynb               # ROOT — universal benchmark runner
├── AnyTimeLLaMa/                 # LLaMa-3-8B early exit (Colab-primary)
├── AnyTimeBert/                  # ElasticBERT (local + Colab)
├── AnyTimeVisionenc/             # MSDNet (TODO)
├── AnyTimeYolo/                  # YOLO early-exit (Colab-primary)
├── analysis/                     # cross-model results
├── .env                          # HF_TOKEN + HF_USER (gitignored)
├── .env.example
└── README.md
```

## Setup

```bash
git clone <this repo>
cd spd
cp .env.example .env              # edit HF_TOKEN + HF_USER

# Single venv for benchmarking all 4 models + shared utils.
# Install torch FIRST from the CUDA-matched index (Blackwell needs cu128+):
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Then everything else:
pip install -r requirements.txt

# Training-only quirks needing separate venvs:
#   AnyTimeBert  -> needs transformers==4.6.1 (legacy ElasticBERT reference)
pip install -r AnyTimeBert/requirements.txt   # in its OWN venv
```

`.env` format:
```
HF_TOKEN=hf_xxx
HF_USER=your-username
```

Loaded automatically by every `train.py` / `benchmark.py` via `shared.load_env()`.

## Workflow per model

Same 3-step pattern across all 4 models:

```python
# 1) prepare data (auto-downloads from HF / Roboflow / torchvision)
from AnyTime{X}.prepare_data import prepare_all; prepare_all()

# 2) train (auto-pushes best checkpoint to HF)
from AnyTime{X}.train import train; train(task_or_dataset)

# 3) benchmark (pulls from HF, runs sweeps, writes CSVs)
# either function:
from AnyTime{X}.benchmark import benchmark
benchmark(model_id="...", ...)
# or via universal notebook:
jupyter notebook benchmark.ipynb     # change one import line, run cells
```

## Universal benchmark notebook

`benchmark.ipynb` at root. **Swap one import line** to pick which model:

```python
from benchmark_config import bert   as cfg
# from benchmark_config import llama  as cfg
# from benchmark_config import vision as cfg
# from benchmark_config import yolo   as cfg

cfg.run_all()
```

Captures **4 dimensions** per run, saved as separate CSVs:

| Dim | CSV | Source |
|-----|-----|--------|
| Quality   | `quality.csv`  | `evaluate_quality()` (no HW measure) |
| Latency   | `latency.csv`  | `profile_hw()` (no quality calc) |
| Memory    | `hardware.csv` | `profile_hw()` |
| Energy    | `energy.csv`   | `profile_hw()` |

Quality + HW kept in **separate passes** so quality-metric overhead doesn't pollute HW measurement.

## Per-model docs

- [AnyTimeLLaMa](./AnyTimeLLaMa/README.md)
- [AnyTimeBert](./AnyTimeBert/README.md)
- [AnyTimeVisionenc](./AnyTimeVisionenc/README.md)
- [AnyTimeYolo](./AnyTimeYolo/README.md)

## Security

- `.env` in `.gitignore` — never commit HF token
- Notebooks scrubbed of hardcoded tokens; use Colab Secrets
- Rotate HF token immediately if shared/leaked
