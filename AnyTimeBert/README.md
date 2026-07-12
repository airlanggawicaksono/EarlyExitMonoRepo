# AnyTimeBert

ElasticBERT-BASE early-exit profiling on 5 GLUE tasks.

**Base model**: [`OpenMOSS-Team/elasticbert-base`](https://huggingface.co/OpenMOSS-Team/elasticbert-base) — pretrained from scratch with multi-exit MLM objective. Best exit-quality available for BERT.

## Layout

```
AnyTimeBert/
├── config.py             # central knobs (paths, HF, hyperparams)
├── prepare_data.py       # pull GLUE from HF -> TSV
├── train.py              # def train(task, **overrides) -> Path
├── benchmark.py          # def profile_hw / evaluate_quality / benchmark
├── reference/            # ElasticBERT repo (untouched)
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
pip install datasets huggingface_hub psutil pynvml
```

`.env` at repo root must contain `HF_TOKEN` + `HF_USER`.

## 1. Prepare data (one-time)

Pulls GLUE from HuggingFace `datasets`, dumps TSV in format reference's loader expects.

```python
from AnyTimeBert.prepare_data import prepare_all
prepare_all()                       # all 5 tasks
# or
prepare_all(only=["RTE"])           # one task
```

Output: `AnyTimeBert/glue_data/{TASK}/{train,dev,test}.tsv`

## 2. Train

Training lives in the self-distill grid — see
`GeneralizableSelfDistilationEarlyExitTraining/backends/bert` (modes
pairwise/segd), driven by the root `train_colab.ipynb`. Checkpoints push to
`{HF_USER}/selfdistill-bert-{task}-{mode}`.

## 3. Benchmark

### Option A: universal root notebook
```bash
jupyter notebook ../benchmark.ipynb
# In the model-pick cell:
#   from benchmark_config import bert as cfg
# Then cfg.run_all()
```

### Option B: from Python
```python
from AnyTimeBert.benchmark import benchmark
hw_path, q_path = benchmark(
    model_id="wicaksonolxn/elasticbert-base-rte-ee",
    task="RTE", strategy="entropy", threshold=0.2,
    data_dir="glue_data/RTE", out_dir="logs/benchmark/RTE/entropy_0.2",
)
```

### Option C: full sweep via config
```python
from benchmark_config import bert as cfg
cfg.run_all()                         # all tasks × strategies × thresholds
cfg.run_all(only_task="RTE")          # one task
cfg.run_all(skip_quality=True)        # HW only
cfg.run_all(skip_hw=True)             # quality only
```

## 4. Outputs

```
AnyTimeBert/logs/benchmark/{TASK}/{strategy}_{threshold}/
├── hw_results.json        # latency + memory + energy (NO quality)
└── quality_results.json   # accuracy + F1 (NO HW)
```

## 5. Export CSVs (cross-task averaged)

Run section 4 of root `benchmark.ipynb`, or:

```python
from shared import write_benchmark_csvs, average_across_tasks
# see benchmark.ipynb cell for example
```

Output: `results/bert/{TASK}/{latency,energy,quality,hardware}.csv`

## Configuration

Edit `config.py`:

| Knob | Default |
|------|---------|
| `HF_MODEL_NAME` | `OpenMOSS-Team/elasticbert-base` |
| `TASKS` | `["SST-2","MRPC","QNLI","RTE","CoLA"]` |
| `TRAIN_BATCH` | 16 (8GB GPU friendly) |
| `NUM_EPOCHS` | 5 |
| `USE_FP16` | True |
| `USE_TORCH_COMPILE` | True (HW pass only) |

Sweep grids in `benchmark_config/bert.py`:
- `SWEEPS = {"entropy": [0.0..0.5], "patience": [0..8]}`

## Troubleshooting

| Error | Fix |
|-------|-----|
| OOM on 8GB | drop `TRAIN_BATCH=8`, raise `GRAD_ACCUM=4` |
| `--fp16` requires apex | drop `USE_FP16=False` |
| HF push fails | check `HF_TOKEN` in `.env`, write permissions |
| `glue_data/X/train.tsv not found` | run `prepare_all()` first |
| `OpenMOSS-Team/elasticbert-base` 404 | network / typo; try `fnlp/elasticbert-base` fallback |
