# AnyTimeBert

Profile ElasticBERT early-exit on 5 GLUE tasks. Local GPU.

## Layout

```
AnyTimeBert/
├── reference/                     # ElasticBERT repo (untouched)
│   ├── finetune-static/           # fine-tune on GLUE
│   └── finetune-dynamic/          # entropy + patience early exit inference
├── scripts/
│   ├── finetune_all.py            # train 5 tasks
│   └── benchmark_all.py           # profile 5 tasks (entropy + patience)
├── hw_profiler.py                 # pynvml + psutil sampler
├── average_results.py             # aggregate CSV across 5 tasks
└── README.md
```

## Tasks (5)

SST-2, MRPC, QNLI, RTE, CoLA. Skip MNLI (too slow on 8GB).

## Pipeline

1. Pretrained weights: `fnlp/elasticbert-base` (auto-downloaded).
2. Fine-tune backbone + 12 exit heads jointly per task.
3. Profile each task at multiple exit thresholds (entropy + patience).
4. Average across 5 tasks → final CSV.

## Run

```bash
# Step 1: fine-tune all 5 (fp16, batch 16)
python scripts/finetune_all.py

# Step 2: profile all 5
python scripts/benchmark_all.py

# Step 3: aggregate
python average_results.py
```

## Hardware

- 8GB GPU primary (train + profile)
- 4GB GPU optional (parallel profile)
