# AnyTimeYolo

YOLOv9 **gelan-m-ee** early-exit model + per-(exit, sub-exit) benchmarks.
6 DDetect exits (E0..E5) × 3 scales (P3/P4/P5) = 18 anytime points.

## Training

Training does NOT live here. It's the self-distill grid:
`GeneralizableSelfDistilationEarlyExitTraining/backends/yolo` (pairwise / segd),
driven by the root `train_colab.ipynb`. Heads train fully on a frozen
gelan-m backbone, seeded from the upstream gelan-m detection head.
Checkpoints push to HF: `<HF_USER>/selfdistill-yolo-coco-<mode>`
(per-stage `head_<k>.pt` files).

## Layout

```
AnyTimeYolo/
├── config.py                       # paths + Roboflow keys (data prep only)
├── src/
│   ├── early_exit/
│   │   ├── model.py                # EarlyExitModel (DetectionModel + N exits)
│   │   ├── loss.py                 # EarlyExitLoss (legacy triple-TAL sampling)
│   │   └── configs/gelan-m-ee.yaml # 6-exit architecture
│   ├── benchmark.py                # pretrained-weights HW + mAP sweeps
│   ├── benchmark_trained.py        # trained selfdistill ckpts HW + mAP sweeps
│   └── prepare_data.py             # Roboflow / local dataset prep
├── model/yolov9/                   # YOLOv9 reference (Colab clone)
└── README.md
```

## Benchmark

All knobs in `benchmark_config/yolo.py` (repo root):

```python
from benchmark_config import yolo
yolo.run_all(skip_quality=False)                  # full sweep, trained weights
yolo.run_all(only_mode="segd", dry_run=True)      # smoke test
```

Outputs `logs/benchmark/yolo/<dataset>/<mode>/exit_<k>_<P3|P4|P5>/
{hw_results.json, quality_results.json}`.

## See also

Root [README](../README.md) for monorepo workflow.
