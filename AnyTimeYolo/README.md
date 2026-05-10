# AnyTimeYolo

YOLOv9 early-exit profiling.

## Two training modes

| Mode | When | File | Notes |
|------|------|------|-------|
| **Colab** | Primary (Roboflow integration) | `scripts/train_colab.ipynb` | uses Colab Secrets for HF_TOKEN + RF_API_KEY |
| **Local** | TODO | `train.py` (pending) | needs YOLOv9 weights + COCO/VOC local |

## Layout (current)

```
AnyTimeYolo/
├── early_exit/                     # exit head implementation
├── model/yolov9/                   # YOLOv9 reference
├── scripts/
│   └── train_colab.ipynb           # Colab train (primary)
└── README.md
```

## Status

Scaffold pending. `train.py` + `benchmark.py` functions to be added matching the BERT pattern.

## Roboflow datasets

In Colab notebook, set `RF_API_KEY` in Colab Secrets. Notebook handles dataset download.

## See also

Root [README](../README.md) for monorepo workflow.
