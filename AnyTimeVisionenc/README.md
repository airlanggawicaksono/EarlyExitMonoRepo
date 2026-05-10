# AnyTimeVisionenc

MSDNet (Multi-Scale Dense Network) early-exit profiling on 5 vision datasets.

**Reference**: [MSDNet-PyTorch](https://github.com/kalviny/MSDNet-PyTorch) — multi-exit by design (each block exits independently).

## Status

Scaffold pending. Will mirror AnyTimeBert structure:

```
AnyTimeVisionenc/
├── config.py
├── prepare_data.py        # CIFAR/SVHN via torchvision, Tiny-ImageNet via HF
├── train.py
├── benchmark.py
├── dataloader_ext.py      # adds SVHN + Tiny-ImageNet to MSDNet
├── reference/             # MSDNet-PyTorch (cloned)
└── README.md
```

## Datasets (5)

| Dataset | Source | Size |
|---------|--------|------|
| CIFAR-10 | torchvision built-in | 60k 32x32 |
| CIFAR-100 | torchvision built-in | 60k 32x32 |
| SVHN | torchvision built-in | 73k 32x32 |
| Tiny-ImageNet | HF datasets | 100k 64x64 |
| ImageNet | manual download (large) | 1.3M 224x224 |

## Coming soon

- `prepare_data.py` — auto-downloads CIFAR/SVHN via torchvision, Tiny-ImageNet via HF
- `train.py` — wraps MSDNet `main.py`, injects shared `TrainingProfiler`
- `benchmark.py` — anytime + dynamic eval, HW + quality split
- ImageNet pretrained checkpoint (Dropbox link in MSDNet repo)

## See also

Root [README](../README.md) for monorepo workflow.
