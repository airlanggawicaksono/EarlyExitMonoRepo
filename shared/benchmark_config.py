"""Global benchmark sweep defaults. Per-model configs override these."""

# Sampling
WARMUP_STEPS = 3
HW_SAMPLE_EVERY_N = 1  # per-step granular (set higher to reduce I/O)

# Inference
DEFAULT_BENCH_BATCH = 1  # 1 = realistic edge inference; >1 = throughput

# Sweep grids — used by per-model benchmark_runner if not overridden
DEFAULT_ENTROPY_THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
DEFAULT_PATIENCE_VALUES = [0, 1, 2, 3, 4, 6, 8]
DEFAULT_FORCED_EXIT_LAYERS = [3, 6, 9, 12]  # generic; BERT-base 12 layers

# HF
HF_PRIVATE_DEFAULT = True  # newly created repos default to private
