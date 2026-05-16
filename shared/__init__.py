"""Shared infrastructure for AnyTime monorepo.

Provides:
- hw_profiler: GPU/CPU/RAM/power sampling (pynvml + psutil)
- training_profiler: per-step HW + loss capture, train_metrics.json writer
- benchmark_profiler: per-sample HW + latency capture, benchmark_results.json writer
- csv_export: latency/energy/quality/hardware CSV with deltas
- averager: cross-task aggregation
- hf_io: auto_push / auto_pull to HuggingFace Hub
"""

from .hw_profiler import (
    sample_hw,
    avg_hw,
    aggregate_hw,
    device_caps,
    gpu_utilization,
    Timer,
)
from .training_profiler import TrainingProfiler
from .benchmark_profiler import BenchmarkProfiler
from .bg_hw_poller import BgHwPoller
from .hf_io import auto_push, auto_pull, push_if_enabled
from .csv_export import write_benchmark_csvs
from .averager import average_across_tasks
from .env_loader import load_env
from .model_metrics import model_metrics, derive_runtime_metrics, count_flops_macs
from .cpu_cache import CacheCounter, is_available as papi_available

__all__ = [
    "sample_hw",
    "avg_hw",
    "aggregate_hw",
    "device_caps",
    "gpu_utilization",
    "Timer",
    "TrainingProfiler",
    "BenchmarkProfiler",
    "BgHwPoller",
    "auto_push",
    "auto_pull",
    "push_if_enabled",
    "write_benchmark_csvs",
    "average_across_tasks",
    "load_env",
    "model_metrics",
    "derive_runtime_metrics",
    "count_flops_macs",
    "CacheCounter",
    "papi_available",
]
