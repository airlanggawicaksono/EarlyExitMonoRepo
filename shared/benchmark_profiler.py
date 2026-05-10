"""Per-sample inference profiler. HW + latency + prediction captured per sample.

Use as context manager:

    with BenchmarkProfiler("benchmark_results.json", task="SST-2") as prof:
        for batch in loader:
            with prof.timer():
                pred = model(batch)
            prof.log_sample(prediction=pred.item(), label=batch["label"].item(),
                            ttft_sec=ttft, exit_layer=exit_idx)

On exit, dumps JSON with per-sample rows + aggregated stats + quality metrics.
"""

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .hw_profiler import aggregate_hw, device_caps, sample_hw, Timer


class BenchmarkProfiler:
    """Per-sample HW + latency capture. Writes benchmark_results.json on exit."""

    def __init__(
        self,
        out_path: str,
        task: str = "",
        strategy: str = "",
        threshold: Any = None,
        meta: Optional[Dict] = None,
        warmup_steps: int = 3,
    ):
        self.out_path = Path(out_path)
        self.task = task
        self.strategy = strategy
        self.threshold = threshold
        self.meta = meta or {}
        self.warmup_steps = warmup_steps

        self.device_caps: Dict = {}
        self.samples: List[Dict] = []
        self.exit_layer_counts: Dict[int, int] = defaultdict(int)
        self._total_start: float = 0.0
        self._n_warmed: int = 0

    def __enter__(self):
        self.device_caps = device_caps()
        self._total_start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.flush()

    # ------------------------------------------------------------------

    def timer(self) -> Timer:
        """CUDA-synced timer for one sample. Returns a Timer (use as `with` block)."""
        return Timer()

    def log_sample(
        self,
        prediction: Any,
        label: Any = None,
        ttft_sec: Optional[float] = None,
        end_to_end_sec: Optional[float] = None,
        exit_layer: Optional[int] = None,
        confidence: Optional[float] = None,
        **extra: Any,
    ) -> None:
        if self._n_warmed < self.warmup_steps:
            self._n_warmed += 1
            return

        hw = sample_hw()
        row: Dict[str, Any] = {
            "idx": len(self.samples),
            "prediction": prediction,
            "label": label,
            "ttft_sec": ttft_sec,
            "end_to_end_sec": end_to_end_sec,
            "exit_layer": exit_layer,
            "confidence": confidence,
            **hw,
            **extra,
        }
        if exit_layer is not None:
            self.exit_layer_counts[int(exit_layer)] += 1
        self.samples.append(row)

    # ------------------------------------------------------------------

    def flush(self) -> None:
        total_time = time.perf_counter() - self._total_start
        n = len(self.samples)
        if n == 0:
            print("[BenchmarkProfiler] no samples logged. Skipping write.")
            return

        ttfts = [s["ttft_sec"] for s in self.samples if isinstance(s.get("ttft_sec"), (int, float))]
        e2es  = [s["end_to_end_sec"] for s in self.samples if isinstance(s.get("end_to_end_sec"), (int, float))]
        hw_only = [{k: v for k, v in s.items() if k in {
            "power_w", "gpu_util_pct", "gpu_mem_util_pct", "gpu_sm_clock_mhz",
            "gpu_mem_clock_mhz", "vram_allocated_gb", "cpu_pct", "ram_used_gb",
        }} for s in self.samples]

        hw_avg = aggregate_hw(hw_only)
        avg_power = hw_avg.get("avg_power_w", 0.0)
        total_energy = avg_power * total_time

        agg = {
            "task": self.task,
            "strategy": self.strategy,
            "threshold": self.threshold,
            "n_samples": n,
            "total_sec": round(total_time, 4),
            "ttft_sec_mean":         round(sum(ttfts) / len(ttfts), 6) if ttfts else 0.0,
            "end_to_end_sec_mean":   round(sum(e2es)  / len(e2es),  6) if e2es  else 0.0,
            "per_sample_sec_mean":   round(sum(e2es)  / len(e2es),  6) if e2es  else 0.0,
            "throughput_samples_per_sec": round(n / total_time, 4) if total_time > 0 else 0.0,
            "total_energy_j":     round(total_energy, 4),
            "joules_per_sample":  round(total_energy / n, 6) if n else 0.0,
        }
        agg.update(hw_avg)
        agg["exit_layer_distribution"] = dict(self.exit_layer_counts)
        agg.update(self.meta)
        agg = self._add_quality(agg)

        out = {
            "device_caps": self.device_caps,
            "config": {
                "task": self.task,
                "strategy": self.strategy,
                "threshold": self.threshold,
                "warmup_steps": self.warmup_steps,
            },
            "aggregate": agg,
            "samples": self.samples,
        }
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[BenchmarkProfiler] wrote {n} samples -> {self.out_path}")

    def _add_quality(self, agg: Dict) -> Dict:
        """Compute accuracy if labels present. Override for other metrics."""
        labels_present = [s for s in self.samples if s.get("label") is not None]
        if not labels_present:
            return agg
        correct = sum(1 for s in labels_present if s["prediction"] == s["label"])
        agg["accuracy"] = round(correct / len(labels_present), 6)
        return agg
