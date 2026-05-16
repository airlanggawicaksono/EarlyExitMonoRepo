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


from .hw_profiler import aggregate_hw, device_caps, sample_hw, Timer
from .cpu_cache import CacheCounter, is_available as _papi_available


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
        self.device_caps["papi_available"] = _papi_available()
        self._total_start = time.perf_counter()
        try:
            import torch as _t
            if _t.cuda.is_available():
                _t.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        self._cache_counter = CacheCounter().__enter__()
        return self

    def __exit__(self, *args):
        try:
            import torch as _t
            if _t.cuda.is_available():
                self.meta["peak_vram_allocated_mb"] = round(
                    _t.cuda.max_memory_allocated() / (1024 ** 2), 2
                )
                self.meta["peak_vram_reserved_mb"] = round(
                    _t.cuda.max_memory_reserved() / (1024 ** 2), 2
                )
        except Exception:
            pass
        try:
            cache_stats = self._cache_counter.read()
            self._cache_counter.__exit__(None, None, None)
            if cache_stats:
                self.meta.update({f"cpu_{k}": v for k, v in cache_stats.items()})
                # LLC miss rate (derived)
                refs = cache_stats.get("llc_references", 0)
                miss = cache_stats.get("llc_misses", 0)
                if refs > 0:
                    self.meta["cpu_llc_miss_ratio"] = round(miss / refs, 4)
        except Exception:
            pass
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

        ttfts = [
            s["ttft_sec"]
            for s in self.samples
            if isinstance(s.get("ttft_sec"), (int, float))
        ]
        e2es = [
            s["end_to_end_sec"]
            for s in self.samples
            if isinstance(s.get("end_to_end_sec"), (int, float))
        ]
        hw_only = [
            {
                k: v
                for k, v in s.items()
                if k
                in {
                    "power_w",  # per-PID (util-attributed)
                    "gpu_sm_clock_mhz", "gpu_mem_clock_mhz",
                    "proc_vram_used_mb",
                    "proc_gpu_util_pct", "proc_gpu_mem_util_pct",
                    "vram_allocated_mb", "vram_reserved_mb",
                    "cpu_cores_used", "ram_used_mb",
                    "proc_cpu_cores_available", "proc_num_threads",
                }
            }
            for s in self.samples
        ]

        hw_avg = aggregate_hw(hw_only)
        # Per-sample energy: sum(power_i * e2e_i) — more accurate than avg*total_time
        total_energy = 0.0
        for s in self.samples:
            p = s.get("power_w", 0.0)
            dt = s.get("end_to_end_sec", 0.0)
            if isinstance(p, (int, float)) and isinstance(dt, (int, float)):
                total_energy += p * dt

        end_to_end_mean = round(sum(e2es) / len(e2es), 6) if e2es else 0.0
        joules_per_sample = round(total_energy / n, 6) if n else 0.0
        agg = {
            "task": self.task,
            "strategy": self.strategy,
            "threshold": self.threshold,
            "n_samples": n,
            "total_sec": round(total_time, 4),
            "ttft_sec_mean": round(sum(ttfts) / len(ttfts), 6) if ttfts else 0.0,
            "end_to_end_sec_mean": end_to_end_mean,
            "per_sample_sec_mean": end_to_end_mean,
            "throughput_samples_per_sec": round(n / total_time, 4)
            if total_time > 0
            else 0.0,
            "total_energy_j": round(total_energy, 4),
            "joules_per_sample": joules_per_sample,
        }
        agg.update(hw_avg)
        agg["exit_layer_distribution"] = dict(self.exit_layer_counts)
        agg.update(self.meta)
        # Derive research metrics if model_metrics present in meta
        mm = {k: v for k, v in self.meta.items() if k in ("flops_G", "macs_G", "params_M")}
        if mm.get("flops_G") and end_to_end_mean:
            agg["achieved_tflops_per_sec"] = round(
                mm["flops_G"] * 1e9 / end_to_end_mean / 1e12, 4
            )
        if joules_per_sample and end_to_end_mean:
            agg["edp_j_s"] = round(joules_per_sample * end_to_end_mean, 6)
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
