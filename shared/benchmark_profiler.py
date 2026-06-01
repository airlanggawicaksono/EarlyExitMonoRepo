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


from .hw_profiler import (
    aggregate_hw,
    device_caps,
    device_energy_mj,
    proc_cpu_times_sec,
    sample_hw,
    Timer,
)
from .cpu_cache import CacheCounter, is_available as _papi_available


def _sample_dt(s: Dict) -> float:
    """Wall time of one sample. e2e for LLM, forward for one-shot backends."""
    for key in ("end_to_end_sec", "forward_sec"):
        v = s.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return 0.0


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
        # NVML hardware energy counter snapshots (mJ). Energy delta over the
        # whole loop is the only NVML reading dense enough to be trustworthy
        # — sub-ms per-sample reads land between counter refreshes.
        self._energy_mj_start: Optional[float] = None
        self._energy_mj_at_warmup_end: Optional[float] = None
        self._cpu_t_at_warmup_end: Optional[float] = None
        self._timed_start_perf: Optional[float] = None

    def __enter__(self):
        self.device_caps = device_caps()
        self.device_caps["papi_available"] = _papi_available()
        self._total_start = time.perf_counter()
        self._energy_mj_start = device_energy_mj()
        try:
            import torch as _t
            if _t.cuda.is_available():
                _t.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        self._cache_counter = CacheCounter().__enter__()
        # If no warmup is requested (caller did warmup externally), capture
        # energy/CPU baselines immediately so flush() has a valid delta.
        if self.warmup_steps == 0:
            self._energy_mj_at_warmup_end = device_energy_mj()
            self._cpu_t_at_warmup_end = proc_cpu_times_sec()
            self._timed_start_perf = time.perf_counter()
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
        forward_sec: Optional[float] = None,
        ttft_sec: Optional[float] = None,
        end_to_end_sec: Optional[float] = None,
        exit_layer: Optional[int] = None,
        confidence: Optional[float] = None,
        **extra: Any,
    ) -> None:
        """Per-sample log.

        Granularity (mutually exclusive — use the right one for the backend):
          forward_sec     one-shot inference (BERT/Vision/YOLO).
          ttft_sec        TTFT = time-to-first-token (LLM only, prefill wall).
          end_to_end_sec  total generation wall (LLM only).
        """
        if self._n_warmed < self.warmup_steps:
            self._n_warmed += 1
            # Reset energy baseline once warmup is done so the global delta
            # excludes warmup work (compile, cudagraph capture, allocator warm).
            if self._n_warmed == self.warmup_steps:
                self._energy_mj_at_warmup_end = device_energy_mj()
                self._cpu_t_at_warmup_end = proc_cpu_times_sec()
                self._timed_start_perf = time.perf_counter()
            return

        hw = sample_hw()
        row: Dict[str, Any] = {
            "idx": len(self.samples),
            "prediction": prediction,
            "label": label,
            "forward_sec": forward_sec,        # one-shot backends (BERT/Vision/YOLO)
            "ttft_sec": ttft_sec,              # LLM-only: prefill time
            "end_to_end_sec": end_to_end_sec,  # LLM-only: full generation wall
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

        # ---- Global counter capture (post-warmup window only) ----------------
        # NVML's hardware energy counter and psutil's CPU-time counter are
        # both too coarse for sub-ms per-sample reads. We take ONE delta over
        # the entire timed loop for each, then stratify per-sample by wall
        # time so the per-sample rows are consistent with the aggregate.
        energy_mj_end = device_energy_mj()
        cpu_t_end = proc_cpu_times_sec()
        timed_e0 = self._energy_mj_at_warmup_end
        timed_c0 = self._cpu_t_at_warmup_end
        timed_t0 = self._timed_start_perf
        if timed_t0 is not None:
            timed_elapsed = max(time.perf_counter() - timed_t0, 1e-9)
        else:
            timed_elapsed = 0.0

        if timed_e0 is not None and energy_mj_end is not None and timed_elapsed > 0:
            measured_energy_j = max(0.0, (energy_mj_end - timed_e0) / 1000.0)
            global_avg_power_w = measured_energy_j / timed_elapsed
        else:
            measured_energy_j = 0.0
            global_avg_power_w = 0.0

        # Fallback when NVML energy counter unavailable (returns None or 0):
        # use per-sample instantaneous power_w already captured by sample_hw().
        # This is less accurate (point-in-time reads vs integrated counter) but
        # correct on GPUs where nvmlDeviceGetTotalEnergyConsumption returns 0.
        _using_counter_energy = measured_energy_j > 0
        if not _using_counter_energy:
            _inst_powers = [
                s.get("power_w", 0.0) for s in self.samples
                if isinstance(s.get("power_w"), (int, float)) and s.get("power_w", 0.0) > 0
            ]
            if _inst_powers:
                global_avg_power_w = sum(_inst_powers) / len(_inst_powers)
                measured_energy_j = sum(
                    s.get("power_w", 0.0) * max(_sample_dt(s), 0.0)
                    for s in self.samples
                    if isinstance(s.get("power_w"), (int, float))
                )

        if timed_c0 is not None and cpu_t_end is not None and timed_elapsed > 0:
            measured_cpu_sec = max(0.0, cpu_t_end - timed_c0)
            global_avg_cpu_cores = measured_cpu_sec / timed_elapsed
        else:
            measured_cpu_sec = 0.0
            global_avg_cpu_cores = 0.0

        # Stratify: per-sample energy and CPU-seconds ∝ that sample's wall time.
        # Wall time = end_to_end_sec for LLM, forward_sec for one-shot backends.
        sum_dt = sum(_sample_dt(s) for s in self.samples)
        for s in self.samples:
            dt = _sample_dt(s)
            if _using_counter_energy:
                # Authoritative NVML counter: stratify proportionally by dt.
                if dt > 0 and sum_dt > 0:
                    s["energy_j"] = round(measured_energy_j * (dt / sum_dt), 6)
                else:
                    s["energy_j"] = 0.0
                s["power_w"] = round(global_avg_power_w, 3)
            else:
                # No counter: use per-sample instantaneous power_w * dt.
                pw = s.get("power_w", 0.0)
                if isinstance(pw, (int, float)):
                    s["energy_j"] = round(pw * dt, 6)
                else:
                    s["energy_j"] = 0.0
            # CPU-cores are assumed constant across the loop.
            s["cpu_cores_used"] = round(global_avg_cpu_cores, 3)

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
        forwards = [
            s["forward_sec"]
            for s in self.samples
            if isinstance(s.get("forward_sec"), (int, float))
        ]
        hw_only = [
            {
                k: v
                for k, v in s.items()
                if k
                in {
                    "power_w",  # counter-derived avg if NVML energy counter available, else instantaneous
                    "energy_j",
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
        # Total energy = direct NVML counter delta over the timed loop.
        # Sum of stratified per-sample `energy_j` would equal this by
        # construction, but using the counter value avoids float drift.
        total_energy = measured_energy_j

        end_to_end_mean = round(sum(e2es) / len(e2es), 6) if e2es else 0.0
        forward_mean = round(sum(forwards) / len(forwards), 6) if forwards else 0.0
        ttft_mean = round(sum(ttfts) / len(ttfts), 6) if ttfts else 0.0
        # per_sample_sec_mean = whatever the backend's natural granularity is.
        # LLM ships ttft + e2e; non-LLM ships forward_sec; pick the populated one.
        per_sample_mean = end_to_end_mean or forward_mean
        joules_per_sample = round(total_energy / n, 6) if n else 0.0
        agg = {
            "task": self.task,
            "strategy": self.strategy,
            "threshold": self.threshold,
            "n_samples": n,
            "total_sec": round(total_time, 4),
            "forward_sec_mean": forward_mean,        # one-shot backends (0 for LLM)
            "ttft_sec_mean": ttft_mean,              # LLM prefill (0 for non-LLM)
            "end_to_end_sec_mean": end_to_end_mean,  # LLM full gen wall (0 for non-LLM)
            "per_sample_sec_mean": per_sample_mean,
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
        if mm.get("flops_G") and per_sample_mean:
            agg["achieved_tflops_per_sec"] = round(
                mm["flops_G"] * 1e9 / per_sample_mean / 1e12, 4
            )
        if joules_per_sample and per_sample_mean:
            agg["edp_j_s"] = round(joules_per_sample * per_sample_mean, 6)
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
