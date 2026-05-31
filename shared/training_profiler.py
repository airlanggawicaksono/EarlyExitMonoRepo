"""Per-step training profiler. HW + loss + lr captured per step.

Use as context manager:

    with TrainingProfiler("train_metrics.json") as prof:
        for step, batch in enumerate(loader):
            loss = model(batch)
            loss.backward()
            optimizer.step()
            prof.log_step(step, loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

On exit, dumps JSON with per-step rows + final aggregate.

For HuggingFace Trainer, use `TrainingMetricsCallback` from this module instead.
"""

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .hw_profiler import (
    device_caps,
    sample_hw,
    proc_cpu_times_sec,
)


class TrainingProfiler:
    """Per-step HW + loss capture. Writes train_metrics.json on exit."""

    def __init__(
        self,
        out_path: str,
        sample_every_n_steps: int = 1,
        seq_length: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        self.out_path = Path(out_path)
        self.sample_every_n_steps = sample_every_n_steps
        self.seq_length = seq_length
        self.batch_size = batch_size

        self.device_caps: Dict = {}
        self.steps: List[Dict] = []
        self._train_start: float = 0.0
        self._step_start: float = 0.0
        self._step_cpu_t0: Optional[float] = None   # per-step CPU-time delta -> cores used
        self._power_samples: List[float] = []       # all power_w readings; energy = mean * total_time
        self._epoch_starts: Dict[int, float] = {}
        self.epochs: List[Dict] = []
        self._epoch_buf: Dict[str, List[float]] = defaultdict(list)
        self._current_epoch: int = 0

    def __enter__(self):
        self.device_caps = device_caps()
        self._train_start = time.perf_counter()
        self._step_start = self._train_start
        self._epoch_starts[0] = self._train_start
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *args):
        self.flush()

    # ------------------------------------------------------------------

    def begin_epoch(self, epoch: int):
        self._current_epoch = epoch
        self._epoch_starts[epoch] = time.perf_counter()
        self._epoch_buf.clear()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def end_epoch(self, epoch: int):
        epoch_time = time.perf_counter() - self._epoch_starts.get(
            epoch, time.perf_counter()
        )
        # Energy = (avg power across all step samples this epoch) × epoch_wall_time.
        # Single multiply at epoch boundary instead of accumulating noisy per-step
        # power_w × elapsed (NVML's 1s sampling window misreads short steps).
        epoch_powers = self._epoch_buf.get("power_w", [])
        avg_p = (sum(epoch_powers) / len(epoch_powers)) if epoch_powers else 0.0
        epoch_energy_j = avg_p * epoch_time
        rec = {
            "epoch": epoch,
            "time_sec": round(epoch_time, 3),
            "avg_power_w": round(avg_p, 2),
            "energy_j": round(epoch_energy_j, 1),
            "energy_wh": round(epoch_energy_j / 3600, 4),
        }
        for k, vals in self._epoch_buf.items():
            if vals:
                rec[f"avg_{k}"] = round(sum(vals) / len(vals), 4)
                rec[f"max_{k}"] = round(max(vals), 4)
        self.epochs.append(rec)

    def step_begin(self):
        self._step_start = time.perf_counter()
        self._step_cpu_t0 = proc_cpu_times_sec()  # CPU (user+sys) seconds at step start

    def log_step(self, step: int, loss: float, lr: float = 0.0, **extra: Any) -> None:
        """Record one step. extra fields go into the row as-is."""
        elapsed = time.perf_counter() - self._step_start

        row: Dict[str, Any] = {
            "step": int(step),
            "epoch": self._current_epoch,
            "loss": float(loss),
            "lr": float(lr),
            "step_time_sec": round(elapsed, 4),
        }
        if self.seq_length and self.batch_size and elapsed > 0:
            row["tokens_per_sec"] = round(
                self.batch_size * self.seq_length / elapsed, 1
            )

        if (step % self.sample_every_n_steps) == 0:
            hw = sample_hw()
            row.update(hw)

            # Power gets buffered (not multiplied per-step). Energy is computed
            # once at end_epoch / flush as mean_power × wall_time. Per-step
            # power_w × elapsed is noisy on short steps because NVML samples
            # power on a ~1s window.
            power_w = float(hw.get("power_w", 0.0))
            self._power_samples.append(power_w)

            # CPU cores used during this step = delta(proc CPU time) / wall.
            # psutil.cpu_percent() needs 100ms warmup -> useless for short steps.
            # Per-PID cpu_times() is monotonic; delta works at any window size.
            cpu_t1 = proc_cpu_times_sec()
            if self._step_cpu_t0 is not None and cpu_t1 is not None and elapsed > 0:
                row["cpu_cores_used"] = round((cpu_t1 - self._step_cpu_t0) / elapsed, 3)

            if torch.cuda.is_available():
                row["vram_peak_gb"] = round(
                    torch.cuda.max_memory_allocated() / (1024**3), 3
                )

            for k, v in hw.items():
                if isinstance(v, (int, float)):
                    self._epoch_buf[k].append(float(v))
            if "cpu_cores_used" in row:
                self._epoch_buf["cpu_cores_used"].append(row["cpu_cores_used"])

        row.update(extra)
        self.steps.append(row)
        self._step_start = time.perf_counter()

    # ------------------------------------------------------------------

    def flush(self) -> None:
        total_time = time.perf_counter() - self._train_start
        # Energy = avg(all power samples across the run) × total wall time.
        # Single multiply at the end; no per-step accumulation drift.
        avg_p = (sum(self._power_samples) / len(self._power_samples)) if self._power_samples else 0.0
        total_energy_j = avg_p * total_time
        out = {
            "device_caps": self.device_caps,
            "summary": {
                "total_time_sec": round(total_time, 3),
                "total_time_min": round(total_time / 60, 3),
                "avg_power_w": round(avg_p, 2),
                "total_energy_j": round(total_energy_j, 1),
                "total_energy_wh": round(total_energy_j / 3600, 4),
                "n_steps": len(self.steps),
                "n_epochs": len(self.epochs),
            },
            "epochs": self.epochs,
            "steps": self.steps,
        }
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(
            f"[TrainingProfiler] wrote {len(self.steps)} step rows + {len(self.epochs)} epoch rows -> {self.out_path}"
        )
