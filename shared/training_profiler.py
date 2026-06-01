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
        self._total_energy_j: float = 0.0           # accumulator of per-step watt × step_time
        self._epoch_starts: Dict[int, float] = {}
        self.epochs: List[Dict] = []
        self._epoch_buf: Dict[str, List[float]] = defaultdict(list)
        self._current_epoch: int = 0
        self._epoch_energy_j: float = 0.0

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
        self._epoch_energy_j = 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def end_epoch(self, epoch: int):
        epoch_time = time.perf_counter() - self._epoch_starts.get(
            epoch, time.perf_counter()
        )
        # Energy accumulated per step as watt × step_time. Mean power kept for
        # reference (the integral is what matters for total joules).
        epoch_powers = self._epoch_buf.get("power_w", [])
        avg_p = (sum(epoch_powers) / len(epoch_powers)) if epoch_powers else 0.0
        rec = {
            "epoch": epoch,
            "time_sec": round(epoch_time, 3),
            "avg_power_w": round(avg_p, 2),
            "energy_j": round(self._epoch_energy_j, 1),
            "energy_wh": round(self._epoch_energy_j / 3600, 4),
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

            # Energy = recorded watt × step elapsed (e2e fwd+bwd+optim). Sum
            # per step gives total joules. Per-step watt itself is kept in the
            # row for later analysis.
            power_w = float(hw.get("power_w", 0.0))
            step_energy_j = power_w * elapsed
            self._total_energy_j += step_energy_j
            self._epoch_energy_j += step_energy_j
            row["step_energy_j"] = round(step_energy_j, 6)
            row["total_energy_j"] = round(self._total_energy_j, 3)

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
        # Total energy = sum of per-step (watt × step_time). avg_power_w kept
        # for reference; energy_j is the load-bearing field.
        all_powers = [
            r.get("power_w") for r in self.steps
            if isinstance(r.get("power_w"), (int, float))
        ]
        avg_p = (sum(all_powers) / len(all_powers)) if all_powers else 0.0
        out = {
            "device_caps": self.device_caps,
            "summary": {
                "total_time_sec": round(total_time, 3),
                "total_time_min": round(total_time / 60, 3),
                "avg_power_w": round(avg_p, 2),
                "total_energy_j": round(self._total_energy_j, 3),
                "total_energy_wh": round(self._total_energy_j / 3600, 4),
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
