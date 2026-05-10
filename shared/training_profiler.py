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
        self._total_energy_j: float = 0.0
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
        rec = {
            "epoch": epoch,
            "time_sec": round(epoch_time, 3),
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

            power_w = hw.get("power_w", 0.0)
            step_energy_j = power_w * elapsed
            self._total_energy_j += step_energy_j
            self._epoch_energy_j += step_energy_j
            row["step_energy_j"] = round(step_energy_j, 4)
            row["total_energy_j"] = round(self._total_energy_j, 1)

            if torch.cuda.is_available():
                row["vram_peak_gb"] = round(
                    torch.cuda.max_memory_allocated() / (1024**3), 3
                )

            for k, v in hw.items():
                if isinstance(v, (int, float)):
                    self._epoch_buf[k].append(float(v))

        row.update(extra)
        self.steps.append(row)
        self._step_start = time.perf_counter()

    # ------------------------------------------------------------------

    def flush(self) -> None:
        total_time = time.perf_counter() - self._train_start
        out = {
            "device_caps": self.device_caps,
            "summary": {
                "total_time_sec": round(total_time, 3),
                "total_time_min": round(total_time / 60, 3),
                "total_energy_j": round(self._total_energy_j, 1),
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
