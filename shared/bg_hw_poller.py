"""Background HW poller. Use when you can't inject hooks into a training loop
(e.g. wrapping a subprocess like MSDNet's main.py).

Spawns a daemon thread that samples HW every N seconds, dumps timeline JSON.

Usage:
    from shared.bg_hw_poller import BgHwPoller

    with BgHwPoller("train_metrics_hw.json", interval_sec=2.0):
        subprocess.run(["python", "main.py", ...], check=True)

Output JSON shape:
    {"device_caps": {...}, "samples": [{"t_sec": 0.0, "power_w": ..., ...}, ...]}
"""

import json
import threading
import time
from pathlib import Path
from typing import Union

from .hw_profiler import device_caps, sample_hw


class BgHwPoller:
    def __init__(self, out_path: Union[str, Path], interval_sec: float = 2.0):
        self.out_path = Path(out_path)
        self.interval = float(interval_sec)
        self._stop = threading.Event()
        self._thread: threading.Thread = None
        self.samples = []
        self.device_caps = {}
        self._start_t = 0.0

    def __enter__(self):
        self.device_caps = device_caps()
        self._start_t = time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval * 2)
        self._dump()

    def _loop(self):
        while not self._stop.is_set():
            try:
                row = {"t_sec": round(time.perf_counter() - self._start_t, 3)}
                row.update(sample_hw())
                self.samples.append(row)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def _dump(self):
        total = time.perf_counter() - self._start_t
        # rough energy = trapz(power × dt) ≈ mean(power) × total
        powers = [s.get("power_w", 0.0) for s in self.samples if s.get("power_w")]
        avg_power = sum(powers) / len(powers) if powers else 0.0
        out = {
            "device_caps": self.device_caps,
            "summary": {
                "total_sec":     round(total, 3),
                "n_samples":     len(self.samples),
                "interval_sec":  self.interval,
                "avg_power_w":   round(avg_power, 2),
                "energy_j":      round(avg_power * total, 1),
                "energy_wh":     round(avg_power * total / 3600, 4),
            },
            "samples": self.samples,
        }
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[BgHwPoller] {len(self.samples)} samples ({total:.1f}s) -> {self.out_path}")
