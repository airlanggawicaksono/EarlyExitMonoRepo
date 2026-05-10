"""Hardware sampling. pynvml (GPU power/util/clocks) + psutil (CPU/RAM) + torch (VRAM).

Extracted from AnyTimeLLaMa/ee/inference.py and AnyTimeLLaMa/ee/callbacks.py.
Single source of truth for HW metrics across all AnyTime models.
"""

import time
from typing import Dict, List

import psutil
import torch

_nvml_available = False
try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_available = True
except Exception:
    _nvml_available = False

_psutil_process = psutil.Process()
_psutil_process.cpu_percent()  # prime baseline


def _get_handle():
    if not _nvml_available:
        return None
    try:
        return pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        return None


def device_caps() -> Dict:
    """Static device capacity. GPU name, total VRAM, power limit, CPU, RAM."""
    caps: Dict = {}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        caps["gpu_name"] = props.name
        caps["gpu_vram_total_gb"] = round(props.total_memory / (1024 ** 3), 2)
    handle = _get_handle()
    if handle is not None:
        try:
            caps["gpu_power_limit_w"] = round(pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0, 1)
        except Exception:
            pass
    caps["cpu_count_physical"] = psutil.cpu_count(logical=False)
    caps["cpu_count_logical"]  = psutil.cpu_count(logical=True)
    caps["ram_total_gb"]       = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    return caps


def gpu_utilization() -> Dict[str, float]:
    """Live GPU snapshot. Util%, mem%, temp, power, clocks. Empty if NVML unavailable."""
    handle = _get_handle()
    if handle is None:
        return {}
    try:
        util  = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp  = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        out = {
            "gpu/utilization_pct": float(util.gpu),
            "gpu/memory_util_pct": float(util.memory),
            "gpu/temperature_c":   float(temp),
            "gpu/power_w":         round(power, 1),
        }
        try:
            out["gpu/sm_clock_mhz"]  = float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
            out["gpu/mem_clock_mhz"] = float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))
        except Exception:
            pass
        return out
    except Exception:
        return {}


def sample_hw() -> Dict[str, float]:
    """One-shot snapshot for benchmark/inference loops. Flat keys."""
    out: Dict[str, float] = {}
    handle = _get_handle()
    if handle is not None:
        try:
            out["power_w"] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            out["gpu_util_pct"]     = float(util.gpu)
            out["gpu_mem_util_pct"] = float(util.memory)
            out["gpu_sm_clock_mhz"]  = float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
            out["gpu_mem_clock_mhz"] = float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))
        except Exception:
            pass
    if torch.cuda.is_available():
        out["vram_allocated_gb"] = round(torch.cuda.memory_allocated() / (1024**3), 3)
    out["cpu_pct"]     = _psutil_process.cpu_percent()
    out["ram_used_gb"] = round(_psutil_process.memory_info().rss / (1024**3), 3)
    return out


def avg_hw(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    """Average two snapshots (start + end of a step)."""
    keys = set(a) | set(b)
    return {k: (a.get(k, 0.0) + b.get(k, 0.0)) / 2.0 for k in keys}


def aggregate_hw(samples: List[Dict[str, float]]) -> Dict[str, float]:
    """Mean across many snapshots. Returns dict of avg_* keyed metrics."""
    if not samples:
        return {}
    keys = samples[0].keys()
    return {f"avg_{k}": round(sum(s.get(k, 0.0) for s in samples) / len(samples), 4) for k in keys}


class Timer:
    """CUDA-aware wall-clock timer. Use as context manager."""

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed_s = time.perf_counter() - self.t0
