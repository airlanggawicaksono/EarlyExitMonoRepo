"""Hardware sampling. pynvml (GPU power/util/clocks) + psutil (CPU/RAM) + torch (VRAM).

Extracted from AnyTimeLLaMa/ee/inference.py and AnyTimeLLaMa/ee/callbacks.py.
Single source of truth for HW metrics across all AnyTime models.
"""

import os
import time
from typing import Dict, List, Optional

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
_OUR_PID = os.getpid()
_LAST_PROC_UTIL_TS = 0  # microseconds; for nvmlDeviceGetProcessUtilization


def _get_handle():
    if not _nvml_available:
        return None
    try:
        return pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        return None


def _probe_power_monitoring(handle) -> bool:
    """Return True if nvmlDeviceGetPowerUsage returns a plausible non-zero value."""
    try:
        w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        return w > 0.0
    except Exception:
        return False


def device_caps() -> Dict:
    """Static device capacity. Absolute units (MB), no percentages."""
    caps: Dict = {}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        caps["gpu_name"] = props.name
        caps["gpu_vram_total_mb"] = round(props.total_memory / (1024 ** 2), 2)
        caps["gpu_multi_processor_count"] = props.multi_processor_count
        caps["gpu_cuda_capability"] = f"{props.major}.{props.minor}"
    handle = _get_handle()
    if handle is not None:
        try:
            caps["gpu_power_limit_w"] = round(
                pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0, 1
            )
            # Peak memory bandwidth (theoretical) = 2 * mem_clock * bus_width / 8
            try:
                mem_clock_max = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                bus_width = pynvml.nvmlDeviceGetMemoryBusWidth(handle)
                caps["gpu_max_mem_bandwidth_gbps"] = round(
                    2 * mem_clock_max * 1e6 * bus_width / 8 / 1e9, 2
                )
            except Exception:
                pass
        except Exception:
            pass
        # Probe power monitoring once at startup so benchmark output documents it.
        caps["gpu_power_monitoring"] = _probe_power_monitoring(handle)
    caps["cpu_max_cores_physical"] = psutil.cpu_count(logical=False)
    caps["cpu_max_cores_logical"] = psutil.cpu_count(logical=True)
    caps["ram_total_mb"] = round(psutil.virtual_memory().total / (1024 ** 2), 2)
    try:
        freq = psutil.cpu_freq()
        if freq:
            caps["cpu_max_freq_mhz"] = round(float(freq.max), 1)
    except Exception:
        pass
    return caps


def gpu_utilization() -> Dict[str, float]:
    """Live GPU snapshot. Util%, mem%, temp, power, clocks. Empty if NVML unavailable."""
    handle = _get_handle()
    if handle is None:
        return {}
    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        out = {
            "gpu/utilization_pct": float(util.gpu),
            "gpu/memory_util_pct": float(util.memory),
            "gpu/temperature_c": float(temp),
            "gpu/power_w": round(power, 1),
        }
        try:
            out["gpu/sm_clock_mhz"] = float(
                pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            )
            out["gpu/mem_clock_mhz"] = float(
                pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            )
        except Exception:
            pass
        return out
    except Exception:
        return {}


def _proc_vram_used_mb(handle) -> float:
    """VRAM used by our PID only (NVML per-process query)."""
    try:
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses_v3(handle)
    except Exception:
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        except Exception:
            return 0.0
    for p in procs:
        if p.pid == _OUR_PID:
            return round(p.usedGpuMemory / (1024 ** 2), 2) if p.usedGpuMemory else 0.0
    return 0.0


def _proc_gpu_util_pct(handle) -> Dict[str, float]:
    """Per-process SM + mem util (NVML driver >=410). Returns dict, may be empty."""
    global _LAST_PROC_UTIL_TS
    try:
        samples = pynvml.nvmlDeviceGetProcessUtilization(handle, _LAST_PROC_UTIL_TS)
    except Exception:
        return {}
    out: Dict[str, float] = {}
    latest_ts = _LAST_PROC_UTIL_TS
    for s in samples:
        if s.timeStamp > latest_ts:
            latest_ts = s.timeStamp
        if s.pid == _OUR_PID:
            out["proc_gpu_util_pct"] = float(s.smUtil)
            out["proc_gpu_mem_util_pct"] = float(s.memUtil)
    _LAST_PROC_UTIL_TS = latest_ts
    return out


def sample_hw() -> Dict:
    """One-shot snapshot. Per-PID metrics only (this process).

    Backends:
      NVML (C, per-PID GPU util + VRAM)
      torch CUDA caching allocator (C++, per-PID VRAM)
      psutil (C, per-PID CPU/RAM)

    Power: NVIDIA HW exposes only device-wide power. We attribute by SM-util share:
        power_w = device_power_w * (proc_gpu_util_pct / device_util_pct)
    Energy (per run) = power_w * elapsed_sec (Joules).

    Clocks are architectural (device-wide but not interference-prone) — kept.
    """
    out: Dict = {}
    handle = _get_handle()
    if handle is not None:
        # Clocks, util, VRAM — all robust on RTX/NVML.
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            device_util = float(util.gpu)
            out["gpu_sm_clock_mhz"] = float(
                pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            )
            out["gpu_mem_clock_mhz"] = float(
                pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            )
            # Per-PID GPU util. NVML's per-process util uses a 1-second
            # sampling window — a sub-ms inference typically falls between
            # windows and reports 0. On a dedicated benchmark GPU, fall back
            # to device-wide util (process == device since nothing else runs).
            proc_util = _proc_gpu_util_pct(handle)
            if not proc_util.get("proc_gpu_util_pct") and device_util > 0:
                proc_util = {
                    "proc_gpu_util_pct": float(device_util),
                    "proc_gpu_mem_util_pct": float(getattr(util, "memory", 0.0)),
                }
            out.update(proc_util)
            # Per-PID VRAM. NVML's `nvmlDeviceGetComputeRunningProcesses` is
            # a snapshot — only reports our PID when it has work *currently*
            # launched on the GPU. Between kernel launches it returns 0. For
            # short inferences this is most samples. We fall back to torch's
            # caching allocator (memory_reserved) which is always accurate
            # and within MBs of NVML when NVML reports.
            nvml_proc_vram = _proc_vram_used_mb(handle)
            if nvml_proc_vram <= 0 and torch.cuda.is_available():
                nvml_proc_vram = round(torch.cuda.memory_reserved() / (1024 ** 2), 2)
            out["proc_vram_used_mb"] = nvml_proc_vram
        except Exception:
            pass

        # Power: separate try so a driver limitation here doesn't drop clocks/util.
        # Some laptop GPUs (e.g. RTX 5050) return 0 from nvmlDeviceGetPowerUsage
        # even when other NVML metrics work. BenchmarkProfiler.flush() uses the
        # global NVML energy counter delta instead; per-sample power_w here is a
        # best-effort instantaneous attribution only.
        try:
            device_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            if device_power > 0:
                proc_u = out.get("proc_gpu_util_pct", 0.0)
                dev_u = out.get("proc_gpu_util_pct", 0.0)  # same as device_util after update
                try:
                    dev_u = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                except Exception:
                    pass
                share = min(proc_u / dev_u, 1.0) if dev_u > 0 else (1.0 if proc_u > 0 else 0.0)
                out["power_w"] = round(device_power * share, 3)
            else:
                out["power_w"] = 0.0
        except Exception:
            out["power_w"] = 0.0

    # Per-PID VRAM via torch C++ allocator
    if torch.cuda.is_available():
        out["vram_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 ** 2), 2)
        out["vram_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 ** 2), 2)

    # Per-PID RAM (psutil.Process)
    out["ram_used_mb"] = round(_psutil_process.memory_info().rss / (1024 ** 2), 2)
    # NOTE: cpu_cores_used is intentionally NOT sampled per-call.
    # psutil.cpu_percent() needs ~100ms between calls to return non-zero;
    # sub-ms inference loops yield all-zeros. BenchmarkProfiler captures
    # one cpu_times() delta over the whole loop and stratifies per-sample
    # by wall time (same approach as NVML energy).
    try:
        cpu_aff = _psutil_process.cpu_affinity() if hasattr(_psutil_process, "cpu_affinity") else None
        if cpu_aff is not None:
            out["proc_cpu_cores_available"] = len(cpu_aff)
    except Exception:
        pass
    try:
        out["proc_num_threads"] = _psutil_process.num_threads()
    except Exception:
        pass
    return out


def avg_hw(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    """Average two snapshots (start + end of a step)."""
    keys = set(a) | set(b)
    return {k: (a.get(k, 0.0) + b.get(k, 0.0)) / 2.0 for k in keys}


def aggregate_hw(samples: List[Dict]) -> Dict[str, float]:
    """Mean + max across many snapshots. Returns dict of avg_*/max_* keys.

    Scalar fields -> avg_* + max_*. List fields (e.g. cpu_per_core_pct) skipped
    (kept per-sample in samples list for downstream slicing).

    Scans union of keys across all samples (not just samples[0]) so a field
    missing in the first row but present later — e.g. NVML power_w briefly
    unavailable at warmup boundary — still aggregates.
    """
    if not samples:
        return {}
    all_keys: set = set()
    for s in samples:
        all_keys.update(s.keys())
    out: Dict[str, float] = {}
    for k in all_keys:
        vals = [s.get(k) for s in samples if isinstance(s.get(k), (int, float))]
        if not vals:
            continue
        out[f"avg_{k}"] = round(sum(vals) / len(vals), 4)
        out[f"max_{k}"] = round(max(vals), 4)
    return out


def _device_energy_mj() -> Optional[float]:
    """NVML hardware energy counter in millijoules (monotonic).

    `nvmlDeviceGetTotalEnergyConsumption` is a hardware counter that integrates
    power over time at the device level. Sub-millisecond resolution. Per-sample
    energy = end_mj - start_mj. Avoids the NVML 1-second util sampling window
    that makes per-process power attribution unreliable for short inferences.

    Device-wide (not per-process). On a dedicated benchmark GPU this matches
    the workload of interest.
    """
    handle = _get_handle()
    if handle is None:
        return None
    try:
        return float(pynvml.nvmlDeviceGetTotalEnergyConsumption(handle))
    except Exception:
        return None


class Timer:
    """CUDA-aware wall-clock timer. Use as context manager.

    Per-sample energy is NOT measured here -- NVML's hardware energy counter
    has an internal refresh cadence (~tens of ms) and sub-millisecond reads
    return delta=0 or wild spikes when they straddle a refresh. BenchmarkProfiler
    instead captures one global energy delta over the entire loop and stratifies
    per-sample energy by each sample's wall time.
    """

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed_s = time.perf_counter() - self.t0
        # Defaults so callers can read these unconditionally; real values are
        # injected by BenchmarkProfiler.flush() after the global energy delta
        # is known.
        self.energy_j = 0.0
        self.power_w = 0.0


# Public alias so other modules can grab the energy counter without poking
# at the private name.
device_energy_mj = _device_energy_mj


def proc_cpu_times_sec() -> Optional[float]:
    """Monotonic per-PID CPU time in seconds (user + system).

    Use deltas across a window to compute average CPU cores used:
        cores = (cpu_t1 - cpu_t0) / wall_elapsed_sec
    Works for any window size; avoids the ~100ms minimum interval that
    `psutil.cpu_percent()` needs to return non-zero.
    """
    try:
        ct = _psutil_process.cpu_times()
        return float(ct.user + ct.system)
    except Exception:
        return None
