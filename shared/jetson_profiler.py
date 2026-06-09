"""Jetson (ARM64 Orin/Xavier/Nano) HW sampling via jetson-stats (jtop).

Drop-in replacement for shared/hw_profiler.sample_hw() when running on Jetson
where NVML lacks power/per-pid util. Returns the SAME keys so downstream
consumers (BenchmarkProfiler, plotters) don't need to branch.

Install on Jetson:
    sudo pip3 install -U jetson-stats
    sudo systemctl restart jtop.service   # if first install

Detection:
    /etc/nv_tegra_release          (Jetson Linux marker file)
    /proc/device-tree/model        (e.g. "NVIDIA Jetson Orin Nano Developer Kit")
"""

import os
from pathlib import Path
from typing import Dict, Optional

_JTOP = None  # lazy persistent client
_JTOP_OK = None


def is_jetson() -> bool:
    if Path("/etc/nv_tegra_release").exists():
        return True
    try:
        model = Path("/proc/device-tree/model").read_bytes().decode(errors="ignore")
        return "jetson" in model.lower() or "tegra" in model.lower()
    except Exception:
        return False


def _ensure_jtop():
    """Lazily start a persistent jtop client. Returns the client or None."""
    global _JTOP, _JTOP_OK
    if _JTOP_OK is False:
        return None
    if _JTOP is not None:
        return _JTOP
    try:
        from jtop import jtop  # type: ignore

        client = jtop()
        client.start()
        # block briefly for first sample
        for _ in range(20):
            if client.ok(spin=True):
                break
        _JTOP = client
        _JTOP_OK = True
        return _JTOP
    except Exception as e:
        print(f"[jetson_profiler] jtop unavailable: {e}")
        _JTOP_OK = False
        return None


def jetson_caps() -> Dict:
    """Static-ish device caps. Matches shared.hw_profiler.device_caps keys
    where applicable."""
    caps: Dict = {"is_jetson": True}
    j = _ensure_jtop()
    if j is None:
        return caps
    try:
        board = getattr(j, "board", None) or {}
        hw = board.get("hardware", {}) if isinstance(board, dict) else {}
        if hw:
            caps["gpu_name"] = hw.get("Model") or hw.get("module") or "Jetson"
        gpu = j.gpu
        if gpu:
            # gpu is a dict-of-dicts keyed by GPU name on newer jtop
            first = next(iter(gpu.values())) if isinstance(gpu, dict) else gpu
            freq = first.get("freq", {}) if isinstance(first, dict) else {}
            if "max" in freq:
                caps["gpu_max_sm_clock_mhz"] = float(freq["max"]) / 1000.0
        mem = j.memory.get("RAM", {}) if hasattr(j, "memory") else {}
        if mem:
            caps["ram_total_mb"] = round(mem.get("tot", 0) / 1024, 2)
        power = j.power if hasattr(j, "power") else {}
        if isinstance(power, dict):
            tot = power.get("tot", {})
            if isinstance(tot, dict) and "power" in tot:
                # tot.power is in mW
                caps["gpu_power_limit_w"] = round(tot["power"] / 1000.0, 1)
        caps["gpu_power_monitoring"] = True
    except Exception as e:
        print(f"[jetson_profiler] caps probe failed: {e}")
    return caps


def _first_gpu(j) -> Dict:
    gpu = getattr(j, "gpu", {}) or {}
    if not gpu:
        return {}
    if isinstance(gpu, dict):
        # might be a dict of named gpus or a flat dict
        first_val = next(iter(gpu.values()))
        if isinstance(first_val, dict):
            return first_val
        return gpu
    return {}


def _gpu_rail_mw(power_dict: dict) -> Optional[float]:
    """Sum any rail name that smells GPU-related."""
    rails = power_dict.get("rail", {}) if isinstance(power_dict, dict) else {}
    if not isinstance(rails, dict):
        return None
    s = 0.0
    found = False
    for name, info in rails.items():
        up = name.upper()
        # Orin Nano: VDD_GPU_SOC, VDD_CPU_GPU_CV combine GPU + something.
        # Take VDD_GPU* preferentially; else fall back to whole-board VDD_IN.
        if "GPU" in up and isinstance(info, dict) and "power" in info:
            s += float(info["power"])
            found = True
    return s if found else None


def sample_jetson_hw() -> Dict:
    """One-shot Jetson snapshot. Same key shape as shared.hw_profiler.sample_hw()."""
    out: Dict = {}
    j = _ensure_jtop()
    if j is None:
        return out
    try:
        # spin once to refresh the latest tegrastats line
        j.ok(spin=True)
        gpu = _first_gpu(j)
        status = gpu.get("status", {}) if isinstance(gpu, dict) else {}
        freq = gpu.get("freq", {}) if isinstance(gpu, dict) else {}
        if "load" in status:
            out["proc_gpu_util_pct"] = float(status["load"])
            out["gpu_util_pct"] = float(status["load"])
        if "cur" in freq:
            # jtop returns kHz
            out["gpu_sm_clock_mhz"] = float(freq["cur"]) / 1000.0
        # EMC (external memory controller) clock = unified DRAM clock — the Jetson
        # analog of GPU memory clock (GPU + CPU share the same LPDDR5).
        emc = getattr(j, "emc", {}) or {}
        if isinstance(emc, dict) and "cur" in emc:
            out["gpu_mem_clock_mhz"] = float(emc["cur"]) / 1000.0

        power = getattr(j, "power", {}) or {}
        tot = power.get("tot", {}) if isinstance(power, dict) else {}
        board_w = round(tot["power"] / 1000.0, 3) if isinstance(tot, dict) and "power" in tot else None
        # Compute-attributable power = GPU/CPU_GPU_CV rail. Orin Nano/NX fuse GPU+CPU+CV
        # onto one VDD_CPU_GPU_CV rail (no GPU-only rail), which is what we want for
        # inference energy — excludes idle IO/SOC/board baseline. Prefer it over total board.
        gpu_mw = _gpu_rail_mw(power)
        if gpu_mw is not None:
            out["power_w"] = round(gpu_mw / 1000.0, 3)       # primary: compute rail
        elif board_w is not None:
            out["power_w"] = board_w                          # fallback: board total
        if board_w is not None:
            out["power_total_board_w"] = board_w             # kept for reference, not used for energy

        # Memory — mirror the x86 categories using torch (per-process, works on Tegra iGPU).
        try:
            import torch
            if torch.cuda.is_available():
                out["vram_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 ** 2), 2)
                out["vram_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 ** 2), 2)
                out["proc_vram_used_mb"] = out["vram_reserved_mb"]  # per-process GPU footprint (no NVML per-pid on Tegra)
        except Exception:
            pass
        # Per-PID process RAM + CPU clock (comparable to x86 path)
        try:
            import psutil
            out["ram_used_mb"] = round(psutil.Process().memory_info().rss / (1024 ** 2), 2)
            cf = psutil.cpu_freq()
            if cf is not None:
                out["cpu_clock_mhz"] = round(float(cf.current), 1)
        except Exception:
            pass
        # Board-wide unified memory used (safety "total" — Jetson shares RAM + VRAM)
        mem = getattr(j, "memory", {}) or {}
        ram = mem.get("RAM", {}) if isinstance(mem, dict) else {}
        if isinstance(ram, dict) and "used" in ram:
            out["unified_mem_used_mb"] = round(ram["used"] / 1024, 2)

        cpu = getattr(j, "cpu", {}) or {}
        if isinstance(cpu, dict):
            total = cpu.get("total", {})
            if isinstance(total, dict) and "user" in total:
                out["cpu_cores_used"] = round(float(total.get("user", 0)) / 100.0
                                              * (os.cpu_count() or 1), 3)

        temp = getattr(j, "temperature", {}) or {}
        if isinstance(temp, dict):
            t = temp.get("GPU", {}) if isinstance(temp.get("GPU", {}), dict) else {}
            if isinstance(t, dict) and "temp" in t:
                out["gpu_temperature_c"] = float(t["temp"])
    except Exception as e:
        print(f"[jetson_profiler] sample failed: {e}")
    return out


def jetson_device_energy_mj() -> Optional[float]:
    """No NVML energy counter on Jetson — return None so callers integrate
    power × elapsed instead."""
    return None


def close():
    global _JTOP
    if _JTOP is not None:
        try:
            _JTOP.close()
        except Exception:
            pass
        _JTOP = None
