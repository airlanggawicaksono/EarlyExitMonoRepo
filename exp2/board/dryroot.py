"""Fake Jetson sysfs tree (K17) for tests and `--dry-run`. stdlib only.

make_dry_root(DIR) builds, under DIR, the files the board code reads and writes. Every board path P maps to DIR + P
(so "/sys/class/devfreq/17000000.gpu/cur_freq" lives at DIR/sys/class/devfreq/17000000.gpu/cur_freq).

    sys/class/devfreq/17000000.gpu/{available_frequencies,min_freq,max_freq,cur_freq,governor,load,device/load}   Hz
    sys/devices/platform/17000000.gpu/load                                                                 permille
    sys/class/hwmon/<vdd_in_hwmon>/{name=ina3221, in1..3_label, in1..3_input (mV), curr1..3_input (mA),
                                    update_interval (ms), samples}      VDD_IN label sits on in<vdd_in_channel>
    sys/class/hwmon/hwmon0 (pwm-fan), hwmon1 (name nvme, but with a VDD_IN label: a decoy)
    sys/class/thermal/thermal_zone{0,1,2}/{type,temp}                   milli-degC; zone types are not in sorted order
    sys/devices/system/cpu/cpu0/cpufreq/{scaling_cur_freq,cpuinfo_cur_freq,scaling_min_freq,scaling_max_freq}   kHz
    proc/meminfo, proc/sys/kernel/random/boot_id
    sys/kernel/debug/bpmp/debug/clk/emc/{rate,min_rate,max_rate,mrq_rate_locked,state}                           Hz
    sys/kernel/debug/bpmp/debug/bwmgr/bwmgr_halt
    etc/nv_tegra_release, etc/nvpmodel_state (power mode id; 1 = 25W, the board default)

The three INA3221 channels carry different values (VDD_IN: 5072 mV / 1203 mA, the K10 example), so a reader that
hard-codes hwmon1 or in1 gets a wrong number. Start state mirrors a stock board: GPU min = lowest level,
max = cur = highest level, EMC at max_rate, mrq_rate_locked 0, state 1, bwmgr_halt 0.
"""
from __future__ import annotations

from pathlib import Path

GPU_DEVFREQ = "sys/class/devfreq/17000000.gpu"
EMC_DIR = "sys/kernel/debug/bpmp/debug/clk/emc"
BWMGR_HALT = "sys/kernel/debug/bpmp/debug/bwmgr/bwmgr_halt"
BOOT_ID = "proc/sys/kernel/random/boot_id"
NVPMODEL_STATE = "etc/nvpmodel_state"
VDD_IN_MV, VDD_IN_MA = 5072, 1203          # 5072 * 1203 / 1000 = 6101.616 mW
_OTHER_CHANNELS = (("VDD_CPU_GPU_CV", 4961, 417), ("VDD_SOC", 4830, 88))
_MEMINFO = (
    "MemTotal:        7803652 kB\nMemFree:         3012344 kB\nMemAvailable:    5234180 kB\n"
    "Buffers:           98276 kB\nCached:          2110532 kB\n")
_THERMAL_ZONES = (("gpu-thermal", 43500), ("cpu-thermal", 45000), ("tj-thermal", 46000))


def _put(root: Path, rel: str, text) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{text}\n", encoding="utf-8", newline="\n")


def _hz(mhz: float) -> int:
    return int(round(mhz * 1e6))


def make_dry_root(dir, *, vdd_in_hwmon: str = "hwmon3", vdd_in_channel: int = 2,
                  gpu_levels_mhz=(306, 408, 510, 612, 714, 816, 918, 1020), emc_min: float = 204.0,
                  emc_max: float = 3199.0, boot_id: str = "dry-boot") -> Path:
    """Create the fake tree under `dir` (created if missing) and return it as a Path."""
    if vdd_in_channel not in (1, 2, 3):
        raise ValueError(f"vdd_in_channel must be 1, 2 or 3, got {vdd_in_channel}")
    levels = sorted(int(m) for m in gpu_levels_mhz)
    if not levels:
        raise ValueError("gpu_levels_mhz is empty")
    if not emc_min < emc_max:
        raise ValueError(f"emc_min {emc_min} must be < emc_max {emc_max}")
    root = Path(dir)
    root.mkdir(parents=True, exist_ok=True)

    # GPU devfreq
    _put(root, f"{GPU_DEVFREQ}/available_frequencies", " ".join(str(_hz(m)) for m in levels))
    _put(root, f"{GPU_DEVFREQ}/min_freq", _hz(levels[0]))
    _put(root, f"{GPU_DEVFREQ}/max_freq", _hz(levels[-1]))
    _put(root, f"{GPU_DEVFREQ}/cur_freq", _hz(levels[-1]))
    _put(root, f"{GPU_DEVFREQ}/governor", "nvhost_podgov")
    for rel in (f"{GPU_DEVFREQ}/load", f"{GPU_DEVFREQ}/device/load", "sys/devices/platform/17000000.gpu/load"):
        _put(root, rel, 273)

    # INA3221 power monitor: VDD_IN on in<vdd_in_channel>, the other channels hold different values
    hw = f"sys/class/hwmon/{vdd_in_hwmon}"
    _put(root, f"{hw}/name", "ina3221")
    others = iter(_OTHER_CHANNELS)
    for ch in (1, 2, 3):
        label, mv, ma = ("VDD_IN", VDD_IN_MV, VDD_IN_MA) if ch == vdd_in_channel else next(others)
        _put(root, f"{hw}/in{ch}_label", label)
        _put(root, f"{hw}/in{ch}_input", mv)
        _put(root, f"{hw}/curr{ch}_input", ma)
    _put(root, f"{hw}/update_interval", 1)
    _put(root, f"{hw}/samples", 4)
    if vdd_in_hwmon != "hwmon0":
        _put(root, "sys/class/hwmon/hwmon0/name", "pwm-fan")
    if vdd_in_hwmon != "hwmon1":
        _put(root, "sys/class/hwmon/hwmon1/name", "nvme")
        _put(root, "sys/class/hwmon/hwmon1/in1_label", "VDD_IN")
        _put(root, "sys/class/hwmon/hwmon1/in1_input", 7)
        _put(root, "sys/class/hwmon/hwmon1/curr1_input", 9)

    # thermal, cpufreq, memory
    for i, (ztype, temp) in enumerate(_THERMAL_ZONES):
        _put(root, f"sys/class/thermal/thermal_zone{i}/type", ztype)
        _put(root, f"sys/class/thermal/thermal_zone{i}/temp", temp)
    cpu = "sys/devices/system/cpu/cpu0/cpufreq"
    _put(root, f"{cpu}/scaling_cur_freq", 1497600)
    _put(root, f"{cpu}/cpuinfo_cur_freq", 1497600)
    _put(root, f"{cpu}/scaling_min_freq", 729600)
    _put(root, f"{cpu}/scaling_max_freq", 1497600)
    (root / "proc").mkdir(parents=True, exist_ok=True)
    (root / "proc/meminfo").write_text(_MEMINFO, encoding="utf-8", newline="\n")
    _put(root, BOOT_ID, boot_id)

    # EMC (BPMP debugfs) and bandwidth manager
    _put(root, f"{EMC_DIR}/rate", _hz(emc_max))
    _put(root, f"{EMC_DIR}/min_rate", _hz(emc_min))
    _put(root, f"{EMC_DIR}/max_rate", _hz(emc_max))
    _put(root, f"{EMC_DIR}/mrq_rate_locked", 0)
    _put(root, f"{EMC_DIR}/state", 1)
    _put(root, BWMGR_HALT, 0)

    # release and power mode
    _put(root, "etc/nv_tegra_release",
         "# R36 (release), REVISION: 4.0, GCID: dry-root, BOARD: generic, EABI: aarch64, DATE: dry-root")
    _put(root, NVPMODEL_STATE, 1)
    return root
