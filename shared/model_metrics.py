"""Static model metrics: params, FLOPs, MACs, model size.

Used once per model load to enrich hw_results.json aggregate. Combine with
runtime latency to derive achieved TFLOPS = FLOPs / latency.

Backends: thop (default) -> torch profiler fallback -> params-only.
"""

from typing import Dict, Optional, Tuple

import torch


def _param_count_bytes(model) -> Tuple[int, int]:
    n = 0
    nb = 0
    for p in model.parameters():
        n += p.numel()
        nb += p.numel() * p.element_size()
    return n, nb


def count_flops_macs(model, dummy_input) -> Optional[Tuple[float, float]]:
    """Returns (flops, macs) as floats, or None if backend fails.

    dummy_input: tensor matching model's expected forward input shape (1 sample).
    """
    try:
        from thop import profile
        was_training = model.training
        model.eval()
        with torch.no_grad():
            macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        if was_training:
            model.train()
        # thop returns MACs; FLOPs ~= 2 * MACs (multiply + add)
        return float(2 * macs), float(macs)
    except Exception as e:
        print(f"[model_metrics] thop failed: {e}")
        return None


def model_metrics(model, dummy_input=None) -> Dict:
    """Returns static metrics dict ready to merge into aggregate JSON.

    Keys: params_M, params_count, model_size_mb, dtype.
    If dummy_input provided AND thop works: flops_G, macs_G.
    """
    n, nb = _param_count_bytes(model)
    out: Dict = {
        "params_count": n,
        "params_M": round(n / 1e6, 3),
        "model_size_mb": round(nb / (1024 ** 2), 3),
    }
    try:
        out["dtype"] = str(next(model.parameters()).dtype)
    except StopIteration:
        out["dtype"] = "unknown"

    if dummy_input is not None:
        fm = count_flops_macs(model, dummy_input)
        if fm is not None:
            flops, macs = fm
            out["flops_G"] = round(flops / 1e9, 4)
            out["macs_G"] = round(macs / 1e9, 4)
    return out


def derive_runtime_metrics(model_metrics_dict: Dict, end_to_end_sec_mean: float,
                           joules_per_sample: float) -> Dict:
    """Combine static + runtime -> achieved_tflops, EDP.

    Call once per run after benchmark complete.
    """
    out: Dict = {}
    if end_to_end_sec_mean and "flops_G" in model_metrics_dict:
        flops = model_metrics_dict["flops_G"] * 1e9
        achieved = flops / end_to_end_sec_mean / 1e12
        out["achieved_tflops_per_sec"] = round(achieved, 4)
    if end_to_end_sec_mean and joules_per_sample:
        out["edp_j_s"] = round(joules_per_sample * end_to_end_sec_mean, 6)
    return out
