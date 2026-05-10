"""Cross-task aggregator. Mean per metric across tasks.

Use when each task (SST-2, MRPC, ...) has its own set of run JSONs
and you want a single averaged set of numbers.

Usage:
    avg = average_across_tasks(
        per_task={"SST-2": "logs/SST-2/", "MRPC": "logs/MRPC/", ...},
        run_pattern="entropy_*/benchmark_results.json",
    )
    # avg["entropy_0.2"] = {"avg_power_w": 145.0, "accuracy": 0.85, ...}
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union


def _load_run_merged(run_dir: Path) -> Dict:
    """Merge hw_results.json + quality_results.json (or fallback legacy)."""
    merged: Dict = {}
    hw = run_dir / "hw_results.json"
    if hw.exists():
        d = json.loads(hw.read_text(encoding="utf-8"))
        merged.update(d.get("aggregate", d))
    q = run_dir / "quality_results.json"
    if q.exists():
        qd = json.loads(q.read_text(encoding="utf-8"))
        if "metrics" in qd:
            merged.update(
                {k: v for k, v in qd["metrics"].items() if isinstance(v, (int, float))}
            )
        for k, v in qd.items():
            if isinstance(v, (int, float)):
                merged[k] = v
    legacy = run_dir / "benchmark_results.json"
    if not merged and legacy.exists():
        d = json.loads(legacy.read_text(encoding="utf-8"))
        merged.update(d.get("aggregate", d))
    return merged


def _load_runs(task_dir: Union[str, Path], run_pattern: str) -> Dict[str, Dict]:
    """Load all run JSONs under a task dir. Key = parent dir name.

    run_pattern can be:
      "*/hw_results.json"          (modern split)
      "*/benchmark_results.json"   (legacy)
      "*"                          (auto-merge from each subdir)
    """
    runs: Dict[str, Dict] = {}
    base = Path(task_dir)
    if run_pattern in ("*", "*/"):
        for sub in base.iterdir():
            if sub.is_dir():
                merged = _load_run_merged(sub)
                if merged:
                    runs[sub.name] = merged
        return runs

    for p in base.glob(run_pattern):
        try:
            run_dir = p.parent
            merged = _load_run_merged(run_dir)
            if merged:
                runs[run_dir.name] = merged
        except Exception as e:
            print(f"[averager] skip {p}: {e}")
    return runs


def _mean(vals: List[float]) -> float:
    nums = [v for v in vals if isinstance(v, (int, float))]
    return sum(nums) / len(nums) if nums else 0.0


def average_across_tasks(
    per_task: Dict[str, Union[str, Path]],
    run_pattern: str = "*",
) -> Dict[str, Dict[str, float]]:
    """Average each run config across N task directories.

    per_task:    {task_name: path/to/task_logs_dir}
    run_pattern: glob within each task dir
    Returns:     {run_key: {metric: avg_value}}
    """
    per_task_runs = {t: _load_runs(d, run_pattern) for t, d in per_task.items()}

    all_run_keys = set()
    for runs in per_task_runs.values():
        all_run_keys.update(runs.keys())

    averaged: Dict[str, Dict[str, float]] = {}
    for run_key in sorted(all_run_keys):
        samples = [
            per_task_runs[t][run_key]
            for t in per_task
            if run_key in per_task_runs.get(t, {})
        ]
        if not samples:
            continue

        merged: Dict[str, List[float]] = defaultdict(list)
        for s in samples:
            for k, v in s.items():
                if isinstance(v, (int, float)):
                    merged[k].append(float(v))

        averaged[run_key] = {k: round(_mean(v), 6) for k, v in merged.items()}

    return averaged


def write_averaged_json(averaged: Dict[str, Dict], out_path: Union[str, Path]) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(averaged, indent=2), encoding="utf-8")
    print(f"[averager] wrote {len(averaged)} runs -> {p}")
