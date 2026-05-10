import argparse
import json
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt


def ensure_writable_dir(preferred: Path) -> Path:
    preferred.mkdir(parents=True, exist_ok=True)
    probe = preferred / ".write_probe.tmp"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return preferred
    except OSError:
        fallback = Path.cwd() / "plots"
        fallback.mkdir(parents=True, exist_ok=True)
        probe_fb = fallback / ".write_probe.tmp"
        probe_fb.write_text("ok", encoding="utf-8")
        probe_fb.unlink(missing_ok=True)
        print(f"Output directory not writable: {preferred}")
        print(f"Using fallback output directory: {fallback}")
        return fallback


def next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    idx = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{idx}{path.suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def resolve_log_dir(raw_log_dir: Path) -> Path:
    candidates = []
    cwd = Path.cwd()

    if raw_log_dir.is_absolute():
        candidates.append(raw_log_dir)
    else:
        candidates.extend(
            [
                cwd / raw_log_dir,
                cwd.parent / raw_log_dir,
                cwd.parent.parent / raw_log_dir,
            ]
        )
        parts = raw_log_dir.parts
        if parts and parts[0].lower() == "analysis" and len(parts) > 1:
            stripped = Path(*parts[1:])
            candidates.extend(
                [
                    cwd / stripped,
                    cwd.parent / stripped,
                    cwd.parent.parent / stripped,
                ]
            )

    seen = set()
    unique_candidates = []
    for cand in candidates:
        resolved = cand.resolve(strict=False)
        key = str(resolved).lower()
        if key not in seen:
            seen.add(key)
            unique_candidates.append(resolved)

    for cand in unique_candidates:
        if (cand / "log_history.json").exists():
            return cand

    tried = "\n".join(f"- {c}" for c in unique_candidates)
    raise FileNotFoundError(
        "Could not locate a valid log directory (missing log_history.json). Tried:\n"
        f"{tried}"
    )


def load_step_df(log_dir: Path) -> pd.DataFrame:
    log_history_path = log_dir / "log_history.json"
    rows = json.loads(log_history_path.read_text(encoding="utf-8"))
    step_rows = [r for r in rows if "step" in r]
    if not step_rows:
        raise ValueError(f"No step rows found in {log_history_path}")
    return pd.DataFrame(step_rows).sort_values("step")


def plot_loss_curves(step_df: pd.DataFrame, out_path: Path) -> None:
    series = [
        "loss_total",
        "loss_base_final",
        "loss_exit_8",
        "loss_exit_16",
        "loss_exit_24",
    ]

    fig, ax = plt.subplots(figsize=(11, 6))
    for key in series:
        if key in step_df.columns:
            ax.plot(step_df["step"], step_df[key], linewidth=2, label=key)

    ax.set_title("Training Loss Curves")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_summary(step_df: pd.DataFrame, out_path: Path) -> None:
    if "loss_total" not in step_df.columns:
        return

    first = step_df.iloc[0]
    last = step_df.iloc[-1]
    best = step_df.loc[step_df["loss_total"].idxmin()]
    drop = ((first["loss_total"] - last["loss_total"]) / first["loss_total"] * 100.0) if first["loss_total"] else 0.0

    lines = [
        "# Loss Summary",
        "",
        f"- Points: {len(step_df)}",
        f"- Step range: {int(first['step'])} -> {int(last['step'])}",
        f"- loss_total: {first['loss_total']:.6f} -> {last['loss_total']:.6f} ({drop:.2f}% drop)",
        f"- best loss_total: {best['loss_total']:.6f} at step {int(best['step'])}",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot loss curves from training logs")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("analysis/llama3.1-8b-early-exit/logs/train"),
        help="Directory containing log_history.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output dir (default: <log-dir>/plots)",
    )
    args = parser.parse_args()

    log_dir = resolve_log_dir(args.log_dir)
    preferred_out_dir = args.out_dir.resolve() if args.out_dir else (log_dir / "plots")
    out_dir = ensure_writable_dir(preferred_out_dir)

    step_df = load_step_df(log_dir)
    perstep_path = out_dir / "perstep.csv"
    try:
        step_df.to_csv(perstep_path, index=False)
    except PermissionError:
        perstep_path = next_available_path(perstep_path)
        step_df.to_csv(perstep_path, index=False)

    loss_plot_path = out_dir / "loss_graph.png"
    try:
        plot_loss_curves(step_df, loss_plot_path)
    except PermissionError:
        loss_plot_path = next_available_path(loss_plot_path)
        plot_loss_curves(step_df, loss_plot_path)

    summary_path = out_dir / "loss_summary.md"
    try:
        write_summary(step_df, summary_path)
    except PermissionError:
        summary_path = next_available_path(summary_path)
        write_summary(step_df, summary_path)

    print(f"Saved: {loss_plot_path}")
    print(f"Saved: {perstep_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
