"""Post-training sync. Two cheap destinations:

  - HF Hub  ← model weights (.pt / adapter files). Free unlimited for public,
              free generous quota for private. One repo per (backend, dataset, mode).
  - Drive   ← logs only (metrics.json, text). Tiny, fits in free 15GB.

The split is intentional: putting checkpoints on Drive eats quota fast; logs on
HF is overkill. Each destination is opt-in (pass None to skip).
"""

import shutil
from pathlib import Path
from typing import Optional


_CKPT_PATTERNS = [
    "**/*.pt", "**/*.bin", "**/*.safetensors",
    "**/adapter_config.json", "**/adapter_model*",
]
_LOG_PATTERNS = ["metrics.json"]


def push_ckpts_to_hf(
    run_dir: Path,
    repo_id: str,
    token: str,
    *,
    private: bool = True,
    commit_message: Optional[str] = None,
):
    """Upload every checkpoint file under run_dir to one HF model repo. Lazy
    import so callers without huggingface_hub installed still work."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True, token=token, private=private, repo_type="model")
    api.upload_folder(
        folder_path=str(run_dir),
        repo_id=repo_id,
        token=token,
        allow_patterns=_CKPT_PATTERNS,
        commit_message=commit_message or f"sync {run_dir.name}",
    )


def copy_logs_to_drive(run_dir: Path, drive_root: Path, label: str):
    """Mirror metrics.json files (and only those) under <drive_root>/<label>/.
    Preserves the per-stage subdir structure so logs are reviewable later."""
    dst_root = drive_root / label
    dst_root.mkdir(parents=True, exist_ok=True)
    for pattern in _LOG_PATTERNS:
        for src in run_dir.rglob(pattern):
            rel = src.relative_to(run_dir)
            dst = dst_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
