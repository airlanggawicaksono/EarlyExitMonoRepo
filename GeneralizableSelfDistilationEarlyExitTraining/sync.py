"""Post-training sync to HF Hub ← model weights (.pt / adapter files). Free
unlimited for public, generous quota for private. One repo per (backend,
dataset, mode). Opt-in (the runner skips it when no token).

Drive sync is NOT here: the notebook rsyncs the whole run tree (out_root) to
Drive on its own background thread.
"""

from pathlib import Path
from typing import Optional


_CKPT_PATTERNS = [
    "**/*.pt", "**/*.bin", "**/*.safetensors",
    "**/adapter_config.json", "**/adapter_model*",
]


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
