"""HuggingFace Hub I/O wrappers. auto_push after train, auto_pull at benchmark.

Requires HF_TOKEN env var or `huggingface-cli login` beforehand.
"""

import os
from pathlib import Path
from typing import Optional, Union


def push_if_enabled(
    local_path: Union[str, Path],
    repo_id: str,
    *,
    enabled: bool = True,
    commit_msg: Optional[str] = None,
    private: bool = True,
    token: Optional[str] = None,
) -> Optional[str]:
    """Shared post-training push wrapper. Returns URL if pushed, None if skipped.

    Used by all 4 AnyTime train.py to standardize push behavior.
    """
    if not enabled:
        print(f"[hf_io] push disabled, skipping {repo_id}")
        return None
    msg = commit_msg or f"Auto-push checkpoint to {repo_id}"
    return auto_push(local_path, repo_id, commit_msg=msg, private=private, token=token)


def auto_push(
    local_path: Union[str, Path],
    repo_id: str,
    commit_msg: str = "Auto-push from AnyTime training",
    private: bool = True,
    token: Optional[str] = None,
) -> str:
    """Upload a local checkpoint folder to HF Hub. Creates repo if missing.

    Returns the repo URL.
    """
    from huggingface_hub import HfApi, create_repo

    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"local_path does not exist: {local_path}")

    api = HfApi()
    token = token or os.environ.get("HF_TOKEN")

    create_repo(repo_id, exist_ok=True, private=private, token=token)
    api.upload_folder(
        folder_path=str(local_path),
        repo_id=repo_id,
        commit_message=commit_msg,
        token=token,
    )
    url = f"https://huggingface.co/{repo_id}"
    print(f"[hf_io] pushed {local_path} -> {url}")
    return url


def auto_pull(
    repo_id: str,
    local_dir: Optional[Union[str, Path]] = None,
    token: Optional[str] = None,
    revision: str = "main",
) -> Path:
    """Download a repo from HF Hub. Returns local snapshot path.

    Caches under ~/.cache/huggingface/hub by default.
    """
    from huggingface_hub import snapshot_download

    token = token or os.environ.get("HF_TOKEN")
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir) if local_dir else None,
        token=token,
        revision=revision,
    )
    print(f"[hf_io] pulled {repo_id} -> {path}")
    return Path(path)
