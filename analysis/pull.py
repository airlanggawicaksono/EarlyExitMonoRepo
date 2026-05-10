import os
from pathlib import Path

from huggingface_hub import login, snapshot_download, whoami

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)
    print("Logged in as:", whoami().get("name", "unknown"))

local_dir = Path(__file__).resolve().parent / "llama3.1-8b-early-exit"
path = snapshot_download(
    repo_type="dataset",
    repo_id="wicaksonolxn/llama3.1-8b-early-exit",
    allow_patterns=["logs/*", "logs/**/*", "logs/**/**/*"],
    local_dir=str(local_dir),
    max_workers=1,
)

print("downloaded in:", path)
