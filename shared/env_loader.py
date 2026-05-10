"""Load .env from repo root. Call load_env() at start of train.py / benchmark.py."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_env(env_file: Path = None) -> dict:
    """Read .env at repo root. Sets os.environ. Returns dict of loaded vars.

    No external dependency (no python-dotenv). Simple KEY=VALUE format.
    Lines starting with # ignored. Quotes stripped.
    """
    f = env_file or (REPO_ROOT / ".env")
    loaded = {}
    if not f.exists():
        return loaded
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and v:
            os.environ.setdefault(k, v)
            loaded[k] = v
    return loaded
