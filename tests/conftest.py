"""Tests of the pre-Exp2 tools at the repo root (shared/, multitenant_*.py). Moved from the root on 2026-10-20.

The tested modules live at the repo root, so the root goes on sys.path for `import multitenant_run` and friends.
"""
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
