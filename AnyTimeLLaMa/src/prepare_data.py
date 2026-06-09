"""Pre-cache C4 subset to disk. Avoids repeated HF dataset pulls during training.

Usage:
    python prepare_data.py
"""

import json
import sys
from pathlib import Path

_HERE  = Path(__file__).resolve().parent     # AnyTimeLLaMa/src/
_MODEL = _HERE.parent
_REPO  = _MODEL.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_MODEL))

import config as C  # type: ignore


def _stream_to_jsonl(split_name: str, n_samples: int, out_path: Path) -> None:
    from datasets import load_dataset

    print(f"[prepare_data] caching C4 {split_name} -> {out_path} ({n_samples} rows)")
    ds = load_dataset("allenai/c4", "en", split=split_name, streaming=True)
    written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            text = row.get("text", "")
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            written += 1
            if written >= n_samples:
                break
    print(f"  wrote {written:,} rows")


def prepare_c4(force: bool = False) -> None:
    train_file = C.C4_CACHE / "c4_train.jsonl"
    val_file = C.C4_CACHE / "c4_validation.jsonl"

    if force or not train_file.exists():
        _stream_to_jsonl("train", C.MAX_TRAIN_SAMPLES, train_file)
    else:
        print(f"[prepare_data] C4 train cache exists: {train_file}")

    if force or not val_file.exists():
        _stream_to_jsonl("validation", C.MAX_VAL_SAMPLES, val_file)
    else:
        print(f"[prepare_data] C4 val cache exists: {val_file}")


def prepare_cnn_dailymail() -> None:
    """CNN/DailyMail used as benchmark dataset. HF datasets caches automatically."""
    from shared import load_hf_dataset, resolve_hf_dataset

    spec = resolve_hf_dataset("cnn_dailymail")
    print(f"[prepare_data] downloading {spec.label}")
    load_hf_dataset(spec)
    print("  cached")


def prepare_all():
    if C.USE_LOCAL_C4_CACHE:
        prepare_c4()
    prepare_cnn_dailymail()
    print("\nAll done.")


if __name__ == "__main__":
    prepare_all()
