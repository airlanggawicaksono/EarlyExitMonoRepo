"""Pull GLUE from HuggingFace + dump TSVs in format reference's load_data expects.

ElasticBERT reference uses Jiant-style TSVs. We don't want manual download.
This script grabs GLUE via `datasets.load_dataset` and writes the TSVs once.

Usage:
    python prepare_data.py
    # or
    from AnyTimeBert.prepare_data import prepare_all
    prepare_all()
"""

import csv
import sys
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

import config as C   # type: ignore


# Map GLUE task name (uppercase ref name) -> (HF dataset config, columns)
TASK_SCHEMA = {
    "SST-2": {"hf": "sst2",  "cols": ["sentence"],              "label": "label"},
    "MRPC":  {"hf": "mrpc",  "cols": ["sentence1","sentence2"], "label": "label"},
    "QNLI":  {"hf": "qnli",  "cols": ["question","sentence"],   "label": "label"},
    "RTE":   {"hf": "rte",   "cols": ["sentence1","sentence2"], "label": "label"},
    "CoLA":  {"hf": "cola",  "cols": ["sentence"],              "label": "label"},
    "MNLI":  {"hf": "mnli",  "cols": ["premise","hypothesis"],  "label": "label"},
    "QQP":   {"hf": "qqp",   "cols": ["question1","question2"], "label": "label"},
    "STS-B": {"hf": "stsb",  "cols": ["sentence1","sentence2"], "label": "label"},
}


def _label_int_to_str(task: str, idx: int) -> str:
    """Reference expects string labels. Map back from HF int -> original."""
    mapping = {
        "SST-2": ["0", "1"],
        "MRPC":  ["0", "1"],
        "QNLI":  ["entailment", "not_entailment"],
        "RTE":   ["entailment", "not_entailment"],
        "CoLA":  ["0", "1"],
        "MNLI":  ["entailment", "neutral", "contradiction"],
        "QQP":   ["0", "1"],
        "STS-B": [str(idx)],   # regression
    }
    table = mapping.get(task, ["0", "1"])
    if task == "STS-B":
        return f"{idx}"
    return table[int(idx)]


def _write_tsv(path: Path, header: list, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {path.name:24s} ({len(rows)} rows)")


def prepare_task(task: str, out_root: Optional[Path] = None) -> Path:
    """Download one GLUE task from HF + dump train.tsv / dev.tsv / test.tsv."""
    from datasets import load_dataset

    schema = TASK_SCHEMA[task]
    out_dir = (out_root or C.GLUE_DIR) / task
    if out_dir.exists() and any(out_dir.glob("*.tsv")):
        print(f"[prepare_data] {task}: already exists at {out_dir}, skip")
        return out_dir

    print(f"[prepare_data] {task}: downloading from HF...")
    ds = load_dataset("glue", schema["hf"])

    cols = schema["cols"]
    label = schema["label"]

    splits = {"train": "train", "dev": "validation", "test": "test"}
    if task == "MNLI":
        splits = {"train": "train", "dev_matched": "validation_matched",
                  "dev_mismatched": "validation_mismatched"}

    for out_name, hf_split in splits.items():
        if hf_split not in ds:
            continue
        sub = ds[hf_split]

        # Reference expects: <id>\t<col1>\t<col2>\t...\t<label>  (header row)
        header = ["id"] + cols + [label]
        rows = []
        for i, ex in enumerate(sub):
            row = [str(i)]
            row.extend(str(ex[c]).replace("\t", " ").replace("\n", " ") for c in cols)
            lab = ex.get(label)
            if lab is None or lab == -1:
                row.append("0")  # test split has no labels
            else:
                row.append(_label_int_to_str(task, lab))
            rows.append(row)

        _write_tsv(out_dir / f"{out_name}.tsv", header, rows)

    return out_dir


def prepare_all(only: Optional[list] = None) -> None:
    tasks = only or C.TASKS
    for t in tasks:
        prepare_task(t)
    print(f"\nAll done. Data at: {C.GLUE_DIR}")


if __name__ == "__main__":
    prepare_all()
