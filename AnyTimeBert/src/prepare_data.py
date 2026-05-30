"""Pull GLUE from HuggingFace + dump TSVs in the exact layout the reference
processors expect.

ElasticBERT reference uses transformers `glue_processors`, which read Jiant-style
TSVs with very specific column orders (per task). We can't just write a uniform
`id,col1,col2,label` -- each task has its own schema.

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

_HERE  = Path(__file__).resolve().parent     # AnyTimeBert/src/
_MODEL = _HERE.parent
_REPO  = _MODEL.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_MODEL))

import config as C  # type: ignore


# Per-task schema mapping HF dataset -> reference TSV layout.
# Each entry: (hf_config, splits_map, write_fn)
# write_fn(out_dir, split_name, hf_split) -> writes train.tsv / dev.tsv / test.tsv

QNLI_RTE_LABEL = {0: "entailment", 1: "not_entailment"}


def _write_tsv(path: Path, header: Optional[list], rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        if header is not None:
            w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {path.name:24s} ({len(rows)} rows)")


def _clean(s) -> str:
    return str(s).replace("\t", " ").replace("\n", " ").replace("\\", "")


# ---- per-task writers --------------------------------------------------------
# Each writer's row layout MUST match what transformers' glue_processors[task]
# reads in _create_examples.

def _emit_sst2(out_dir: Path, split_name: str, sub) -> None:
    # processor: text=line[0], label=line[1]; expects header at line 0
    rows = []
    for ex in sub:
        rows.append([_clean(ex["sentence"]), str(ex["label"])])
    _write_tsv(out_dir / f"{split_name}.tsv", ["sentence", "label"], rows)


def _emit_mrpc(out_dir: Path, split_name: str, sub) -> None:
    # processor: text_a=line[3], text_b=line[4], label=line[0]
    rows = []
    for i, ex in enumerate(sub):
        rows.append([
            str(ex["label"]),
            str(ex.get("idx", i)),
            str(ex.get("idx", i)),
            _clean(ex["sentence1"]),
            _clean(ex["sentence2"]),
        ])
    _write_tsv(
        out_dir / f"{split_name}.tsv",
        ["Quality", "#1 ID", "#2 ID", "#1 String", "#2 String"],
        rows,
    )


def _emit_qnli(out_dir: Path, split_name: str, sub) -> None:
    # processor: id=line[0], text_a=line[1], text_b=line[2], label=line[-1]
    rows = []
    for i, ex in enumerate(sub):
        lab = ex["label"]
        lab_str = QNLI_RTE_LABEL.get(lab, "entailment") if lab != -1 else "entailment"
        rows.append([str(i), _clean(ex["question"]), _clean(ex["sentence"]), lab_str])
    _write_tsv(out_dir / f"{split_name}.tsv",
               ["index", "question", "sentence", "label"], rows)


def _emit_rte(out_dir: Path, split_name: str, sub) -> None:
    # processor: id=line[0], text_a=line[1], text_b=line[2], label=line[-1]
    rows = []
    for i, ex in enumerate(sub):
        lab = ex["label"]
        lab_str = QNLI_RTE_LABEL.get(lab, "entailment") if lab != -1 else "entailment"
        rows.append([str(i), _clean(ex["sentence1"]), _clean(ex["sentence2"]), lab_str])
    _write_tsv(out_dir / f"{split_name}.tsv",
               ["index", "sentence1", "sentence2", "label"], rows)


def _emit_cola(out_dir: Path, split_name: str, sub) -> None:
    # processor: text_index=3, label=line[1]; does NOT skip line 0 -> NO header
    rows = []
    for ex in sub:
        rows.append(["src", str(ex["label"]), "*", _clean(ex["sentence"])])
    _write_tsv(out_dir / f"{split_name}.tsv", None, rows)


def _emit_stsb(out_dir: Path, split_name: str, sub) -> None:
    # processor: id=line[0], text_a=line[7], text_b=line[8], label=line[-1]
    rows = []
    for i, ex in enumerate(sub):
        # pad up to index 7/8 with empty strings
        row = [str(i)] + [""] * 6 + [_clean(ex["sentence1"]), _clean(ex["sentence2"])]
        row.append(f"{ex['label']:.3f}")
        rows.append(row)
    header = ["index"] + [f"_{i}" for i in range(6)] + ["sentence1", "sentence2", "score"]
    _write_tsv(out_dir / f"{split_name}.tsv", header, rows)


def _emit_mnli(out_dir: Path, split_name: str, sub) -> None:
    # processor: id=line[0], text_a=line[8], text_b=line[9], label=line[-1]
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    rows = []
    for i, ex in enumerate(sub):
        lab = ex["label"]
        lab_str = label_map.get(lab, "entailment") if lab != -1 else "entailment"
        row = [str(i)] + [""] * 7 + [_clean(ex["premise"]), _clean(ex["hypothesis"]), lab_str]
        rows.append(row)
    header = ["index"] + [f"_{i}" for i in range(7)] + ["premise", "hypothesis", "label"]
    _write_tsv(out_dir / f"{split_name}.tsv", header, rows)


def _emit_qqp(out_dir: Path, split_name: str, sub) -> None:
    # processor: id=line[0], text_a=line[3], text_b=line[4], label=line[5]
    rows = []
    for i, ex in enumerate(sub):
        rows.append([
            str(i), str(i), str(i),
            _clean(ex["question1"]),
            _clean(ex["question2"]),
            str(ex["label"]),
        ])
    _write_tsv(out_dir / f"{split_name}.tsv",
               ["id", "qid1", "qid2", "question1", "question2", "is_duplicate"],
               rows)


TASK_TABLE = {
    # task_name: (hf_config, splits, emit_fn)
    "SST-2": ("sst2", {"train": "train", "dev": "validation"}, _emit_sst2),
    "MRPC":  ("mrpc", {"train": "train", "dev": "validation"}, _emit_mrpc),
    "QNLI":  ("qnli", {"train": "train", "dev": "validation"}, _emit_qnli),
    "RTE":   ("rte",  {"train": "train", "dev": "validation"}, _emit_rte),
    "CoLA":  ("cola", {"train": "train", "dev": "validation"}, _emit_cola),
    "STS-B": ("stsb", {"train": "train", "dev": "validation"}, _emit_stsb),
    "MNLI":  ("mnli", {"train": "train",
                        "dev_matched": "validation_matched",
                        "dev_mismatched": "validation_mismatched"}, _emit_mnli),
    "QQP":   ("qqp",  {"train": "train", "dev": "validation"}, _emit_qqp),
}


def prepare_task(task: str, out_root: Optional[Path] = None) -> Path:
    """Download one GLUE task from HF + dump TSVs in reference layout."""
    from datasets import load_dataset

    if task not in TASK_TABLE:
        raise ValueError(f"unknown task {task!r}, add to TASK_TABLE")
    hf_config, splits, emit = TASK_TABLE[task]

    out_dir = (out_root or C.GLUE_DIR) / task
    if out_dir.exists() and any(out_dir.glob("*.tsv")):
        print(f"[prepare_data] {task}: already exists at {out_dir}, skip")
        return out_dir

    print(f"[prepare_data] {task}: downloading from HF...")
    # canonical "glue" id was removed in datasets 3.x -> use the relocated repo
    ds = load_dataset("nyu-mll/glue", hf_config)

    for split_name, hf_split in splits.items():
        if hf_split not in ds:
            continue
        emit(out_dir, split_name, ds[hf_split])

    return out_dir


def prepare_all(only: Optional[list] = None) -> None:
    tasks = only or C.TASKS
    for t in tasks:
        prepare_task(t)
    print(f"\nAll done. Data at: {C.GLUE_DIR}")


if __name__ == "__main__":
    prepare_all()
