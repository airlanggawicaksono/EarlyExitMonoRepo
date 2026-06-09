"""Hugging Face dataset aliases used by benchmarks.

The benchmark configs use stable local names such as "gsm8k" so result
directories stay readable. Resolve those names here before calling
datasets.load_dataset, because several old short dataset IDs now need
namespaced Hub repo IDs.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union


@dataclass(frozen=True)
class HFDatasetSpec:
    alias: str
    repo_id: str
    config: Optional[str] = None
    default_split: str = "test"
    trust_remote_code: bool = False

    @property
    def label(self) -> str:
        if self.config is None:
            return self.repo_id
        return f"{self.repo_id}/{self.config}"


_DATASET_ALIASES = {
    "cnn_dailymail": HFDatasetSpec(
        alias="cnn_dailymail",
        repo_id="abisee/cnn_dailymail",
        config="3.0.0",
    ),
    "gsm8k": HFDatasetSpec(
        alias="gsm8k",
        repo_id="openai/gsm8k",
        config="main",
    ),
    "arc_challenge": HFDatasetSpec(
        alias="arc_challenge",
        repo_id="allenai/ai2_arc",
        config="ARC-Challenge",
    ),
    "arc_easy": HFDatasetSpec(
        alias="arc_easy",
        repo_id="allenai/ai2_arc",
        config="ARC-Easy",
    ),
    "hellaswag": HFDatasetSpec(
        alias="hellaswag",
        repo_id="Rowan/hellaswag",
        default_split="validation",
        trust_remote_code=True,
    ),
    "mmlu": HFDatasetSpec(
        alias="mmlu",
        repo_id="cais/mmlu",
        config="all",
    ),
    "imagenet-1k": HFDatasetSpec(
        alias="imagenet-1k",
        repo_id="imagenet-1k",
        default_split="validation",
    ),
}


DatasetInput = Union[str, HFDatasetSpec, Tuple[str, Optional[str], str]]


def resolve_hf_dataset(dataset: DatasetInput) -> HFDatasetSpec:
    """Return the Hub repo/config/split for a local benchmark dataset name."""
    if isinstance(dataset, HFDatasetSpec):
        return dataset
    if isinstance(dataset, tuple):
        repo_id, config, split = dataset
        return HFDatasetSpec(alias=repo_id, repo_id=repo_id, config=config, default_split=split)
    if dataset in _DATASET_ALIASES:
        return _DATASET_ALIASES[dataset]
    default_split = "validation" if dataset == "imagenet-1k" else "test"
    return HFDatasetSpec(alias=dataset, repo_id=dataset, default_split=default_split)


def load_hf_dataset(dataset: DatasetInput, split: Optional[str] = None, **kwargs: Any):
    """Load a benchmark dataset through the shared alias resolver."""
    from datasets import load_dataset

    spec = resolve_hf_dataset(dataset)
    args = (spec.repo_id,) if spec.config is None else (spec.repo_id, spec.config)
    if spec.trust_remote_code and "trust_remote_code" not in kwargs:
        kwargs["trust_remote_code"] = True
    return load_dataset(*args, split=split or spec.default_split, **kwargs)
