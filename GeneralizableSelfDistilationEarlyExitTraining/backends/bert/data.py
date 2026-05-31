"""All data IO. GLUE dataloaders reusing the repo's ElasticBERT pipeline.

load_and_cache_examples_glue returns a TensorDataset of
(input_ids, attention_mask, token_type_ids, labels).
"""

import argparse

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset

from . import bootstrap  # noqa: F401  (injects sys.path)
from transformers import BertTokenizer, glue_processors  # type: ignore
from load_data import load_and_cache_examples_glue  # type: ignore

# train -> shuffle; dev/test -> sequential. Dict dispatch, no branch.
_SAMPLER = {"train": RandomSampler}


def _limit(dataset, n):
    """Cap dataset to first n items (dry-run). n=None -> full dataset."""
    if n is None:
        return dataset
    return Subset(dataset, list(range(min(n, len(dataset)))))


def count_labels(task: str) -> int:
    return len(glue_processors[task.lower()]().get_labels())


def _eval_args(cfg, data_type: str) -> argparse.Namespace:
    return argparse.Namespace(
        task_name=cfg.task.lower(),
        data_dir=str(cfg.data_dir / cfg.task),
        output_dir=str(cfg.run_dir),
        max_seq_length=cfg.max_seq_length,
        per_gpu_eval_batch_size=cfg.batch_size,
        n_gpu=1,
        local_rank=-1,
        model_name_or_path=cfg.model_id,
        overwrite_cache=False,
        model_type="elasticbert",
    )


def build_loader(cfg, data_type: str = "train") -> DataLoader:
    tokenizer = BertTokenizer.from_pretrained(cfg.model_id, do_lower_case=True)
    dataset = load_and_cache_examples_glue(
        _eval_args(cfg, data_type), cfg.task.lower(), tokenizer, data_type=data_type
    )
    dataset = _limit(dataset, cfg.max_train_samples)
    sampler = _SAMPLER.get(data_type, SequentialSampler)(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=cfg.batch_size)
