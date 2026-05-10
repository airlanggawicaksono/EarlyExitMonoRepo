"""ElasticBERT benchmark. TWO separate passes:

profile_hw(...)        -> hw_results.json       (latency + memory + energy, NO quality)
evaluate_quality(...)  -> quality_results.json  (accuracy/F1, NO HW measurement)
benchmark(...)         -> runs both
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple, Union

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

_HERE  = Path(__file__).resolve().parent     # AnyTimeBert/src/
_MODEL = _HERE.parent                         # AnyTimeBert/
_REPO  = _MODEL.parent                        # spd/
sys.path.insert(0, str(_REPO))                                          # `import shared`
sys.path.insert(0, str(_MODEL))                                         # `import config`
sys.path.insert(0, str(_MODEL / "reference"))                            # `import elue`
sys.path.insert(0, str(_MODEL / "reference" / "finetune-dynamic"))       # `models.*`, `load_data`

import config as C  # type: ignore
from transformers import BertTokenizer, glue_processors, glue_compute_metrics

from models.configuration_elasticbert import ElasticBertConfig  # type: ignore
from load_data import load_and_cache_examples_glue  # type: ignore

from shared import BenchmarkProfiler


def _load_model(
    model_id: str, strategy: str, num_labels: int, compile_model: bool = False
):
    cfg = ElasticBertConfig.from_pretrained(
        model_id,
        num_labels=num_labels,
        num_hidden_layers=C.NUM_HIDDEN_LAYERS,
        num_output_layers=C.NUM_OUTPUT_LAYERS,
    )
    if strategy == "entropy":
        from models.modeling_elasticbert_entropy import (
            ElasticBertForSequenceClassification,
        )  # type: ignore
    else:
        from models.modeling_elasticbert_patience import (
            ElasticBertForSequenceClassification,
        )  # type: ignore
    model = ElasticBertForSequenceClassification.from_pretrained(model_id, config=cfg)
    model.cuda().eval()
    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[bert.benchmark] torch.compile enabled")
        except Exception as e:
            print(f"[bert.benchmark] torch.compile failed: {e}")
    return model


def _load_loader(
    model_id: str,
    task: str,
    data_dir: Union[str, Path],
    out_dir: Path,
    max_seq_length: int,
):
    tokenizer = BertTokenizer.from_pretrained(model_id, do_lower_case=True)
    eval_args = argparse.Namespace(
        task_name=task.lower(),
        data_dir=str(data_dir),
        output_dir=str(out_dir),
        max_seq_length=max_seq_length,
        per_gpu_eval_batch_size=1,
        n_gpu=1,
        local_rank=-1,
        model_name_or_path=model_id,
        overwrite_cache=False,
        model_type="elasticbert",
    )
    eval_dataset = load_and_cache_examples_glue(
        eval_args, eval_args.task_name, tokenizer, data_type="dev"
    )
    return tokenizer, DataLoader(
        eval_dataset, sampler=SequentialSampler(eval_dataset), batch_size=1
    )


def _set_strategy(model, strategy, threshold):
    if strategy == "entropy":
        model.elasticbert.set_early_exit_entropy(float(threshold))
        model.elasticbert.set_eval_state(True)
    else:
        model.elasticbert.set_patience(int(threshold))
    model.elasticbert.reset_stats()


# =============================================================================
# HW pass — pure latency + memory + energy. NO quality.
# =============================================================================
def profile_hw(
    model_id: str,
    task: str,
    strategy: str,
    threshold: Union[int, float],
    data_dir: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    max_seq_length: int = 128,
    warmup_steps: int = 3,
    use_torch_compile: bool = True,
) -> Path:
    out_dir = Path(out_dir)
    out_path = out_dir / "hw_results.json"

    processor = glue_processors[task.lower()]()
    num_labels = len(processor.get_labels())
    model = _load_model(model_id, strategy, num_labels, compile_model=use_torch_compile)
    _set_strategy(model, strategy, threshold)
    _, loader = _load_loader(model_id, task, data_dir, out_dir, max_seq_length)

    with BenchmarkProfiler(
        out_path=out_path,
        task=task,
        strategy=strategy,
        threshold=threshold,
        warmup_steps=warmup_steps,
    ) as prof:
        for batch in tqdm(loader, desc=f"HW {task} {strategy}={threshold}"):
            ids, mask, types = [b.cuda() for b in batch[:3]]
            with prof.timer() as t:
                with torch.no_grad():
                    _ = model(input_ids=ids, attention_mask=mask, token_type_ids=types)
            prof.log_sample(
                prediction=None,
                label=None,
                ttft_sec=t.elapsed_s,
                end_to_end_sec=t.elapsed_s,
            )
    return out_path


# =============================================================================
# Quality pass — pure correctness eval. NO HW sampling.
# =============================================================================
def evaluate_quality(
    model_id: str,
    task: str,
    strategy: str,
    threshold: Union[int, float],
    data_dir: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    max_seq_length: int = 128,
) -> Path:
    out_dir = Path(out_dir)
    out_path = out_dir / "quality_results.json"

    processor = glue_processors[task.lower()]()
    num_labels = len(processor.get_labels())
    model = _load_model(model_id, strategy, num_labels, compile_model=False)
    _set_strategy(model, strategy, threshold)
    _, loader = _load_loader(model_id, task, data_dir, out_dir, max_seq_length)

    preds, labels = [], []
    for batch in tqdm(loader, desc=f"Q  {task} {strategy}={threshold}"):
        ids, mask, types, label = [b.cuda() for b in batch[:4]]
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask, token_type_ids=types)
        preds.append(out[0].argmax(-1).item())
        labels.append(label.item())

    metrics = glue_compute_metrics(
        task.lower(), torch.tensor(preds).numpy(), torch.tensor(labels).numpy()
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "task": task,
                "strategy": strategy,
                "threshold": threshold,
                "n_samples": len(preds),
                **metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[evaluate_quality] {metrics}")
    return out_path


# =============================================================================
# Combined: run both passes
# =============================================================================
def benchmark(
    model_id: str,
    task: str,
    strategy: str,
    threshold: Union[int, float],
    data_dir: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    max_seq_length: int = 128,
    bench_batch: int = 1,
    warmup_steps: int = 3,
    use_torch_compile: bool = True,
) -> Tuple[Path, Path]:
    hw = profile_hw(
        model_id,
        task,
        strategy,
        threshold,
        data_dir,
        out_dir,
        max_seq_length=max_seq_length,
        warmup_steps=warmup_steps,
        use_torch_compile=use_torch_compile,
    )
    q = evaluate_quality(
        model_id,
        task,
        strategy,
        threshold,
        data_dir,
        out_dir,
        max_seq_length=max_seq_length,
    )
    return hw, q
