"""ElasticBERT-large family adapter (K6 FamilyAdapter).

One request = one SST-2 sentence padded to 128 tokens, run through the first (exit + 1) encoder layers.
A checkpoint is a layer index. `advance(state, c)` runs embeddings plus the extended attention mask on the first
call, then encoder layers up to and including layer c, and keeps the hidden state of every checkpoint it reached.
The server stops at the first late checkpoint, but that segment has already run, so `finish(state, reached)` must
read the hidden state of the last on-time checkpoint, not the latest one: pooler[reached], classifier, argmax().item().
A failing layer raises. There is no fallback path.

Ported from AnyTimeBert/src/benchmark.py (`_load_model`, `_broadcast_pooler`); that module imports `shared` and is
not imported. The ElasticBERT model definition is imported inside functions from AnyTimeBert/reference.
torch, transformers are imported inside functions only.
Weights are the pretrained ElasticBERT-large with a random 2-class head (weight_source "pretrained"): the cost of every
exit is the same as for a fine-tuned head, and the trained poolers of one run are not needed.
"""
import csv
import random
import sys
import time
from pathlib import Path

from exp2 import contracts
from exp2.contracts import EXIT_RANGE
from exp2.serve import families

MODEL_ID = "OpenMOSS-Team/elasticbert-large"  # benchmark_config/bert.py HF_PRETRAINED_MODEL
NUM_LABELS = 2  # SST-2
MAX_SEQ_LENGTH = 128  # benchmark_config/bert.py MAX_SEQ_LENGTH
PREPROCESS_MODES = ("per_request", "preload")
INPUT_MODES = ("synthetic", "dataset")
SPECIAL_TOKEN_END = 104  # BERT uncased: [PAD] [UNK] [CLS] [SEP] [MASK] sit below this id; synthetic ids stay above
REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = REPO_ROOT / "AnyTimeBert" / "reference" / "finetune-static"
SST2_PATH = REPO_ROOT / "AnyTimeBert" / "glue_data" / "SST-2" / "dev.tsv"


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _log(msg):
    print(f"[exp2.fam_bert] {msg}", file=sys.stderr)


# --- model loading (ported from AnyTimeBert/src/benchmark.py:44-112) -------------------------------------

def _elasticbert():
    """Return (ElasticBertConfig, ElasticBertForSequenceClassification) from the reference tree."""
    ref = str(REFERENCE_DIR)
    if ref not in sys.path:
        sys.path.insert(0, ref)
    from models.configuration_elasticbert import ElasticBertConfig
    from models.modeling_elasticbert import ElasticBertForSequenceClassification

    return ElasticBertConfig, ElasticBertForSequenceClassification


def _broadcast_pooler(model):
    """Clone the one trained pooler into every empty pooler slot. Returns the number of slots filled.

    ElasticBert with num_output_layers=1 trains a single pooler at current_pooler_num (= num_hidden_layers - 1) and
    leaves the other slots None. Every exit needs a real pooler, so each empty slot gets a copy of the trained one
    (the standard anytime-eval baseline when no per-layer pooler is trained).
    """
    enc = model.elasticbert.encoder
    if not hasattr(enc, "pooler") or enc.pooler is None:
        return 0
    src_idx = enc.current_pooler_num
    if src_idx is None:
        return 0
    src = enc.pooler[src_idx]
    if src is None:
        return 0
    src_sd = src.state_dict()
    n_filled = 0
    for i in range(len(enc.pooler)):
        if i == src_idx:
            continue
        if enc.pooler[i] is None:
            new_p = type(src)(model.config)
            new_p.load_state_dict(src_sd)
            enc.pooler[i] = new_p
            n_filled += 1
    if n_filled:
        _log(f"broadcast trained pooler (slot {src_idx}) -> {n_filled} other slots")
    return n_filled


def _prepare(model):
    """Broadcast the pooler, move to the GPU (CPU when there is none), eval mode, no gradients."""
    import torch

    _broadcast_pooler(model)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _load_model(model_id, num_labels):
    """Load ElasticBert with the FULL layer stack of the checkpoint (24 for large).

    The exit is applied at run time by how far advance() goes, so one set of (compiled) layer modules serves every
    exit. num_hidden_layers is left at the checkpoint's native depth: forcing 12 broke elasticbert-large exits >= 12.
    """
    config_cls, model_cls = _elasticbert()
    cfg = config_cls.from_pretrained(model_id, num_labels=num_labels, num_output_layers=1)
    model = model_cls.from_pretrained(model_id, config=cfg, ignore_mismatched_sizes=True)
    return _prepare(model)


def _check_poolers(model, exits):
    """Raise unless every exit has a pooler module (a None slot would only fail at the first request)."""
    pooler = getattr(model.elasticbert.encoder, "pooler", None)
    if pooler is None:
        raise RuntimeError("the ElasticBERT checkpoint has no pooler modules")
    for e in exits:
        if e >= len(pooler) or pooler[e] is None:
            raise RuntimeError(f"no pooler for exit {e} after the broadcast ({len(pooler)} slots)")


def _compile_per_layer(enc, enable):
    """Wrap each encoder layer in torch.compile (shared by every exit). Returns the layer count. Errors propagate."""
    if not enable:
        return 0
    import torch

    n = len(enc.layer)
    for i in range(n):
        enc.layer[i] = torch.compile(enc.layer[i])
    return n


# --- SST-2 input helpers (stdlib only) -----------------------------------------------------------------------

def read_sst2_sentences(path):
    """Sentences of a GLUE SST-2 tsv (header `sentence<TAB>label`), raw: no stripping, quotes are data."""
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader, None)
        if not header or header[0] != "sentence":
            raise ValueError(f"{path}: expected a GLUE tsv with a 'sentence' header, got {header!r}")
        sentences = [row[0] for row in reader if row]
    if not sentences:
        raise ValueError(f"{path}: no sentences")
    return sentences


def pick_indices(total, n, seed):
    """n row indices out of `total`: a seeded shuffle, repeated from the start when n > total."""
    if total < 1:
        raise ValueError("cannot pick from an empty pool")
    order = list(range(total))
    random.Random(seed).shuffle(order)
    return [order[i % total] for i in range(n)]


class BertState:
    """Per-request state: input tensors, running hidden state, and the hidden state of every checkpoint reached."""

    __slots__ = ("exit", "ids", "mask", "types", "ckpts", "hidden", "ext_mask", "n_done", "hiddens", "logits")

    def __init__(self, exit, ids, mask, types, ckpts):
        self.exit = exit
        self.ids = ids
        self.mask = mask
        self.types = types
        self.ckpts = ckpts
        self.hidden = None  # output of the last finished layer; None until the embeddings ran
        self.ext_mask = None
        self.n_done = 0  # layers finished
        self.hiddens = {}  # checkpoint -> hidden state after that layer
        self.logits = None  # set by finish()


class BertAdapter:
    family = "bert"

    def __init__(self):
        self._model = None
        self._n_layers = None
        self._device = None
        self._model_id = MODEL_ID
        self._sst2_path = SST2_PATH
        self._preprocess = "per_request"
        self._inputs = None
        self._sync = None

    # ---- helpers
    def _check_exit(self, exit):
        hi = EXIT_RANGE["bert"][1] if self._n_layers is None else self._n_layers - 1
        if not _is_int(exit) or not EXIT_RANGE["bert"][0] <= exit <= hi:
            raise ValueError(f"bert exit {exit!r} outside {EXIT_RANGE['bert'][0]}..{hi}")

    def _require_loaded(self):
        if self._model is None:
            raise RuntimeError("load() must run first")

    # ---- model access (overridden by tests)
    def _build_model(self, cfg):
        return _load_model(self._model_id, NUM_LABELS)

    def _tokenizer(self):
        from transformers import BertTokenizer

        return BertTokenizer.from_pretrained(self._model_id, do_lower_case=True)

    def _synthetic_batch(self, n, seed):
        import torch

        gen = torch.Generator().manual_seed(seed)
        vocab = self._model.config.vocab_size
        ids = torch.randint(min(SPECIAL_TOKEN_END, vocab // 2), vocab, (n, MAX_SEQ_LENGTH), generator=gen)
        return ids, torch.ones_like(ids), torch.zeros_like(ids)

    def _dataset_batch(self, n, seed):
        """n SST-2 dev sentences tokenized and padded to MAX_SEQ_LENGTH: (input_ids, attention_mask, token_type_ids)."""
        sentences = read_sst2_sentences(self._sst2_path)
        picked = [sentences[i] for i in pick_indices(len(sentences), n, seed)]
        enc = self._tokenizer()(picked, max_length=MAX_SEQ_LENGTH, padding="max_length", truncation=True,
                                return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"], enc["token_type_ids"]

    def _embed(self, state):
        """Segment 0: embeddings and the extended attention mask. Returns (hidden, ext_mask)."""
        import torch

        with torch.no_grad():
            bert = self._model.elasticbert
            hidden = bert.embeddings(input_ids=state.ids, position_ids=None, token_type_ids=state.types)
            return hidden, bert.get_extended_attention_mask(state.mask, state.ids.shape)

    def _layer(self, state, i):
        """Encoder layer i on the running hidden state. Returns the new hidden state."""
        import torch

        with torch.no_grad():
            return self._model.elasticbert.encoder.layer[i](state.hidden, state.ext_mask, False)[0]

    def _head(self, state, reached):
        """pooler[reached] and the classifier on the hidden state of checkpoint `reached`; the class id is the D2H."""
        import torch

        with torch.no_grad():
            pooled = self._model.elasticbert.encoder.pooler[reached](state.hiddens[reached])
            state.logits = self._model.classifier(pooled)
            return int(state.logits.argmax(dim=-1).item())

    # ---- FamilyAdapter
    def load(self, exits, cfg):
        if not exits:
            raise ValueError("load() needs at least one exit")
        for e in exits:
            self._check_exit(e)
        preprocess = cfg.get("preprocess", "per_request")
        if preprocess not in PREPROCESS_MODES:
            raise ValueError(f"preprocess {preprocess!r} not in {PREPROCESS_MODES}")
        t0 = time.perf_counter()
        self._model_id = cfg.get("model_id", MODEL_ID)
        self._sst2_path = cfg.get("sst2_path", SST2_PATH)
        self._preprocess = preprocess
        self._model = self._build_model(cfg)
        self._n_layers = len(self._model.elasticbert.encoder.layer)
        for e in exits:
            self._check_exit(e)
        _check_poolers(self._model, exits)
        self._device = next(self._model.parameters()).device
        compiled = _compile_per_layer(self._model.elasticbert.encoder, bool(cfg.get("compile", False)))
        self._sync = families.make_sync(cfg.get("sync", "event"))
        return {"load_s": time.perf_counter() - t0, "n_layers": self._n_layers, "compiled_layers": compiled}

    def make_inputs(self, n, seed, mode):
        if mode not in INPUT_MODES:
            raise ValueError(f"input mode {mode!r} not in {INPUT_MODES}")
        if not _is_int(n) or n < 1:
            raise ValueError(f"n must be a positive int, got {n!r}")
        self._require_loaded()
        ids, mask, types = self._synthetic_batch(n, seed) if mode == "synthetic" else self._dataset_batch(n, seed)
        for name, t in (("input_ids", ids), ("attention_mask", mask), ("token_type_ids", types)):
            if tuple(t.shape) != (n, MAX_SEQ_LENGTH):
                raise ValueError(f"{name} shape {tuple(t.shape)}, expected {(n, MAX_SEQ_LENGTH)}")
        if self._preprocess == "preload":
            ids, mask, types = (t.to(self._device) for t in (ids, mask, types))
        self._inputs = [(ids[i:i + 1], mask[i:i + 1], types[i:i + 1]) for i in range(n)]

    def checkpoints(self, exit):
        self._check_exit(exit)
        return contracts.default_checkpoints("bert", exit)

    def begin(self, input_idx, exit):
        if self._inputs is None:
            raise RuntimeError("make_inputs() must run before begin()")
        if not _is_int(input_idx) or not 0 <= input_idx < len(self._inputs):
            raise ValueError(f"input_idx {input_idx!r} outside 0..{len(self._inputs) - 1}")
        self._check_exit(exit)
        ids, mask, types = self._inputs[input_idx]
        if self._preprocess == "per_request":  # the per-request host work: H2D copy of the three tensors
            ids, mask, types = ids.to(self._device), mask.to(self._device), types.to(self._device)
        return BertState(exit, ids, mask, types, contracts.default_checkpoints("bert", exit))

    def advance(self, state, ckpt):
        if not _is_int(ckpt) or ckpt not in state.ckpts or ckpt < state.n_done:
            raise ValueError(f"ckpt {ckpt!r} must be a checkpoint of {state.ckpts} after layer {state.n_done - 1}")
        if state.hidden is None:
            state.hidden, state.ext_mask = self._embed(state)
        for i in range(state.n_done, ckpt + 1):
            state.hidden = self._layer(state, i)
            state.n_done = i + 1
        state.hiddens[ckpt] = state.hidden

    def sync(self):
        if self._sync is None:
            raise RuntimeError("load() must run before sync()")
        self._sync()

    def finish(self, state, reached):
        """Head at checkpoint `reached` (one the request advanced to) and a small D2H. None: no-op."""
        if reached is None:
            return None
        if not _is_int(reached) or reached not in state.hiddens:
            raise ValueError(f"reached {reached!r} is not a checkpoint advanced to: {sorted(state.hiddens)}")
        return self._head(state, reached)

    def mem_stats(self):
        return families.mem_stats()

    def reset_peak(self):
        families.reset_peak()
