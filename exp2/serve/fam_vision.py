"""ViT-large family adapter (K6 FamilyAdapter).

One request = one 224x224 image through the first `exit` transformer blocks of google/vit-large-patch16-224.
A checkpoint is a block count c (4, 8, ..., plus the target). `advance(state, c)` runs the embeddings on the first
call, then blocks up to c, and keeps the hidden state of every checkpoint it reached. The server may run a late
segment anyway, so `finish(state, reached)` reads the hidden state of `reached`, not the latest one:
heads[reached - 1] on h[:, 0], then a small D2H (the class id). A failing block raises. There is no fallback path.

Ported from AnyTimeVisionenc/src/benchmark_pretrained_vit.py (`_load_pretrained_vit`, WITHOUT its
`_broadcast_or_random_head`, which loads a second ViT-large of about 1.2 GB) and
benchmark_trained_vit.py (`_get_block_list_holder`, `_trained_forward_exit`). Those modules import `shared`
and are not imported. MultiExitViT is not imported either: its package __init__ pulls in the training stack.
Heads are random-init Linear(hidden, NUM_LABELS), one per block, same cost as trained heads (HW cost only).
As in `_trained_forward_exit` there is no final layernorm and dropout is the identity in eval.
torch, transformers and datasets are imported inside functions only.

Inputs: "synthetic" = seeded random normalized pixels. "dataset" = CIFAR-10 test images through the HF image
processor. preprocess "per_request" keeps raw inputs on the host and `begin` does processing and the H2D copy;
"preload" converts once in make_inputs and keeps tensors on the device.
"""
import time

from exp2 import contracts
from exp2.contracts import EXIT_RANGE
from exp2.serve import families

MODEL_ID = "google/vit-large-patch16-224"
DATASET = "uoft-cs/cifar10"
NUM_LABELS = 10  # CIFAR-10
PREPROCESS_MODES = ("per_request", "preload")
INPUT_MODES = ("synthetic", "dataset")


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _block_list(backbone):
    """The ModuleList of transformer blocks: ViTModel.layers (transformers >= 5) or .encoder.layer (< 5)."""
    layers = getattr(backbone, "layers", None)
    if layers is not None:
        return layers
    enc = getattr(backbone, "encoder", None)
    for attr in ("layer", "layers"):
        layers = getattr(enc, attr, None)
        if layers is not None:
            return layers
    raise AttributeError("ViT transformer block list not found (tried .layers, .encoder.layer, .encoder.layers)")


def _block_output(out):
    """ViTLayer returns a tuple in transformers 4.x and a tensor in 5.x."""
    return out[0] if isinstance(out, (tuple, list)) else out


def _load_model(model_id, num_labels):
    """(backbone, heads): pretrained ViTModel without pooler, one random head per block, eval, frozen, on the GPU."""
    import torch
    from transformers import ViTModel

    backbone = ViTModel.from_pretrained(model_id, add_pooling_layer=False)
    heads = torch.nn.ModuleList(
        [torch.nn.Linear(backbone.config.hidden_size, num_labels) for _ in _block_list(backbone)])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone, heads = backbone.to(device).eval(), heads.to(device).eval()
    for p in list(backbone.parameters()) + list(heads.parameters()):
        p.requires_grad = False
    return backbone, heads


def _compile_per_block(blocks, enable):
    """Wrap each block in torch.compile (shared by every exit). Returns the block count. Errors propagate."""
    if not enable:
        return 0
    import torch

    for i in range(len(blocks)):
        blocks[i] = torch.compile(blocks[i])
    return len(blocks)


class VisionState:
    """Per-request state: pixels, running hidden state, and the hidden state of every checkpoint reached."""

    __slots__ = ("exit", "pixels", "ckpts", "hidden", "n_done", "hiddens", "logits")

    def __init__(self, exit, pixels, ckpts):
        self.exit = exit
        self.pixels = pixels
        self.ckpts = ckpts
        self.hidden = None  # output of the last finished block; None until the embeddings ran
        self.n_done = 0  # blocks finished
        self.hiddens = {}  # checkpoint -> hidden state after that many blocks
        self.logits = None  # set by finish()


class VisionAdapter:
    family = "vision"

    def __init__(self):
        self._backbone = None
        self._heads = None
        self._blocks = None
        self._n_layers = None
        self._device = None
        self._model_id = MODEL_ID
        self._preprocess = "per_request"
        self._inputs = None
        self._pixelize = None  # raw image -> (1, 3, S, S) host tensor; set only for dataset + per_request
        self._sync = None

    # ---- helpers
    def _check_exit(self, exit):
        hi = EXIT_RANGE["vision"][1] if self._n_layers is None else self._n_layers
        if not _is_int(exit) or not EXIT_RANGE["vision"][0] <= exit <= hi:
            raise ValueError(f"vision exit {exit!r} outside {EXIT_RANGE['vision'][0]}..{hi}")

    def _require_loaded(self):
        if self._backbone is None:
            raise RuntimeError("load() must run first")

    # ---- model access (overridden by tests)
    def _build_model(self, cfg):
        return _load_model(self._model_id, NUM_LABELS)

    def _synthetic_batch(self, n, seed):
        import torch

        c = self._backbone.config
        gen = torch.Generator().manual_seed(seed)
        return torch.randn(n, c.num_channels, c.image_size, c.image_size, generator=gen)

    def _dataset_images(self, n, seed):
        """n CIFAR-10 test images (PIL, RGB); a seeded shuffle, repeated from the start when n > len."""
        from datasets import load_dataset

        ds = load_dataset(DATASET, split="test").shuffle(seed=seed)
        return [ds[i % len(ds)]["img"].convert("RGB") for i in range(n)]

    def _processor(self):
        from transformers import AutoImageProcessor

        return AutoImageProcessor.from_pretrained(self._model_id)

    def _embed(self, state):
        """Segment 0: patch + position embeddings. Returns the hidden state before block 0."""
        import torch

        with torch.no_grad():
            return self._backbone.embeddings(state.pixels)

    def _block(self, state, i):
        """Block i on the running hidden state. Returns the new hidden state."""
        import torch

        with torch.no_grad():
            return _block_output(self._blocks[i](state.hidden))

    def _head(self, state, reached):
        """heads[reached - 1] on the CLS token after `reached` blocks; the class id is the D2H."""
        import torch

        with torch.no_grad():
            state.logits = self._heads[reached - 1](state.hiddens[reached][:, 0])
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
        self._preprocess = preprocess
        self._backbone, self._heads = self._build_model(cfg)
        self._blocks = _block_list(self._backbone)
        self._n_layers = len(self._blocks)
        for e in exits:
            self._check_exit(e)
        self._device = next(self._backbone.parameters()).device
        compiled = _compile_per_block(self._blocks, bool(cfg.get("compile", False)))
        self._sync = families.make_sync(cfg.get("sync", "event"))
        return {"load_s": time.perf_counter() - t0, "n_layers": self._n_layers, "compiled_layers": compiled}

    def make_inputs(self, n, seed, mode):
        if mode not in INPUT_MODES:
            raise ValueError(f"input mode {mode!r} not in {INPUT_MODES}")
        if not _is_int(n) or n < 1:
            raise ValueError(f"n must be a positive int, got {n!r}")
        self._require_loaded()
        pixelize = None
        if mode == "synthetic":
            px = self._synthetic_batch(n, seed)
            items = [px[i:i + 1] for i in range(n)]
        else:
            items = self._dataset_images(n, seed)
            if len(items) != n:
                raise ValueError(f"dataset gave {len(items)} images, expected {n}")
            proc = self._processor()

            def pixelize(img):
                return proc(img, return_tensors="pt")["pixel_values"]

            if self._preprocess == "preload":
                items, pixelize = [pixelize(img) for img in items], None
        if pixelize is None:
            c = self._backbone.config
            want = (1, c.num_channels, c.image_size, c.image_size)
            for t in items:
                if tuple(t.shape) != want:
                    raise ValueError(f"pixel shape {tuple(t.shape)}, expected {want}")
            if self._preprocess == "preload":
                items = [t.to(self._device) for t in items]
        self._inputs, self._pixelize = items, pixelize

    def checkpoints(self, exit):
        self._check_exit(exit)
        return contracts.default_checkpoints("vision", exit)

    def begin(self, input_idx, exit):
        if self._inputs is None:
            raise RuntimeError("make_inputs() must run before begin()")
        if not _is_int(input_idx) or not 0 <= input_idx < len(self._inputs):
            raise ValueError(f"input_idx {input_idx!r} outside 0..{len(self._inputs) - 1}")
        self._check_exit(exit)
        x = self._inputs[input_idx]
        if self._preprocess == "per_request":  # the per-request host work: image processing, then the H2D copy
            if self._pixelize is not None:
                x = self._pixelize(x)
            x = x.to(self._device)
        return VisionState(exit, x, contracts.default_checkpoints("vision", exit))

    def advance(self, state, ckpt):
        if not _is_int(ckpt) or ckpt not in state.ckpts or ckpt <= state.n_done:
            raise ValueError(f"ckpt {ckpt!r} must be a checkpoint of {state.ckpts} after {state.n_done} blocks")
        if state.hidden is None:
            state.hidden = self._embed(state)
        for i in range(state.n_done, ckpt):
            state.hidden = self._block(state, i)
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
