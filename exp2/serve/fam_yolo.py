"""YOLOv9 gelan-m-ee family adapter (K6 FamilyAdapter).

One request = one 640x640 letterboxed image through backbone modules 0..YOLO_EXIT_MAX_DEPTH[exit], then detection
head 22 + reached over all 3 scales (P3/P4/P5). A checkpoint is an exit index c: `advance(state, c)` runs the modules
after the last one done, up to YOLO_EXIT_MAX_DEPTH[c]. Nothing restarts and nothing is recomputed: the feature maps
that heads read (the model's `save` list) stay in `state.y`, so the head of any earlier checkpoint can still run
after a later segment has used the GPU. `finish(state, reached)` runs only that head, with no NMS, and copies one
float to the host (the best class score) so the call returns after the head finished.

Ported from AnyTimeYolo/src/benchmark.py (`_load_ee_model`, the module loop of `_forward_to_exit`); that module is
not imported. Differences from the port:
  - the loop is resumable (the legacy one restarts at module 0 for every exit);
  - compile errors and bad checkpoints raise: no `except` fallback on the hot path;
  - no sys.modules purge: one family runs per process, so `models`/`utils` cannot collide with another family;
  - no upstream-head broadcast: exp2 measures time, energy and depth, not mAP. The checkpoint load is strict=False and
    the number of missing and unexpected keys is returned by load().
torch and the yolov9 packages (cv2, ...) are imported inside functions only.

Inputs: preprocess "per_request" keeps the letterboxed uint8 image on the host and `begin` does the H2D copy and
uint8 -> float / 255. "preload" converts once in make_inputs and keeps float tensors on the device.
"""
import random
import sys
import time
from pathlib import Path

from exp2 import contracts
from exp2.contracts import EXIT_RANGE, YOLO_EXIT_MAX_DEPTH
from exp2.serve import families

IMG_SIZE = 640
HEAD_OFFSET = 22  # module index of exit head 0 in gelan-m-ee.yaml; backbone and FPN are modules 0..21
PREPROCESS_MODES = ("per_request", "preload")
INPUT_MODES = ("synthetic", "dataset")

YOLO_ROOT = Path(__file__).resolve().parents[2] / "AnyTimeYolo"
EE_YAML = YOLO_ROOT / "src" / "early_exit" / "configs" / "gelan-m-ee.yaml"
WEIGHTS = YOLO_ROOT / "ckpts" / "gelan-m.pt"
DATA_DIR = YOLO_ROOT / "datasets" / "coco"
PRETRAINED_URL = "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-m.pt"
VAL_DIRS = ("val", "valid/images", "valid", "images/val2017", "val2017")


def _ensure_import_paths():
    """Make `early_exit` (AnyTimeYolo/src) and the yolov9 packages `models`, `utils` importable. Idempotent."""
    for p in (YOLO_ROOT / "model" / "yolov9", YOLO_ROOT / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _val_path(data_dir):
    """Validation image folder under data_dir (same search order as the legacy `_load_loader`)."""
    base = Path(data_dir)
    found = next((base / p for p in VAL_DIRS if (base / p).exists()), None)
    if found is not None:
        return found
    cands = (list(base.rglob("images/val2017")) + list(base.rglob("val2017"))
             + list(base.rglob("valid/images")) + list(base.rglob("valid")))
    return cands[0] if cands else base / "val"


def _compile_backbone(model, enable):
    """Wrap each of the 22 backbone/FPN modules in torch.compile (heads stay eager). Returns the count."""
    if not enable:
        return 0
    import torch

    for idx in range(HEAD_OFFSET):
        model.model[idx] = torch.compile(model.model[idx])
    return HEAD_OFFSET


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


class YoloState:
    """Per-request state: `y[i]` is the output of module i if a head or a later module reads it, else None.

    len(y) is the index of the next module to run. `out` is the output of the last module run.
    `ckpt` is the last checkpoint (exit index) fully advanced to, -1 before the first one.
    """

    __slots__ = ("exit", "out", "y", "ckpt")

    def __init__(self, exit, out):
        self.exit = exit
        self.out = out
        self.y = []
        self.ckpt = -1


class YoloAdapter:
    family = "yolo"

    def __init__(self):
        self._model = None
        self._save = frozenset()
        self._device = None
        self._img_size = IMG_SIZE
        self._data_dir = DATA_DIR
        self._preprocess = "per_request"
        self._inputs = None
        self._sync = None
        self._load_report = {}

    # ---- helpers
    def _check_exit(self, exit):
        lo, hi = EXIT_RANGE["yolo"]
        if not _is_int(exit) or not lo <= exit <= hi:
            raise ValueError(f"yolo exit {exit!r} outside {lo}..{hi}")

    def _require_loaded(self):
        if self._model is None:
            raise RuntimeError("load() must run first")

    # ---- model and torch access (overridden by tests)
    def _build_model(self, cfg):
        """EarlyExitModel from the yaml plus the checkpoint, on the GPU if there is one, in eval mode."""
        import torch

        _ensure_import_paths()
        from early_exit.model import EarlyExitModel  # type: ignore

        ee_yaml = Path(cfg.get("ee_yaml", EE_YAML))
        weights = Path(cfg.get("weights", WEIGHTS))
        if not weights.is_file():
            raise FileNotFoundError(f"yolo weights not found: {weights} (cfg 'weights'; download {PRETRAINED_URL})")
        model = EarlyExitModel(str(ee_yaml), ch=3)
        ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        if hasattr(state, "state_dict"):
            state = state.state_dict()
        missing, unexpected = model.load_state_dict(state, strict=False)
        self._load_report = {"missing_keys": len(missing), "unexpected_keys": len(unexpected)}
        model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    def _no_grad(self):
        import torch

        return torch.no_grad()

    def _to_input(self, raw):
        """Host uint8 (1, 3, S, S) -> float32 in [0, 1] on the model device (the legacy `.to(device).float() / 255`)."""
        return raw.to(self._device).float() / 255.0

    def _summarize(self, out):
        """Head output (pred, feats) -> the best class score as a Python float: a 4-byte D2H, no NMS."""
        pred = out[0] if isinstance(out, (tuple, list)) else out
        return pred[:, 4:, :].amax().item()

    def _synthetic_images(self, n, seed):
        import torch

        gen = torch.Generator().manual_seed(seed)
        return torch.randint(0, 256, (n, 3, self._img_size, self._img_size), generator=gen, dtype=torch.uint8)

    def _open_dataset(self):
        _ensure_import_paths()
        from utils.dataloaders import LoadImagesAndLabels  # type: ignore

        return LoadImagesAndLabels(str(_val_path(self._data_dir)), img_size=self._img_size, batch_size=1,
                                   augment=False, hyp=None, rect=False)

    def _dataset_images(self, n, seed):
        """n letterboxed uint8 images (n, 3, S, S), chosen by seed. rect=False: fixed shape, no recompiles."""
        import torch

        ds = self._open_dataset()
        if len(ds) < n:
            raise RuntimeError(f"only {len(ds)} images in {self._data_dir}, need {n}")
        return torch.stack([ds[i][0] for i in random.Random(seed).sample(range(len(ds)), n)])

    # ---- FamilyAdapter
    def load(self, exits, cfg):
        if not exits:
            raise ValueError("load() needs at least one exit")
        for e in exits:
            self._check_exit(e)
        preprocess = cfg.get("preprocess", "per_request")
        if preprocess not in PREPROCESS_MODES:
            raise ValueError(f"preprocess {preprocess!r} not in {PREPROCESS_MODES}")
        img_size = cfg.get("img_size", IMG_SIZE)
        if not _is_int(img_size) or img_size < 32:
            raise ValueError(f"img_size {img_size!r} must be an int >= 32")
        t0 = time.perf_counter()
        model = self._build_model(cfg)
        need = HEAD_OFFSET + max(exits) + 1  # the head of the deepest exit is module HEAD_OFFSET + exit
        if len(model.model) < need:
            raise ValueError(f"model has {len(model.model)} modules, exit {max(exits)} needs {need}")
        self._model = model
        self._save = frozenset(model.save)
        self._preprocess = preprocess
        self._img_size = img_size
        self._data_dir = Path(cfg.get("data_dir", DATA_DIR))
        self._device = next(model.parameters()).device
        compiled = _compile_backbone(model, bool(cfg.get("compile", False)))
        self._sync = families.make_sync(cfg.get("sync", "event"))
        return {"load_s": time.perf_counter() - t0, "n_modules": len(model.model), "compiled_modules": compiled,
                **self._load_report}

    def make_inputs(self, n, seed, mode):
        if mode not in INPUT_MODES:
            raise ValueError(f"input mode {mode!r} not in {INPUT_MODES}")
        if not _is_int(n) or n < 1:
            raise ValueError(f"n must be a positive int, got {n!r}")
        self._require_loaded()
        images = self._synthetic_images(n, seed) if mode == "synthetic" else self._dataset_images(n, seed)
        want = (n, 3, self._img_size, self._img_size)
        if tuple(images.shape) != want:
            raise ValueError(f"image batch shape {tuple(images.shape)}, expected {want}")
        views = [images[i:i + 1] for i in range(n)]
        self._inputs = views if self._preprocess == "per_request" else [self._to_input(v) for v in views]

    def checkpoints(self, exit):
        self._check_exit(exit)
        return contracts.default_checkpoints("yolo", exit)

    def begin(self, input_idx, exit):
        if self._inputs is None:
            raise RuntimeError("make_inputs() must run before begin()")
        if not _is_int(input_idx) or not 0 <= input_idx < len(self._inputs):
            raise ValueError(f"input_idx {input_idx!r} outside 0..{len(self._inputs) - 1}")
        self._check_exit(exit)
        x = self._inputs[input_idx]
        if self._preprocess == "per_request":
            x = self._to_input(x)  # the per-request host work: H2D copy and uint8 -> float
        return YoloState(exit, x)

    def advance(self, state, ckpt):
        """Run the modules after the last one done, up to YOLO_EXIT_MAX_DEPTH[ckpt]. An error propagates; a failed
        call leaves `state` at a module boundary (len(y) = next module), so a later call resumes there."""
        if not _is_int(ckpt) or not state.ckpt < ckpt <= state.exit:
            raise ValueError(f"ckpt {ckpt!r} must be an int in ({state.ckpt}, {state.exit}]")
        self._require_loaded()
        model = self._model
        max_d = YOLO_EXIT_MAX_DEPTH[ckpt]
        with self._no_grad():
            while len(state.y) <= max_d:
                idx = len(state.y)
                m = model.model[idx]
                state.out = m(model._resolve_input(m, state.out, state.y))
                state.y.append(state.out if idx in self._save else None)
        state.ckpt = ckpt

    def sync(self):
        if self._sync is None:
            raise RuntimeError("load() must run before sync()")
        self._sync()

    def finish(self, state, reached):
        """Run head 22 + reached on its three saved maps; return the best class score. None: no-op.

        Reads only maps saved at or before checkpoint `reached`, so segments that ran after the last on-time
        checkpoint do not change the result.
        """
        if reached is None:
            return None
        if not _is_int(reached) or not 0 <= reached <= state.ckpt:
            raise ValueError(f"reached {reached!r} outside 0..{state.ckpt}")
        self._require_loaded()
        model = self._model
        head = model.model[HEAD_OFFSET + reached]
        with self._no_grad():
            out = head(model._resolve_input(head, state.out, state.y))
        return self._summarize(out)

    def mem_stats(self):
        return families.mem_stats()

    def reset_peak(self):
        families.reset_peak()
