"""Import hygiene for exp2.

Runtime layer: every exp2 module is imported in a fresh interpreter with a blocker on torch, transformers and
cv2. The module passes only if it neither loads nor tries to load them (a swallowed ImportError still counts).
The module list is discovered from the files on disk, so nothing is ever skipped.

Static layer (AST): no `import shared`, `bench_jetson`, `multitenant_run`, no AnyTime*/src/benchmark* module,
no scipy/pandas/sklearn, and torch/transformers/cv2 only inside families.py and fam_*.py.

Each checker has its own tests on bad snippets and on a throw-away package tree, so a weakened checker fails here.
"""
import ast
import functools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXP2 = ROOT / "exp2"
HEAVY = ("torch", "transformers", "cv2")
FORBIDDEN_TOP = ("shared", "bench_jetson", "multitenant_run")
BANNED_THIRD_PARTY = ("scipy", "pandas", "sklearn")

PROBE = r'''
import importlib, json, sys
BLOCKED = ("torch", "transformers", "cv2")
attempts = []


class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            attempts.append(name)
            raise ImportError("blocked heavy import: " + name)
        return None


sys.meta_path.insert(0, Blocker())
error = None
try:
    importlib.import_module(sys.argv[1])
except BaseException as exc:
    error = type(exc).__name__ + ": " + str(exc)
loaded = sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)
telemetry = "exp2.board.telemetry" in sys.modules
print(json.dumps({"error": error, "attempts": attempts, "loaded": loaded, "telemetry": telemetry}))
'''


# ---------------------------------------------------------------- runtime probe
def run_probe(module, root=ROOT):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([sys.executable, "-c", PROBE, module], cwd=str(root), env=env,
                          capture_output=True, text=True, timeout=180)
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")]
    assert lines, "probe crashed for %s (rc=%s): %s" % (module, proc.returncode, proc.stderr[-800:])
    return json.loads(lines[-1])


def light_problems(result):
    out = []
    if result["error"]:
        out.append("import failed: " + result["error"])
    if result["attempts"]:
        out.append("tried to import heavy modules: %s" % result["attempts"])
    if result["loaded"]:
        out.append("heavy modules in sys.modules: %s" % result["loaded"])
    return out


def discover_modules(root=EXP2, package="exp2"):
    """Dotted names of every non-test module under `root`, found on disk."""
    mods = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).with_suffix("")
        if "__pycache__" in rel.parts or "tests" in rel.parts:
            continue
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        mods.append(".".join([package] + parts))
    return mods


@functools.lru_cache(maxsize=None)
def _probe_cached(module):
    return run_probe(module)


MODULES = ["exp2"] + [m for m in discover_modules() if m != "exp2"]
if (EXP2 / "tests" / "mocks.py").exists():
    MODULES.append("exp2.tests.mocks")
NO_TELEMETRY = ["exp2"] + [m for m in ("exp2.serve.dispatcher", "exp2.policies")
                           if (ROOT / (m.replace(".", "/") + ".py")).exists()]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports_without_heavy_deps(module):
    assert light_problems(_probe_cached(module)) == []


@pytest.mark.parametrize("module", NO_TELEMETRY)
def test_dispatcher_and_policies_do_not_load_telemetry(module):
    assert _probe_cached(module)["telemetry"] is False


def _write_tree(tmp_path, files):
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return tmp_path


@pytest.fixture
def mini_tree(tmp_path):
    return _write_tree(tmp_path, {
        "exp2/__init__.py": "",
        "exp2/board/__init__.py": "",
        "exp2/board/telemetry.py": "X = 1\n",
        "exp2/serve/__init__.py": "",
        "exp2/clean.py": "def f():\n    import torch\n    return torch\n",
        "exp2/top_torch.py": "import torch\n",
        "exp2/swallowed.py": "try:\n    import torch\nexcept ImportError:\n    torch = None\n",
        "exp2/from_tf.py": "from transformers import AutoModel\n",
        "exp2/uses_cv2.py": "import cv2\n",
        "exp2/serve/dispatcher.py": "from exp2.board import telemetry\n",
        "exp2/serve/lean.py": "import json\n",
    })


def test_probe_accepts_lazy_import_inside_function(mini_tree):
    assert light_problems(run_probe("exp2.clean", mini_tree)) == []


@pytest.mark.parametrize("module,needle", [
    ("exp2.top_torch", "import failed"),
    ("exp2.swallowed", "tried to import heavy modules"),
    ("exp2.from_tf", "tried to import heavy modules"),
    ("exp2.uses_cv2", "tried to import heavy modules"),
])
def test_probe_flags_heavy_import_at_module_level(mini_tree, module, needle):
    problems = light_problems(run_probe(module, mini_tree))
    assert any(needle in p for p in problems), problems


def test_probe_detects_telemetry_loaded_by_dispatcher(mini_tree):
    assert run_probe("exp2.serve.dispatcher", mini_tree)["telemetry"] is True
    assert run_probe("exp2.serve.lean", mini_tree)["telemetry"] is False


def test_discover_modules_lists_packages_and_modules_but_not_tests(mini_tree):
    _write_tree(mini_tree, {"exp2/tests/__init__.py": "", "exp2/tests/test_x.py": ""})
    found = discover_modules(mini_tree / "exp2")
    assert "exp2" in found and "exp2.board.telemetry" in found and "exp2.serve.dispatcher" in found
    assert not any(m.startswith("exp2.tests") for m in found)


def test_module_list_is_not_empty_and_covers_s0_modules():
    assert len(MODULES) >= 2
    assert "exp2" in MODULES


# ---------------------------------------------------------------- static import rules
def import_problems(src, filename="<src>", allow_heavy=False):
    tree = ast.parse(src, filename=filename)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets = [(a.name, []) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            targets = [(node.module, [a.name for a in node.names])]
        else:
            continue
        for mod, imported in targets:
            parts = mod.split(".")
            top = parts[0]
            bench = [p for p in parts[1:] + imported if p.startswith("benchmark")]
            reason = None
            if top in FORBIDDEN_TOP:
                reason = "imports %s" % top
            elif top.startswith("AnyTime") and bench:
                reason = "imports an AnyTime benchmark module (port the code instead)"
            elif top.startswith("benchmark") or ("src" in parts and bench):
                reason = "imports a benchmark module (port the code instead)"
            elif top in BANNED_THIRD_PARTY:
                reason = "uses %s (numpy and stdlib only)" % top
            elif top in HEAVY and not allow_heavy:
                reason = "imports %s outside families.py / fam_*.py" % top
            if reason:
                out.append("%s:%s: %s" % (filename, node.lineno, reason))
    return out


BAD_IMPORTS = {
    "shared": "import shared\n",
    "shared_sub": "from shared.env import load_env\n",
    "bench_jetson": "import bench_jetson\n",
    "bench_jetson_from": "from bench_jetson import setup\n",
    "multitenant_run": "import multitenant_run as m\n",
    "anytime_dotted": "import AnyTimeBert.src.benchmark\n",
    "anytime_from": "from AnyTimeLLaMa.src.benchmark import run\n",
    "anytime_from_src": "from AnyTimeYolo.src import benchmark\n",
    "anytime_vit_file": "from AnyTimeVision.src.benchmark_pretrained_vit import x\n",
    "bare_benchmark": "import benchmark\n",
    "src_benchmark": "from src.benchmark import f\n",
    "src_import_benchmark": "from src import benchmark_x\n",
    "scipy": "import scipy.stats\n",
    "pandas": "import pandas as pd\n",
    "sklearn": "from sklearn.linear_model import Ridge\n",
    "torch_lazy": "def f():\n    import torch\n",
    "transformers": "from transformers import AutoModel\n",
    "cv2": "import cv2\n",
}


@pytest.mark.parametrize("name", sorted(BAD_IMPORTS))
def test_import_checker_flags(name):
    assert import_problems(BAD_IMPORTS[name], name), "checker missed %s" % name


def test_import_checker_accepts_model_definition_imports_and_heavy_in_family_files():
    ok = ("import json\nimport numpy as np\nfrom exp2 import contracts\n"
          "def load():\n    from early_exit.model import build\n    from models.modeling_elasticbert import X\n"
          "    from backends.vision.model import Y\n    return build, X, Y\n")
    assert import_problems(ok) == []
    assert import_problems("def f():\n    import torch\n    import cv2\n", allow_heavy=True) == []


def test_import_checker_does_not_flag_relative_or_substring_names():
    assert import_problems("from . import shared_utils\nimport sharedstuff\nimport my_benchmark_free\n") == []


def _exp2_sources():
    return sorted(p for p in EXP2.rglob("*.py") if "__pycache__" not in p.parts)


def _allows_heavy(path):
    rel = path.relative_to(EXP2).as_posix()
    return rel == "serve/families.py" or (rel.startswith("serve/fam_") and rel.endswith(".py")) \
        or rel.startswith("tests/") \
        or rel == "board/scripts/lp_probe.py"   # board-only probe: reads dynamo counters and CUDA allocation


@pytest.mark.parametrize("path", _exp2_sources(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_exp2_file_obeys_import_rules(path):
    problems = import_problems(path.read_text(encoding="utf-8"), path.relative_to(ROOT).as_posix(),
                               allow_heavy=_allows_heavy(path))
    assert problems == []


# ---------------------------------------------------------------- tree hygiene (.gitignore traps)
def ignored_paths(paths):
    # bytes + NUL separators: text mode would turn "\n" into "\r\n" on Windows and git would see "\r" in each path
    data = b"".join(p.encode("utf-8") + b"\0" for p in paths)
    proc = subprocess.run(["git", "check-ignore", "--no-index", "--stdin", "-z"], cwd=str(ROOT), input=data,
                          capture_output=True, timeout=60)
    assert proc.returncode in (0, 1), "git check-ignore failed: %s" % proc.stderr.decode("utf-8", "replace")
    return sorted(p.decode("utf-8") for p in proc.stdout.split(b"\0") if p)


def test_gitignore_check_detects_the_known_traps():
    traps = ["exp2/analysis/x.py", "exp2/logs/y.json", "exp2/data/z.json", "exp2/tests/fixtures/a.bin",
             "exp2/tests/fixtures/b.pt", "exp2/tests/fixtures/freeze.local", "exp2/runs/r.py", "exp2/cache/c.py"]
    fine = ["exp2/analyze/x.py", "exp2/serve/ok.py", "exp2/tests/fixtures/freeze.json"]
    assert ignored_paths(traps + fine) == sorted(traps)


def test_no_exp2_path_is_gitignored():
    files = [p.relative_to(ROOT).as_posix() for p in EXP2.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    assert files
    assert ignored_paths(files) == []
