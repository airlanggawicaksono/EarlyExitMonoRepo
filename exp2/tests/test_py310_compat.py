"""Python 3.10 compatibility gate for exp2 (the Jetson venv is 3.10, the laptop is 3.13).

Two layers:
  1. scan_source() is the checker. Its own tests feed it bad and good snippets, so a checker
     that stops detecting something fails here.
  2. The repo test applies the checker to every exp2/**/*.py file and to multitenant_run.py.
     The file list is discovered at import time, so a new module is covered with no edit here.
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXP2 = ROOT / "exp2"
FEATURE_VERSION = (3, 10)

# (module, attribute) pairs that do not exist on python 3.10 or numpy 1.26.
FORBIDDEN_ATTRS = {
    ("datetime", "UTC"),
    ("typing", "Self"),
    ("typing", "override"),
    ("math", "cbrt"),
    ("numpy", "trapz"),
    ("numpy", "trapezoid"),
    ("itertools", "batched"),
}
FORBIDDEN_MODULES = {"tomllib"}
FORBIDDEN_NAMES = {"ExceptionGroup", "BaseExceptionGroup"}


def _quote_of(segment):
    """Return the opening quote ('"', "'", or a triple quote) of a string literal source segment."""
    s = segment.lstrip("rRbBfFuU")
    if s[:3] in ('"""', "'''"):
        return s[:3]
    return s[:1] if s[:1] in ("'", '"') else None


def _same_quote_clash(outer, inner):
    """True when `inner` cannot sit inside the expression part of an f-string quoted with `outer` on 3.10."""
    if outer is None or inner is None or outer[0] != inner[0]:
        return False
    return len(outer) == 1 or len(inner) == 3


def _fstring_findings(tree, src):
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr):
            continue
        outer = _quote_of(ast.get_source_segment(src, node) or "")
        for fv in (v for v in node.values if isinstance(v, ast.FormattedValue)):
            expr_src = ast.get_source_segment(src, fv.value) or ""
            if "\\" in expr_src:
                out.append((fv.value.lineno, "backslash inside f-string expression (3.12+ only)"))
            for sub in ast.walk(fv.value):
                is_str = isinstance(sub, ast.JoinedStr) or (
                    isinstance(sub, ast.Constant) and isinstance(sub.value, str))
                if is_str and _same_quote_clash(outer, _quote_of(ast.get_source_segment(src, sub) or "")):
                    out.append((sub.lineno, "f-string nested with the same quote (3.12+ only)"))
    return out


def scan_source(src, filename="<src>"):
    """Return a list of 'file:line: problem' strings; empty means the source is 3.10-clean."""
    try:
        tree = ast.parse(src, filename=filename, feature_version=FEATURE_VERSION)
    except SyntaxError as exc:
        return ["%s:%s: syntax not valid on 3.10: %s" % (filename, exc.lineno, exc.msg)]
    found = []
    alias = {}  # local name -> imported module
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                alias[a.asname or a.name.split(".")[0]] = a.name.split(".")[0]
                if a.name.split(".")[0] in FORBIDDEN_MODULES:
                    found.append((node.lineno, "import %s" % a.name))
        elif isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            if top in FORBIDDEN_MODULES:
                found.append((node.lineno, "import from %s" % node.module))
            for a in node.names:
                if (top, a.name) in FORBIDDEN_ATTRS:
                    found.append((node.lineno, "from %s import %s" % (node.module, a.name)))
                if a.name in FORBIDDEN_NAMES:
                    found.append((node.lineno, "import of %s" % a.name))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            mod = alias.get(node.value.id)
            if (mod, node.attr) in FORBIDDEN_ATTRS:
                found.append((node.lineno, "%s.%s" % (mod, node.attr)))
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            found.append((node.lineno, "name %s" % node.id))
        elif type(node).__name__ == "TryStar":
            found.append((node.lineno, "except* (3.11+ only)"))
    found += _fstring_findings(tree, src)
    return ["%s:%s: %s" % (filename, line, msg) for line, msg in sorted(found)]


# ---------------------------------------------------------------- checker self-tests
BAD_SNIPPETS = {
    "tomllib": "import tomllib\n",
    "tomllib_from": "from tomllib import loads\n",
    "datetime_UTC": "import datetime\nx = datetime.UTC\n",
    "datetime_UTC_from": "from datetime import UTC\n",
    "typing_Self": "import typing\nS = typing.Self\n",
    "typing_Self_from": "from typing import Self\n",
    "typing_override": "from typing import override\n",
    "typing_override_attr": "import typing as t\n@t.override\ndef f(): pass\n",
    "math_cbrt": "import math\nx = math.cbrt(8)\n",
    "np_trapz": "import numpy as np\nx = np.trapz([1], [1])\n",
    "numpy_trapz": "import numpy\nx = numpy.trapz([1], [1])\n",
    "np_trapezoid": "import numpy as np\nx = np.trapezoid([1], [1])\n",
    "np_trapz_from": "from numpy import trapz\n",
    "itertools_batched": "import itertools\nx = list(itertools.batched([1], 1))\n",
    "itertools_batched_from": "from itertools import batched\n",
    "ExceptionGroup": "x = ExceptionGroup('m', [ValueError()])\n",
    "BaseExceptionGroup": "raise BaseExceptionGroup('m', [ValueError()])\n",
    "except_star": "try:\n    pass\nexcept* ValueError:\n    pass\n",
    "pep695_def": "def f[T](x: T) -> T:\n    return x\n",
    "type_statement": "type X = int\n",
    "fstring_same_double": 'd = {"k": 1}\nx = f"{d["k"]}"\n',
    "fstring_same_single": "d = {'k': 1}\nx = f'{d['k']}'\n",
    "fstring_triple_nested_triple": 'x = f"""{"""a"""}"""\n',
    "fstring_nested_fstring": 'x = f"{f"{1}"}"\n',
    "fstring_backslash": "x = f\"{'\\n'.join([])}\"\n",
}

GOOD_SOURCE = '''
import math
import numpy as np
from datetime import datetime, timezone
from itertools import islice
from typing import Optional


def trapezoid(t, y):
    return sum((t[i + 1] - t[i]) * (y[i + 1] + y[i]) / 2 for i in range(len(t) - 1))


d = {"k": 1}
a = f"{d['k']}"
b = f'{d["k"]}'
c = f"""{d["k"]}"""
e = f"{d['k']:>4} {math.sqrt(4)!r}"
n = chr(10)
g = f"{n.join(['a', 'b'])}"
now = datetime.now(timezone.utc)
arr = np.sum(np.asarray([1.0, 2.0]))
try:
    pass
except (ValueError, KeyError):
    pass
match d:
    case {"k": 1}:
        pass
'''


@pytest.mark.parametrize("name", sorted(BAD_SNIPPETS))
def test_checker_flags_forbidden_construct(name):
    findings = scan_source(BAD_SNIPPETS[name], filename=name)
    assert findings, "checker missed %s" % name


def test_checker_accepts_clean_source():
    assert scan_source(GOOD_SOURCE) == []


def test_checker_reports_file_and_line():
    findings = scan_source("x = 1\nimport tomllib\n", filename="demo.py")
    assert findings == ["demo.py:2: import tomllib"]


# ---------------------------------------------------------------- the repo scan
def _repo_files():
    files = sorted(p for p in EXP2.rglob("*.py") if "__pycache__" not in p.parts)
    files.append(ROOT / "multitenant_run.py")
    return files


REPO_FILES = _repo_files()


def test_file_discovery_covers_exp2_and_multitenant_run():
    rels = {p.relative_to(ROOT).as_posix() for p in REPO_FILES}
    assert "multitenant_run.py" in rels
    assert "exp2/tests/test_py310_compat.py" in rels
    assert len(rels) >= 4


@pytest.mark.parametrize("path", REPO_FILES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_file_is_python310_clean(path):
    src = path.read_text(encoding="utf-8")
    findings = scan_source(src, filename=path.relative_to(ROOT).as_posix())
    assert findings == []
