"""Folder layout rules of the T2-41 refactor (plans/T2-41.md), checked by reading imports (ast), not by importing.

Adapted from Phase 1's T1-28 layout test. S0 holds only the rules that pass on today's structure; each step adds
its own (OLD_PATHS grows with every moved module, STAGES / FRAME_ALLOWED are filled when the stage folders exist,
the langgraph rule narrows to src/kosis_rag/graph/ at S4). Exceptions are listed with their reason and the step
that removes them.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MODULE = SRC / "kosis_rag"  # the search module (online)
BUILD = SRC / "kosis_rag_build"  # the offline DB build package (from S6a; absent until then)
# Code outside src that imports the module: an old path must be gone from all of it (P2-R6, no shims)
CALLERS = tuple(ROOT / d for d in ("backend", "evals", "scripts", "tests", "lab"))

# Stage folders of kosis_rag in data-flow order (a stage may import the ones before it, never the ones after it).
# Placeholder until S2 creates them; the order rule is added then.
STAGES = ("parse", "retrieve", "rerank", "gate")
# Old module paths nothing may import any more, added by the step that removes them (S1: kosis_rag.schemas, ...).
OLD_PATHS: tuple[str, ...] = ()
# Files allowed to import pandas / pyarrow. Placeholder: not enforced at S0 (no rule for frame libraries yet).
FRAME_ALLOWED: tuple[str, ...] = ()

# langgraph: only the graph files (S4: src/kosis_rag/graph/ and, from S6a, src/kosis_rag_build/graph/).
LANGGRAPH_ALLOWED = ("src/kosis_rag/graph.py", "src/kosis_rag/hq/graph.py")
# Top-level packages src must never import (entry points and experiments depend on src, not the other way).
NOT_FROM_SRC = ("backend", "evals", "scripts", "lab", "tests")
# (file, imported name) pairs allowed against NOT_FROM_SRC, with the reason.
SRC_EXCEPTIONS = {
    # The hq build CLI prints the run's cost from evals/pricing.json through a function-level import that falls
    # back when evals is not importable (cost_line). Removed at S6a, when the CLI moves to kosis_rag_build.
    ("src/kosis_rag/hq/build.py", "evals.ops"),
    ("src/kosis_rag/hq/build.py", "evals.ops.cost"),
    ("src/kosis_rag/hq/build.py", "evals.ops.load_pricing"),
}


def _py_files(*roots: Path) -> list[Path]:
    return sorted(p for root in roots if root.is_dir() for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _module_of(path: Path) -> str:
    rel = path.relative_to(SRC if path.is_relative_to(SRC) else ROOT).with_suffix("")
    parts = rel.parts[:-1] if rel.name == "__init__" else rel.parts
    return ".".join(parts)


def _imports(path: Path) -> list[str]:
    """Absolute names of every import in the file, function-level ones included."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = _module_of(path) if path.name == "__init__.py" else _module_of(path).rpartition(".")[0]
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")[: len(package.split(".")) - node.level + 1]
                module = ".".join([*base, node.module] if node.module else base)
            else:
                module = node.module or ""
            names.append(module)
            names += [f"{module}.{alias.name}" for alias in node.names]
    return names


def _hits(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)


SRC_FILES = _py_files(MODULE, BUILD)
CALLER_FILES = _py_files(*CALLERS)


def test_files_are_found():
    assert len(SRC_FILES) > 20 and len(CALLER_FILES) > 50


@pytest.mark.parametrize("path", SRC_FILES, ids=_rel)
def test_src_import_rules(path: Path):
    rel = _rel(path)
    names = _imports(path)
    problems = []
    if rel not in LANGGRAPH_ALLOWED:
        problems += [n for n in names if _hits(n, ("langgraph",))]
    problems += [n for n in names if _hits(n, NOT_FROM_SRC) and (rel, n) not in SRC_EXCEPTIONS]
    if path.is_relative_to(MODULE):  # the search module never imports the offline build package
        problems += [n for n in names if _hits(n, ("kosis_rag_build",))]
    problems += [n for n in names if _hits(n, OLD_PATHS)]
    assert not problems, f"{rel}: imports break the layout rules: {sorted(set(problems))}"


@pytest.mark.parametrize("path", CALLER_FILES, ids=_rel)
def test_callers_use_no_old_path(path: Path):
    problems = [n for n in _imports(path) if _hits(n, OLD_PATHS)]
    assert not problems, f"{_rel(path)}: imports an old path: {sorted(set(problems))}"


def test_exceptions_are_still_needed():
    """An exception whose import is gone must be dropped from SRC_EXCEPTIONS."""
    for rel, name in SRC_EXCEPTIONS:
        assert name in _imports(ROOT / rel), f"stale exception: {rel} no longer imports {name}"


def test_every_package_is_in_pyproject():
    listed = set(tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["setuptools"]["packages"])
    found = {_module_of(p) for p in SRC_FILES if p.name == "__init__.py"}
    assert found == listed, f"missing in pyproject: {sorted(found - listed)}, not a package: {sorted(listed - found)}"
