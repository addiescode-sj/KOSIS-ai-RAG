"""Folder layout rules of the T2-41 refactor (plans/T2-41.md), checked by reading imports (ast), not by importing.

Adapted from Phase 1's T1-28 layout test. S0 held only the rules that pass on the old structure; each step adds
its own (OLD_PATHS grows with every moved module, STAGES / FRAME_ALLOWED are filled when the stage folders exist,
the langgraph rule narrows to src/kosis_rag/graph/ at S4). S1 adds: the old schemas/filters/fields/roles/vocab/text/
loader/vectorstore/model paths, shared/ imports only schemas, Phase 1 only through the listed files, chromadb and
sentence_transformers / torch in one adapter each, the Phase 1 copy headers, fields -> filters one way and an
__init__ without side effects. S2 adds: the old query / retriever paths and the names moved out of modules that stay
(OLD_NAMES), the stage order for parse/ and retrieve/ (a stage imports schemas, shared and the stages before it only;
never adapters, graph, legacy or the old top-level modules), and no self.last_* state in retrieve/.
Exceptions are listed with their reason and the step that removes them.
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
# S2: parse and retrieve; S3 adds rerank and gate.
STAGES = ("parse", "retrieve")
# What every stage may import from kosis_rag besides the stages before it (and itself).
STAGE_BASE = ("kosis_rag.schemas", "kosis_rag.shared")
# Old module paths nothing may import any more, added by the step that removes them (docs/migration.md).
OLD_PATHS: tuple[str, ...] = (
    # S1: moved to schemas/ shared/ adapters/ (kosis_rag.schemas is now a package that re-exports nothing)
    "kosis_rag.schemas", "kosis_rag.filters", "kosis_rag.fields", "kosis_rag.roles", "kosis_rag.vocab",
    "kosis_rag.text", "kosis_rag.loader", "kosis_rag.vectorstore", "kosis_rag.model",
    # S2: moved to parse/ retrieve/ legacy/retrievers.py shared/tokenize.py
    "kosis_rag.query", "kosis_rag.retriever",
)
# Names moved out of a module that stays (S2), checked like OLD_PATHS: no import of them, and the module no longer
# defines them at top level (docs/migration.md).
OLD_NAMES: dict[str, tuple[str, ...]] = {
    # S2: the online part of hq/index.py went to retrieve/hq.py, retrieve/constants.py, adapters/files.py,
    # shared/constants.py and legacy/retrievers.py; its CLI main stays until S6a
    "kosis_rag.hq.index": ("HqRetriever", "hq_bm25", "max_pool", "ckpt8", "collection_name", "read_questions",
                           "HQ_DIR", "COLLECTION_PREFIX", "DOC_FORMAT", "DESC_SUFFIX", "N_QUESTIONS"),
    "kosis_rag.index": ("COLLECTION_NAME",),  # S2: shared/constants.py
    "kosis_rag.pipeline": ("bm25_text",),  # S2: retrieve/bm25.py
}
# Packages that took an old module's name: their own modules are new paths, not the old one (kosis_rag.schemas.io).
REUSED_NAMES = ("kosis_rag.schemas",)
# Files allowed to import pandas / pyarrow. Placeholder: not enforced at S0 (no rule for frame libraries yet).
FRAME_ALLOWED: tuple[str, ...] = ()

# S1: shared/ imports only schemas and itself; never a stage, graph, adapters or llm (nor the old top-level modules
# they are made from: pipeline, retriever, rerank, query, chain ...).
SHARED = MODULE / "shared"
SHARED_ALLOWED = ("kosis_rag.schemas", "kosis_rag.shared")
# S1: Phase 1 (kosis_ai, kosis_chat) only through these files; the S3 llm/ copies empty the list.
PHASE1 = ("kosis_ai", "kosis_chat")
PHASE1_ALLOWED = ("src/kosis_rag/chain.py", "src/kosis_rag/graph.py", "src/kosis_rag/hcx.py", "src/kosis_rag/clova.py",
                  "src/kosis_rag/tools.py", "src/kosis_rag/hq/graph.py")
# S1: client libraries in one adapter each. chromadb: adapters/chroma.py (index.py and hq/index.py take a client as an
# argument, so they need no exception); sentence_transformers / torch: adapters/models.py (kure_embedder moved there).
LIBRARY_ALLOWED = {
    "chromadb": ("src/kosis_rag/adapters/chroma.py",),
    "sentence_transformers": ("src/kosis_rag/adapters/models.py",),
    "torch": ("src/kosis_rag/adapters/models.py",),
}
# Files that are copies of Phase 1 code (P2-D0): first line names the source at the pinned commit.
PHASE1_COPIES = {
    "src/kosis_rag/schemas/common.py": "# copied from Phase 1 src/kosis_ai/schemas.py@e248a45",
    "src/kosis_rag/shared/units.py": "# copied from Phase 1 src/kosis_ai/normalize.py@e248a45",
}

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


def _new_modules() -> tuple[str, ...]:
    """Modules of the packages in REUSED_NAMES, e.g. kosis_rag.schemas.io (found on disk)."""
    out = []
    for package in REUSED_NAMES:
        folder = SRC.joinpath(*package.split("."))
        out += [f"{package}.{p.stem}" for p in folder.glob("*.py") if p.stem != "__init__"]
    return tuple(out)


NEW_MODULES = _new_modules()


def _old(names: list[str]) -> list[str]:
    """Imported names on an old path. The bare name of a reused package (from kosis_rag.schemas import io) and its
    new modules are fine; a name taken from the package itself (from kosis_rag.schemas import TableRecord) is not.
    A name in OLD_NAMES (from kosis_rag.hq.index import HqRetriever) is old too."""
    moved = tuple(f"{module}.{name}" for module, names_ in OLD_NAMES.items() for name in names_)
    return [n for n in names if (_hits(n, OLD_PATHS) and n not in REUSED_NAMES and not _hits(n, NEW_MODULES))
            or n in moved]


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
    problems += _old(names)
    if path.is_relative_to(SHARED):
        problems += [n for n in names if _hits(n, ("kosis_rag",)) and not _hits(n, SHARED_ALLOWED)]
    if rel not in PHASE1_ALLOWED:
        problems += [n for n in names if _hits(n, PHASE1)]
    for library, allowed in LIBRARY_ALLOWED.items():
        if rel not in allowed:
            problems += [n for n in names if _hits(n, (library,))]
    stage = _stage_of(path)
    if stage is not None:  # S2: schemas <- shared <- parse <- retrieve <- ...; nothing else of kosis_rag
        allowed = STAGE_BASE + tuple(f"kosis_rag.{s}" for s in STAGES[: STAGES.index(stage) + 1])
        problems += [n for n in names if _hits(n, ("kosis_rag",)) and not _hits(n, allowed)]
    assert not problems, f"{rel}: imports break the layout rules: {sorted(set(problems))}"


def _stage_of(path: Path) -> str | None:
    """The stage folder a src file is in (parse, retrieve, ...), or None."""
    if not path.is_relative_to(MODULE):
        return None
    top = path.relative_to(MODULE).parts[0]
    return top if top in STAGES else None


def test_stage_folders_exist():
    for stage in STAGES:
        assert (MODULE / stage / "__init__.py").is_file(), stage


def test_moved_names_are_gone_from_their_old_module():
    """S2: a name moved out of a module that stays is no longer defined there (no re-export, P2-R6)."""
    for module, moved in OLD_NAMES.items():
        tree = ast.parse(SRC.joinpath(*module.split(".")).with_suffix(".py").read_text(encoding="utf-8"))
        defined = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                defined.add(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                defined |= {t.id for t in targets if isinstance(t, ast.Name)}
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                defined |= {(a.asname or a.name).split(".")[0] for a in node.names}
        assert not defined & set(moved), f"{module} still defines {sorted(defined & set(moved))}"


def test_retrieve_keeps_no_last_state():
    """S2 (P2-D17): retrieve/ returns what it removed / relaxed; no self.last_* (only legacy/retrievers.py keeps it)."""
    for path in _py_files(MODULE / "retrieve"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        hits = [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                and n.value.id == "self" and n.attr.startswith("last_")]
        assert not hits, f"{_rel(path)}: self state {hits}"


@pytest.mark.parametrize("path", CALLER_FILES, ids=_rel)
def test_callers_use_no_old_path(path: Path):
    problems = _old(_imports(path))
    assert not problems, f"{_rel(path)}: imports an old path: {sorted(set(problems))}"


def test_exceptions_are_still_needed():
    """An exception whose import is gone must be dropped from SRC_EXCEPTIONS / PHASE1_ALLOWED."""
    for rel, name in SRC_EXCEPTIONS:
        assert name in _imports(ROOT / rel), f"stale exception: {rel} no longer imports {name}"
    for rel in PHASE1_ALLOWED:
        assert any(_hits(n, PHASE1) for n in _imports(ROOT / rel)), f"stale exception: {rel} no longer imports Phase 1"


def test_old_modules_are_gone():
    """No shims (P2-R6): a moved module's file is deleted, so an old import fails instead of working on."""
    for old in OLD_PATHS:
        if old in REUSED_NAMES:
            folder = SRC.joinpath(*old.split("."))
            assert not folder.with_suffix(".py").exists() and (folder / "__init__.py").is_file(), old
            tree = ast.parse((folder / "__init__.py").read_text(encoding="utf-8"))
            assert all(isinstance(n, ast.Expr) for n in tree.body), f"{old}/__init__.py re-exports names"
        else:
            assert not SRC.joinpath(*old.split(".")).with_suffix(".py").exists(), f"{old} still exists"


def test_fields_and_filters_import_one_way():
    """S1: filters imports fields at the top; fields never imports filters (the old cycle needed a lazy import)."""
    fields, filters = SHARED / "fields.py", SHARED / "filters.py"
    assert not [n for n in _imports(fields) if _hits(n, ("kosis_rag.shared.filters",))]
    top = [n.module for n in ast.parse(filters.read_text(encoding="utf-8")).body if isinstance(n, ast.ImportFrom)]
    assert "kosis_rag.shared.fields" in top
    for path in (fields, filters):  # no function-level kosis_rag import left in either
        tree = ast.parse(path.read_text(encoding="utf-8"))
        nested = [n for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) for n in ast.walk(f)
                  if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("kosis_rag")]
        assert not nested, f"{_rel(path)}: function-level import {[n.module for n in nested]}"


@pytest.mark.parametrize("rel", sorted(PHASE1_COPIES))
def test_phase1_copies_name_their_source(rel: str):
    first = (ROOT / rel).read_text(encoding="utf-8").splitlines()[0]
    assert first == PHASE1_COPIES[rel]


def test_package_init_has_no_side_effects():
    """S1 (P2-D5): kosis_rag/__init__.py is a docstring only (it used to set KOSIS_WORKSPACE)."""
    tree = ast.parse((MODULE / "__init__.py").read_text(encoding="utf-8"))
    assert all(isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) for n in tree.body)


def test_every_package_is_in_pyproject():
    listed = set(tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["setuptools"]["packages"])
    found = {_module_of(p) for p in SRC_FILES if p.name == "__init__.py"}
    assert found == listed, f"missing in pyproject: {sorted(found - listed)}, not a package: {sorted(listed - found)}"
