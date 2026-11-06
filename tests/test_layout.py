"""Folder layout rules of the T2-41 refactor (plans/T2-41.md), checked by reading imports (ast), not by importing.

Adapted from Phase 1's T1-28 layout test. S0 held only the rules that pass on the old structure; each step adds
its own (OLD_PATHS grows with every moved module, STAGES / FRAME_ALLOWED are filled when the stage folders exist,
the langgraph rule narrows to src/kosis_rag/graph/ at S4). S1 adds: the old schemas/filters/fields/roles/vocab/text/
loader/vectorstore/model paths, shared/ imports only schemas, Phase 1 only through the listed files, chromadb and
sentence_transformers / torch in one adapter each, the Phase 1 copy headers, fields -> filters one way and an
__init__ without side effects. S2 adds: the old query / retriever paths and the names moved out of modules that stay
(OLD_NAMES), the stage order for parse/ and retrieve/ (a stage imports schemas, shared and the stages before it only;
never adapters, graph, legacy or the old top-level modules), and no self.last_* state in retrieve/. S3 adds: the old
rerank / hcx / clova paths (kosis_rag.rerank is now a package), the names chain.py gave to gate/, the stages rerank and
gate (rerank may also import llm/), llm/ importing only itself, no Phase 1 import anywhere in src (the llm/ copies),
urllib.request / openai / langchain_naver only in llm/, and the rerank prompts as package data. S4 adds: the old
graph / pipeline paths (kosis_rag.graph is now a package), the names chain.py and legacy/retrievers.py no longer
define, langgraph only in src/kosis_rag/graph/, no .env / environment read in src (the CLIs listed until S6a),
adapters/ imported only by graph/, api.py and llm/, and no positional load_resources(spec, options) call anywhere.
S5 adds: the old chain / tools paths (the answer moved to backend/app/respond, the search tool to legacy/tools.py),
backend importing first-party code only from kosis_rag (legacy included) and itself, get_stream_writer only in
src/kosis_rag/graph/ (and the backend), the refusal sentences named only in gate/texts.py and api.py, and the guide
prompt under backend/app/respond/prompts.
S6a adds: the old index / metadata / hq paths and the names that left shared/vocab.py and shared/tabletext.py (the
offline builds are the package kosis_rag_build now), the core rule that nothing in kosis_rag imports or names
kosis_rag_build, kosis_rag_build importing only kosis_rag.{schemas,shared,adapters,llm,rerank.stage1} and itself,
langgraph only in src/kosis_rag/graph/ and src/kosis_rag_build/graph/, the environment read only in
src/kosis_rag_build/cli.py, no src -> evals exception, both packages in pyproject and the hq prompt as package data.
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
BUILD = SRC / "kosis_rag_build"  # the offline DB build package (S6a)
# S6a: what kosis_rag_build may import from kosis_rag (besides itself): never parse, retrieve, gate, graph, api or
# legacy.
BUILD_ALLOWED = ("kosis_rag.schemas", "kosis_rag.shared", "kosis_rag.adapters", "kosis_rag.llm", "kosis_rag.rerank.stage1",
                 "kosis_rag_build")
# (file, imported name) pairs allowed against BUILD_ALLOWED, with the reason.
BUILD_EXCEPTIONS: set[tuple[str, str]] = set()  # S6a review: question-index naming moved to kosis_rag.shared.hqnames
# Code outside src that imports the module: an old path must be gone from all of it (P2-R6, no shims)
CALLERS = tuple(ROOT / d for d in ("backend", "evals", "scripts", "tests", "lab"))

# Stage folders of kosis_rag in data-flow order (a stage may import the ones before it, never the ones after it).
# S2: parse and retrieve; S3 adds rerank and gate.
STAGES = ("parse", "retrieve", "rerank", "gate")
# What every stage may import from kosis_rag besides the stages before it (and itself).
STAGE_BASE = ("kosis_rag.schemas", "kosis_rag.shared")
# S3: what a stage may import besides STAGE_BASE and the stages before it: rerank builds its LLM clients from llm/.
STAGE_EXTRA = {"rerank": ("kosis_rag.llm",)}
# S3: llm/ (transport and the Phase 1 client copies) imports only itself from kosis_rag.
LLM = MODULE / "llm"
LLM_ALLOWED = ("kosis_rag.llm",)
# Old module paths nothing may import any more, added by the step that removes them (docs/migration.md).
OLD_PATHS: tuple[str, ...] = (
    # S1: moved to schemas/ shared/ adapters/ (kosis_rag.schemas is now a package that re-exports nothing)
    "kosis_rag.schemas", "kosis_rag.filters", "kosis_rag.fields", "kosis_rag.roles", "kosis_rag.vocab",
    "kosis_rag.text", "kosis_rag.loader", "kosis_rag.vectorstore", "kosis_rag.model",
    # S2: moved to parse/ retrieve/ legacy/retrievers.py shared/tokenize.py
    "kosis_rag.query", "kosis_rag.retriever",
    # S3: moved to rerank/ (a package now: stage1 stage2 selectors hcx node ...), llm/ and legacy/rerankers.py
    "kosis_rag.rerank", "kosis_rag.hcx", "kosis_rag.clova",
    # S4: graph.py -> graph/ (a package now: search nodes_api spans) + schemas/state.py; pipeline.py ->
    # schemas/resources.py, adapters/loader.py, legacy/pipeline.py (SearchSettings / from_env deleted)
    "kosis_rag.graph", "kosis_rag.pipeline",
    # S5: chain.py -> backend/app/respond/ (texts patterns prompts verify render node state graph legacy),
    # kosis_rag.legacy.search; tools.py -> kosis_rag.legacy.tools (search_tables) + backend.app.respond.legacy
    "kosis_rag.chain", "kosis_rag.tools",
    # S6a: the offline builds -> kosis_rag_build (index.py -> chroma.py, metadata.py -> tables.py, hq/ -> hq/ +
    # schemas.py + graph/hq_build.py + cli.py; kosis_rag.hq is gone as a whole)
    "kosis_rag.index", "kosis_rag.metadata", "kosis_rag.hq",
)
# Names moved out of a module that stays (S2), checked like OLD_PATHS: no import of them, and the module no longer
# defines them at top level (docs/migration.md).
OLD_NAMES: dict[str, tuple[str, ...]] = {
    # (S2: the online part of hq/index.py and index.COLLECTION_NAME moved out; S6a deleted both modules, so they are
    # in OLD_PATHS now)
    # (S3/S4: the names chain.py gave to gate/, llm/providers.py and backend/app/config.py; S5 deleted chain.py, so it
    # is in OLD_PATHS now)
    # S4: the environment is the caller's (backend/app/config.py): connect takes ChromaConfig
    "kosis_rag.legacy.retrievers": ("env_chroma_config",),
    # S6a: the builders and CLIs moved to kosis_rag_build (sigungu.py, cli.py sigungu / text)
    "kosis_rag.shared.vocab": ("build_sigungu", "main", "SIGUNGU_SOURCE"),
    "kosis_rag.shared.tabletext": ("main",),
}
# Packages that took an old module's name: their own modules are new paths, not the old one (kosis_rag.schemas.io).
REUSED_NAMES = ("kosis_rag.schemas", "kosis_rag.rerank", "kosis_rag.graph")
# Files allowed to import pandas / pyarrow. Placeholder: not enforced at S0 (no rule for frame libraries yet).
FRAME_ALLOWED: tuple[str, ...] = ()

# S1: shared/ imports only schemas and itself; never a stage, graph, adapters or llm (nor the old top-level modules
# they are made from: pipeline, retriever, rerank, query, chain ...).
SHARED = MODULE / "shared"
SHARED_ALLOWED = ("kosis_rag.schemas", "kosis_rag.shared")
# S1: Phase 1 (kosis_ai, kosis_chat) only through listed files; S3: the llm/ copies emptied the list (P2-D0: no Phase 1
# import anywhere in src).
PHASE1 = ("kosis_ai", "kosis_chat")
PHASE1_ALLOWED: tuple[str, ...] = ()
# S1: client libraries in one adapter each. chromadb: adapters/chroma.py (kosis_rag_build/chroma.py and
# hq/questions_index.py take a client as an argument and open it through the adapter, so they need no exception); sentence_transformers / torch: adapters/models.py (kure_embedder moved there).
# S3: the LLM transports in llm/ (an entry ending in "/" is a folder). chain.py builds its ChatClovaX through
# llm.providers.clova_chat, so it needs no exception.
LIBRARY_ALLOWED = {
    "chromadb": ("src/kosis_rag/adapters/chroma.py",),
    "sentence_transformers": ("src/kosis_rag/adapters/models.py",),
    "torch": ("src/kosis_rag/adapters/models.py",),
    "urllib.request": ("src/kosis_rag/llm/",),
    "openai": ("src/kosis_rag/llm/openai.py",),
    "langchain_naver": ("src/kosis_rag/llm/providers.py",),
}
# Files that are copies of Phase 1 code (P2-D0): first line names the source at the pinned commit.
PHASE1_COPIES = {
    "src/kosis_rag/schemas/common.py": "# copied from Phase 1 src/kosis_ai/schemas.py@e248a45",
    "src/kosis_rag/shared/units.py": "# copied from Phase 1 src/kosis_ai/normalize.py@e248a45",
    "src/kosis_rag/llm/base.py": "# copied from Phase 1 kosis_chat/llm/model.py@e248a45",
    "src/kosis_rag/llm/hcx.py": "# copied from Phase 1 kosis_chat/llm/model.py@e248a45",
}
# S3 (P2-D22): prompt files that must ship as package data: package -> glob in [tool.setuptools.package-data]
PACKAGE_DATA = {"kosis_rag.rerank": ["prompts/*.md"], "kosis_rag_build.hq": ["prompts/*.md"]}

# S5: the token stream writer only where the graph runs: src/kosis_rag/graph/ (the module's nodes write none since
# guide moved out) and the backend (backend/app/respond/node.py guide); checked for src only (backend may use it).
STREAM_WRITER = "langgraph.config.get_stream_writer"
STREAM_WRITER_ALLOWED = ("src/kosis_rag/graph/",)
# S5: the refusal sentences are named (in any form, docstrings included) only where they are defined and read.
REFUSAL_NAMES = ("REFUSAL_TEXT", "REFUSALS")
REFUSAL_FILES = {"src/kosis_rag/gate/texts.py", "src/kosis_rag/api.py"}
# S5: backend imports first-party code from kosis_rag (legacy included) and itself only; never Phase 1, the offline
# build package, evals, scripts, lab or tests.
BACKEND = ROOT / "backend"
BACKEND_FORBIDDEN = ("kosis_ai", "kosis_chat", "kosis_rag_build", "evals", "scripts", "lab", "tests")
# S5: the guide prompt, a backend file (not package data of src), read through importlib.resources.
GUIDE_PROMPT = ROOT / "backend" / "app" / "respond" / "prompts" / "guide-v1.md"

# langgraph: only the graph folders (S4: src/kosis_rag/graph/; S6a: src/kosis_rag_build/graph/, no exception left).
# "/" at the end = a folder.
LANGGRAPH_ALLOWED = ("src/kosis_rag/graph/", "src/kosis_rag_build/graph/")
# S4 (§0.6): no .env and no environment read in src (dotenv import, os.environ, os.getenv). S6a: only the build CLI
# reads the shell environment (no .env), as the fallback of its arguments (plans/T2-41.md S6a: kosis_rag_build/cli.py).
ENV_ALLOWED: tuple[str, ...] = ()  # 미결 3: the build env is read by scripts/kosis_build.py, outside src
# S4: adapters/ (files, Chroma, models, loader) is imported only by graph/, api.py and llm/ (and itself).
ADAPTERS = "kosis_rag.adapters"
ADAPTERS_IMPORTERS = ("src/kosis_rag/adapters/", "src/kosis_rag/graph/", "src/kosis_rag/api.py", "src/kosis_rag/llm/")
ADAPTERS_EXCEPTIONS = {
    # (S6a: the offline CLIs left for kosis_rag_build, which may import kosis_rag.adapters)
    # The old retriever classes' connect() opens its collection and embedder (evals, lab and adapters.loader call it).
    "src/kosis_rag/legacy/retrievers.py",
}
# S4: callers whose old import is left on purpose, with the reason (checked by test_exceptions_are_still_needed).
CALLER_EXCEPTIONS = {
    # evals are not modified by the refactor (user decision 11/05 21:17): SearchSettings is gone and the L1 runner's
    # _staged_settings fails when called until the evals owner rewrites it (refactor/phase2/03 §2).
    ("evals/runners/retrieval.py", "kosis_rag.pipeline"),
    ("evals/runners/retrieval.py", "kosis_rag.pipeline.SearchSettings"),
}
# S4: files allowed a positional load_resources(...) call (spec and options are keyword-only), with the reason.
LOAD_RESOURCES_POSITIONAL = {
    # evals construction lines not modified (03 §2): the call fails with TypeError until the evals owner fixes it
    "evals/runners/retrieval.py",
}
# Top-level packages src must never import (entry points and experiments depend on src, not the other way).
NOT_FROM_SRC = ("backend", "evals", "scripts", "lab", "tests")
# (file, imported name) pairs allowed against NOT_FROM_SRC, with the reason.
SRC_EXCEPTIONS: set[tuple[str, str]] = set()  # S6a: the hq build's evals.ops cost line is gone (tokens in build.json)


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
    A name in OLD_NAMES (from kosis_rag.shared.vocab import build_sigungu) is old too."""
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
    if not _allowed(rel, LANGGRAPH_ALLOWED):
        problems += [n for n in names if _hits(n, ("langgraph",))]
    if not _allowed(rel, STREAM_WRITER_ALLOWED):
        problems += [n for n in names if n == STREAM_WRITER]
    if path.is_relative_to(MODULE) and not _allowed(rel, ADAPTERS_IMPORTERS) and rel not in ADAPTERS_EXCEPTIONS:
        problems += [n for n in names if _hits(n, (ADAPTERS,))]
    problems += [n for n in names if _hits(n, NOT_FROM_SRC) and (rel, n) not in SRC_EXCEPTIONS]
    if path.is_relative_to(MODULE):  # the search module never imports the offline build package
        problems += [n for n in names if _hits(n, ("kosis_rag_build",))]
    if path.is_relative_to(BUILD):  # S6a: the build package uses the module's lower layers only
        problems += [n for n in names if _hits(n, ("kosis_rag", "kosis_rag_build")) and not _hits(n, BUILD_ALLOWED)
                     and (rel, n) not in BUILD_EXCEPTIONS]
    problems += _old(names)
    if path.is_relative_to(SHARED):
        problems += [n for n in names if _hits(n, ("kosis_rag",)) and not _hits(n, SHARED_ALLOWED)]
    if path.is_relative_to(LLM):
        problems += [n for n in names if _hits(n, ("kosis_rag",)) and not _hits(n, LLM_ALLOWED)]
    if rel not in PHASE1_ALLOWED:
        problems += [n for n in names if _hits(n, PHASE1)]
    for library, allowed in LIBRARY_ALLOWED.items():
        if not _allowed(rel, allowed):
            problems += [n for n in names if _hits(n, (library,))]
    stage = _stage_of(path)
    if stage is not None:  # S2: schemas <- shared <- parse <- retrieve <- rerank <- gate; nothing else of kosis_rag
        allowed = (STAGE_BASE + tuple(f"kosis_rag.{s}" for s in STAGES[: STAGES.index(stage) + 1])
                   + STAGE_EXTRA.get(stage, ()))
        problems += [n for n in names if _hits(n, ("kosis_rag",)) and not _hits(n, allowed)]
    assert not problems, f"{rel}: imports break the layout rules: {sorted(set(problems))}"


def _allowed(rel: str, allowed: tuple[str, ...]) -> bool:
    """rel is one of the listed files or inside a listed folder (an entry ending in "/")."""
    return any(rel.startswith(a) if a.endswith("/") else rel == a for a in allowed)


def _env_reads(path: Path) -> list[str]:
    """What in the file reads .env or the environment: a dotenv import, os.environ / os.getenv, or a from-import of
    environ / getenv from os."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits = [n for n in _imports(path) if _hits(n, ("dotenv",)) or n in ("os.environ", "os.getenv")]
    hits += [f"os.{n.attr}" for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
             and n.value.id == "os" and n.attr in ("environ", "getenv")]
    return hits


@pytest.mark.parametrize("path", _py_files(MODULE, BUILD), ids=_rel)
def test_src_reads_no_env(path: Path):
    """S4 (§0.6): the module reads no .env and no environment; the caller declares the settings. S6a: the build
    package neither, except its CLI (shell environment only, never .env)."""
    rel = _rel(path)
    if rel in ENV_ALLOWED:
        return
    assert not _env_reads(path), f"{rel}: reads .env or the environment: {sorted(set(_env_reads(path)))}"


def test_env_exceptions_are_still_needed():
    for rel in ENV_ALLOWED:
        assert _env_reads(ROOT / rel), f"stale exception: {rel} no longer reads the environment"
        assert not [n for n in _imports(ROOT / rel) if _hits(n, ("dotenv",))], f"{rel} reads .env"


def _positional_load_resources(path: Path) -> list[int]:
    """Lines of calls load_resources(<positional>, ...) in the file (by name or as an attribute)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and n.args
            and ((isinstance(n.func, ast.Name) and n.func.id == "load_resources")
                 or (isinstance(n.func, ast.Attribute) and n.func.attr == "load_resources"))]


def test_load_resources_is_called_by_keyword_only():
    """S4 (11/05 external review P1-2b): spec and options are passed by name everywhere (the signature makes them
    keyword-only, and search_config_from_env returns options first, so a positional call would swap them)."""
    import inspect

    from kosis_rag.adapters.loader import load_resources

    params = inspect.signature(load_resources).parameters
    assert all(params[name].kind is inspect.Parameter.KEYWORD_ONLY for name in ("spec", "options", "llm"))
    found = {_rel(p): lines for p in _py_files(MODULE, BUILD, *CALLERS) if (lines := _positional_load_resources(p))}
    assert set(found) == LOAD_RESOURCES_POSITIONAL, f"positional load_resources calls: {found}"


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
    problems = [n for n in _old(_imports(path)) if (_rel(path), n) not in CALLER_EXCEPTIONS]
    assert not problems, f"{_rel(path)}: imports an old path: {sorted(set(problems))}"


def test_exceptions_are_still_needed():
    """An exception whose import is gone must be dropped from SRC_EXCEPTIONS / PHASE1_ALLOWED."""
    for rel, name in SRC_EXCEPTIONS:
        assert name in _imports(ROOT / rel), f"stale exception: {rel} no longer imports {name}"
    for rel in PHASE1_ALLOWED:
        assert any(_hits(n, PHASE1) for n in _imports(ROOT / rel)), f"stale exception: {rel} no longer imports Phase 1"
    for rel, name in CALLER_EXCEPTIONS:
        assert name in _imports(ROOT / rel), f"stale exception: {rel} no longer imports {name}"
    for rel in ADAPTERS_EXCEPTIONS:
        assert any(_hits(n, (ADAPTERS,)) for n in _imports(ROOT / rel)), f"stale exception: {rel} imports no adapter"
    for rel, name in BUILD_EXCEPTIONS:
        assert name in _imports(ROOT / rel), f"stale exception: {rel} no longer imports {name}"


def test_old_modules_are_gone():
    """No shims (P2-R6): a moved module's file is deleted, so an old import fails instead of working on."""
    assert not (MODULE / "hq").exists(), "kosis_rag/hq/ is still there (S6a: kosis_rag_build/hq)"
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


def test_rerank_prompts_are_package_data():
    """S3 (P2-D22): every prompt .md under a package is declared as package data, so a wheel ships it."""
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["setuptools"].get(
        "package-data", {})
    assert {k: sorted(v) for k, v in declared.items()} == {k: sorted(v) for k, v in PACKAGE_DATA.items()}
    for package, globs in PACKAGE_DATA.items():
        folder = SRC.joinpath(*package.split("."))
        assert all(list(folder.glob(g)) for g in globs), f"{package}: no file matches {globs}"
    found = {p.relative_to(SRC).as_posix() for p in SRC.rglob("prompts/*.md")}
    covered = {p.relative_to(SRC).as_posix() for package, globs in PACKAGE_DATA.items()
               for g in globs for p in SRC.joinpath(*package.split(".")).glob(g)}
    assert found == covered, f"prompt files not declared as package data: {sorted(found - covered)}"


@pytest.mark.parametrize("path", _py_files(BACKEND), ids=_rel)
def test_backend_imports_only_kosis_rag_and_itself(path: Path):
    """S5: the app layer depends on the search module (kosis_rag, legacy included), never the other way, and on no
    Phase 1, build, eval, script, lab or test code."""
    problems = [n for n in _imports(path) if _hits(n, BACKEND_FORBIDDEN)]
    assert not problems, f"{_rel(path)}: imports {sorted(set(problems))}"


def test_refusal_sentences_are_named_only_in_gate_texts_and_api():
    """S5: the module's refusal sentences are defined in gate/texts.py and read in api.to_output (SearchOutput
    .refusal_text); the app reads them from the output (backend/app/respond)."""
    found = {_rel(p) for p in _py_files(MODULE, BUILD)
             if any(name in p.read_text(encoding="utf-8") for name in REFUSAL_NAMES)}
    assert found == REFUSAL_FILES


def test_the_answer_lives_in_the_backend():
    """S5: chain.py and tools.py are gone; the guide prompt is a backend file and src has no prompt for it."""
    assert GUIDE_PROMPT.is_file() and not list(SRC.rglob("guide-*.md"))
    assert (BACKEND / "app" / "respond" / "__init__.py").is_file()


def test_the_module_never_names_the_build_package():
    """S6a core rule (plans/T2-41.md Done: grep -rn kosis_rag_build src/kosis_rag gives nothing): not in an import,
    not in a docstring or message either, so the search module reads as complete on its own."""
    found = [_rel(p) for p in _py_files(MODULE) if "kosis_rag_build" in p.read_text(encoding="utf-8")]
    found += [p.relative_to(ROOT).as_posix() for p in MODULE.rglob("*.md") if "kosis_rag_build" in p.read_text("utf-8")]
    assert not found


def test_importing_the_module_loads_no_build_code():
    """S6a: python -c "import kosis_rag.api; import sys; assert 'kosis_rag_build' not in sys.modules"."""
    import subprocess
    import sys

    code = ("import sys, kosis_rag.api, kosis_rag.graph.search, kosis_rag.adapters.loader; "
            "bad = sorted(m for m in sys.modules if m.split('.')[0] == 'kosis_rag_build'); assert not bad, bad")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert done.returncode == 0, done.stderr[-2000:]


def test_the_build_package_is_complete():
    """S6a: the files of plans/T2-41.md S6a (folder tree) are there, the graphs in graph/ and the node bodies outside."""
    for rel in ("cli.py", "schemas.py", "tables.py", "sigungu.py", "chroma.py", "graph/index_build.py",
                "graph/hq_build.py", "hq/prompt.py", "hq/rules.py", "hq/patterns.py", "hq/constants.py", "hq/nodes.py",
                "hq/report.py", "hq/questions_index.py", "hq/prompts/hq-v1.md"):
        assert (BUILD / rel).is_file(), rel
