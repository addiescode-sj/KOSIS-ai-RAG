"""T2-41 S4: scripts/parity_l1.py builds its preset graph from the new settings objects (not through evals), and its
config fingerprint keeps the S0 keys and values (compare refuses dumps whose fingerprints differ).

The S0 fingerprint function is copied below verbatim (it read resources.settings, the old settings object); it runs on an
old-shaped object holding the same fakes, and must give what the S4 function gives on the new resources.
No Chroma, no model: the fakes of tests/test_graph.py.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from kosis_rag.adapters.loader import load_resources
from kosis_rag.schemas.resources import IndexSpec, SearchOptions
from tests.test_graph import KEYS, RECORDS, CatalogRetriever, _pick, _stage1

pytestmark = pytest.mark.offline
ROOT = Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location("parity_l1", ROOT / "scripts" / "parity_l1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = _tool()


def s0_config_fingerprint(preset: str, resources: Any) -> dict[str, Any]:
    """scripts/parity_l1.py config_fingerprint at S0 (resources.settings = the old settings object), copied verbatim."""
    file_sha256, _collection = TOOL.file_sha256, TOOL._collection
    s = resources.settings
    dense_name, dense_meta = _collection(resources.dense)
    hq_name, _ = _collection(resources.hq)
    weights = {"dense": 1.0, "bm25": 1.0}
    if resources.hq is not None:
        weights["hq"] = float(s.hq_weight)
    if resources.hq_bm25 is not None:
        weights["hq_bm25"] = float(s.hq_weight)
    stage1 = resources.stage1_score is not None
    stage2 = resources.select is not None
    tables = Path(s.tables)
    return {
        "preset": preset,
        "tables_sha256": file_sha256(tables) if tables.is_file() else None,
        "collection": dense_name,
        "embed": dense_meta.get("embed_ckpt") or dense_meta.get("embed_model"),
        "collection_doc_format": dense_meta.get("doc_format"),
        "text_format": resources.text_format,
        "cond_filter": bool(s.cond_filter),
        "pool": int(s.pool),
        "lists": list(weights),
        "weights": weights,
        "stages": {"dense": True, "bm25": True, "hq": resources.hq is not None,
                   "hq_bm25": resources.hq_bm25 is not None, "rerank1": stage1, "rerank2": stage2},
        "hq_build": s.hq_build if resources.hq is not None else None,
        "hq_collection": hq_name,
        "rerank1_ckpt": s.rerank_ckpt if stage1 else None,
        "rerank1_mode": s.rerank_mode if stage1 else None,
        "rerank1_n": int(s.rerank_n) if stage1 else None,
        "rerank2": resources.stage2_name if stage2 else None,
        "rerank2_k": int(s.stage2_k) if stage2 else None,
    }


class Collection(SimpleNamespace):
    """A Chroma collection stand-in: name and metadata, as _collection reads them."""


class NamedDense(CatalogRetriever):
    collection: Any = None
    doc_format: str | None = "n2-L60"


def _old_shaped(resources) -> SimpleNamespace:
    """The same resources as the S0 code saw them: one `settings` object with the old settings fields."""
    o, spec = resources.options, resources.spec
    settings = SimpleNamespace(**o.model_dump(), tables=spec.tables or TOOL_TABLES, collection=spec.chroma.collection,
                               text_format=spec.text_format, embed_ckpt=spec.embed_ckpt, rerank_ckpt=spec.rerank_ckpt,
                               device=spec.device, hq_build=spec.hq_build)
    return SimpleNamespace(settings=settings, **{k: getattr(resources, k) for k in (
        "dense", "hq", "hq_bm25", "stage1_score", "select", "text_format", "stage2_name")})


TOOL_TABLES = ROOT / "data" / "tables.jsonl"  # the old settings' tables default (= adapters.files.TABLES)


@pytest.mark.parametrize("hq", [False, True])
@pytest.mark.parametrize("stages", [False, True])
@pytest.mark.parametrize("tables", [None, "file"])
def test_fingerprint_keys_and_values_stay_the_s0_ones(tmp_path, hq, stages, tables):
    path = None
    if tables:
        path = tmp_path / "tables.jsonl"
        path.write_text("\n".join(r.model_dump_json() for r in RECORDS) + "\n", encoding="utf-8")
    dense = NamedDense(keys=KEYS, collection=Collection(name="kosis_a_bgeft_x", metadata={
        "embed_ckpt": "data/models/bge-m3-seen-e2.pt", "doc_format": "n2-L60"}))
    hq_dense = NamedDense(keys=[KEYS[-1]], collection=Collection(name="kosis_hq__x", metadata={}))
    spec = IndexSpec(tables=path, hq_build="hq-x" if hq else None, rerank_ckpt="r.pt" if stages else None)
    options = SearchOptions(cond_filter=True, stage2="none", hq_weight=0.5, hq_bm25=hq)
    resources = load_resources(spec=spec, options=options, records=RECORDS, dense=dense, hq=hq_dense,
                               hq_bm25=CatalogRetriever(keys=KEYS) if hq else None, sigungu={},
                               stage1_score=_stage1 if stages else None, select=_pick() if stages else None)
    new = TOOL.config_fingerprint("staged", resources)
    old = s0_config_fingerprint("staged", _old_shaped(resources))
    assert new == old
    assert list(new) == list(old)
    assert new["stages"]["hq"] is hq and new["stages"]["rerank1"] is stages
    if tables:
        assert new["tables_sha256"] == TOOL.file_sha256(path)


def test_preset_graph_is_built_from_the_new_settings_objects(monkeypatch):
    """build_preset_graph: search_config_from_shell(**STAGED[preset]) -> load_resources(spec=, options=, llm=) -> graph;
    the empty rerank_ckpt of a preset means stage 1 off, as the runner's _staged_settings read it at S0."""
    import kosis_rag.adapters.loader as loader

    for key in ("RERANK_CKPT", "STAGE2", "COND_FILTER", "HQ_BUILD", "CHROMA_COLLECTION"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RERANK_CKPT", "data/models/r.pt")
    seen = []

    def fake_load(**kw):
        seen.append(kw)
        return load_resources(**kw, records=RECORDS, dense=CatalogRetriever(keys=KEYS), sigungu={},
                              stage1_score=_stage1 if kw["spec"].rerank_ckpt else None)

    monkeypatch.setattr(loader, "load_resources", fake_load)
    assert TOOL.preset_overrides("hybrid+cond") == {"cond_filter": True, "rerank_ckpt": None, "stage2": "none"}
    _, plain = TOOL.build_preset_graph("hybrid+cond")
    graph, staged = TOOL.build_preset_graph("staged")
    assert set(seen[0]) == {"spec", "options", "llm"}  # by name only
    assert plain.spec.rerank_ckpt is None and plain.stage1_score is None and plain.options.cond_filter
    assert staged.spec.rerank_ckpt == "data/models/r.pt" and staged.stage1_score is not None
    assert "rerank1" in graph.get_graph().nodes and "rerank2" not in graph.get_graph().nodes
