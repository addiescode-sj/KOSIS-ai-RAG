"""The table index (T2-24, versioned A builds in T2-05): one Chroma document per table, built from the table metadata
only (tables.jsonl). T2-41 S6a: was index.py of the search package (its CLI is `python -m kosis_rag_build.cli chroma`, the graph
graph/index_build.build_chroma_graph over the nodes at the end of this file).

Values are not loaded. Embedding every row (617M) was measured at ~72 days and ~47 GB of vectors on an M3,
and the values are fetched from Parquet with pandas as in stage 1, so the search document only carries what
helps find a table: name, survey, organization, real period range, cycle, menu paths, items and labels.

Embeddings are computed here and passed to Chroma (`upsert(embeddings=...)`), so the collection never relies on
Chroma's default (English) embedding function; queries must be embedded with the same model.
`TableRecord.search_text` (all labels, up to 22,929 chars) stays the BM25 document; the embedding document below
summarizes region classes by level and caps the other labels per class so every table fits the model's input.

    python -m kosis_rag_build.cli chroma
    python -m kosis_rag_build.cli chroma --model bge-m3-ft --ckpt data/models/bge-m3-seen-e2.pt   # T2-06 collection

`--model bge-m3-ft` loads the fine-tuned bge-m3 (kosis_rag.adapters.models) and embeds the text format it was trained on
(kosis_rag.shared.tabletext, `--text-format`) into its own collection (FT_COLLECTION_PREFIX); the checkpoint id goes into the collection metadata
so `ChromaTableRetriever.connect` embeds questions with the same weights. Metadata (filters) is the same either way.
EMBED_MODEL / kure_embedder live in kosis_rag.adapters.models, region_summary in kosis_rag.shared.tabletext and the
COLLECTION_NAME pattern in kosis_rag.shared.constants (the search reads them too).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, NamedTuple

from kosis_rag.adapters import files, models
from kosis_rag.adapters.models import EMBED_MODEL, kure_embedder
from kosis_rag.schemas.catalog import TableRecord
from kosis_rag.shared import constants
from kosis_rag.shared import tabletext as text
from kosis_rag.shared.catalog import _META_FIELDS
from kosis_rag.shared.fields import table_fields
from kosis_rag.shared.roles import table_roles
from kosis_rag.shared.tabletext import region_summary
from kosis_rag_build.schemas import IndexBuildState

DOC_FORMAT = "meta-region-items-head-measure3"  # T2-30 KOS-109: names that list or summarize what is counted; before: measure2 (KOS-97), measure (T2-27)
# Per-table fields as list metadata (T2-30 KOS-101). Bump it whenever to_metadata changes: it is part of the build name.
META_FORMAT = "fields2"  # fields1: A3 (KOS-101); fields2 adds f_implied, f_fixed, f_measure_keys (KOS-109)
COLLECTION_PREFIX = "kosis_kure__"  # T2-30: content-named builds; before T2-30 "kosis_a_kure_<YYYYMMDDHHMM>" (kept on servers)
FT_COLLECTION_PREFIX = "kosis_a_bgeft_"  # T2-06: fine-tuned bge-m3 over a kosis_rag.shared.tabletext format
LABEL_CAP = 100  # per non-region class
UNIT_SUFFIX = re.compile(r"\s*\(([^()]*)\)\s*$")


def clean_item(item: str) -> tuple[str, str | None]:
    """'혼인건수 (건)' -> ('혼인건수', '건'), '가구_계' -> ('가구 계', None)."""
    m = UNIT_SUFFIX.search(item)
    name, unit = (item[: m.start()], m.group(1).strip()) if m else (item, None)
    name = name.replace("_", " ").replace("ㆍ", "·").replace("–", "-")
    return re.sub(r"\s+", " ", name).strip(), unit


def to_document(record: TableRecord) -> str:
    """Embedding text (T2-25 1+2+3): a summary line first, then listing facts, the real period range, cleaned items
    with their units on one line, region classes as level counts and the other classes as capped labels.

    Region labels were 81% of all labels and, listed, made a table read like a list of place names; the level
    summary says which regions a table covers in a few tokens. Exact place names stay in `search_text` (BM25).
    The summary line also names what the table counts (T2-27), so "출생아 수" is not read off "출생아수별 부인".
    """
    info = record.info
    cleaned = [clean_item(i) for i in record.items]
    items = list(dict.fromkeys(name for name, _ in cleaned))
    units = list(dict.fromkeys(unit for _, unit in cleaned if unit))
    regions = " / ".join(region_summary(c.levels) for c in record.classes if c.levels) or "지역 구분 없음"
    measure, _ = table_roles(record)
    counted = f" | 세는 대상: {measure}" if measure else ""
    lines = [
        f"{info.table_name}{counted} | {'·'.join(record.prd_se)} {record.start}~{record.end} | 지역 단위: {regions} | 항목: {', '.join(items[:8])}",
        f"통계표: {info.table_name}",
        f"조사명: {info.survey_name} | 작성기관: {info.organization}" + (f" | 주제: {record.topic}" if record.topic else ""),
        f"수록기간: {record.start}~{record.end} ({'·'.join(record.prd_se)}), 목록 표기 {info.period}",
        "분류 경로: " + " / ".join(record.menu_paths),
        "항목: " + ", ".join(items),
    ]
    if units:
        lines.append("단위: " + ", ".join(units))
    for c in record.classes:
        if c.levels:
            lines.append(f"{c.name}: 지역 단위 {region_summary(c.levels)}")
            continue
        shown = c.labels[:LABEL_CAP]
        more = f" 외 {c.n_labels - len(shown)}개" if c.n_labels > len(shown) else ""
        lines.append(f"{c.name}: " + ", ".join(shown) + more)
    return "\n".join(lines)


def to_metadata(record: TableRecord) -> dict:
    """Chroma metadata: the listing scalars, plus the per-table fields (T2-30) as `f_*` lists for `$contains` filters.

    Keys of shared.catalog._META_FIELDS stay as they are, so shared.catalog.to_candidate works unchanged. Chroma rejects None and
    empty lists, so a field the table does not have is left out; a table with no administrative class gets
    f_national instead of f_regions. No load time per document, so reloading a build gives the same metadata.
    """
    info = record.info
    f = table_fields(record)
    meta = {name: getattr(info, name) for name in _META_FIELDS}
    meta |= {
        "start": record.start, "end": record.end,
        "cycle": "|".join(record.cycle), "prd_se": "|".join(record.prd_se),
        "topic": record.topic, "unit": record.unit, "n_labels": sum(c.n_labels for c in record.classes),
        "part": "core",
        "f_years": f.available_years, "f_cycle": f.cycle, "f_regions": f.regions, "f_national": f.regions is None,
        "f_region_level": f.region_level, "f_measure": f.measure, "f_unit": f.measure_unit, "f_dims": f.dimension_keys,
        "f_implied": f.implied_keys, "f_fixed": f.fixed, "f_measure_keys": f.measure_keys,
    }
    return {k: v for k, v in meta.items() if v is not None and v != []}


def collection_name(data_sha256: str) -> str:
    """The build name from what went into it: same tables.jsonl, document and metadata format -> same name."""
    digest = hashlib.sha256(f"{data_sha256}|{DOC_FORMAT}|{META_FORMAT}".encode()).hexdigest()[:8]
    name = f"{COLLECTION_PREFIX}{DOC_FORMAT}__{digest}"
    if not constants.COLLECTION_NAME.match(name) or ".." in name:
        raise ValueError(f"{name!r} is not a valid Chroma collection name; DOC_FORMAT must use a-z, 0-9, - and _ only")
    return name


def build_record(index_version: str, data_sha256: str, n_docs: int, dim: int, max_tokens: int, over_limit: int,
                 embed_model: str = EMBED_MODEL, doc_format: str = DOC_FORMAT, embed_ckpt: str | None = None) -> dict:
    """What the collection was built from; stored as collection metadata so every teammate sees the same record.
    index_version is when it was last loaded (a reload of the same build updates it)."""
    record = {
        "index_version": index_version, "data_sha256": data_sha256, "doc_format": doc_format, "meta_format": META_FORMAT,
        "embed_model": embed_model,
        "dim": dim, "normalized": True, "hnsw:space": "cosine", "n_docs": n_docs, "max_doc_tokens": max_tokens,
        "over_token_limit": over_limit,
    }
    if embed_ckpt:
        record["embed_ckpt"] = embed_ckpt
    return record


def load(client, ids: list[str], docs: list[str], metas: list[dict], embeddings: list[list[float]],
         name: str, record: dict | None = None) -> int:
    """Upsert by table key and drop ids that are no longer in the input, so reloading gives the same collection."""
    collection = client.get_or_create_collection(
        name, embedding_function=None,
        metadata=record or {"embed_model": models.EMBED_MODEL, "doc_format": DOC_FORMAT, "dim": len(embeddings[0]), "hnsw:space": "cosine"},
    )
    if record and collection.metadata != record:  # get_or_create keeps the first loader's record: refresh it
        collection.modify(metadata={k: v for k, v in record.items() if not k.startswith("hnsw:")})
    step = client.get_max_batch_size()
    for i in range(0, len(ids), step):
        collection.upsert(ids=ids[i:i + step], documents=docs[i:i + step], metadatas=metas[i:i + step],
                          embeddings=embeddings[i:i + step])
    stale = sorted(set(collection.get(include=[])["ids"]) - set(ids))
    if stale:
        collection.delete(ids=stale)
    return collection.count()


def write_build(path: Path, *, name: str, ids: list[str], docs: list[str], metas: list[dict], vectors: list,
                record: dict) -> None:
    """One build as a .npz: vectors as float32, everything else as one JSON string (so reading never unpickles)."""
    import numpy as np

    text = json.dumps({"name": name, "ids": ids, "docs": docs, "metas": metas, "record": record}, ensure_ascii=False)
    np.savez_compressed(path, vectors=np.asarray(vectors, dtype=np.float32), text=np.array(text))


def read_build(path: Path) -> dict:
    """The build write_build wrote: {name, ids, docs, metas, record, vectors (list of lists)}; ValueError if they disagree."""
    import numpy as np

    with np.load(path, allow_pickle=False) as f:
        build = json.loads(str(f["text"]))
        build["vectors"] = f["vectors"].tolist()
    if not len(build["ids"]) == len(build["docs"]) == len(build["metas"]) == len(build["vectors"]):
        raise ValueError(f"{path}: ids, docs, metas and vectors have different lengths")
    return build


def list_builds(client, active: str | None = None, out: Callable[[str], None] = print) -> int:
    """Every table collection on the server with what it was built from; * marks `active` (CHROMA_COLLECTION, the
    one the search uses). In: client = a Chroma client (kosis_rag.adapters.chroma.chroma_client). Side effects:
    reads the server, prints one line per build. Node: none (cli chroma --list)."""
    for c in sorted(client.list_collections(), key=lambda c: (c.metadata or {}).get("index_version") or ""):
        m = c.metadata or {}
        if "doc_format" not in m:
            continue  # not a table build (e.g. lab collections)
        mark = "*" if c.name == active else " "
        out(f"{mark} {c.name}  {m.get('doc_format')}  meta {m.get('meta_format', '-')}  docs {c.count()}  "
            f"loaded {m.get('index_version', '-')}  data {str(m.get('data_sha256', '-'))[:16]}")
    if active and active not in {c.name for c in client.list_collections()}:
        out(f"CHROMA_COLLECTION={active} is not on this server")
    return 0


def _batches(items: Iterable, size: int):
    items = list(items)
    for i in range(0, len(items), size):
        yield items[i:i + size]


# --- the embedder of a build ---

class DocEmbedder(NamedTuple):
    """What the embed node needs: embed(texts) -> vectors, count_tokens(text), the model's token limit, and the
    build-record fields it adds (bge-m3-ft: embed_model, doc_format, embed_ckpt; KURE: none)."""

    embed: Callable[[list[str]], list[list[float]]]
    count_tokens: Callable[[str], int]
    max_tokens: int
    info: dict[str, Any]


def load_doc_embedder(state: IndexBuildState, log: Callable[[str], None] = print) -> DocEmbedder:
    """The model of state.model: KURE-v1 (kure_embedder) or the fine-tuned bge-m3 (models.load_embedder over
    state.ckpt and the text format build_docs chose). Side effects: loads the model (downloads on first use)."""
    if state.model == "bge-m3-ft":
        embedder = models.load_embedder(state.ckpt, device=state.device, doc_format=state.fmt)
        log(f"bge-m3 checkpoint: {embedder.ckpt}")
        return DocEmbedder(embedder.embed_documents, embedder.count_tokens, models.DOC_TOKENS,
                           {"embed_model": embedder.name, "doc_format": state.fmt, "embed_ckpt": embedder.ckpt})
    embed, count_tokens, max_tokens = kure_embedder(state.device)
    return DocEmbedder(embed, count_tokens, max_tokens, {})


# --- nodes (graph/index_build.build_chroma_graph) ---

def _timed(name: str, trace: dict, started: float, **extra: Any) -> dict[str, dict[str, Any]]:
    return {**trace, name: {"seconds": time.perf_counter() - started, **extra}}


def read_records(state: IndexBuildState) -> dict:
    """The registry, its sha256, the collection name, table keys and metadata.

    In: state.tables, state.model, state.collection, state.index_version. Out: {"records", "data_sha256", "name",
    "ids", "metas", "trace"}. Side effects: reads tables.jsonl. Node: read_records.
    """
    started = time.perf_counter()
    data_sha256 = hashlib.sha256(state.tables.read_bytes()).hexdigest()
    ft = state.model == "bge-m3-ft"
    name = state.collection or (FT_COLLECTION_PREFIX + state.index_version if ft else collection_name(data_sha256))
    records = files.read_records(state.tables)
    return {"records": records, "data_sha256": data_sha256, "name": name, "ids": [r.info.key for r in records],
            "metas": [to_metadata(r) for r in records], "trace": _timed("read_records", state.trace, started)}


def make_build_docs(log: Callable[[str], None] = print):
    """The build_docs node: bge-m3-ft embeds the kosis_rag.shared.tabletext format (state.text_format, else its
    DOC_FORMAT; a deprecated one is warned about), KURE the to_document text."""

    def build_docs(state: IndexBuildState) -> dict:
        """In: state.records, state.model, state.text_format. Out: {"docs", "fmt", "trace"}. Node: build_docs."""
        started = time.perf_counter()
        if state.model == "bge-m3-ft":
            fmt = state.text_format or text.DOC_FORMAT
            if fmt in text.DEPRECATED:
                log(f"warning: {fmt} is deprecated (docs/reports/t2-06-n3-text-dropped.md); use {text.DOC_FORMAT}")
            docs = [text.build_doc(r, fmt)["embed_text"] for r in state.records]
            log(f"text format: {fmt}")
        else:
            fmt, docs = None, [to_document(r) for r in state.records]
        return {"docs": docs, "fmt": fmt, "trace": _timed("build_docs", state.trace, started)}

    return build_docs


def make_embed(embedder: DocEmbedder | Callable[[IndexBuildState], DocEmbedder] | None = None,
               log: Callable[[str], None] = print):
    """The embed node over `embedder` (a DocEmbedder, a function of the state giving one, or None: load_doc_embedder).
    Documents are embedded 64 at a time, shortest first (similar lengths per batch), and put back in input order."""

    def embed(state: IndexBuildState) -> dict:
        """In: state.docs (+ the model fields). Out: {"vectors", "record" (build_record), "trace"}.
        Side effects: loads and runs the embedding model. Node: embed."""
        started = time.perf_counter()
        e = embedder if isinstance(embedder, DocEmbedder) else (embedder or load_doc_embedder)(state)
        docs = state.docs
        tokens = [e.count_tokens(d) for d in docs]
        over = sum(t > e.max_tokens for t in tokens)
        log(f"tokens per document: max {max(tokens)} (model limit {e.max_tokens}), over limit {over}")
        order = sorted(range(len(docs)), key=lambda i: len(docs[i]))  # similar lengths per batch
        vectors: list = [None] * len(docs)
        t0 = time.perf_counter()
        for chunk in _batches(order, 64):
            for i, v in zip(chunk, e.embed([docs[i] for i in chunk])):
                vectors[i] = v
        log(f"embedded {len(docs)} documents ({time.perf_counter() - t0:.0f}s)")
        record = build_record(state.index_version, state.data_sha256, len(docs), len(vectors[0]), max(tokens), over,
                              **e.info)
        return {"vectors": vectors, "record": record, "trace": _timed("embed", state.trace, started)}

    return embed


def make_load(client: Any = None):
    """The load node: upsert into Chroma through `client` (None: kosis_rag.adapters.chroma.chroma_client(state.chroma),
    opened when the node runs)."""

    def load_node(state: IndexBuildState) -> dict:
        """In: state.ids, docs, metas, vectors, name, record. Out: {"count", "trace"}. Side effects: writes the
        collection. Node: load."""
        started = time.perf_counter()
        c = client
        if c is None:
            from kosis_rag.adapters.chroma import chroma_client

            c = chroma_client(state.chroma)
        count = load(c, state.ids, state.docs, state.metas, state.vectors, state.name, state.record)
        return {"count": count, "trace": _timed("load", state.trace, started)}

    return load_node


def write_build_node(state: IndexBuildState) -> dict:
    """write_build to state.export instead of Chroma. Out: {"count", "trace"}. Side effects: writes the .npz.
    Node: write_build."""
    started = time.perf_counter()
    write_build(state.export, name=state.name, ids=state.ids, docs=state.docs, metas=state.metas,
                vectors=state.vectors, record=state.record)
    return {"count": len(state.ids), "trace": _timed("write_build", state.trace, started)}
