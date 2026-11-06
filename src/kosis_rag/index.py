"""Load the tables into ChromaDB (T2-24, versioned A builds in T2-05): one document per table, built from the table metadata only (tables.jsonl).

Values are not loaded. Embedding every row (617M) was measured at ~72 days and ~47 GB of vectors on an M3,
and the values are fetched from Parquet with pandas as in stage 1, so the search document only carries what
helps find a table: name, survey, organization, real period range, cycle, menu paths, items and labels.

Embeddings are computed here and passed to Chroma (`upsert(embeddings=...)`), so the collection never relies on
Chroma's default (English) embedding function; queries must be embedded with the same model.
`TableRecord.search_text` (all labels, up to 22,929 chars) stays the BM25 document; the embedding document below
summarizes region classes by level and caps the other labels per class so every table fits the model's input.

    PYTHONPATH=.:src:vendor/phase1:vendor/phase1/src python -m kosis_rag.index
    ... python -m kosis_rag.index --model bge-m3-ft --ckpt data/models/bge-m3-seen-e2.pt   # T2-06 collection

`--model bge-m3-ft` loads the fine-tuned bge-m3 (kosis_rag.adapters.models) and embeds the text format it was trained on
(kosis_rag.shared.tabletext, `--text-format`) into its own collection (FT_COLLECTION_PREFIX); the checkpoint id goes into the collection metadata
so `ChromaTableRetriever.connect` embeds questions with the same weights. Metadata (filters) is the same either way.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import time
from datetime import datetime
from collections.abc import Iterable
from pathlib import Path

from dotenv import load_dotenv

from kosis_rag.adapters import models
from kosis_rag.adapters.models import EMBED_MODEL, kure_embedder
from kosis_rag.schemas.catalog import TableRecord
from kosis_rag.schemas.resources import ChromaConfig
from kosis_rag.shared.catalog import _META_FIELDS
from kosis_rag.shared.fields import table_fields
from kosis_rag.shared.roles import table_roles
from kosis_rag.shared.tabletext import region_summary

REPO = Path(__file__).resolve().parents[2]
DOC_FORMAT = "meta-region-items-head-measure3"  # T2-30 KOS-109: names that list or summarize what is counted; before: measure2 (KOS-97), measure (T2-27)
# Per-table fields as list metadata (T2-30 KOS-101). Bump it whenever to_metadata changes: it is part of the build name.
META_FORMAT = "fields2"  # fields1: A3 (KOS-101); fields2 adds f_implied, f_fixed, f_measure_keys (KOS-109)
COLLECTION_PREFIX = "kosis_kure__"  # T2-30: content-named builds; before T2-30 "kosis_a_kure_<YYYYMMDDHHMM>" (kept on servers)
FT_COLLECTION_PREFIX = "kosis_a_bgeft_"  # T2-06: fine-tuned bge-m3 over a kosis_rag.shared.tabletext format
COLLECTION_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{1,510}[a-z0-9]$")  # Chroma: 3-512 chars, alphanumeric at both ends
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
    if not COLLECTION_NAME.match(name) or ".." in name:
        raise ValueError(f"{name!r} is not a valid Chroma collection name; DOC_FORMAT must use a-z, 0-9, - and _ only")
    return name


def read_records(path: Path) -> list[TableRecord]:
    with path.open(encoding="utf-8") as f:
        return [TableRecord.model_validate_json(line) for line in f]


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
    import json

    import numpy as np

    text = json.dumps({"name": name, "ids": ids, "docs": docs, "metas": metas, "record": record}, ensure_ascii=False)
    np.savez_compressed(path, vectors=np.asarray(vectors, dtype=np.float32), text=np.array(text))


def read_build(path: Path) -> dict:
    """The build write_build wrote: {name, ids, docs, metas, record, vectors (list of lists)}; ValueError if they disagree."""
    import json

    import numpy as np

    with np.load(path, allow_pickle=False) as f:
        build = json.loads(str(f["text"]))
        build["vectors"] = f["vectors"].tolist()
    if not len(build["ids"]) == len(build["docs"]) == len(build["metas"]) == len(build["vectors"]):
        raise ValueError(f"{path}: ids, docs, metas and vectors have different lengths")
    return build


def _chroma_config() -> ChromaConfig:
    """CHROMA_HOST / CHROMA_PORT from the environment or .env, as vectorstore.chroma_address read them (T2-41 S1;
    the build CLI takes them as arguments from S6a)."""
    load_dotenv()
    return ChromaConfig(host=os.getenv("CHROMA_HOST") or "localhost", port=int(os.getenv("CHROMA_PORT") or 8001))


def _batches(items: Iterable, size: int):
    items = list(items)
    for i in range(0, len(items), size):
        yield items[i:i + size]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load the table metadata into ChromaDB (T2-24).")
    parser.add_argument("--tables", type=Path, default=REPO / "data" / "tables.jsonl")
    parser.add_argument("--model", choices=("kure", "bge-m3-ft"), default="kure",
                        help="kure: KURE-v1 over the T2-27 text (default); bge-m3-ft: fine-tuned bge-m3 (T2-06)")
    parser.add_argument("--ckpt", default=None, help="bge-m3-ft: checkpoint .pt or run folder (default: EMBED_CKPT in .env)")
    parser.add_argument("--text-format", default=None,
                        help="bge-m3-ft: kosis_rag.shared.tabletext format (default: TEXT_FORMAT in .env, else n2-L60)")
    parser.add_argument("--collection", default=None,
                        help=f"default: kure {COLLECTION_PREFIX}<doc_format>__<hash of tables.jsonl and formats>; "
                             f"bge-m3-ft {FT_COLLECTION_PREFIX}<YYYYMMDDHHMM>")
    parser.add_argument("--device", default=None, help="cpu, mps or cuda (default: auto)")
    parser.add_argument("--list", action="store_true", help="list the builds on the server and which one is active")
    parser.add_argument("--export", type=Path, default=None,
                        help="write the build to this .npz instead of Chroma (e.g. embed on a Colab GPU, T2-11)")
    parser.add_argument("--import", dest="import_", type=Path, default=None,
                        help="load a build written by --export into Chroma (no embedding)")
    args = parser.parse_args(argv)
    if args.list:
        return list_builds()
    if args.import_:
        from kosis_rag.adapters.chroma import chroma_client

        build = read_build(args.import_)
        count = load(chroma_client(_chroma_config()), build["ids"], build["docs"], build["metas"], build["vectors"], build["name"],
                     build["record"])
        print(f"collection {build['name']}: {count} documents")
        print(f"to search it, set CHROMA_COLLECTION={build['name']} in .env")
        return 0 if count == len(build["ids"]) else 1

    index_version = datetime.now().strftime("%Y%m%d%H%M")
    data_sha256 = hashlib.sha256(args.tables.read_bytes()).hexdigest()
    ft = args.model == "bge-m3-ft"
    name = args.collection or (FT_COLLECTION_PREFIX + index_version if ft else collection_name(data_sha256))
    records = read_records(args.tables)
    ids = [r.info.key for r in records]
    metas = [to_metadata(r) for r in records]
    if ft:
        from kosis_rag.adapters import models as model
        from kosis_rag.shared import tabletext as text

        load_dotenv()
        fmt = args.text_format or os.getenv("TEXT_FORMAT") or text.DOC_FORMAT
        if fmt in text.DEPRECATED:
            print(f"warning: {fmt} is deprecated (docs/reports/t2-06-n3-text-dropped.md); use {text.DOC_FORMAT}")
        docs = [text.build_doc(r, fmt)["embed_text"] for r in records]
        embedder = model.load_embedder(args.ckpt or os.getenv("EMBED_CKPT"), device=args.device, doc_format=fmt)
        embed, count_tokens, max_tokens = embedder.embed_documents, embedder.count_tokens, model.DOC_TOKENS
        info = {"embed_model": embedder.name, "doc_format": fmt, "embed_ckpt": embedder.ckpt}
        print(f"text format: {fmt}")
        print(f"bge-m3 checkpoint: {embedder.ckpt}")
    else:
        docs = [to_document(r) for r in records]
        embed, count_tokens, max_tokens = kure_embedder(args.device)
        info = {}
    tokens = [count_tokens(d) for d in docs]
    print(f"tokens per document: max {max(tokens)} (model limit {max_tokens}), over limit {sum(t > max_tokens for t in tokens)}")
    order = sorted(range(len(docs)), key=lambda i: len(docs[i]))  # similar lengths per batch
    vectors: list = [None] * len(docs)
    started = time.perf_counter()
    for chunk in _batches(order, 64):
        for i, v in zip(chunk, embed([docs[i] for i in chunk])):
            vectors[i] = v
    print(f"embedded {len(docs)} documents ({time.perf_counter() - started:.0f}s)")

    record = build_record(index_version, data_sha256, len(docs), len(vectors[0]), max(tokens),
                          sum(t > max_tokens for t in tokens), **info)
    if args.export:
        write_build(args.export, name=name, ids=ids, docs=docs, metas=metas, vectors=vectors, record=record)
        print(f"wrote {args.export}; load it with: python -m kosis_rag.index --import {args.export.name}")
        return 0

    from kosis_rag.adapters.chroma import chroma_client

    count = load(chroma_client(_chroma_config()), ids, docs, metas, vectors, name, record)
    print(f"collection {name}: {count} documents")
    print(f"to search it, set CHROMA_COLLECTION={name} in .env (the previous builds stay; switch back the same way)")
    return 0 if count == len(ids) else 1


def list_builds() -> int:
    """Every table collection on the server with what it was built from; * marks CHROMA_COLLECTION (the active one)."""
    from kosis_rag.adapters.chroma import chroma_client

    load_dotenv()
    active = os.getenv("CHROMA_COLLECTION")
    client = chroma_client(_chroma_config())
    for c in sorted(client.list_collections(), key=lambda c: (c.metadata or {}).get("index_version") or ""):
        m = c.metadata or {}
        if "doc_format" not in m:
            continue  # not a table build (e.g. lab collections)
        mark = "*" if c.name == active else " "
        print(f"{mark} {c.name}  {m.get('doc_format')}  meta {m.get('meta_format', '-')}  docs {c.count()}  "
              f"loaded {m.get('index_version', '-')}  data {str(m.get('data_sha256', '-'))[:16]}")
    if active and active not in {c.name for c in client.list_collections()}:
        print(f"CHROMA_COLLECTION={active} is not on this server")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
