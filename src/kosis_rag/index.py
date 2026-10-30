"""Load the tables into ChromaDB (T2-24, versioned A builds in T2-05): one document per table, built from the table metadata only (tables.jsonl).

Values are not loaded. Embedding every row (617M) was measured at ~72 days and ~47 GB of vectors on an M3,
and the values are fetched from Parquet with pandas as in stage 1, so the search document only carries what
helps find a table: name, survey, organization, real period range, cycle, menu paths, items and labels.

Embeddings are computed here and passed to Chroma (`upsert(embeddings=...)`), so the collection never relies on
Chroma's default (English) embedding function; queries must be embedded with the same model.
`TableRecord.search_text` (all labels, up to 22,929 chars) stays the BM25 document; the embedding document below
summarizes region classes by level and caps the other labels per class so every table fits the model's input.

    PYTHONPATH=.:src:vendor/phase1:vendor/phase1/src python -m kosis_rag.index
"""

from __future__ import annotations

import argparse
import hashlib
import re
import time
from datetime import datetime
from collections.abc import Callable, Iterable
from pathlib import Path

from kosis_rag.loader import _META_FIELDS
from kosis_rag.metadata import REGION_LEVELS
from kosis_rag.schemas import TableRecord

REPO = Path(__file__).resolve().parents[2]
EMBED_MODEL = "nlpai-lab/KURE-v1"
DOC_FORMAT = "meta-region-items-head"  # T2-25 1+2+3; before: "meta-labels100" (kosis_a_kure_202610291341)
COLLECTION_PREFIX = "kosis_a_kure_"  # A: one vector per table; the build version follows (T2-05)
LABEL_CAP = 100  # per non-region class
UNIT_SUFFIX = re.compile(r"\s*\(([^()]*)\)\s*$")
SIDO_SHORT = {
    "서울특별시": "서울", "부산광역시": "부산", "대구광역시": "대구", "인천광역시": "인천", "광주광역시": "광주",
    "대전광역시": "대전", "울산광역시": "울산", "세종특별자치시": "세종", "경기도": "경기", "강원도": "강원",
    "강원특별자치도": "강원", "충청북도": "충북", "충청남도": "충남", "전라북도": "전북", "전북특별자치도": "전북",
    "전라남도": "전남", "경상북도": "경북", "경상남도": "경남", "제주특별자치도": "제주", "제주도": "제주",
}

Embed = Callable[[list[str]], list[list[float]]]


def clean_item(item: str) -> tuple[str, str | None]:
    """'혼인건수 (건)' -> ('혼인건수', '건'), '가구_계' -> ('가구 계', None)."""
    m = UNIT_SUFFIX.search(item)
    name, unit = (item[: m.start()], m.group(1).strip()) if m else (item, None)
    name = name.replace("_", " ").replace("ㆍ", "·").replace("–", "-")
    return re.sub(r"\s+", " ", name).strip(), unit


def sido_names(labels: list[str]) -> list[str]:
    """The 시도 in a label list, in the short form questions use (서울특별시 -> 서울); old and new names merge."""
    return list(dict.fromkeys(SIDO_SHORT[label] for label in labels if label in SIDO_SHORT))


def region_summary(levels: dict[str, int], sido: list[str] | None = None) -> str:
    """{'전국': 1, '시도': 17, '시군구': 250} -> '전국·시도(17)·시군구(250)'; with sido: '시도(17: 서울, 부산, …)'."""
    parts = []
    for lv in REGION_LEVELS:
        if lv not in levels:
            continue
        if lv == "전국":
            parts.append(lv)
        elif lv == "시도" and sido:
            parts.append(f"{lv}({levels[lv]:,}: {', '.join(sido)})")
        else:
            parts.append(f"{lv}({levels[lv]:,})")
    return "·".join(parts)


def to_document(record: TableRecord) -> str:
    """Embedding text (T2-25 1+2+3): a summary line first, then listing facts, the real period range, cleaned items
    with their units on one line, region classes as level counts and the other classes as capped labels.

    Region labels were 81% of all labels and, listed, made a table read like a list of place names; the level
    summary says which regions a table covers in a few tokens. Exact place names stay in `search_text` (BM25).
    """
    info = record.info
    cleaned = [clean_item(i) for i in record.items]
    items = list(dict.fromkeys(name for name, _ in cleaned))
    units = list(dict.fromkeys(unit for _, unit in cleaned if unit))
    regions = " / ".join(region_summary(c.levels) for c in record.classes if c.levels) or "지역 구분 없음"
    lines = [
        f"{info.table_name} | {'·'.join(record.prd_se)} {record.start}~{record.end} | 지역 단위: {regions} | 항목: {', '.join(items[:8])}",
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


def to_metadata(record: TableRecord, index_version: str = "") -> dict:
    """Scalars only (Chroma). Keys of loader._META_FIELDS stay as they are, so loader.to_candidate works unchanged."""
    info = record.info
    meta = {name: getattr(info, name) for name in _META_FIELDS}
    meta |= {
        "start": record.start, "end": record.end,
        "cycle": "|".join(record.cycle), "prd_se": "|".join(record.prd_se),
        "topic": record.topic, "unit": record.unit, "n_labels": sum(c.n_labels for c in record.classes),
        "part": "core", "index_version": index_version or None,
    }
    return {k: v for k, v in meta.items() if v is not None}  # Chroma rejects None


def read_records(path: Path) -> list[TableRecord]:
    with path.open(encoding="utf-8") as f:
        return [TableRecord.model_validate_json(line) for line in f]


def kure_embedder(device: str | None = None) -> tuple[Embed, Callable[[str], int], int]:
    """(embed documents, count tokens, max tokens) for KURE-v1; embeddings are L2-normalized for cosine search."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL, device=device)

    def embed(texts: list[str]) -> list[list[float]]:
        return model.encode(texts, batch_size=8, normalize_embeddings=True, convert_to_numpy=True).tolist()

    return embed, lambda text: len(model.tokenizer(text)["input_ids"]), model.max_seq_length


def build_record(index_version: str, data_sha256: str, n_docs: int, dim: int, max_tokens: int, over_limit: int) -> dict:
    """What the collection was built from; stored as collection metadata so every teammate sees the same record."""
    return {
        "index_version": index_version, "data_sha256": data_sha256, "doc_format": DOC_FORMAT, "embed_model": EMBED_MODEL,
        "dim": dim, "normalized": True, "hnsw:space": "cosine", "n_docs": n_docs, "max_doc_tokens": max_tokens,
        "over_token_limit": over_limit,
    }


def load(client, ids: list[str], docs: list[str], metas: list[dict], embeddings: list[list[float]],
         name: str, record: dict | None = None) -> int:
    """Upsert by table key and drop ids that are no longer in the input, so reloading gives the same collection."""
    collection = client.get_or_create_collection(
        name, embedding_function=None,
        metadata=record or {"embed_model": EMBED_MODEL, "doc_format": DOC_FORMAT, "dim": len(embeddings[0]), "hnsw:space": "cosine"},
    )
    step = client.get_max_batch_size()
    for i in range(0, len(ids), step):
        collection.upsert(ids=ids[i:i + step], documents=docs[i:i + step], metadatas=metas[i:i + step],
                          embeddings=embeddings[i:i + step])
    stale = sorted(set(collection.get(include=[])["ids"]) - set(ids))
    if stale:
        collection.delete(ids=stale)
    return collection.count()


def _batches(items: Iterable, size: int):
    items = list(items)
    for i in range(0, len(items), size):
        yield items[i:i + size]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load the table metadata into ChromaDB (T2-24).")
    parser.add_argument("--tables", type=Path, default=REPO / "data" / "tables.jsonl")
    parser.add_argument("--collection", default=None, help=f"default: {COLLECTION_PREFIX}<YYYYMMDDHHMM>, a new build")
    parser.add_argument("--device", default=None, help="cpu, mps or cuda (default: auto)")
    args = parser.parse_args(argv)

    index_version = datetime.now().strftime("%Y%m%d%H%M")
    name = args.collection or COLLECTION_PREFIX + index_version
    records = read_records(args.tables)
    ids = [r.info.key for r in records]
    docs = [to_document(r) for r in records]
    metas = [to_metadata(r, index_version) for r in records]

    embed, count_tokens, max_tokens = kure_embedder(args.device)
    tokens = [count_tokens(d) for d in docs]
    print(f"tokens per document: max {max(tokens)} (model limit {max_tokens}), over limit {sum(t > max_tokens for t in tokens)}")
    order = sorted(range(len(docs)), key=lambda i: len(docs[i]))  # similar lengths per batch
    vectors: list = [None] * len(docs)
    started = time.perf_counter()
    for chunk in _batches(order, 64):
        for i, v in zip(chunk, embed([docs[i] for i in chunk])):
            vectors[i] = v
    print(f"embedded {len(docs)} documents ({time.perf_counter() - started:.0f}s)")

    from kosis_rag.vectorstore import chroma_client

    record = build_record(index_version, hashlib.sha256(args.tables.read_bytes()).hexdigest(), len(docs),
                          len(vectors[0]), max(tokens), sum(t > max_tokens for t in tokens))
    count = load(chroma_client(), ids, docs, metas, vectors, name, record)
    print(f"collection {name}: {count} documents")
    return 0 if count == len(ids) else 1


if __name__ == "__main__":
    raise SystemExit(main())
