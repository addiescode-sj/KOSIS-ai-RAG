"""Shared bits for the local ChromaDB lab (no Docker): where the store lives, which model, and what text goes in.

The store is a plain folder (chroma/, gitignored) opened with chromadb.PersistentClient, so nothing needs a server.
Two text versions can sit side by side, which is how KOS-88 compares before/after:
    raw   record["search_text"] as KOS-24 built it
    norm  normalize_text.build_doc(...)["embed_text"] (rules version + label cap go into the tag, e.g. n2-L60)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root (lab/ sits directly under it)
sys.path.insert(0, str(ROOT / "src"))  # kosis_rag.retriever.tokenize, the team's BM25 tokenizer

from normalize_text import RULES_VERSION, TABLES, build_doc, load_records, norm_query  # noqa: E402

CHROMA_DIR = ROOT / "chroma"  # already in .gitignore
MANIFEST_DIR = CHROMA_DIR / "manifests"  # one JSON per collection (KOS-90)
DEFAULT_MODEL = "BAAI/bge-m3"
TEXTS = ("raw", "norm")

# Some embedding models were trained with role prefixes; using the wrong (or no) prefix costs recall.
PREFIX = {
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
    "intfloat/multilingual-e5-base": ("query: ", "passage: "),
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
}


def source_sha(path: Path = TABLES) -> str:
    """sha256 of tables.jsonl; the first 8 hex chars go into the collection name, so a rebuilt file gets a new one."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_tag(text: str, max_labels: int) -> str:
    return "raw" if text == "raw" else f"{RULES_VERSION}-L{max_labels}"


def collection_name(model: str, text: str, max_labels: int, sha: str) -> str:
    """KOS-90 naming: {model}__{text rules}__{source sha8}, e.g. bge-m3__n2-L60__1a2b3c4d."""
    return f"{model.split('/')[-1]}__{text_tag(text, max_labels)}__{sha[:8]}"


def docs(records: list[dict], text: str, max_labels: int = 60) -> list[dict]:
    """Per table: key, embed (dense passage), bm25 (sparse text), fields. raw uses search_text for both texts,
    but the parsed fields either way, so both collections carry the same metadata filters."""
    out = []
    for r in records:
        d = build_doc(r, max_labels=max_labels)
        raw = text == "raw"
        out.append({"key": d["key"], "embed": r["search_text"] if raw else d["embed_text"],
                    "bm25": r["search_text"] if raw else d["bm25_text"], "fields": d["fields"]})
    return out


def query_text(question: str, text: str) -> str:
    """The query gets the same surface rules as the documents it is matched against."""
    return question if text == "raw" else norm_query(question)


def metadata(rec: dict, fields: dict) -> dict:
    """Filters and pointers only; the numbers stay in data/values/*.parquet and summarize_stat reads them.
    Chroma 1.5.9 takes str/int/float/bool and non-empty lists of one type, never None."""
    info = rec["info"]
    meta = {
        "key": info["key"],
        "org_id": info["org_id"],
        "tbl_id": info["tbl_id"],
        "table_name": info["table_name"],
        "survey_name": info["survey_name"],
        "organization": info["organization"],
        "url": info["url"],
        "parquet": f"data/values/{info['key']}.parquet",
        "start": rec.get("start"),
        "end": rec.get("end"),
        "unit": rec.get("unit"),
        "cycle": rec.get("cycle") or None,  # list, e.g. ["M", "Y"]; filter with {"cycle": {"$contains": "M"}}
        "n_items": len(rec.get("items", [])),
        "region_level": fields.get("지역 수준"),
        "target": fields.get("대상"),
    }
    return {k: v for k, v in meta.items() if v is not None and v != ""}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_model(name: str, max_tokens: int):
    """sentence-transformers model on GPU in fp16 when there is one (bge-m3 fp16 peaks near 1.3GB on the 2070)."""
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(name, device="cuda" if torch.cuda.is_available() else "cpu")
    if model.device.type == "cuda":
        model.half()
    model.max_seq_length = max_tokens
    return model


def encode(model, model_name: str, texts: list[str], role: str, batch: int = 16):
    """role 'query' or 'passage'. normalize -> unit vectors, so cosine similarity == dot product."""
    prefix = PREFIX.get(model_name, ("", ""))[0 if role == "query" else 1]
    return model.encode([prefix + t for t in texts], batch_size=batch, normalize_embeddings=True,
                        show_progress_bar=len(texts) > 100)


def model_revision(name: str) -> str | None:
    """Commit hash of the cached Hugging Face snapshot, so a later re-download can be told apart."""
    try:
        from huggingface_hub import try_to_load_from_cache

        parts = Path(try_to_load_from_cache(name, "config.json")).parts  # .../snapshots/<commit>/config.json
        return parts[parts.index("snapshots") + 1]
    except Exception:
        return None


def bm25_tokenize():
    """The team's BM25 tokenizer (words + Korean bigrams), so lab numbers match src/kosis_rag/retriever.py."""
    try:
        from kosis_rag.retriever import tokenize
    except ImportError:  # langchain-core missing in this env: same rule, copied
        import re

        word = re.compile(r"[0-9A-Za-z가-힣]+")

        def tokenize(text: str) -> list[str]:
            out = []
            for w in word.findall(text.lower()):
                out += [w, *(w[i : i + 2] for i in range(len(w) - 1))]
            return out
    return tokenize


def client():
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))


def open_collection(store, model: str, text: str, max_labels: int = 60):
    """The collection built from the current tables.jsonl; None when it has not been built yet."""
    name = collection_name(model, text, max_labels, source_sha())
    return store.get_collection(name) if name in [c.name for c in store.list_collections()] else None

