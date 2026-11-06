"""The question index (T2-39): one Chroma document per hypothetical question, searched as table lists.

    set PYTHONPATH=.;src;vendor/phase1;vendor/phase1/src          (Windows cmd)
    python -m kosis_rag.hq.index --build <build_id> --ckpt data/models/unseen-e1.pt
    python -m kosis_rag.hq.index --build <build_id> --ckpt data/models/unseen-e1.pt --desc   # S1-desc ablation

Questions are embedded with the same fine-tuned embedder as the service's table collection (the query side reuses
the dense retriever's query embedder, so both live in one space) after the same surface rules (text.norm_query).
Collection `kosis_hq__<build_id>__<ckpt8>`; its metadata records embed_model, embed_ckpt, doc_format "hq" and
hq_build, and `HqRetriever.connect` refuses a collection embedded with other weights than the dense side.
`--desc` indexes each table's one-sentence description instead (build id `<build_id>-desc`).

HqRetriever: the top `n_questions` questions -> max-pool per table (a table's score is its best question) -> top k
tables as Documents whose text is the table's n2 passage, so rerank1 / rerank2 see what they were trained on.
hq_bm25: BM25 with one document per table = its kept questions joined (the S2 list).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field
from rank_bm25 import BM25Okapi

from kosis_rag.retriever import BM25TableRetriever, env_chroma_config, tokenize
from kosis_rag.schemas.catalog import TableRecord
from kosis_rag.shared import tabletext as text
from kosis_rag.shared.catalog import to_document
from kosis_rag.shared.filters import TableFilter, filter_with_relaxation, relax, soft_rank

REPO = Path(__file__).resolve().parents[3]
HQ_DIR = REPO / "data" / "hq"
COLLECTION_PREFIX = "kosis_hq__"
DOC_FORMAT = "hq"  # collection metadata doc_format of a question index
DESC_SUFFIX = "-desc"  # build id suffix of the description-only index (S1-desc)
N_QUESTIONS = 200  # questions fetched before pooling per table
_INVALID = re.compile(r"[^a-z0-9_-]+")
log = logging.getLogger(__name__)


def ckpt8(ckpt_id: str | None) -> str:
    """8 characters naming the embedder weights: the sha part of model.ckpt_id ('<file>@<sha12>'), else a hash.

    Side effects: none. Node: none (names the collection).
    """
    if not ckpt_id:
        return "none"
    if "@" in ckpt_id and re.fullmatch(r"[0-9a-f]{8,}", ckpt_id.rsplit("@", 1)[1]):
        return ckpt_id.rsplit("@", 1)[1][:8]
    return "base" if ckpt_id == "base" else hashlib.sha256(ckpt_id.encode()).hexdigest()[:8]


def collection_name(build_id: str, ckpt_id: str | None) -> str:
    """kosis_hq__<build_id>__<ckpt8>, lower-cased with characters Chroma refuses replaced by '-' (3-512 chars,
    alphanumeric at both ends, no '..').
    Side effects: ValueError for a name Chroma refuses. Node: none (index CLI, load_resources).
    """
    from kosis_rag.index import COLLECTION_NAME

    name = _INVALID.sub("-", f"{COLLECTION_PREFIX}{build_id}__{ckpt8(ckpt_id)}".lower())[:512].strip("-_")
    if not COLLECTION_NAME.match(name) or ".." in name:
        raise ValueError(f"{name!r} is not a valid Chroma collection name")
    return name


def read_questions(build_id: str, root: Path = HQ_DIR) -> list[dict]:
    """The rows to index for a build: {"id", "key", "type", "text"}.

    In: build_id = a kosis_rag.hq.build id, or '<id>-desc' for its descriptions (one row per table, type
        'description'); root = data/hq. Out: rows in file order. Side effects: reads the build folder;
        FileNotFoundError when the build is missing.
    Node: none directly (load_resources and the index CLI call it; retrieve_hq_bm25 searches what it gives).
    """
    if build_id.endswith(DESC_SUFFIX):
        path = root / build_id[: -len(DESC_SUFFIX)] / "descriptions.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [{"id": f"{r['key']}#d", "key": r["key"], "type": "description", "text": r["description"]} for r in rows]
    path = root / build_id / "questions.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def max_pool(hits: list[tuple[str, str, str, float]]) -> list[dict]:
    """Questions -> tables: each table's best question score, ties keep the hit order.

    In: hits = (question id, table key, question text, score) best first. Out: one dict per table
        {key, score, best_id, best_question, n} ordered by score (n = how many of the hits were its questions).
    Side effects: none. Node: retrieve_hq (through HqRetriever).
    """
    tables: dict[str, dict] = {}
    for qid, key, question, score in hits:
        t = tables.get(key)
        if t is None:
            tables[key] = {"key": key, "score": score, "best_id": qid, "best_question": question, "n": 1}
        else:
            t["n"] += 1
            if score > t["score"]:
                t.update(score=score, best_id=qid, best_question=question)
    return sorted(tables.values(), key=lambda t: -t["score"])  # stable


def _catalog(key: str, records: dict[str, TableRecord]) -> dict:
    return to_document(records[key].info).metadata


class HqRetriever(BaseRetriever):
    """Top-k tables by their closest hypothetical questions (max-pool over the top n_questions; score = 1 - cosine
    distance of the best question). Filters like ChromaTableRetriever: `with_filter` decides the TableFilter on
    `records`, relaxes it when no table meets it (last_relaxed) and sends `key $in` to Chroma.

    Documents carry the table's catalog metadata, key, score, hq_best_question, hq_best_id and hq_n, and the
    table's n2 text as page_content (text_of), so the stages after fuse treat them like dense hits.
    A question whose table key is not in `records` (an index built from another tables.jsonl) is skipped, counted in
    last_missing and logged, so guide / refuse never see a table without catalog fields.
    """

    collection: Any
    embed_query: Callable[[str], list[float]]
    text_of: dict[str, str] = Field(default_factory=dict)  # key -> n2 embedding text (the re-ranker passage)
    records: dict[str, TableRecord] = Field(default_factory=dict)
    k: int = 50
    n_questions: int = N_QUESTIONS
    n_docs: int | None = None  # questions in the collection (n_results never asks for more)
    table_filter: TableFilter = Field(default_factory=TableFilter)
    last_removed: dict[str, str] = Field(default_factory=dict)
    last_relaxed: list[str] = Field(default_factory=list)
    last_missing: int = 0  # tables of the last query's hits not in records (skipped)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def connect(cls, build_id: str, dense: Any, *, records: dict[str, TableRecord], text_of: dict[str, str],
                k: int = 50, name: str | None = None) -> HqRetriever:
        """Open the question collection of `build_id` for the dense side's embedder.

        In: build_id = HQ_BUILD; dense = the table retriever (ChromaTableRetriever: its embed_query and its
            collection's embed_model / embed_ckpt are reused); records, text_of = the registry and n2 passages;
            k = tables returned; name = collection override. Out: HqRetriever.
        Side effects: connects to Chroma; ValueError when the collection was embedded with other weights.
        Node: none (load_resources calls it; retrieve_hq searches it).
        """
        from kosis_rag.adapters.chroma import chroma_client

        meta = (getattr(getattr(dense, "collection", None), "metadata", None) or {})
        ckpt = meta.get("embed_ckpt") or meta.get("embed_model")
        name = name or collection_name(build_id, ckpt)
        collection = chroma_client(env_chroma_config()).get_collection(name, embedding_function=None)
        hq_meta = collection.metadata or {}
        if (hq_meta.get("embed_ckpt") or hq_meta.get("embed_model")) != ckpt or hq_meta.get("doc_format") != DOC_FORMAT:
            raise ValueError(f"question collection {name} was embedded with {hq_meta.get('embed_ckpt')!r} "
                             f"({hq_meta.get('doc_format')!r}); the dense side uses {ckpt!r}")
        return cls(collection=collection, embed_query=dense.embed_query, text_of=text_of, records=records, k=k,
                   n_docs=collection.count())

    def with_filter(self, f: TableFilter, records: dict[str, TableRecord] | None = None) -> HqRetriever:
        """A copy that applies `f` (as ChromaTableRetriever.with_filter).

        Side effects: none. Node: retrieve_hq (through retriever.search_list).
        """
        return self.model_copy(update={"table_filter": f, "records": records or self.records, "last_removed": {},
                                       "last_relaxed": [], "last_missing": 0})

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        """Tables for the question through their questions.

        In: query = the user question. Out: at most k Documents, best table first (soft-ranked under the filter);
            tables not in records are skipped before the top k is taken.
        Side effects: one Chroma query; sets last_removed / last_relaxed / last_missing; a warning when a hit's
            table is not in records. Node: retrieve_hq.
        """
        extra: dict = {}
        self.last_removed, self.last_relaxed, self.last_missing = {}, [], 0
        f = self.table_filter
        if not f.is_empty():
            if not self.records:
                raise ValueError("a table filter needs records (tables.jsonl) to decide on")
            keys, self.last_relaxed, self.last_removed = filter_with_relaxation(self.records, f)
            f = relax(f, self.last_relaxed)
            if not keys:  # no table holds the requested years
                return []
            extra = {"where": {"key": {"$in": sorted(keys)}}}
        n = self.n_questions if self.n_docs is None else min(self.n_questions, self.n_docs)
        if n <= 0:
            return []
        res = self.collection.query(query_embeddings=[self.embed_query(query)], n_results=n,
                                    include=["documents", "metadatas", "distances"], **extra)
        hits = [(qid, meta["key"], doc, 1.0 - dist)
                for qid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0],
                                                res["distances"][0])]
        pooled = max_pool(hits)
        known = [t for t in pooled if t["key"] in self.records]
        self.last_missing = len(pooled) - len(known)
        if self.last_missing:
            log.warning("question index: %d table(s) of this query are not in tables.jsonl and were skipped "
                        "(index built from another registry?)", self.last_missing)
        docs = [Document(page_content=self.text_of.get(t["key"], ""), id=t["key"], metadata={
            **_catalog(t["key"], self.records), "key": t["key"], "score": t["score"],
            "hq_best_question": t["best_question"], "hq_best_id": t["best_id"], "hq_n": t["n"]})
            for t in known[: self.k]]
        return soft_rank(docs, self.records, f)


def hq_bm25(rows: list[dict], records: dict[str, TableRecord], text_of: dict[str, str], k: int = 50
            ) -> BM25TableRetriever:
    """BM25 over the questions, one document per table (its questions joined, after text.norm_query).

    In: rows = read_questions(build); records = the registry (catalog metadata); text_of = key -> n2 text (returned
        as page_content, so later stages see the table text); k = tables per search.
    Out: a BM25TableRetriever (questions with text.norm_query, like the table BM25). Tables without a kept question
        are not in it, nor tables missing from records (skipped, with one warning giving their count).
    Side effects: builds the index in memory; logs a warning when rows name tables not in records.
    Node: none (load_resources builds it; retrieve_hq_bm25 searches it).
    """
    joined: dict[str, list[str]] = {}
    missing: set[str] = set()
    for r in rows:
        if r["key"] not in records:
            missing.add(r["key"])
            continue
        joined.setdefault(r["key"], []).append(text.norm_query(r["text"]))
    if missing:
        log.warning("question BM25: %d table(s) of the build are not in tables.jsonl and were skipped", len(missing))
    keys = list(joined)
    docs = [Document(page_content=text_of.get(key, ""), metadata=_catalog(key, records), id=key) for key in keys]
    bm25 = BM25Okapi([tokenize(" ".join(joined[key])) for key in keys])
    return BM25TableRetriever(docs=docs, k=k, bm25=bm25, query_text=text.norm_query)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load a hypothetical question build into Chroma (T2-39).")
    parser.add_argument("--build", required=True, help="build id printed by python -m kosis_rag.hq.build")
    parser.add_argument("--ckpt", default=None, help="fine-tuned bge-m3 .pt (default: EMBED_CKPT in .env)")
    parser.add_argument("--desc", action="store_true", help="index the descriptions instead (build id <id>-desc)")
    parser.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    parser.add_argument("--collection", default=None, help="override the collection name")
    args = parser.parse_args(argv)

    from dotenv import load_dotenv

    from kosis_rag.adapters import models as model
    from kosis_rag.adapters.chroma import chroma_client
    from kosis_rag.index import load

    load_dotenv(REPO / ".env")
    build = args.build + (DESC_SUFFIX if args.desc and not args.build.endswith(DESC_SUFFIX) else "")
    rows = read_questions(build)
    if not rows:
        raise SystemExit(f"build {build} has no rows")
    embedder = model.load_embedder(args.ckpt or os.getenv("EMBED_CKPT"), device=args.device, doc_format=text.DOC_FORMAT)
    texts = [text.norm_query(r["text"]) for r in rows]
    vectors = embedder.embed_documents(texts)
    name = args.collection or collection_name(build, embedder.ckpt)
    digest = hashlib.sha256("\n".join(f"{r['id']}\t{r['text']}" for r in rows).encode("utf-8")).hexdigest()
    record = {"index_version": datetime.now().strftime("%Y%m%d%H%M"), "doc_format": DOC_FORMAT, "hq_build": build,
              "embed_model": embedder.name, "embed_ckpt": embedder.ckpt, "data_sha256": digest,
              "dim": len(vectors[0]), "normalized": True, "hnsw:space": "cosine", "n_docs": len(rows),
              "n_tables": len({r["key"] for r in rows})}
    count = load(chroma_client(env_chroma_config()), [r["id"] for r in rows], texts, [{"key": r["key"], "type": r["type"]} for r in rows],
                 vectors, name, record)
    print(f"collection {name}: {count} questions over {record['n_tables']} tables (embedder {embedder.ckpt})")
    print(f"to search with it, set HQ_BUILD={build} in .env (empty = off); HQ_BM25=on adds the question BM25")
    return 0 if count == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
