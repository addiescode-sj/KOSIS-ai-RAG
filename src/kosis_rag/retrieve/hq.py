"""Searching the question index (T2-39; T2-41 S2: the online part of hq/index.py).

Questions are embedded with the same fine-tuned embedder as the service's table collection (the query side reuses
the dense side's query embedder, so both live in one space) after the same surface rules (tabletext.norm_query).
Collection `kosis_hq__<build_id>__<ckpt8>`; its metadata records embed_model, embed_ckpt, doc_format "hq" and hq_build.
This module only reads a question collection and questions.jsonl that were built already (`python -m
kosis_rag.hq.index` loads them; plans/T2-41.md 미결 12).

HqIndex: the top `n_questions` questions -> max-pool per table (a table's score is its best question) -> top k
tables as Documents whose text is the table's n2 passage, so rerank1 / rerank2 see what they were trained on.
hq_bm25: BM25 with one document per table = its kept questions joined (the S2 list).
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from kosis_rag.retrieve.bm25 import Bm25Index
from kosis_rag.retrieve.constants import HQ_COLLECTION_PREFIX, HQ_K, HQ_N_QUESTIONS
from kosis_rag.schemas.catalog import TableRecord
from kosis_rag.shared import tabletext as text
from kosis_rag.shared.catalog import to_document
from kosis_rag.shared.constants import COLLECTION_NAME
from kosis_rag.shared.filters import TableFilter, filter_with_relaxation, relax, soft_rank
from kosis_rag.shared.tokenize import tokenize

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
    name = _INVALID.sub("-", f"{HQ_COLLECTION_PREFIX}{build_id}__{ckpt8(ckpt_id)}".lower())[:512].strip("-_")
    if not COLLECTION_NAME.match(name) or ".." in name:
        raise ValueError(f"{name!r} is not a valid Chroma collection name")
    return name


def max_pool(hits: list[tuple[str, str, str, float]]) -> list[dict]:
    """Questions -> tables: each table's best question score, ties keep the hit order.

    In: hits = (question id, table key, question text, score) best first. Out: one dict per table
        {key, score, best_id, best_question, n} ordered by score (n = how many of the hits were its questions).
    Side effects: none. Node: retrieve_hq (through HqIndex.search).
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


@dataclass
class HqIndex:
    """A question collection searched as a table list; `search` is stateless (removed / relaxed / missing returned).

    Documents carry the table's catalog metadata, key, score, hq_best_question, hq_best_id and hq_n, and the
    table's n2 text as page_content (text_of), so the stages after fuse treat them like dense hits.
    """

    collection: Any
    embed_query: Callable[[str], list[float]]
    text_of: dict[str, str] = field(default_factory=dict)  # key -> n2 embedding text (the re-ranker passage)
    records: dict[str, TableRecord] = field(default_factory=dict)  # registry (catalog fields; filter default)
    k: int = HQ_K
    n_questions: int = HQ_N_QUESTIONS
    n_docs: int | None = None  # questions in the collection (n_results never asks for more)

    def search(self, query: str, *, vector: list[float] | None = None, table_filter: TableFilter | None = None,
               records: dict[str, TableRecord] | None = None
               ) -> tuple[list[Document], dict[str, str], list[str], int]:
        """Tables for the question through their questions.

        In: query = the user question; vector = its embedding when already made (the embed node); table_filter =
            conditions (None or empty: none; decided on `records`, default self.records, relaxed when no table meets
            it, then sent as `key $in`).
        Out: (docs, removed, relaxed, missing): at most k Documents, best table first (soft-ranked under the filter);
            a question whose table is not in the registry (an index built from another tables.jsonl) is skipped
            before the top k is taken and counted in missing, so guide / refuse never see a table without catalog
            fields.
        Side effects: one Chroma query (none when no table holds the requested years); a warning when a hit's table
            is not in the registry; ValueError for a non-empty filter without records. Node: retrieve_hq.
        """
        records = records or self.records
        f = table_filter if table_filter is not None else TableFilter()
        extra: dict = {}
        removed: dict[str, str] = {}
        relaxed: list[str] = []
        if not f.is_empty():
            if not records:
                raise ValueError("a table filter needs records (tables.jsonl) to decide on")
            keys, relaxed, removed = filter_with_relaxation(records, f)
            f = relax(f, relaxed)
            if not keys:  # no table holds the requested years
                return [], removed, relaxed, 0
            extra = {"where": {"key": {"$in": sorted(keys)}}}
        n = self.n_questions if self.n_docs is None else min(self.n_questions, self.n_docs)
        if n <= 0:
            return [], removed, relaxed, 0
        res = self.collection.query(query_embeddings=[vector if vector is not None else self.embed_query(query)],
                                    n_results=n, include=["documents", "metadatas", "distances"], **extra)
        hits = [(qid, meta["key"], doc, 1.0 - dist)
                for qid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0],
                                                res["distances"][0])]
        pooled = max_pool(hits)
        known = [t for t in pooled if t["key"] in records]
        missing = len(pooled) - len(known)
        if missing:
            log.warning("question index: %d table(s) of this query are not in tables.jsonl and were skipped "
                        "(index built from another registry?)", missing)
        docs = [Document(page_content=self.text_of.get(t["key"], ""), id=t["key"], metadata={
            **_catalog(t["key"], records), "key": t["key"], "score": t["score"],
            "hq_best_question": t["best_question"], "hq_best_id": t["best_id"], "hq_n": t["n"]})
            for t in known[: self.k]]
        return soft_rank(docs, records, f), removed, relaxed, missing


def hq_bm25(rows: list[dict], records: dict[str, TableRecord], text_of: dict[str, str], k: int = HQ_K) -> Bm25Index:
    """BM25 over the questions, one document per table (its questions joined, after text.norm_query).

    In: rows = adapters.files.read_questions(build); records = the registry (catalog metadata); text_of = key -> n2
        text (returned as page_content, so later stages see the table text); k = tables per search.
    Out: a Bm25Index (questions with text.norm_query, like the table BM25). Tables without a kept question are not in
        it, nor tables missing from records (skipped, with one warning giving their count).
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
    return Bm25Index(docs=docs, bm25=bm25, k=k, query_text=text.norm_query)
