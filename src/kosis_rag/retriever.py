"""BM25, ChromaDB and hybrid (RRF) retrievers. Each reports a score in Document.metadata["score"] (the no-answer check
needs it); the hybrid keeps the dense cosine there and orders by RRF (T2-06)."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field
from rank_bm25 import BM25Okapi

from kosis_rag.filters import TableFilter, filter_tables, soft_rank, why_excluded
from kosis_rag.schemas import TableRecord

_WORD = re.compile(r"[0-9A-Za-z가-힣]+")


def tokenize(text: str) -> list[str]:
    """Words plus Korean character bigrams, so '출생아수' also matches '출생아 수'."""
    # TODO(T2-04): placeholder tokenizer (no morphological analyzer). Replace with the Korean tokenizer T2-04 picks
    # (e.g. Kiwi); BM25 is rebuilt in memory, so changing this function is the whole swap.
    tokens: list[str] = []
    for word in _WORD.findall(text.lower()):
        tokens.append(word)
        tokens += [word[i : i + 2] for i in range(len(word) - 1)]
    return tokens


class BM25TableRetriever(BaseRetriever):
    """Top-k tables by BM25. Score 0 means no query token appears in any table text."""

    docs: list[Document]
    k: int = 20  # upper bound; callers slice to their own k (SearchArgs.k <= 20)
    bm25: Any = None
    query_text: Callable[[str], str] | None = None  # e.g. text.norm_query when the documents are n2 text
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_documents(cls, docs: list[Document], k: int = 20,
                       query_text: Callable[[str], str] | None = None) -> BM25TableRetriever:
        return cls(docs=docs, k=k, bm25=BM25Okapi([tokenize(d.page_content) for d in docs]), query_text=query_text)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        scores = self.bm25.get_scores(tokenize(self.query_text(query) if self.query_text else query))
        top = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)[: self.k]
        return [
            Document(page_content=self.docs[i].page_content, metadata={**self.docs[i].metadata, "score": float(scores[i])}, id=self.docs[i].id)
            for i in top
            if scores[i] > 0
        ]


class ChromaTableRetriever(BaseRetriever):
    """Top-k tables by cosine similarity over a T2-24 collection. Score = 1 - cosine distance (higher is closer).

    The collection holds vectors we computed ourselves (embedding_function=None), so the query must be embedded
    with the same model; `connect` refuses a collection built with another model.
    A `TableFilter` (T2-05, via `with_filter`) is decided on `records` (tables.jsonl) and sent as one `key $in` condition.
    """

    collection: Any
    embed_query: Callable[[str], list[float]]
    k: int = 20  # upper bound; callers slice to their own k (SearchArgs.k <= 20)
    records: dict[str, TableRecord] = Field(default_factory=dict)  # registry the filter is decided on
    doc_format: str | None = None  # collection metadata doc_format (kosis_rag.text format for fine-tuned collections)
    table_filter: TableFilter = Field(default_factory=TableFilter)
    last_removed: dict[str, str] = Field(default_factory=dict)  # why each table was filtered out on the last query
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def connect(cls, name: str | None = None, k: int = 20, device: str | None = None,
                embed_ckpt: str | None = None) -> ChromaTableRetriever:
        """Open CHROMA_COLLECTION (or `name`) on the shared server and load the query embedder it was built with.

        KURE-v1 collections (T2-24 text) need nothing more. Fine-tuned bge-m3 collections (T2-06, kosis_rag.text
        formats) need the same checkpoint (`embed_ckpt`, else EMBED_CKPT in .env); a different file is refused, and
        questions get the same surface rules as the documents.
        """
        from kosis_rag.index import EMBED_MODEL, kure_embedder
        from kosis_rag.vectorstore import chroma_client

        client = chroma_client()  # also loads .env
        name = name or os.getenv("CHROMA_COLLECTION")
        if not name:
            raise ValueError("Set CHROMA_COLLECTION in .env to the collection built by `python -m kosis_rag.index`")
        collection = client.get_collection(name, embedding_function=None)
        meta = collection.metadata or {}
        built_with = meta.get("embed_model")
        if built_with == EMBED_MODEL:
            embed, _, _ = kure_embedder(device)
            return cls(collection=collection, embed_query=lambda text: embed([text])[0], k=k,
                       doc_format=meta.get("doc_format"))
        from kosis_rag import model, text

        if built_with != model.EMBED_BASE:
            raise ValueError(f"collection {name} was embedded with {built_with!r}; known: {EMBED_MODEL}, {model.EMBED_BASE}")
        fmt = meta.get("doc_format")
        if fmt not in text.FORMATS:
            raise ValueError(f"collection {name} holds {fmt!r} text; this code knows {', '.join(text.FORMATS)}")
        embedder = model.load_embedder(embed_ckpt or os.getenv("EMBED_CKPT"), device=device, doc_format=fmt)
        if embedder.ckpt != meta.get("embed_ckpt"):
            raise ValueError(f"collection {name} was embedded with {meta.get('embed_ckpt')!r}, "
                             f"the query embedder loaded {embedder.ckpt!r}; set EMBED_CKPT to the same file")
        return cls(collection=collection, embed_query=lambda q: embedder.embed_query(text.norm_query(q)), k=k,
                   doc_format=fmt)

    def with_filter(self, f: TableFilter, records: dict[str, TableRecord] | None = None) -> ChromaTableRetriever:
        """A copy that applies `f`; the tools keep their search_tables(question, k) signature."""
        return self.model_copy(update={"table_filter": f, "records": records or self.records, "last_removed": {}})

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        # TODO(T2-09): Chroma always returns k hits, so the chain's "no candidates" refusal never fires here;
        # refusal relies on the LLM's NO_MATCH until a cosine threshold is calibrated on the L1 dev split.
        extra, n = {}, self.k
        self.last_removed = {}
        if not self.table_filter.is_empty():
            if not self.records:
                raise ValueError("a table filter needs records (tables.jsonl) to decide on")
            keys, self.last_removed = filter_tables(self.records, self.table_filter)
            if not keys:  # no table meets the filter; the reasons are in last_removed
                return []
            extra, n = {"where": {"key": {"$in": sorted(keys)}}}, min(self.k, len(keys))
        res = self.collection.query(query_embeddings=[self.embed_query(query)], n_results=n,
                                    include=["documents", "metadatas", "distances"], **extra)
        docs = [
            Document(page_content=doc, metadata={**meta, "score": 1.0 - dist}, id=key)
            for key, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]
        return soft_rank(docs, self.records, self.table_filter)


class HybridRetriever(BaseRetriever):
    """Reciprocal rank fusion of several retrievers by table key (T2-06): RRF = sum over lists of 1 / (c + rank).

    Each list is cut to its top `k_each` before fusing. With `condition` (question -> TableFilter, e.g.
    `parse_query(q).to_filter()`), a retriever that has `with_filter` (Chroma) filters before its search and the others
    are filtered after, so every list holds only tables that meet the question's conditions (give BM25 a large k).

    Order comes from RRF, but metadata["score"] stays the score of `retrievers[score_from]` (the dense cosine), since
    RRF values sit in a narrow band (1/61..2/61) and the no-answer threshold (T2-09) needs a calibrated score.
    A table only the other lists found gets the lowest score of that list, marked `score_from="other"`.
    """

    retrievers: list[BaseRetriever]
    names: list[str] = Field(default_factory=list)  # one per retriever, for the per-list ranks in metadata
    k: int = 50
    k_each: int = 50
    c: int = 60
    score_from: int = 0
    condition: Callable[[str], TableFilter] | None = None
    records: dict[str, TableRecord] = Field(default_factory=dict)  # registry the condition is decided on
    last_removed: dict[str, str] = Field(default_factory=dict)  # why each table was filtered out on the last query
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _lists(self, query: str) -> list[list[Document]]:
        f = self.condition(query) if self.condition else None
        self.last_removed = {}
        if f is None or f.is_empty():
            return [r.invoke(query)[: self.k_each] for r in self.retrievers]
        if not self.records:
            raise ValueError("a condition needs records (tables.jsonl) to decide on")
        lists = []
        for r in self.retrievers:
            if hasattr(r, "with_filter"):
                filtered: Any = r.with_filter(f, self.records)
                docs = filtered.invoke(query)
                self.last_removed.update(filtered.last_removed)
            else:
                docs = []
                for d in r.invoke(query):
                    record = self.records.get(d.metadata["key"])
                    reason = why_excluded(record, f) if record is not None else None
                    if reason is None:
                        docs.append(d)
                    else:
                        self.last_removed[d.metadata["key"]] = reason
                docs = soft_rank(docs, self.records, f)
            lists.append(docs[: self.k_each])
        return lists

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        lists = self._lists(query)
        names = self.names or [f"r{i}" for i in range(len(lists))]
        rrf: dict[str, float] = {}
        ranks: dict[str, dict[str, int]] = {}
        first: dict[str, Document] = {}
        for name, docs in zip(names, lists):
            for rank, d in enumerate(docs, 1):
                key = d.metadata["key"]
                rrf[key] = rrf.get(key, 0.0) + 1.0 / (self.c + rank)
                ranks.setdefault(key, {})[name] = rank
                first.setdefault(key, d)
        anchor = lists[self.score_from] if self.score_from < len(lists) else []
        own = {d.metadata["key"]: d.metadata.get("score") for d in anchor}
        floor = min((v for v in own.values() if v is not None), default=None)
        order = sorted(rrf, key=lambda key: -rrf[key])  # stable: ties keep first-seen order
        out = []
        for key in order[: self.k]:
            d = first[key]
            score, source = (own[key], names[self.score_from]) if key in own else (floor, "other")
            out.append(Document(page_content=d.page_content, id=d.id, metadata={
                **d.metadata, "score": score, "score_from": source, "rrf_score": rrf[key],
                **{f"rank_{n}": r for n, r in ranks[key].items()}}))
        return out
