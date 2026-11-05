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

from kosis_rag.filters import TableFilter, filter_with_relaxation, relax, soft_rank, why_excluded
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
    When no table meets it, the conditions are relaxed one at a time (filters.filter_with_relaxation, T2-29); the
    dropped ones are in `last_relaxed`. Only a year no table holds still gives no result.
    """

    collection: Any
    embed_query: Callable[[str], list[float]]
    k: int = 20  # upper bound; callers slice to their own k (SearchArgs.k <= 20)
    records: dict[str, TableRecord] = Field(default_factory=dict)  # registry the filter is decided on
    doc_format: str | None = None  # collection metadata doc_format (kosis_rag.text format for fine-tuned collections)
    table_filter: TableFilter = Field(default_factory=TableFilter)
    last_removed: dict[str, str] = Field(default_factory=dict)  # why each table was filtered out on the last query
    last_relaxed: list[str] = Field(default_factory=list)  # conditions dropped on the last query (filters.RELAX_ORDER)
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
        return self.model_copy(update={"table_filter": f, "records": records or self.records, "last_removed": {},
                                       "last_relaxed": []})

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        # TODO(T2-09): Chroma always returns k hits, so the chain's "no candidates" refusal never fires here;
        # refusal relies on the LLM's NO_MATCH until a cosine threshold is calibrated on the L1 dev split.
        extra, n = {}, self.k
        self.last_removed, self.last_relaxed = {}, []
        f = self.table_filter
        if not f.is_empty():
            if not self.records:
                raise ValueError("a table filter needs records (tables.jsonl) to decide on")
            keys, self.last_relaxed, self.last_removed = filter_with_relaxation(self.records, f)
            f = relax(f, self.last_relaxed)
            if not keys:  # no table holds the requested years (the one condition never relaxed); reasons in last_removed
                return []
            extra, n = {"where": {"key": {"$in": sorted(keys)}}}, min(self.k, len(keys))
        res = self.collection.query(query_embeddings=[self.embed_query(query)], n_results=n,
                                    include=["documents", "metadatas", "distances"], **extra)
        docs = [
            Document(page_content=doc, metadata={**meta, "score": 1.0 - dist}, id=key)
            for key, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]
        return soft_rank(docs, self.records, f)


def condition_filter(table_filter: TableFilter | None, records: dict[str, TableRecord] | None) -> TableFilter | None:
    """The filter every list is searched under: hybrid_search's relaxation step, shared by the graph's lists (T2-39).

    In: table_filter = conditions from the question (None or empty: no filtering); records = the registry the filter
        is decided on. Out: the filter as given when it is None or empty; else the filter with the conditions no table
        meets dropped (filters.filter_with_relaxation, T2-29). A filter the caller relaxed already comes back the same.
    Side effects: raises ValueError when a non-empty filter has no records to decide on.
    Node: condition (and hybrid_search for the old path).
    """
    if table_filter is None or table_filter.is_empty():
        return table_filter
    if not records:
        raise ValueError("a condition needs records (tables.jsonl) to decide on")
    _, dropped, _ = filter_with_relaxation(records, table_filter)
    return relax(table_filter, dropped)


def search_list(query: str, retriever: BaseRetriever, *, table_filter: TableFilter | None = None,
                records: dict[str, TableRecord] | None = None, k_each: int = 50, filtered: bool | None = None
                ) -> tuple[list[Document], dict[str, str]]:
    """One retriever's ranked list under the question's conditions (one list of hybrid_search, T2-39).

    In: query = the user question; retriever = one list's retriever; table_filter = the filter from condition_filter
        (None or empty: no filtering; soft signals only re-rank); records = the registry; k_each = top kept;
        filtered = take the filtering path (default: when table_filter is non-empty). hybrid_search decides it on the
        filter before relaxation, as it always did, so a filter relaxed to empty still takes the filtering path there.
    Out: (docs, removed). A retriever with with_filter (Chroma, HqRetriever) filters before its search, the others
        (BM25) are filtered after and soft-ranked; then the top k_each. removed = table key -> why the filter dropped it.
    Side effects: calls the retriever (Chroma query / BM25 scoring); ValueError for a non-empty filter without records.
    Node: retrieve_dense, retrieve_bm25, retrieve_hq, retrieve_hq_bm25 (and hybrid_search).
    """
    if filtered is None:
        filtered = table_filter is not None and not table_filter.is_empty()
    if not filtered:
        docs = retriever.invoke(query)[:k_each]
        if table_filter is not None and records and table_filter.is_soft():
            docs = soft_rank(docs, records, table_filter)  # soft signals only re-rank
        return docs, {}
    if not records:
        raise ValueError("a condition needs records (tables.jsonl) to decide on")
    removed: dict[str, str] = {}
    if hasattr(retriever, "with_filter"):
        scoped: Any = retriever.with_filter(table_filter, records)
        docs = scoped.invoke(query)
        removed.update(scoped.last_removed)
    else:
        docs = []
        for d in retriever.invoke(query):
            record = records.get(d.metadata["key"])
            reason = why_excluded(record, table_filter) if record is not None else None
            if reason is None:
                docs.append(d)
            else:
                removed[d.metadata["key"]] = reason
        docs = soft_rank(docs, records, table_filter)
    return docs[:k_each], removed


def rrf_fuse(lists: list[list[Document]], names: list[str] | None = None, *, k: int = 50, c: int = 60,
             score_from: int = 0, weights: list[float] | None = None) -> list[Document]:
    """Fuse ranked lists by weighted reciprocal rank over table keys (T2-06; weights T2-39).

    In: lists = ranked lists (dense first); names = one label per list for rank_<name> (default r0, r1, ...);
        k = tables returned; c = RRF constant; score_from = index of the list whose score stays in metadata["score"];
        weights = one weight per list (default 1.0 each: RRF = sum of 1 / (c + rank)).
    Out: at most k tables in fused order (ties keep first-seen order); each carries metadata score (the score_from
        list's score, or that list's lowest score for a table only other lists found), score_from (its name or
        "other"), rrf_score and rank_<name> per list it appeared in. The first list a table appears in gives its text.
    Side effects: none.
    Node: fuse (and hybrid_search).
    """
    names = names or [f"r{i}" for i in range(len(lists))]
    weights = weights or [1.0] * len(lists)
    rrf: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    first: dict[str, Document] = {}
    for name, w, docs in zip(names, weights, lists):
        for rank, d in enumerate(docs, 1):
            key = d.metadata["key"]
            rrf[key] = rrf.get(key, 0.0) + w / (c + rank)
            ranks.setdefault(key, {})[name] = rank
            first.setdefault(key, d)

    # keep the score_from list's score (the no-answer threshold is calibrated on it, T2-09)
    anchor = lists[score_from] if score_from < len(lists) else []
    own = {d.metadata["key"]: d.metadata.get("score") for d in anchor}
    floor = min((v for v in own.values() if v is not None), default=None)
    order = sorted(rrf, key=lambda key: -rrf[key])  # stable: ties keep first-seen order
    out = []
    for key in order[:k]:
        d = first[key]
        score, source = (own[key], names[score_from]) if key in own else (floor, "other")
        out.append(Document(page_content=d.page_content, id=d.id, metadata={
            **d.metadata, "score": score, "score_from": source, "rrf_score": rrf[key],
            **{f"rank_{n}": r for n, r in ranks[key].items()}}))
    return out


def hybrid_search(query: str, retrievers: list[BaseRetriever], *, names: list[str] | None = None,
                  table_filter: TableFilter | None = None, records: dict[str, TableRecord] | None = None,
                  k: int = 50, k_each: int = 50, c: int = 60, score_from: int = 0
                  ) -> tuple[list[Document], dict[str, str]]:
    """Search every retriever under the question's conditions and fuse the lists by reciprocal rank (T2-06).

    In: query = the user question; retrievers = the lists to fuse (dense first, then BM25); names = one label per
        retriever for the rank_<name> metadata (default r0, r1, ...); table_filter = conditions from the question
        (None or empty: no filtering); records = the registry the filter is decided on (key -> TableRecord);
        k = tables returned; k_each = top of each list kept before fusing; c = RRF constant;
        score_from = index of the retriever whose score stays in metadata["score"].
    Out: (docs, removed). docs = at most k tables in RRF order (ties keep first-seen order); each carries
         metadata score (the score_from list's score, or that list's lowest score for a table only other lists
         found), score_from (its name or "other"), rrf_score and rank_<name> per list it appeared in.
         removed = table key -> why the filter dropped it.
    Side effects: calls each retriever (Chroma query / BM25 scoring); raises ValueError when a non-empty filter has
        no records to decide on. A filter no table meets is relaxed once here for both lists
        (filters.filter_with_relaxation, T2-29); a caller that wants the dropped conditions computes them first and
        passes the relaxed filter (HybridRetriever).
    Node: not a node itself since T2-39: the graph runs its three parts as nodes (condition_filter in condition,
        search_list in retrieve_*, rrf_fuse in fuse); the old path (HybridRetriever) calls it.
    """
    # 1. one list per retriever: a retriever with with_filter (Chroma) filters before its search, the others after
    filtered = table_filter is not None and not table_filter.is_empty()
    table_filter = condition_filter(table_filter, records)
    lists: list[list[Document]] = []
    removed: dict[str, str] = {}
    for r in retrievers:
        docs, dropped = search_list(query, r, table_filter=table_filter, records=records, k_each=k_each,
                                    filtered=filtered)
        lists.append(docs)
        removed.update(dropped)

    # 2. fuse by RRF over table keys, keeping the score_from list's score
    return rrf_fuse(lists, names, k=k, c=c, score_from=score_from), removed


class HybridRetriever(BaseRetriever):
    """Reciprocal rank fusion of several retrievers by table key (T2-06): RRF = sum over lists of 1 / (c + rank).

    Each list is cut to its top `k_each` before fusing. With `condition` (question -> TableFilter, e.g.
    `parse_query(q).to_filter()`), a retriever that has `with_filter` (Chroma) filters before its search and the others
    are filtered after, so every list holds only tables that meet the question's conditions (give BM25 a large k).

    Order comes from RRF, but metadata["score"] stays the score of `retrievers[score_from]` (the dense cosine), since
    RRF values sit in a narrow band (1/61..2/61) and the no-answer threshold (T2-09) needs a calibrated score.
    A table only the other lists found gets the lowest score of that list, marked `score_from="other"`.

    The work is done by `hybrid_search`; this class is the retriever-shaped wrapper the old path (build_search,
    chain.answer, the eval runners) uses. Side effects: sets `self.last_removed` and `self.last_relaxed` on every
    query (tests read them).
    Node: not used by the graph (its condition, retrieve_* and fuse nodes run the parts of hybrid_search).
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
    last_relaxed: list[str] = Field(default_factory=list)  # conditions dropped on the last query (T2-29, D3)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        """Fused tables for the question: the condition (if any) is read here, then hybrid_search does the work.

        In: query = the user question. Out: hybrid_search's documents. Side effects: calls `self.condition`
        (parse_query in the pipeline) and the retrievers; replaces `self.last_removed` and `self.last_relaxed`.
        Node: not used by the graph.
        """
        f = self.condition(query) if self.condition else None
        self.last_removed, self.last_relaxed = {}, []
        if f is not None and not f.is_empty() and self.records:
            _, self.last_relaxed, _ = filter_with_relaxation(self.records, f)
            f = relax(f, self.last_relaxed)
        docs, self.last_removed = hybrid_search(
            query, self.retrievers, names=self.names, table_filter=f, records=self.records,
            k=self.k, k_each=self.k_each, c=self.c, score_from=self.score_from)
        return docs
