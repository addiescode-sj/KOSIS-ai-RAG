"""BM25, ChromaDB, hybrid (RRF), condition and question-index retrievers (T2-41 S2: from retriever.py, query.py and
hq/index.py). LangChain BaseRetriever wrappers whose bodies call the stage functions (retrieve/, parse/), P2-D18.

Each reports a score in Document.metadata["score"] (the no-answer check needs it); the hybrid keeps the dense cosine
there and orders by RRF (T2-06). Unlike the stage functions they keep what the last query removed / relaxed in
`last_*` attributes, as their callers (build_search, chain.answer, the eval runners, lab, tests) read them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from kosis_rag.parse.constants import MAX_RESULTS
from kosis_rag.parse.node import parse_query
from kosis_rag.parse.schemas import ParsedQuery
from kosis_rag.retrieve.bm25 import Bm25Index, bm25_index, bm25_search
from kosis_rag.retrieve.constants import HQ_DOC_FORMAT, HQ_N_QUESTIONS
from kosis_rag.retrieve.dense import DenseIndex
from kosis_rag.retrieve.fuse import rrf_fuse
from kosis_rag.retrieve.hq import HqIndex, collection_name
from kosis_rag.retrieve.lists import condition_filter, search_list
from kosis_rag.schemas.catalog import TableRecord
from kosis_rag.schemas.resources import ChromaConfig
from kosis_rag.shared.filters import TableFilter, filter_with_relaxation, relax, soft_rank, why_excluded


class BM25TableRetriever(BaseRetriever):
    """Top-k tables by BM25 (retrieve.bm25.bm25_search). Score 0 means no query token appears in any table text."""

    docs: list[Document]
    k: int = 20  # upper bound; callers slice to their own k (SearchArgs.k <= 20)
    bm25: Any = None
    query_text: Callable[[str], str] | None = None  # e.g. text.norm_query when the documents are n2 text
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_documents(cls, docs: list[Document], k: int = 20,
                       query_text: Callable[[str], str] | None = None) -> BM25TableRetriever:
        return cls.from_index(bm25_index(docs, k=k, query_text=query_text))

    @classmethod
    def from_index(cls, index: Bm25Index) -> BM25TableRetriever:
        """The retriever over a retrieve.bm25.Bm25Index (bm25_text, hq_bm25)."""
        return cls(docs=index.docs, k=index.k, bm25=index.bm25, query_text=index.query_text)

    @property
    def index(self) -> Bm25Index:
        return Bm25Index(docs=self.docs, bm25=self.bm25, k=self.k, query_text=self.query_text)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return bm25_search(self.index, query)


class ChromaTableRetriever(BaseRetriever):
    """Top-k tables by cosine similarity over a T2-24 collection (retrieve.dense.DenseIndex.search). Score = 1 - cosine
    distance (higher is closer).

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
    doc_format: str | None = None  # collection metadata doc_format (kosis_rag.shared.tabletext format for fine-tuned collections)
    table_filter: TableFilter = Field(default_factory=TableFilter)
    last_removed: dict[str, str] = Field(default_factory=dict)  # why each table was filtered out on the last query
    last_relaxed: list[str] = Field(default_factory=list)  # conditions dropped on the last query (filters.RELAX_ORDER)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def connect(cls, name: str | None = None, k: int = 20, device: str | None = None,
                embed_ckpt: str | None = None, *, chroma: ChromaConfig | None = None) -> ChromaTableRetriever:
        """Open `name` (else chroma.collection) on the shared server and load the query embedder it was built with.

        KURE-v1 collections (T2-24 text) need nothing more. Fine-tuned bge-m3 collections (T2-06, kosis_rag.shared.tabletext
        formats) need the same checkpoint (`embed_ckpt`); a different file is refused, and questions get the same
        surface rules as the documents. T2-41 S4: reads no .env and no environment — the server is `chroma`
        (default ChromaConfig(): localhost:8001) and the caller names the collection and the checkpoint.
        """
        from kosis_rag.adapters.chroma import chroma_client
        from kosis_rag.adapters.models import EMBED_MODEL, kure_embedder

        chroma = chroma or ChromaConfig()
        name = name or chroma.collection
        if not name:
            raise ValueError("no collection: pass ChromaConfig(collection=...) (backend: CHROMA_COLLECTION in the shell "
                             "environment) naming the collection built by `python -m kosis_rag.index`")
        client = chroma_client(chroma)
        collection = client.get_collection(name, embedding_function=None)
        meta = collection.metadata or {}
        built_with = meta.get("embed_model")
        if built_with == EMBED_MODEL:
            embed, _, _ = kure_embedder(device)
            return cls(collection=collection, embed_query=lambda text: embed([text])[0], k=k,
                       doc_format=meta.get("doc_format"))
        from kosis_rag.adapters import models as model
        from kosis_rag.shared import tabletext as text

        if built_with != model.EMBED_BASE:
            raise ValueError(f"collection {name} was embedded with {built_with!r}; known: {EMBED_MODEL}, {model.EMBED_BASE}")
        fmt = meta.get("doc_format")
        if fmt not in text.FORMATS:
            raise ValueError(f"collection {name} holds {fmt!r} text; this code knows {', '.join(text.FORMATS)}")
        embedder = model.load_embedder(embed_ckpt, device=device, doc_format=fmt)
        if embedder.ckpt != meta.get("embed_ckpt"):
            raise ValueError(f"collection {name} was embedded with {meta.get('embed_ckpt')!r}, "
                             f"the query embedder loaded {embedder.ckpt!r}; set EMBED_CKPT to the same file")
        return cls(collection=collection, embed_query=lambda q: embedder.embed_query(text.norm_query(q)), k=k,
                   doc_format=fmt)

    @property
    def index(self) -> DenseIndex:
        """The stateless dense index over this retriever's collection and embedder."""
        return DenseIndex(collection=self.collection, embed_query=self.embed_query, k=self.k, doc_format=self.doc_format)

    def with_filter(self, f: TableFilter, records: dict[str, TableRecord] | None = None) -> ChromaTableRetriever:
        """A copy that applies `f`; the tools keep their search_tables(question, k) signature."""
        return self.model_copy(update={"table_filter": f, "records": records or self.records, "last_removed": {},
                                       "last_relaxed": []})

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        self.last_removed, self.last_relaxed = {}, []
        docs, self.last_removed, self.last_relaxed, _ = self.index.search(query, table_filter=self.table_filter,
                                                                          records=self.records)
        return docs


class HqRetriever(BaseRetriever):
    """Top-k tables by their closest hypothetical questions (retrieve.hq.HqIndex.search: max-pool over the top
    n_questions; score = 1 - cosine distance of the best question). Filters like ChromaTableRetriever: `with_filter`
    decides the TableFilter on `records`, relaxes it when no table meets it (last_relaxed) and sends `key $in`.

    Documents carry the table's catalog metadata, key, score, hq_best_question, hq_best_id and hq_n, and the
    table's n2 text as page_content (text_of). A question whose table key is not in `records` is skipped, counted in
    last_missing and logged.
    """

    collection: Any
    embed_query: Callable[[str], list[float]]
    text_of: dict[str, str] = Field(default_factory=dict)  # key -> n2 embedding text (the re-ranker passage)
    records: dict[str, TableRecord] = Field(default_factory=dict)
    k: int = 50
    n_questions: int = HQ_N_QUESTIONS
    n_docs: int | None = None  # questions in the collection (n_results never asks for more)
    table_filter: TableFilter = Field(default_factory=TableFilter)
    last_removed: dict[str, str] = Field(default_factory=dict)
    last_relaxed: list[str] = Field(default_factory=list)
    last_missing: int = 0  # tables of the last query's hits not in records (skipped)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def connect(cls, build_id: str, dense: Any, *, records: dict[str, TableRecord], text_of: dict[str, str],
                k: int = 50, name: str | None = None, chroma: ChromaConfig | None = None) -> HqRetriever:
        """Open the question collection of `build_id` for the dense side's embedder.

        In: build_id = HQ_BUILD; dense = the table retriever (ChromaTableRetriever: its embed_query and its
            collection's embed_model / embed_ckpt are reused); records, text_of = the registry and n2 passages;
            k = tables returned; name = collection override; chroma = the server (default ChromaConfig(); T2-41 S4:
            no environment read). Out: HqRetriever.
        Side effects: connects to Chroma; ValueError when the collection was embedded with other weights.
        Node: none (load_resources calls it; retrieve_hq searches it).
        """
        from kosis_rag.adapters.chroma import chroma_client

        meta = (getattr(getattr(dense, "collection", None), "metadata", None) or {})
        ckpt = meta.get("embed_ckpt") or meta.get("embed_model")
        name = name or collection_name(build_id, ckpt)
        collection = chroma_client(chroma or ChromaConfig()).get_collection(name, embedding_function=None)
        hq_meta = collection.metadata or {}
        if (hq_meta.get("embed_ckpt") or hq_meta.get("embed_model")) != ckpt or hq_meta.get("doc_format") != HQ_DOC_FORMAT:
            raise ValueError(f"question collection {name} was embedded with {hq_meta.get('embed_ckpt')!r} "
                             f"({hq_meta.get('doc_format')!r}); the dense side uses {ckpt!r}")
        return cls(collection=collection, embed_query=dense.embed_query, text_of=text_of, records=records, k=k,
                   n_docs=collection.count())

    @property
    def index(self) -> HqIndex:
        """The stateless question index over this retriever's collection, embedder, passages and registry."""
        return HqIndex(collection=self.collection, embed_query=self.embed_query, text_of=self.text_of,
                       records=self.records, k=self.k, n_questions=self.n_questions, n_docs=self.n_docs)

    def with_filter(self, f: TableFilter, records: dict[str, TableRecord] | None = None) -> HqRetriever:
        """A copy that applies `f` (as ChromaTableRetriever.with_filter).

        Side effects: none. Node: retrieve_hq (through retrieve.lists.search_list).
        """
        return self.model_copy(update={"table_filter": f, "records": records or self.records, "last_removed": {},
                                       "last_relaxed": [], "last_missing": 0})

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        """Tables for the question through their questions (HqIndex.search); sets last_removed / last_relaxed /
        last_missing. Node: retrieve_hq."""
        self.last_removed, self.last_relaxed, self.last_missing = {}, [], 0
        docs, self.last_removed, self.last_relaxed, self.last_missing = self.index.search(
            query, table_filter=self.table_filter)
        return docs


class ConditionRetriever(BaseRetriever):
    """Parses each question (parse.node.parse_query with `sigungu`) and applies the conditions to `base`.

    A base with `with_filter` (the Chroma retriever) filters before the vector search; any other base (BM25) is asked
    for its candidates first and filtered after, so give it a large k. At most MAX_RESULTS (10) tables come back.
    When no table meets the conditions they are relaxed one at a time (filters.filter_with_relaxation); the dropped
    ones are in `last_relaxed`. T2-41 S2: the 시군구 dictionary is the `sigungu` field (default {}: 시군구 names are not
    read); it used to be query.SIGUNGU, loaded from data/sigungu.json at import.
    """

    base: BaseRetriever
    records: dict[str, TableRecord]
    region_mode: Literal["hard", "soft"] = "hard"
    k: int = MAX_RESULTS
    sigungu: dict[str, list[str]] = Field(default_factory=dict)  # 시군구 -> 시도 names (adapters.files.load_sigungu)
    last_query: ParsedQuery | None = None
    last_removed: dict[str, str] = Field(default_factory=dict)
    last_relaxed: list[str] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        self.last_query = parsed = parse_query(query, sigungu=self.sigungu)
        f = parsed.to_filter(self.region_mode)
        _, self.last_relaxed, _ = filter_with_relaxation(self.records, f)
        f = relax(f, self.last_relaxed)
        if hasattr(self.base, "with_filter"):
            filtered: Any = self.base.with_filter(f, self.records)
            docs = filtered.invoke(query)
            self.last_removed = dict(filtered.last_removed)
            return docs[: min(self.k, MAX_RESULTS)]
        docs, self.last_removed = [], {}
        for doc in self.base.invoke(query):
            record = self.records.get(doc.metadata["key"])
            reason = why_excluded(record, f) if record is not None else None
            if reason is None:
                docs.append(doc)
            else:
                self.last_removed[doc.metadata["key"]] = reason
        return soft_rank(docs, self.records, f)[: min(self.k, MAX_RESULTS)]


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
    Node: not a node itself since T2-39: the graph runs its three parts as nodes (decide_condition in condition,
        search_list in retrieve_*, rrf_fuse in fuse); the old path (HybridRetriever) calls it.
    """
    # 1. one list per retriever: a retriever with with_filter (Chroma) filters before its search, the others after
    filtered = table_filter is not None and not table_filter.is_empty()
    table_filter = condition_filter(table_filter, records)
    lists: list[list[Document]] = []
    removed: dict[str, str] = {}
    for r in retrievers:
        docs, dropped, _, _ = search_list(query, r, table_filter=table_filter, records=records, k_each=k_each,
                                          filtered=filtered)
        lists.append(docs)
        removed.update(dropped)

    # 2. fuse by RRF over table keys, keeping the score_from list's score
    return rrf_fuse(lists, names, k=k, c=c, score_from=score_from), removed


class HybridRetriever(BaseRetriever):
    """Reciprocal rank fusion of several retrievers by table key (T2-06): RRF = sum over lists of 1 / (c + rank).

    Each list is cut to its top `k_each` before fusing. With `condition` (question -> TableFilter, e.g.
    `parse_query(q, sigungu=...).to_filter()`), a retriever that has `with_filter` (Chroma) filters before its search
    and the others are filtered after, so every list holds only tables that meet the question's conditions (give BM25
    a large k).

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
