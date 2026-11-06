"""One ranked list under the question's conditions (T2-39; T2-41 S2: from retriever.py).

condition_filter is the hybrid's relaxation step; search_list searches one list. A list is a DenseIndex / HqIndex
(filter before the search, stateless), a Bm25Index (filter after), or a retriever-shaped object: one with
`with_filter` (legacy ChromaTableRetriever / HqRetriever, test stand-ins) filters before its search, one without
(BM25TableRetriever) is filtered after. Nothing here keeps state: what was removed or relaxed is returned.
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from kosis_rag.retrieve.bm25 import Bm25Index
from kosis_rag.retrieve.dense import DenseIndex
from kosis_rag.retrieve.hq import HqIndex
from kosis_rag.schemas.catalog import TableRecord
from kosis_rag.shared.filters import TableFilter, filter_with_relaxation, relax, soft_rank, why_excluded

INDEXES = (DenseIndex, HqIndex)  # the stateless pre-filtering lists (search(query, vector=, table_filter=, records=))


def condition_filter(table_filter: TableFilter | None, records: dict[str, TableRecord] | None) -> TableFilter | None:
    """The filter every list is searched under: hybrid_search's relaxation step, shared by the graph's lists (T2-39).

    In: table_filter = conditions from the question (None or empty: no filtering); records = the registry the filter
        is decided on. Out: the filter as given when it is None or empty; else the filter with the conditions no table
        meets dropped (filters.filter_with_relaxation, T2-29). A filter the caller relaxed already comes back the same.
    Side effects: raises ValueError when a non-empty filter has no records to decide on.
    Node: condition (parse.node.decide_condition runs the same step) and hybrid_search for the old path.
    """
    if table_filter is None or table_filter.is_empty():
        return table_filter
    if not records:
        raise ValueError("a condition needs records (tables.jsonl) to decide on")
    _, dropped, _ = filter_with_relaxation(records, table_filter)
    return relax(table_filter, dropped)


def _ranked(retriever: Any, query: str) -> tuple[list[Document], int]:
    """A list's own ranking with no filter, and how many hits it skipped (HqIndex / HqRetriever: missing tables)."""
    if isinstance(retriever, INDEXES):
        docs, _, _, missing = retriever.search(query)
        return docs, missing
    if isinstance(retriever, Bm25Index):
        return retriever.search(query), 0
    docs = retriever.invoke(query)
    return docs, getattr(retriever, "last_missing", 0)


def search_list(query: str, retriever: Any, *, table_filter: TableFilter | None = None,
                records: dict[str, TableRecord] | None = None, k_each: int = 50, filtered: bool | None = None,
                vector: list[float] | None = None) -> tuple[list[Document], dict[str, str], list[str], int]:
    """One list's ranked tables under the question's conditions (one list of hybrid_search, T2-39).

    In: query = the user question; retriever = one list (see the module docstring); table_filter = the filter from
        condition_filter (None or empty: no filtering; soft signals only re-rank); records = the registry; k_each =
        top kept; filtered = take the filtering path (default: when table_filter is non-empty). hybrid_search decides
        it on the filter before relaxation, as it always did, so a filter relaxed to empty still takes the filtering
        path there. vector = the question's embedding from the embed node: a DenseIndex / HqIndex searches with it,
        a retriever-shaped list gets a copy whose embed_query returns it (so the embedder is not called again).
    Out: (docs, removed, relaxed, missing). A pre-filtering list filters before its search, the others are filtered
        after and soft-ranked; then the top k_each. removed = table key -> why the filter dropped it; relaxed =
        conditions the list itself dropped (none for a filter condition_filter relaxed already); missing = hits
        skipped because their table is not in the registry (question index only).
    Side effects: calls the list (Chroma query / BM25 scoring); ValueError for a non-empty filter without records.
    Node: retrieve_dense, retrieve_bm25, retrieve_hq, retrieve_hq_bm25 (through retrieve.node.retrieve_list; and
        hybrid_search).
    """
    if filtered is None:
        filtered = table_filter is not None and not table_filter.is_empty()
    indexed = isinstance(retriever, INDEXES)
    if vector is not None and not indexed:  # embedded once by the embed node: this copy returns it
        retriever = retriever.model_copy(update={"embed_query": lambda _question, v=vector: v})
    if not filtered:
        if indexed:
            docs, _, _, missing = retriever.search(query, vector=vector)
        else:
            docs, missing = _ranked(retriever, query)
        docs = docs[:k_each]
        if table_filter is not None and records and table_filter.is_soft():
            docs = soft_rank(docs, records, table_filter)  # soft signals only re-rank
        return docs, {}, [], missing
    if not records:
        raise ValueError("a condition needs records (tables.jsonl) to decide on")
    removed: dict[str, str] = {}
    relaxed: list[str] = []
    missing = 0
    if indexed:
        docs, dropped, relaxed, missing = retriever.search(query, vector=vector, table_filter=table_filter,
                                                           records=records)
        removed.update(dropped)
    elif hasattr(retriever, "with_filter"):
        scoped: Any = retriever.with_filter(table_filter, records)
        docs = scoped.invoke(query)
        removed.update(scoped.last_removed)
        relaxed = list(getattr(scoped, "last_relaxed", []))
        missing = getattr(scoped, "last_missing", 0)
    else:
        docs = []
        ranked, missing = _ranked(retriever, query)
        for d in ranked:
            record = records.get(d.metadata["key"])
            reason = why_excluded(record, table_filter) if record is not None else None
            if reason is None:
                docs.append(d)
            else:
                removed[d.metadata["key"]] = reason
        docs = soft_rank(docs, records, table_filter)
    return docs[:k_each], removed, relaxed, missing
