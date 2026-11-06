"""Dense search over a T2-24 Chroma table collection (T2-41 S2: the body of retriever.ChromaTableRetriever).

Score = 1 - cosine distance (higher is closer). The collection holds vectors we computed ourselves
(embedding_function=None), so the query must be embedded with the same model; opening the collection and loading that
embedder is the caller's job (legacy ChromaTableRetriever.connect until S4, then adapters/loader).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from kosis_rag.retrieve.constants import LIST_K
from kosis_rag.schemas.catalog import TableRecord
from kosis_rag.shared.filters import TableFilter, filter_with_relaxation, relax, soft_rank


@dataclass
class DenseIndex:
    """A table collection and its query embedder; `search` is stateless (what it removed / relaxed is returned)."""

    collection: Any
    embed_query: Callable[[str], list[float]]
    k: int = LIST_K  # upper bound; callers slice to their own k (SearchArgs.k <= 20)
    doc_format: str | None = None  # collection metadata doc_format (a kosis_rag.shared.tabletext format, fine-tuned)

    def search(self, query: str, *, vector: list[float] | None = None, table_filter: TableFilter | None = None,
               records: dict[str, TableRecord] | None = None
               ) -> tuple[list[Document], dict[str, str], list[str], int]:
        """Top-k tables by cosine similarity, under the question's conditions.

        In: query = the user question; vector = its embedding when already made (the embed node), else embed_query is
            called; table_filter = conditions (None or empty: no filtering; soft signals only re-rank); records = the
            registry the filter is decided on (key -> TableRecord).
        Out: (docs, removed, relaxed, missing). A non-empty filter is decided on records and sent as one `key $in`
            condition; when no table meets it the conditions are relaxed one at a time (filters.filter_with_relaxation,
            T2-29) and the dropped ones are `relaxed`. Only a year no table holds gives no result (no query is made).
            removed = table key -> why the filter dropped it; missing is always 0 (every hit is a registry table).
        Side effects: one Chroma query (and the embedder when no vector is given); ValueError for a non-empty filter
            without records. Node: retrieve_dense (through retrieve.lists.search_list).
        """
        # TODO(T2-09): Chroma always returns k hits, so the chain's "no candidates" refusal never fires here;
        # refusal relies on the LLM's NO_MATCH until a cosine threshold is calibrated on the L1 dev split.
        f = table_filter if table_filter is not None else TableFilter()
        records = records or {}
        extra, n = {}, self.k
        removed: dict[str, str] = {}
        relaxed: list[str] = []
        if not f.is_empty():
            if not records:
                raise ValueError("a table filter needs records (tables.jsonl) to decide on")
            keys, relaxed, removed = filter_with_relaxation(records, f)
            f = relax(f, relaxed)
            if not keys:  # no table holds the requested years (the one condition never relaxed); reasons in removed
                return [], removed, relaxed, 0
            extra, n = {"where": {"key": {"$in": sorted(keys)}}}, min(self.k, len(keys))
        res = self.collection.query(query_embeddings=[vector if vector is not None else self.embed_query(query)],
                                    n_results=n, include=["documents", "metadatas", "distances"], **extra)
        docs = [
            Document(page_content=doc, metadata={**meta, "score": 1.0 - dist}, id=key)
            for key, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]
        return soft_rank(docs, records, f), removed, relaxed, 0
