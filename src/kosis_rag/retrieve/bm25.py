"""BM25 over table texts (T2-41 S2: from retriever.BM25TableRetriever and pipeline.bm25_text).

Score 0 means no query token appears in any table text; those tables are not returned. Each hit carries its BM25
score in Document.metadata["score"].
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from kosis_rag.retrieve.constants import LIST_K
from kosis_rag.schemas.catalog import TableRecord
from kosis_rag.shared import tabletext as text
from kosis_rag.shared.catalog import to_document
from kosis_rag.shared.tokenize import tokenize


@dataclass
class Bm25Index:
    """An in-memory BM25 index: the documents, their BM25Okapi scorer, the top k and the question surface rule."""

    docs: list[Document]
    bm25: Any
    k: int = LIST_K  # upper bound; callers slice to their own k
    query_text: Callable[[str], str] | None = None  # e.g. tabletext.norm_query when the documents are n2 text

    def search(self, query: str) -> list[Document]:
        """bm25_search(self, query)."""
        return bm25_search(self, query)


def bm25_index(docs: list[Document], k: int = LIST_K, query_text: Callable[[str], str] | None = None) -> Bm25Index:
    """Index `docs` by their page_content (shared.tokenize). Side effects: none (in memory). Node: none."""
    return Bm25Index(docs=docs, bm25=BM25Okapi([tokenize(d.page_content) for d in docs]), k=k, query_text=query_text)


def bm25_search(index: Bm25Index, query: str) -> list[Document]:
    """Top-k documents by BM25 for the question.

    In: index = a Bm25Index; query = the user question (index.query_text applied first, when set).
    Out: at most index.k copies of the documents with metadata score = the BM25 score, best first (ties keep index
        order); score 0 is left out. Side effects: none. Node: retrieve_bm25, retrieve_hq_bm25 (through search_list).
    """
    scores = index.bm25.get_scores(tokenize(index.query_text(query) if index.query_text else query))
    top = sorted(range(len(index.docs)), key=lambda i: scores[i], reverse=True)[: index.k]
    return [
        Document(page_content=index.docs[i].page_content, metadata={**index.docs[i].metadata, "score": float(scores[i])}, id=index.docs[i].id)
        for i in top
        if scores[i] > 0
    ]


def bm25_text(records: list[TableRecord], k: int, fmt: str = text.DOC_FORMAT) -> Bm25Index:
    """BM25 over the full table text (every label) of `fmt`, questions with the same surface rules (lab F0).

    In: records = every table; k = tables each search returns; fmt = the table text format (kosis_rag.shared.tabletext).
    Out: a Bm25Index whose documents carry the catalog metadata (key, ids, names, period, url).
    Side effects: builds the BM25 index in memory. Node: none directly (built by load_resources; the retrieve_bm25
    node searches it).
    """
    docs = []
    for r in records:
        catalog = to_document(r.info)
        docs.append(Document(page_content=text.build_doc(r, fmt)["bm25_text"], metadata=catalog.metadata, id=catalog.id))
    return bm25_index(docs, k=k, query_text=text.norm_query)
