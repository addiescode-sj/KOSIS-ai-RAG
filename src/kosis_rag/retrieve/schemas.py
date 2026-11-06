"""What the retrieve stage's steps read and write (T2-41 S2). BoundaryModel: strict, extra fields refused."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from pydantic import Field

from kosis_rag.shared.boundary import BoundaryModel
from kosis_rag.shared.filters import TableFilter


class EmbedInput(BoundaryModel):
    """In of embed: the user question."""

    question: str


class EmbedOutput(BoundaryModel):
    """Out of embed: list name -> question vector (lists sharing one embed_query share one vector), embedder calls."""

    query_vecs: dict[str, Any] = Field(default_factory=dict)  # list name -> vector (list of floats)
    calls: int = 0


class ListInput(BoundaryModel):
    """In of retrieve_list: the question, the shared filter (from condition) and this list's vector (from embed)."""

    question: str
    table_filter: TableFilter | None = None
    vector: Any = None  # list of floats, or None: the list embeds the question itself


class ListOutput(BoundaryModel):
    """Out of retrieve_list (= search_list's tuple): ranked docs, why tables were dropped, relaxed, missing."""

    docs: list[Document] = Field(default_factory=list)
    removed: dict[str, str] = Field(default_factory=dict)
    relaxed: list[str] = Field(default_factory=list)
    missing: int = 0


class FuseInput(BoundaryModel):
    """In of fuse: each list's ranked docs and removed tables, by list name (the graph's runs / run_removed)."""

    runs: dict[str, list[Document]] = Field(default_factory=dict)
    run_removed: dict[str, dict[str, str]] = Field(default_factory=dict)


class FuseOutput(BoundaryModel):
    """Out of fuse: the candidate pool in RRF order, and key -> why filtered out (lists merged in fuse order)."""

    docs: list[Document] = Field(default_factory=list)
    removed: dict[str, str] = Field(default_factory=dict)
