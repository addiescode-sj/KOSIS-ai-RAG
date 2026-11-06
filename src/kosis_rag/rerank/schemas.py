"""What the rerank stage's steps read and write (T2-41 S3). BoundaryModel: strict, extra fields refused."""

from __future__ import annotations

from langchain_core.documents import Document
from pydantic import Field

from kosis_rag.shared.boundary import BoundaryModel


class Stage1Input(BoundaryModel):
    """In of rerank1: the question and the candidates in their current (fused) order."""

    question: str
    docs: list[Document] = Field(default_factory=list)


class Stage1Output(BoundaryModel):
    """Out of rerank1: the top rerank_n re-ordered (stage1_* metadata), then the rest as given."""

    docs: list[Document] = Field(default_factory=list)


class Stage2Input(BoundaryModel):
    """In of rerank2: the question and the candidates in their current order."""

    question: str
    docs: list[Document] = Field(default_factory=list)


class Stage2Output(BoundaryModel):
    """Out of rerank2: picked docs first, then the rest; every doc carries the stage2_* metadata."""

    docs: list[Document] = Field(default_factory=list)
