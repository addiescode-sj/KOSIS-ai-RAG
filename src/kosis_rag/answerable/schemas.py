"""What the answerability stage's step reads and writes (KOS-193). BoundaryModel: strict, extra fields refused."""

from __future__ import annotations

from langchain_core.documents import Document
from pydantic import Field

from kosis_rag.shared.boundary import BoundaryModel


class AnswerInput(BoundaryModel):
    """In of answerable: the question and the candidates in their current (stage 1) order."""

    question: str
    docs: list[Document] = Field(default_factory=list)


class AnswerOutput(BoundaryModel):
    """Out of answerable: the docs in the same order with answer_* metadata, the best logit of the scored docs (None
    when none was scored), whether the question is refused (best logit < answer_theta) and how many docs were scored."""

    docs: list[Document] = Field(default_factory=list)
    top_score: float | None = None
    refused: bool = False
    n_scored: int = 0
