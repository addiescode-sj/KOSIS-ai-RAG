"""Phase 2 I/O contract: search candidates and the guide answer. Shared types stay in kosis_ai.schemas (Phase 1)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from kosis_ai.schemas import Source


class TableCandidate(BaseModel):
    """One table proposed for a question. tbl_id is kept from load to answer so Hit@k can be graded on it."""

    key: str  # "<org_id>_<tbl_id>", same as TableInfo.key
    org_id: str
    tbl_id: str
    table_name: str
    survey_name: str
    organization: str
    period: str  # listed period, e.g. "년 (2015 ~ 2025)"
    url: str
    score: float | None = None  # retriever score; None when the retriever gives none

    def to_source(self) -> Source:
        return Source(org_id=self.org_id, tbl_id=self.tbl_id, survey_nm=self.survey_name, tbl_nm=self.table_name, url=self.url)


class SearchArgs(BaseModel):
    """Tool input for search_tables / answer_table_question."""

    question: str = Field(min_length=1, description="User question in Korean")
    k: int = Field(default=5, ge=1, le=20, description="Number of candidate tables")


class SearchResult(BaseModel):
    """Return value of answer_table_question: candidates, the sentence, and how the sentence was made."""

    question: str
    candidates: list[TableCandidate] = Field(default_factory=list)
    no_answer: bool = False
    answer: str  # final text; citations are appended by code
    answer_origin: str  # "llm" | "template" (LLM failed or failed verification) | "refusal"
    llm_status: str | None = None  # Phase 1 LLMResult.status when the model was called
