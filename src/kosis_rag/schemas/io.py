"""Phase 2 I/O contract: search candidates, tool and API inputs, and the guide answer (T2-41 S1: from schemas.py);
the module's own input and output, SearchInput / SearchOutput (S4, plans/T2-41.md 입출력 계약; fixed at S5).

Source comes from the Phase 1 copy in `kosis_rag.schemas.common` (P2-D0).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from kosis_rag.schemas.common import Source


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


class AskArgs(BaseModel):
    """Input of the answer API (/search, /search/stream): k is the guide LLM's candidate budget, a server setting."""

    question: str = Field(min_length=1, description="사용자 질문 (한국어)")
    k: int | None = Field(default=None, ge=1, le=20,
                          description="안내 LLM에 넘길 후보 표 수. 생략하면 서버 설정(ANSWER_K, 기본 5). "
                                      "답변에 보여 줄 표는 LLM이 고른 표(picked)다")


class SearchResult(BaseModel):
    """Return value of answer_table_question: candidates, the sentence, and how the sentence was made."""

    question: str
    candidates: list[TableCandidate] = Field(default_factory=list)  # the tables shown to the guide LLM (top k)
    picked: list[TableCandidate] = Field(default_factory=list)  # the ones the answer recommends and cites (T2-39); [] on refusal
    no_answer: bool = False
    no_answer_reason: str | None = None  # out_of_scope | no_item when the stage 2 LLM said why (T2-09)
    answer: str  # final text; citations are appended by code
    answer_origin: str  # "llm" | "template" (LLM failed or failed verification) | "refusal"
    llm_status: str | None = None  # Phase 1 LLMResult.status when the model was called


class SearchInput(BaseModel):
    """Input of kosis_rag.api.SearchModule.search (T2-41 S4)."""

    schema_version: Literal["2.0"] = "2.0"
    question: str = Field(min_length=1)
    k: int | None = Field(default=None, ge=1, le=20)  # candidate budget docs[:k]; None = options.answer_k (as AskArgs.k)


class SearchOutput(BaseModel):
    """Output of SearchModule.search (T2-41 S4): gate's decision and the tables, no sentence (that is the app's).

    route "found" means gate refused nothing: there are candidates (stage 2 picked some, or stage 2 is off). It does
    not mean Phase 1 can compute on them; hand Phase 1 `picked[0].key` (the tables.csv key "101_DT_..."), not tbl_id.
    """

    schema_version: Literal["2.0"] = "2.0"
    question: str
    route: Literal["found", "refuse"]
    refusal_reason: str | None = None  # gate: stage2_reason (out_of_scope | no_item) or None
    refusal_text: str | None = None  # on refuse: gate.texts REFUSALS.get(reason, REFUSAL_TEXT) (code text, no LLM)
    picked: list[TableCandidate] = Field(default_factory=list)  # found: stage 2's picks (rank order), else candidates
    candidates: list[TableCandidate] = Field(default_factory=list)  # docs[:k] (kept on refuse when there are docs)
    spans: list[dict[str, Any]] = Field(default_factory=list)  # {"node": name, **trace[name]} in node order
