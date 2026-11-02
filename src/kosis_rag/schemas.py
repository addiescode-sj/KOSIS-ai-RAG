"""Phase 2 I/O contract: table records, search candidates and the guide answer. Shared models (TableInfo, Source, ...) come from Phase 1 `kosis_ai.schemas`."""

from __future__ import annotations

from pydantic import BaseModel, Field

from kosis_ai.schemas import Source, TableInfo


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


class TableClass(BaseModel):
    """One classification of a table with every label that appears in its values."""

    name: str  # e.g. "행정구역별"
    labels: list[str] = Field(default_factory=list)
    n_labels: int = 0  # len(labels), kept as a scalar for vector-store metadata
    # region classes only: label count per level from the KOSIS codes, e.g. {"전국": 1, "시도": 17, "시군구": 250}
    levels: dict[str, int] = Field(default_factory=dict)


class TableRecord(BaseModel):
    """One line of data/tables.jsonl (T2-01): what stage 2 searches and filters on.

    Listed facts come from tables.csv, the catalog and the KOSIS table metadata; periods, items and labels come
    from the values themselves, so a 5-year survey keeps cycle ["F"] while its periods say which years exist.
    """

    info: TableInfo
    table_name_en: str | None = None
    topic: str | None = None
    menu_paths: list[str] = Field(default_factory=list)
    unit: str | None = None  # table-wide unit from the metadata (kosis_ai.normalize._meta_unit), None when not a known unit
    cycle: list[str] = Field(default_factory=list)  # survey cycle codes from the metadata, e.g. ["F"], ["M", "Q", "Y"]
    contain_period: str = ""  # metadata wording, e.g. "5년 1995 ~ 2020"
    prd_se: list[str] = Field(default_factory=list)  # period codes found in the values
    periods: list[str] = Field(default_factory=list)  # every period found in the values, sorted
    start: int | None = None  # first year in periods
    end: int | None = None  # last year in periods
    items: list[str] = Field(default_factory=list)
    classes: list[TableClass] = Field(default_factory=list)
    search_text: str = ""  # one document for BM25 and embeddings (T2-04, T2-05)
    issues: list[str] = Field(default_factory=list)  # where the sources disagree, e.g. "end_differs:2024!=2025"
