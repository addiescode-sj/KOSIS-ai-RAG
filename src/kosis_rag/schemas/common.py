# copied from Phase 1 src/kosis_ai/schemas.py@e248a45
"""Phase 1 shared models Phase 2 uses (P2-D0): TableInfo (a row of data/tables.csv), Source and MAX_DIMS.

A copy, so Phase 2 does not import Phase 1 for its own records; tests/test_phase1_contract.py checks it still
matches the Phase 1 models (fields, types, defaults, validation, JSON of the sample catalog). Change it only together
with Phase 1.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_DIMS = 8  # dim1..dim8 columns in the Parquet files


class Contract(BaseModel):
    """Base of every shared model: unknown fields are an error, never dropped."""

    model_config = ConfigDict(extra="forbid")


class Source(Contract):
    org_id: str = Field(pattern=r"^\d{3}$")  # KOSIS agency code, e.g. "101"
    tbl_id: str = Field(min_length=1)
    survey_nm: str = Field(min_length=1)
    tbl_nm: str = Field(min_length=1)
    url: str | None = None

    def citation(self) -> str:
        """'KOSIS / survey / table (url)' — built by code, never by the LLM."""
        text = f"KOSIS / {self.survey_nm} / {self.tbl_nm}"
        return f"{text} ({self.url})" if self.url else text


class TableInfo(Contract):
    """One row of data/tables.csv: a table in the provided scope and where its values are."""

    key: str  # "<org_id>_<tbl_id>"
    org_id: str
    tbl_id: str
    survey_name: str
    organization: str
    table_name: str
    period: str  # listed period, e.g. "년 (2015 ~ 2025)"
    source: Literal["api", "zip"]
    coverage: str  # e.g. "A_정상", "B_수록시점차이", "C_수집누락"
    missing_periods: str = ""  # empty when nothing is missing
    rows: int
    numeric_rows: int
    symbol_rows: int
    parquet: str  # path relative to data/, e.g. "values/101_DT_1B8000H.parquet"
    url: str
    menu_path: str

    def to_source(self) -> Source:
        return Source(org_id=self.org_id, tbl_id=self.tbl_id, survey_nm=self.survey_name, tbl_nm=self.table_name, url=self.url)
