"""Table registry records (data/tables.jsonl, T2-01) (T2-41 S1: from schemas.py).

TableInfo comes from the Phase 1 copy in `kosis_rag.schemas.common` (P2-D0).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from kosis_rag.schemas.common import TableInfo


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
    unit: str | None = None  # table-wide unit from the metadata (shared.units.meta_unit, Phase 1 copy), None when not a known unit
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
