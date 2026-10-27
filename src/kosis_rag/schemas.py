"""Stage 2 I/O contract. Shared models (TableInfo, Source, ...) come from stage 1 `kosis_ai.schemas`."""

from __future__ import annotations

from pydantic import BaseModel, Field

from kosis_ai.schemas import TableInfo


class TableClass(BaseModel):
    """One classification of a table with every label that appears in its values."""

    name: str  # e.g. "행정구역별"
    labels: list[str] = Field(default_factory=list)
    n_labels: int = 0  # len(labels), kept as a scalar for vector-store metadata


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
