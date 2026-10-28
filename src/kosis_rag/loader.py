"""Load the Phase 1 table catalog (data/tables.csv) as TableInfo rows and turn each table into one Document."""

from __future__ import annotations

import csv
from pathlib import Path

from langchain_core.documents import Document

from kosis_ai.schemas import TableInfo

from kosis_rag.schemas import TableCandidate

# Fields copied into Document.metadata; enough to rebuild a TableCandidate without the catalog.
_META_FIELDS = ("key", "org_id", "tbl_id", "table_name", "survey_name", "organization", "period", "url")


def load_catalog(path: str | Path) -> list[TableInfo]:
    """Read tables.csv; every row is validated by the Phase 1 TableInfo schema (raises on a bad row)."""
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return [TableInfo.model_validate(row) for row in csv.DictReader(f)]


def to_document(info: TableInfo) -> Document:
    """One table = one Document (no chunking), so a hit is always a whole table with its id."""
    # ponytail: catalog fields only; category / item names from table_metadata/*.json go here when a test shows the gain.
    text = "\n".join([
        f"통계표: {info.table_name}",
        f"조사명: {info.survey_name}",
        f"작성기관: {info.organization}",
        f"수록기간: {info.period}",
        f"분류 경로: {info.menu_path}",
    ])
    return Document(page_content=text, metadata={name: getattr(info, name) for name in _META_FIELDS}, id=info.key)


def to_candidate(doc: Document) -> TableCandidate:
    return TableCandidate(**{name: doc.metadata[name] for name in _META_FIELDS}, score=doc.metadata.get("score"))
