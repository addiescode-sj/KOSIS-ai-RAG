"""Structured filters for table search (T2-05, D): year, cycle and organization decided on the table registry.

Filters are decided in Python on the registry (tables.jsonl, 2,364 records), not by Chroma range queries:
a table passes only if its real periods contain the requested years, so a table whose listed range covers a year
it does not actually hold (5-year censuses, collection gaps) is left out, and the reason is kept per table.
`retriever.ChromaTableRetriever` then gives Chroma one condition, `key $in <passing keys>`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from kosis_rag.schemas import TableRecord

# Same organization under its old and new names (통계청 became 국가데이터처)
ORG_ALIASES = {"통계청": "국가데이터처"}


class TableFilter(BaseModel):
    """Conditions taken from the question (T2-14 fills these); empty means no filtering."""

    years: list[int] = Field(default_factory=list)  # requested years, e.g. [2023] or [2019, 2023]
    cycle: str | None = None  # period code: Y, M, Q, F (multi-year), IR
    organization: str | None = None
    # strict: every requested year present; partial: at least one. On the golden dev set strict removed every gold
    # table for 2 of 14 questions with years (both ask for two years), partial removed none (plans/T2-05.md).
    mode: Literal["strict", "partial"] = "strict"

    def is_empty(self) -> bool:
        return not (self.years or self.cycle or self.organization)


def _org(name: str) -> str:
    name = name.strip()
    return ORG_ALIASES.get(name, name)


def why_excluded(record: TableRecord, f: TableFilter) -> str | None:
    """None when the table meets the filter, else the reason in words (for the no-answer message and the harness)."""
    if f.organization and _org(record.info.organization) != _org(f.organization):
        return f"작성기관 {record.info.organization} (요청 {f.organization})"
    if f.cycle and f.cycle not in set(record.cycle) | set(record.prd_se):
        return f"주기 {'·'.join(record.cycle or record.prd_se)} (요청 {f.cycle})"
    if f.years:
        years = {int(p[:4]) for p in record.periods if p[:4].isdigit()}
        if not years:
            return "실제 수록 시점 없음"
        hit = [y for y in f.years if y in years]
        ok = len(hit) == len(f.years) if f.mode == "strict" else bool(hit)
        if not ok:
            missing = [y for y in f.years if y not in years]
            cycle = f", 주기 {'·'.join(record.cycle)}" if record.cycle else ""
            return f"요청 {'·'.join(map(str, missing))}년 없음 (실제 {min(years)}~{max(years)}, {len(years)}개 시점{cycle})"
    return None


def filter_tables(records: dict[str, TableRecord], f: TableFilter) -> tuple[set[str], dict[str, str]]:
    """(keys that pass, {removed key: reason})."""
    passed, removed = set(), {}
    for key, record in records.items():
        reason = why_excluded(record, f)
        if reason is None:
            passed.add(key)
        else:
            removed[key] = reason
    return passed, removed
