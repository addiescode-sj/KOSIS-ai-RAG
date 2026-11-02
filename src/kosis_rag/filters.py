"""Structured filters for table search (T2-05, D; T2-14): year, cycle, organization and 시도 decided on the registry.

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
# 시도 label -> its current official name; old names (직할시, 강원도, 전라북도, 제주도) and bare short labels map to it
SIDO = {
    "서울특별시": ["서울"], "부산광역시": ["부산직할시", "부산"], "대구광역시": ["대구직할시", "대구"],
    "인천광역시": ["인천직할시", "인천"], "광주광역시": ["광주직할시", "광주"], "대전광역시": ["대전직할시", "대전"],
    "울산광역시": ["울산"], "세종특별자치시": ["세종"], "경기도": ["경기"], "강원특별자치도": ["강원도", "강원"],
    "충청북도": ["충북"], "충청남도": ["충남"], "전북특별자치도": ["전라북도", "전북"], "전라남도": ["전남"],
    "경상북도": ["경북"], "경상남도": ["경남"], "제주특별자치도": ["제주도", "제주"],
}
SIDO_LABELS = {name: name for name in SIDO} | {old: name for name, olds in SIDO.items() for old in olds}
MIN_SIDO = 3  # a class is administrative only with at least this many 시도 labels ("국적(지역)별" has none)


class TableFilter(BaseModel):
    """Conditions taken from the question (T2-14 fills these); empty means no filtering."""

    years: list[int] = Field(default_factory=list)  # requested years, e.g. [2023] or [2019, 2023]
    cycle: str | None = None  # period code: Y, M, Q, F (multi-year), IR
    organization: str | None = None
    # strict: every requested year present; partial: at least one. On the golden dev set strict removed every gold
    # table for 2 of 14 questions with years (both ask for two years), partial removed none (plans/T2-05.md).
    mode: Literal["strict", "partial"] = "strict"
    regions: list[str] = Field(default_factory=list)  # official 시도 names, e.g. ["서울특별시"]
    # hard: drop a table with an administrative class that lacks every requested 시도; soft: keep it, rank it lower.
    # Tables without an administrative class are never dropped (most national tables).
    region_mode: Literal["hard", "soft"] = "hard"

    def is_empty(self) -> bool:
        return not (self.years or self.cycle or self.organization or (self.regions and self.region_mode == "hard"))


def table_regions(record: TableRecord) -> set[str] | None:
    """Official 시도 names the table's administrative classes hold; None when it has no administrative class."""
    found = [{SIDO_LABELS[x] for x in c.labels if x in SIDO_LABELS} for c in record.classes]
    found = [names for names in found if len(names) >= MIN_SIDO]
    return set().union(*found) if found else None


def lacks_region(record: TableRecord, f: TableFilter) -> str | None:
    """Reason when the table has 시도 rows but none of the requested ones, else None (soft mode ranks on this)."""
    if not f.regions:
        return None
    held = table_regions(record)
    if held is None or held & set(f.regions):
        return None
    return f"요청 지역 {'·'.join(f.regions)} 없음 (시도 {len(held)}개 수록)"


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
    if f.region_mode == "hard":
        return lacks_region(record, f)
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


def soft_rank(docs: list, records: dict[str, TableRecord], f: TableFilter) -> list:
    """Soft region mode: tables lacking the requested 시도 move behind the others, order otherwise kept."""
    if not f.regions or f.region_mode != "soft":
        return docs

    def lacks(doc) -> bool:
        record = records.get(doc.metadata["key"])
        return record is not None and lacks_region(record, f) is not None

    return sorted(docs, key=lacks)  # stable: tables with the region (or without 시도 rows) first
