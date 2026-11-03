"""Structured filters for table search (T2-05, D; T2-14; T2-27): year, cycle, organization, 시도, breakdowns and what
a table counts, decided on the registry.

Filters are decided in Python on the registry (tables.jsonl, 2,364 records), not by Chroma range queries:
a table passes only if its real periods contain the requested years, so a table whose listed range covers a year
it does not actually hold (5-year censuses, collection gaps) is left out, and the reason is kept per table.
`retriever.ChromaTableRetriever` then gives Chroma one condition, `key $in <passing keys>`.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from kosis_rag.roles import table_roles
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
REGION_LEVEL_DIMS = {"시도", "시군구", "읍면동"}  # decided on TableClass.levels when present (T2-25, PR #10)
# A requested cycle is met by data that holds its points as they are: annual data holds every 5th year, so a
# "5년마다" (F) question can use a yearly table. Monthly data does not hold quarter values (they need adding up).
CYCLE_MET_BY = {"F": {"F", "Y"}}
SEX = re.compile(r"(?:^|\s|의\s?)성$|남녀")  # "성", "가구주의 성"; not "세대구성"


class TableFilter(BaseModel):
    """Conditions taken from the question (T2-14 fills these); empty means no filtering."""

    years: list[int] = Field(default_factory=list)  # requested years, e.g. [2023] or [2019, 2023]
    cycle: str | None = None  # period code: Y, M, Q, F (multi-year), IR
    organization: str | None = None
    # strict: every requested year present; partial: at least one. On the golden dev set strict removed every gold
    # table for 2 of 14 questions with years (both ask for two years), partial removed none (plans/T2-05.md).
    mode: Literal["strict", "partial"] = "strict"
    # a range ("2015~2020", "1995년부터", "2040년까지", "2000년대"): a table passes with any point inside it
    period_from: int | None = None
    period_to: int | None = None
    regions: list[str] = Field(default_factory=list)  # official 시도 names, e.g. ["서울특별시"]
    # hard: drop a table with an administrative class that lacks every requested 시도; soft: keep it, rank it lower.
    # Tables without an administrative class are never dropped (most national tables).
    region_mode: Literal["hard", "soft"] = "hard"
    dimensions: list[str] = Field(default_factory=list)  # breakdowns the table must all have, e.g. ["성", "연령"]
    measures: list[str] = Field(default_factory=list)  # words one of which the counted thing must hold, e.g. ["혼인"]

    def is_empty(self) -> bool:
        """Whether the filter would keep every table (no condition set; soft-mode regions only re-rank).

        In: self. Out: True when no condition can remove a table. Side effects: none.
        Node: retrieve (through hybrid_search).
        """
        return not (self.years or self.period_from or self.period_to or self.cycle or self.organization
                    or self.dimensions or self.measures
                    or (self.regions and self.region_mode == "hard"))


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
    """None when the table meets the filter, else the reason in words (for the no-answer message and the harness).

    In: record = one table from the registry; f = the question's conditions.
    Out: None (kept) or the first failing condition in words: organization, cycle, period range, years, breakdowns /
        measure (T2-27), then 시도 in hard mode.
    Side effects: none.
    Node: retrieve (through hybrid_search, for retrievers without with_filter, and filter_tables).
    """
    if f.organization and _org(record.info.organization) != _org(f.organization):
        return f"작성기관 {record.info.organization} (요청 {f.organization})"
    if f.cycle and not CYCLE_MET_BY.get(f.cycle, {f.cycle}) & (set(record.cycle) | set(record.prd_se)):
        return f"주기 {'·'.join(record.cycle or record.prd_se)} (요청 {f.cycle})"
    if f.period_from or f.period_to:
        years = {int(p[:4]) for p in record.periods if p[:4].isdigit()}
        lo, hi = f.period_from or -1, f.period_to or 10_000
        if not any(lo <= y <= hi for y in years):
            asked = f"{f.period_from or ''}~{f.period_to or ''}"
            actual = f"실제 {min(years)}~{max(years)}" if years else "실제 수록 시점 없음"
            return f"요청 기간 {asked} 자료 없음 ({actual})"
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
    if reason := lacks_role(record, f):
        return reason
    if f.region_mode == "hard":
        return lacks_region(record, f)
    return None


def has_dimension(record: TableRecord, dim: str, names: list[str]) -> bool:
    """Whether the table breaks down by `dim`; `names` are its breakdown names (from its name and classes)."""
    if dim in REGION_LEVEL_DIMS:
        levels = {lv for c in record.classes for lv in (getattr(c, "levels", None) or {})}
        if levels:
            return dim in levels
        return any(dim in n or "행정구역" in n for n in names)  # no level counts: go by names
    if dim == "성":
        return any(SEX.search(n) for n in names)
    return any(dim in n for n in names)


def lacks_role(record: TableRecord, f: TableFilter) -> str | None:
    """Reason when the table misses a requested breakdown or counts something else, else None (T2-27)."""
    if not (f.dimensions or f.measures):
        return None
    measure, dims = table_roles(record)
    names = dims + [c.name for c in record.classes]
    missing = [d for d in f.dimensions if not has_dimension(record, d, names)]
    if missing:
        return f"분류 {'·'.join(missing)} 없음 (분류: {'·'.join(dims) or '없음'})"
    if f.measures and measure and not any(word in measure for word in f.measures):
        return f"세는 대상 {measure} (요청 {'·'.join(f.measures)})"  # a name that says nothing is kept
    return None


def filter_tables(records: dict[str, TableRecord], f: TableFilter) -> tuple[set[str], dict[str, str]]:
    """(keys that pass, {removed key: reason}).

    Does: decides the filter on the whole registry before a vector search (Chroma gets the passing keys).
    In: records = key -> TableRecord; f = the question's conditions.
    Out: (passing keys, removed key -> why_excluded reason).
    Side effects: none.
    Node: retrieve (through ChromaTableRetriever.with_filter inside hybrid_search).
    """
    passed, removed = set(), {}
    for key, record in records.items():
        reason = why_excluded(record, f)
        if reason is None:
            passed.add(key)
        else:
            removed[key] = reason
    return passed, removed


def soft_rank(docs: list, records: dict[str, TableRecord], f: TableFilter) -> list:
    """Soft region mode: tables lacking the requested 시도 move behind the others, order otherwise kept.

    In: docs = candidates in rank order (metadata["key"] set); records = the registry; f = the question's conditions.
    Out: docs unchanged unless f asks for regions in soft mode; then tables with the region (or without 시도 rows)
        first, both groups in their input order.
    Side effects: none.
    Node: retrieve (through hybrid_search).
    """
    if not f.regions or f.region_mode != "soft":
        return docs

    def lacks(doc) -> bool:
        record = records.get(doc.metadata["key"])
        return record is not None and lacks_region(record, f) is not None

    return sorted(docs, key=lacks)  # stable: tables with the region (or without 시도 rows) first
