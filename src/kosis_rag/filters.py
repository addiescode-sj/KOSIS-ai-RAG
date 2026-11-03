"""Structured filters for table search (T2-05, D; T2-14; T2-27; T2-29): year, cycle, organization, 시도, breakdowns and
what a table counts, decided on the registry.

Filters are decided in Python on the registry (tables.jsonl, 2,364 records), not by Chroma range queries:
a table passes only if its real periods contain the requested years, so a table whose listed range covers a year
it does not actually hold (5-year censuses, collection gaps) is left out, and the reason is kept per table.
`retriever.ChromaTableRetriever` then gives Chroma one condition, `key $in <passing keys>`.
When no table passes, `filter_with_relaxation` drops the conditions one at a time (T2-29, D1: measures, breakdowns,
시도 hard -> soft, cycle) and keeps the years, so a wrong reading of the question lowers ranks instead of refusing.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from kosis_rag.roles import table_roles
from kosis_rag.schemas import TableRecord
from kosis_rag.vocab import SIDO, SIDO_LABELS, dimension_keys  # noqa: F401  (SIDO·SIDO_LABELS re-exported, T2-30 imports)

# Same organization under its old and new names (통계청 became 국가데이터처)
ORG_ALIASES = {"통계청": "국가데이터처"}
MIN_SIDO = 3  # a class is administrative only with at least this many 시도 labels ("국적(지역)별" has none)
REGION_LEVEL_DIMS = {"시도", "시군구", "읍면동"}  # decided on TableClass.levels when present (T2-25, PR #10)
# A requested cycle is met by data that holds its points as they are: annual data holds every 5th year, so a
# "5년마다" (F) question can use a yearly table. Monthly data does not hold quarter values (they need adding up).
CYCLE_MET_BY = {"F": {"F", "Y"}}
# Met only loosely (T2-29, B6): a "연도별" (Y) question usually means "over time", so a 5-year table is kept and
# ranked lower (soft_rank) instead of dropped.
CYCLE_SOFT = {"Y": {"F"}}
SEX = re.compile(r"(?:^|\s|의\s?)성$|남녀")  # "성", "가구주의 성"; not "세대구성"
# a table with no administrative class whose name starts with one 시도 is that 시도's table (T2-29, 분석 #22: 33 tables
# such as "서울특별시 총조사인구 총괄(...)"); full names only, followed by a non-Hangul character
TABLE_SIDO = re.compile("^(" + "|".join(sorted((x for x in SIDO_LABELS if len(x) >= 3), key=len, reverse=True))
                        + r")(?![가-힣])")
# relaxation order (T2-29, D1); years and period are never dropped (D2)
RELAX_ORDER = ("measures", "dimensions", "regions", "cycle")


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
    dimensions: list[str] = Field(default_factory=list)  # breakdown keys the table must all have, e.g. ["성", "연령"]
    measures: list[str] = Field(default_factory=list)  # words one of which the counted thing must hold, e.g. ["혼인"]
    # words the counted thing should hold (T2-29, C7: 인구, 가구, 외국인 ...): rank only, never drop
    soft_measures: list[str] = Field(default_factory=list)

    def is_empty(self) -> bool:
        """Whether the filter would keep every table (no condition set; soft signals only re-rank).

        In: self. Out: True when no condition can remove a table. Side effects: none.
        Node: retrieve (through hybrid_search).
        """
        return not (self.years or self.period_from or self.period_to or self.cycle or self.organization
                    or self.dimensions or self.measures
                    or (self.regions and self.region_mode == "hard"))

    def is_soft(self) -> bool:
        """Whether the filter has a signal that only re-ranks (soft 시도, soft measure words, a Y cycle).

        In: self. Out: True when soft_rank could change an order. Side effects: none. Node: retrieve.
        """
        return bool((self.regions and self.region_mode == "soft") or self.soft_measures or self.cycle in CYCLE_SOFT)


def table_regions(record: TableRecord) -> set[str] | None:
    """Official 시도 names the table's administrative classes hold; None when it has no administrative class.

    A table with no administrative class whose name starts with a 시도 ("부산광역시 총조사인구 총괄(…)") holds that
    one 시도 (T2-29, 분석 #22).
    """
    found = [{SIDO_LABELS[x] for x in c.labels if x in SIDO_LABELS} for c in record.classes]
    found = [names for names in found if len(names) >= MIN_SIDO]
    if found:
        return set().union(*found)
    m = TABLE_SIDO.match(record.info.table_name.strip())
    return {SIDO_LABELS[m.group(1)]} if m else None


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


def _codes(record: TableRecord) -> set[str]:
    return set(record.cycle) | set(record.prd_se)


def lacks_cycle(record: TableRecord, f: TableFilter) -> str | None:
    """Reason when the table meets the requested cycle only loosely (CYCLE_SOFT), else None (soft_rank uses this)."""
    if not f.cycle or CYCLE_MET_BY.get(f.cycle, {f.cycle}) & _codes(record):
        return None
    if CYCLE_SOFT.get(f.cycle, set()) & _codes(record):
        return f"주기 {'·'.join(record.cycle or record.prd_se)} (요청 {f.cycle}, 순위만 낮춤)"
    return None


def why_excluded(record: TableRecord, f: TableFilter) -> str | None:
    """None when the table meets the filter, else the reason in words (for the no-answer message and the harness).

    In: record = one table from the registry; f = the question's conditions.
    Out: None (kept) or the first failing condition in words: organization, cycle, period range, years, breakdowns /
        measure (T2-27), then 시도 in hard mode. A cycle met only loosely (CYCLE_SOFT) and a reversed period range
        (period_from > period_to) do not exclude.
    Side effects: none.
    Node: retrieve (through hybrid_search, for retrievers without with_filter, and filter_tables).
    """
    if f.organization and _org(record.info.organization) != _org(f.organization):
        return f"작성기관 {record.info.organization} (요청 {f.organization})"
    if f.cycle and not (CYCLE_MET_BY.get(f.cycle, {f.cycle}) | CYCLE_SOFT.get(f.cycle, set())) & _codes(record):
        return f"주기 {'·'.join(record.cycle or record.prd_se)} (요청 {f.cycle})"
    if (f.period_from or f.period_to) and not (f.period_from and f.period_to and f.period_from > f.period_to):
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
    """Whether the table breaks down by the key `dim`; `names` are its breakdown names (from its name and classes).

    A name counts when its standard key (vocab.dimension_keys: 가구의 크기 -> 가구원수, 배우관계 -> 혼인상태, 도 -> 시도)
    is `dim`, or when it holds `dim` as text (T2-29, 분석 #26).
    """
    keys = {k for n in names for k in dimension_keys(n)}
    if dim in REGION_LEVEL_DIMS:
        levels = {lv for c in record.classes for lv in (getattr(c, "levels", None) or {})}
        if levels:
            return dim in levels
        return dim in keys or "행정구역" in keys or any(dim in n or "행정구역" in n for n in names)  # no level counts
    if dim == "성":
        return "성" in keys or any(SEX.search(n) for n in names)
    return dim in keys or any(dim in n for n in names)


def lacks_role(record: TableRecord, f: TableFilter) -> str | None:
    """Reason when the table misses a requested breakdown or counts something else, else None (T2-27).

    A breakdown the table holds outside its classes (fields.implied: 남자/여자 items -> 성, 년생별 -> 연령) counts
    (T2-29, C6), but not one fixed to a single value (a women-only table does not break down by 성).
    """
    if not (f.dimensions or f.measures):
        return None
    measure, dims = table_roles(record)
    names = dims + [c.name for c in record.classes]
    missing = [d for d in f.dimensions if not has_dimension(record, d, names)]
    if missing:
        from kosis_rag.fields import implied  # here, not at the top: fields imports this module

        implied_keys, fixed = implied(record, measure, [])
        held = {k for k in implied_keys if not any(x.startswith(f"{k}=") for x in fixed)}
        missing = [d for d in missing if d not in held]
    if missing:
        return f"분류 {'·'.join(missing)} 없음 (분류: {'·'.join(dims) or '없음'})"
    if f.measures and measure and not any(word in measure for word in f.measures):
        return f"세는 대상 {measure} (요청 {'·'.join(f.measures)})"  # a name that says nothing is kept
    return None


def lacks_measure(record: TableRecord, f: TableFilter) -> str | None:
    """Reason when the table counts something that holds none of the soft measure words, else None (soft_rank)."""
    if not f.soft_measures:
        return None
    measure, _ = table_roles(record)
    if not measure or any(word in measure for word in f.soft_measures):
        return None  # a name that says nothing is not ranked down
    return f"세는 대상 {measure} (요청 {'·'.join(f.soft_measures)}, 순위만 낮춤)"


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


def _is_set(f: TableFilter, name: str) -> bool:
    if name == "regions":
        return bool(f.regions) and f.region_mode == "hard"
    return bool(getattr(f, name))


def relax(f: TableFilter, names: list[str]) -> TableFilter:
    """The filter with the named conditions dropped: measures, dimensions, cycle emptied; regions turned soft.

    In: f = the question's conditions; names = entries of RELAX_ORDER. Out: a copy; f is left as it is.
    Side effects: none. Node: retrieve (through filter_with_relaxation and its callers).
    """
    update: dict = {}
    for name in names:
        if name == "regions":
            update["region_mode"] = "soft"
        elif name == "measures":
            update["measures"] = []
        elif name == "dimensions":
            update["dimensions"] = []
        elif name == "cycle":
            update["cycle"] = None
        else:
            raise ValueError(f"unknown condition {name!r}; known: {', '.join(RELAX_ORDER)}")
    return f.model_copy(update=update) if update else f


def filter_with_relaxation(records: dict[str, TableRecord], f: TableFilter
                           ) -> tuple[set[str], list[str], dict[str, str]]:
    """filter_tables, then, while no table passes, drop one condition at a time in RELAX_ORDER and filter again.

    Does: the step-wise relaxation (T2-29, D1-D3): measures -> dimensions -> regions (hard -> soft) -> cycle.
        Years and the period range are never dropped (D2): a year no table holds stays a refusal.
    In: records = key -> TableRecord; f = the question's conditions.
    Out: (passing keys, conditions dropped so that a table could pass, removed key -> reason under the filter the
        keys were decided on, i.e. `relax(f, dropped)`). An empty filter gives every key. When nothing passes even
        with every droppable condition gone, the years alone refuse: (set(), [], removed under `f` itself).
    Side effects: none.
    Node: retrieve (ChromaTableRetriever, HybridRetriever, ConditionRetriever and the graph's retrieve node).
    """
    if f.is_empty():
        return set(records), [], {}
    keys, removed = filter_tables(records, f)
    if keys:
        return keys, [], removed
    dropped: list[str] = []
    g = f
    for name in RELAX_ORDER:
        if not _is_set(g, name):
            continue
        dropped.append(name)
        g = relax(g, [name])
        keys, relaxed_removed = filter_tables(records, g) if not g.is_empty() else (set(records), {})
        if keys:
            return keys, dropped, relaxed_removed
    return set(), [], removed


def soft_rank(docs: list, records: dict[str, TableRecord], f: TableFilter) -> list:
    """Soft signals: tables lacking the requested 시도 (soft mode), the soft measure words or the exact cycle
    (CYCLE_SOFT) move behind the others, order otherwise kept.

    In: docs = candidates in rank order (metadata["key"] set); records = the registry; f = the question's conditions.
    Out: docs unchanged unless f has a soft signal (is_soft); then tables that meet every signal (or say nothing
        about it) first, the 시도 signal outranking the measure signal, which outranks the cycle signal; ties keep
        the input order.
    Side effects: none.
    Node: retrieve (through hybrid_search).
    """
    if not f.is_soft():
        return docs

    def misses(doc) -> tuple[bool, bool, bool]:
        record = records.get(doc.metadata["key"])
        if record is None:
            return (False, False, False)
        return (f.region_mode == "soft" and lacks_region(record, f) is not None,
                lacks_measure(record, f) is not None, lacks_cycle(record, f) is not None)

    return sorted(docs, key=misses)  # stable: tables with the region (or without 시도 rows) first
