"""Parse stage step functions (T2-14, T2-29; T2-41 S2: from query.py and graph.py's parse / condition nodes).

parse_query reads the search conditions a question states (years, period range, cycle, 시도, breakdowns, what is
counted) by rule, no model call. The 시군구 -> 시도 dictionary is passed in (P2-D9: SearchResources.sigungu, loaded
once by the caller); this module reads no file, at import or later.

Step functions (the graph's nodes call them): parse_question(ParseInput, resources) -> ParsedQuery and
decide_condition(ConditionInput, resources) -> ConditionOutput. `resources` is the search resources
(schemas.resources.SearchResources): parse reads its sigungu, condition its records and options.cond_filter.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from kosis_rag.parse.patterns import (BACK, BEFORE_GWANGJU, BETWEEN, BIRTH, CYCLES, DECADE, DIM_BEFORE, DIMENSION, HALF,
                                      MEASURES, MONTH_RANGE, MONTH_TO_YEAR, NOT_GYEONGGI, ONE_SIDO, OPEN_FROM,
                                      OPEN_FROM_DECADE, OPEN_TO, QUARTER, RANGE, RECENT, REGION, SGG_FRONT, YEAR,
                                      YEAR_MONTH)
from kosis_rag.parse.schemas import ConditionInput, ConditionOutput, ParsedQuery, ParseInput
from kosis_rag.shared.filters import filter_with_relaxation, relax
from kosis_rag.shared.vocab import DIMENSION_WORDS, REGION_WORDS, SIDO_ORDER


def _blank(m: re.Match) -> str:
    return " " * len(m[0])


def _month_spans(question: str) -> tuple[list[tuple[str, str]], str]:
    """Month-level periods (YYYYMM, YYYYMM or YYYY) in the question, and the question with them blanked out."""
    found: list[tuple[str, str]] = []
    for m in MONTH_RANGE.finditer(question):
        y1, m1, y2, m2 = m.groups()
        found.append((f"{y1}{int(m1):02d}", f"{y2 or y1}{int(m2):02d}"))
    question = MONTH_RANGE.sub(_blank, question)
    for m in MONTH_TO_YEAR.finditer(question):
        found.append((f"{m[1]}{int(m[2]):02d}", m[3]))
    question = MONTH_TO_YEAR.sub(_blank, question)
    for m in QUARTER.finditer(question):
        q = int(m[2])
        found.append((f"{m[1]}{3 * q - 2:02d}", f"{m[1]}{3 * q:02d}"))
    question = QUARTER.sub(_blank, question)
    for m in HALF.finditer(question):
        found.append((f"{m[1]}01", f"{m[1]}06") if m[2] == "상" else (f"{m[1]}07", f"{m[1]}12"))
    question = HALF.sub(_blank, question)
    return [(a, b) if a <= b else (b, a) for a, b in found], question


def _ranges(question: str) -> tuple[list[tuple[int | None, int | None]], str]:
    """Closed, open and decade ranges in the question, and the question with them blanked out. A reversed range
    ("2020~2015년") is put the right way round."""
    found: list[tuple[int | None, int | None]] = []
    for pattern, to_range in ((RANGE, lambda m: (int(m[1]), int(m[2]))), (BETWEEN, lambda m: (int(m[1]), int(m[2]))),
                              (OPEN_FROM, lambda m: (int(m[1]), None)), (OPEN_FROM_DECADE, lambda m: (int(m[1]) * 10, None)),
                              (DECADE, lambda m: (int(m[1]) * 10, int(m[1]) * 10 + 9)),
                              (OPEN_TO, lambda m: (None, int(m[1])))):
        for m in pattern.finditer(question):
            a, b = to_range(m)
            found.append((b, a) if a is not None and b is not None and a > b else (a, b))
        question = pattern.sub(_blank, question)
    return found, question


@lru_cache(maxsize=4)
def _sigungu_pattern(labels: tuple[str, ...]) -> re.Pattern | None:
    names = sorted((x for x in labels if len(x) >= 2 and x not in REGION_WORDS), key=len, reverse=True)
    return re.compile(rf"{SGG_FRONT}({'|'.join(map(re.escape, names))}){BACK}") if names else None


def _regions(question: str, sigungu: dict[str, list[str]]) -> list[str]:
    """Official 시도 names the question names, in SIDO order (C1 boundaries and exceptions, C2 wide names, C3 시군구)."""
    found: set[str] = set()
    for m in REGION.finditer(question):
        word = m[1] or m[2]
        if word == "경기" and NOT_GYEONGGI.match(question, m.start()):
            continue
        if word == "광주" and BEFORE_GWANGJU.search(question, 0, m.start()):
            continue
        found.update(REGION_WORDS[word])
    pattern = _sigungu_pattern(tuple(sigungu))
    for m in pattern.finditer(question) if pattern else ():
        names = sigungu[m[1]]
        before = ONE_SIDO.search(question, 0, m.start())  # "서울 중구", "서울중구": the 시도 right before it
        if before:
            found.update(REGION_WORDS[before[1]])
        elif len(names) == 1:
            found.add(names[0])  # a name in several 시도 (중구, 남구 …) is read only with its 시도 in front
    return [name for name in SIDO_ORDER if name in found]


def _dimensions(question: str) -> list[str]:
    """Breakdown keys the question asks for; a list before 별 ("성 및 연령별", "시도·시군구별") gives each known word."""
    dims: list[str] = []
    for m in DIMENSION.finditer(question):
        if m[2]:
            dims.append("성")  # bare 남녀
            continue
        chain, head = [m[1]], question[: m.start()]
        while b := DIM_BEFORE.search(head):
            chain.append(b[1])
            head = head[: b.start()]
        dims += [DIMENSION_WORDS[re.sub(r"\s+", "", w)] for w in reversed(chain)]
    return list(dict.fromkeys(dims))


def _measures(question: str) -> list[str]:
    """Concepts the question counts, in MEASURES order; each match is blanked before the next concept is looked for."""
    found: list[str] = []
    for concept, pattern in MEASURES:
        for m in pattern.finditer(question):
            found.append(concept.format(n=m[1]) if "{n}" in concept else concept)
        question = pattern.sub(_blank, question)
    return list(dict.fromkeys(found))


def parse_query(question: str, *, sigungu: dict[str, list[str]] | None = None) -> ParsedQuery:
    """Read the search conditions the question states (years, period range, cycle, 시도, breakdowns, measure) by rule.

    In: question = the user question in Korean; sigungu = 시군구 -> 시도 names (the caller loads it once:
        adapters.files.load_sigungu, held as SearchResources.sigungu; None or {} = 시군구 names are not read).
    Out: ParsedQuery; a field the question does not state stays empty. A range (years, months, a quarter or half)
        wins over single years; a single month ("2023년 3월") keeps the year and sets period_start/end as YYYYMM.
        Birth years ("1993년생") are neither years nor a period.
    Side effects: none (regex only, no model call).
    Node: parse.
    """
    text = BIRTH.sub(_blank, question)
    spans, text = _month_spans(text)
    ranges, rest = _ranges(text)
    years = [] if ranges or spans else sorted({int(y) for y in YEAR.findall(rest)})  # a range wins over single years
    months = [f"{y}{int(m):02d}" for y, m in YEAR_MONTH.findall(rest)]
    if ranges or spans:
        starts = [str(a) for a, _ in ranges if a is not None] + [a for a, _ in spans]
        ends = [str(b) for _, b in ranges if b is not None] + [b for _, b in spans]
        period_start = min(starts) if starts else None  # "YYYY" and "YYYYMM" compare as strings: year first
        period_end = max(ends) if ends else None  # open only when no end at all ("1995년부터"; not "A 이후 B까지")
    elif months:
        period_start, period_end = min(months), max(months)
    elif years:
        period_start, period_end = str(years[0]), str(years[-1])
    else:
        period_start = period_end = None
    cycle = next((code for code, pattern in CYCLES if pattern.search(question)), None)
    return ParsedQuery(years=years, period_start=period_start, period_end=period_end, cycle=cycle,
                       regions=_regions(question, sigungu or {}),
                       recent=bool(RECENT.search(question)), dimensions=_dimensions(question),
                       measures=_measures(question))


def parse_question(inp: ParseInput, resources: Any) -> ParsedQuery:
    """Read the question's search conditions once (the parse node's step).

    In: inp.question; resources.sigungu = 시군구 -> 시도 names (P2-D9; {} = 시군구 names are not read).
    Out: parse_query(question, sigungu=resources.sigungu).
    Side effects: none (rules only). Node: parse.
    """
    return parse_query(inp.question, sigungu=resources.sigungu)


def decide_condition(inp: ConditionInput, resources: Any) -> ConditionOutput:
    """Decide once the filter every list is searched under (the condition node's step).

    In: inp.query (its filter is used only when resources.options.cond_filter is on); resources.records = the
        registry the filter is decided on.
    Out: ConditionOutput(table_filter = the filter after relaxation (None when off or no condition), relaxed =
        conditions dropped because no table met them (filters.RELAX_ORDER; years are never dropped)).
    Side effects: none (rules over the registry); ValueError for a non-empty condition without records.
    Node: condition.
    """
    records = resources.records
    table_filter = inp.query.to_filter() if resources.options.cond_filter and inp.query is not None else None
    relaxed: list[str] = []
    if table_filter is not None and not table_filter.is_empty():
        _, relaxed, _ = filter_with_relaxation(records, table_filter)
        table_filter = relax(table_filter, relaxed)
    # the hybrid's own relaxation step (retrieve.lists.condition_filter), kept as it was: a no-op on a relaxed filter
    if table_filter is not None and not table_filter.is_empty():
        if not records:
            raise ValueError("a condition needs records (tables.jsonl) to decide on")
        _, dropped, _ = filter_with_relaxation(records, table_filter)
        table_filter = relax(table_filter, dropped)
    return ConditionOutput(table_filter=table_filter, relaxed=relaxed)
