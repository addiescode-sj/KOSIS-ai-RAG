"""Question -> search conditions (T2-14): years, period range, cycle and 시도 pulled out by rules, no model call.

`ParsedQuery` is what the parser understood; `to_filter` turns it into the registry filter (filters.TableFilter) and
`to_query_item_fields` into the L2 golden fields (evals.schemas.QueryItem), so the filter and the L2 grade read the
same output. Two-digit years ("23년") are not read: "20년간" and "23년" cannot be told apart by rule.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, ConfigDict, Field

from kosis_rag.filters import SIDO, TableFilter, soft_rank, why_excluded
from kosis_rag.schemas import TableRecord

YEAR = re.compile(r"(?<!\d)(19[5-9]\d|20[0-4]\d)(?!\d)(?!\s*년대)")
RANGE = re.compile(r"(?<!\d)((?:19|20)\d\d)\s*년?\s*(?:~|-|–|부터)\s*((?:19|20)\d\d)(?!\d)")
YEAR_MONTH = re.compile(r"(?<!\d)((?:19|20)\d\d)\s*년\s*(1[0-2]|0?[1-9])\s*월")
RECENT = re.compile(r"최근|지난\s*\d|올해|작년|지난해|전년")
CYCLES = [  # most specific first: a question that says 월별 and 연간 wants the monthly table
    ("M", re.compile(r"월별|매월|월간|매달|월마다|개월")),
    ("Q", re.compile(r"분기")),
    ("F", re.compile(r"5년\s*(?:마다|주기|단위|간격)")),
    ("Y", re.compile(r"연간|연도별|년도별|연별|매년|해마다")),
]
# Question wording -> official 시도 names; a wide name stands for several (충청도 = 충북 + 충남, 수도권 = 서울·인천·경기)
REGION_WORDS: dict[str, list[str]] = {
    "수도권": ["서울특별시", "인천광역시", "경기도"],
    "충청도": ["충청북도", "충청남도"], "전라도": ["전북특별자치도", "전라남도"], "경상도": ["경상북도", "경상남도"],
}
for name, olds in SIDO.items():
    for word in [name, *olds]:  # the short form also covers "서울시", "부산시" ...
        REGION_WORDS.setdefault(word, [name])
REGION = re.compile("|".join(sorted(map(re.escape, REGION_WORDS), key=len, reverse=True)))
SIDO_ORDER = list(SIDO)


class ParsedQuery(BaseModel):
    """What the rules found in the question; empty fields mean the question did not say."""

    years: list[int] = Field(default_factory=list)
    period_start: str | None = None  # "YYYY" or "YYYYMM"
    period_end: str | None = None
    cycle: Literal["Y", "Q", "M", "F"] | None = None
    regions: list[str] = Field(default_factory=list)  # official 시도 names
    recent: bool = False  # "최근 N년", "작년" ...: relative to the latest data, so no year is fixed here

    def to_filter(self, region_mode: Literal["hard", "soft"] = "hard") -> TableFilter:
        # one year: the table must hold it; two or more: any of them (strict removed gold tables on dev, plans/T2-05.md)
        return TableFilter(years=self.years, cycle=self.cycle, regions=self.regions, region_mode=region_mode,
                           mode="strict" if len(self.years) == 1 else "partial")

    def to_query_item_fields(self) -> dict:
        """The L2 fields a rule parser answers (answerable is T2-09's call, not the parser's)."""
        return {"period_start": self.period_start, "period_end": self.period_end, "cycle": self.cycle,
                "regions": self.regions}


def parse_query(question: str) -> ParsedQuery:
    ranges = [(int(a), int(b)) for a, b in RANGE.findall(question)]
    years = sorted({int(y) for y in YEAR.findall(question)})
    months = [f"{y}{int(m):02d}" for y, m in YEAR_MONTH.findall(question)]
    if ranges:
        start, end = min(a for a, _ in ranges), max(b for _, b in ranges)
        period_start, period_end = str(start), str(end)
    elif months:
        period_start, period_end = min(months), max(months)
    elif years:
        period_start, period_end = str(years[0]), str(years[-1])
    else:
        period_start = period_end = None
    cycle = next((code for code, pattern in CYCLES if pattern.search(question)), None)
    found = {name for word in REGION.findall(question) for name in REGION_WORDS[word]}
    return ParsedQuery(years=years, period_start=period_start, period_end=period_end, cycle=cycle,
                       regions=[name for name in SIDO_ORDER if name in found], recent=bool(RECENT.search(question)))


class ConditionRetriever(BaseRetriever):
    """Parses each question and applies the conditions to `base`.

    A base with `with_filter` (the Chroma retriever) filters before the vector search; any other base (BM25) is asked
    for its candidates first and filtered after, so give it a large k.
    """

    base: BaseRetriever
    records: dict[str, TableRecord]
    region_mode: Literal["hard", "soft"] = "hard"
    k: int = 20
    last_query: ParsedQuery | None = None
    last_removed: dict[str, str] = Field(default_factory=dict)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        self.last_query = parsed = parse_query(query)
        f = parsed.to_filter(self.region_mode)
        if hasattr(self.base, "with_filter"):
            filtered: Any = self.base.with_filter(f, self.records)
            docs = filtered.invoke(query)
            self.last_removed = dict(filtered.last_removed)
            return docs[: self.k]
        docs, self.last_removed = [], {}
        for doc in self.base.invoke(query):
            record = self.records.get(doc.metadata["key"])
            reason = why_excluded(record, f) if record is not None else None
            if reason is None:
                docs.append(doc)
            else:
                self.last_removed[doc.metadata["key"]] = reason
        return soft_rank(docs, self.records, f)[: self.k]
