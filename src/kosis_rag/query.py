"""Question -> search conditions (T2-14): years, period range, cycle and 시도 pulled out by rules, no model call.

`ParsedQuery` is what the parser understood; `to_filter` turns it into the registry filter (filters.TableFilter).
Its period and region fields follow evals.schemas.QueryItem (L2), so the evaluation layer can grade them as they are.
Two-digit years ("23년") are not read: "20년간" and "23년" cannot be told apart by rule.
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
# Breakdowns the question asks for ("성별", "연령별", "시군구별" ...) -> the breakdown a table must have (T2-27).
# Only known words: "월별"·"연도별" are cycles, and an unknown "X별" could drop the right table.
DIMENSION_WORDS = {
    "성": "성", "남녀": "성", "연령": "연령", "나이": "연령", "시도": "시도", "시군구": "시군구", "읍면동": "읍면동",
    "혼인상태": "혼인상태", "교육정도": "교육정도", "학력": "교육정도", "국적": "국적", "직업": "직업", "산업": "산업",
    "가구원수": "가구원수", "출생지": "출생지", "태어난 곳": "출생지",
}
DIMENSION = re.compile(r"(" + "|".join(sorted(DIMENSION_WORDS, key=len, reverse=True)) + r")\s*별|(남녀)")
# What the question counts -> words the table's counted thing must hold. 혼인상태·출생지 are breakdowns, not these.
MEASURES = [
    (re.compile(r"(?:혼인|결혼)\s*(?:건수|율|률|횟수|쌍)"), ["혼인"]),
    (re.compile(r"이혼\s*(?:건수|율|률)"), ["이혼"]),
    (re.compile(r"출생아|출생\s*(?:건수|률|율)|출산율"), ["출생", "출산", "생아"]),  # "…별 적생아" is a birth table
    (re.compile(r"사망자|사망\s*(?:건수|률|율)"), ["사망"]),
]
MAX_RESULTS = 10  # candidates handed on after filtering


class ParsedQuery(BaseModel):
    """What the rules found in the question; empty fields mean the question did not say."""

    years: list[int] = Field(default_factory=list)
    period_start: str | None = None  # "YYYY" or "YYYYMM"
    period_end: str | None = None
    cycle: Literal["Y", "Q", "M", "F"] | None = None
    regions: list[str] = Field(default_factory=list)  # official 시도 names
    recent: bool = False  # "최근 N년", "작년" ...: relative to the latest data, so no year is fixed here
    dimensions: list[str] = Field(default_factory=list)  # breakdowns asked for, e.g. ["성", "연령"]
    measures: list[str] = Field(default_factory=list)  # what is counted, e.g. ["혼인"]

    def to_filter(self, region_mode: Literal["hard", "soft"] = "hard") -> TableFilter:
        # one year: the table must hold it; two or more: any of them (strict removed gold tables on dev, plans/T2-05.md)
        return TableFilter(years=self.years, cycle=self.cycle, regions=self.regions, region_mode=region_mode,
                           mode="strict" if len(self.years) == 1 else "partial", dimensions=self.dimensions,
                           measures=self.measures)


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
    dims = [DIMENSION_WORDS[a or b] for a, b in DIMENSION.findall(question)]
    measures = [w for pattern, words in MEASURES if pattern.search(question) for w in words]
    return ParsedQuery(years=years, period_start=period_start, period_end=period_end, cycle=cycle,
                       regions=[name for name in SIDO_ORDER if name in found], recent=bool(RECENT.search(question)),
                       dimensions=list(dict.fromkeys(dims)), measures=list(dict.fromkeys(measures)))


class ConditionRetriever(BaseRetriever):
    """Parses each question and applies the conditions to `base`.

    A base with `with_filter` (the Chroma retriever) filters before the vector search; any other base (BM25) is asked
    for its candidates first and filtered after, so give it a large k. At most MAX_RESULTS (10) tables come back.
    """

    base: BaseRetriever
    records: dict[str, TableRecord]
    region_mode: Literal["hard", "soft"] = "hard"
    k: int = MAX_RESULTS
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
            return docs[: min(self.k, MAX_RESULTS)]
        docs, self.last_removed = [], {}
        for doc in self.base.invoke(query):
            record = self.records.get(doc.metadata["key"])
            reason = why_excluded(record, f) if record is not None else None
            if reason is None:
                docs.append(doc)
            else:
                self.last_removed[doc.metadata["key"]] = reason
        return soft_rank(docs, self.records, f)[: min(self.k, MAX_RESULTS)]
