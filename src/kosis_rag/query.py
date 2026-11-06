"""Question -> search conditions (T2-14, T2-29): years, period range, cycle, 시도, breakdowns and what is counted,
pulled out by rules, no model call.

`ParsedQuery` is what the parser understood; `to_filter` turns it into the registry filter (filters.TableFilter).
Its period and region fields follow evals.schemas.QueryItem (L2), so the evaluation layer can grade them as they are.
Two-digit years ("23년") are not read: "20년간" and "23년" cannot be told apart by rule (T2-29, B1).
The word lists for 시도 and breakdowns live in `vocab` (T2-29, C5); this module only has the question patterns.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any, Literal

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, ConfigDict, Field

from kosis_rag.adapters.files import load_sigungu
from kosis_rag.schemas.catalog import TableRecord
from kosis_rag.shared.filters import TableFilter, filter_with_relaxation, relax, soft_rank, why_excluded
from kosis_rag.shared.vocab import DIMENSION_WORDS, REGION_WORDS, SIDO

log = logging.getLogger(__name__)

# --- years and periods (T2-29 step 1, 분석 #1~#10) ---
Y = r"(?:19|20)\d\d"  # one year rule for every pattern: the data runs 1925~2072, so 19xx and 20xx are all read
D = r"(?:19|20)\d"  # a decade's first three digits
DASH = r"[~\-–—―∼～〜]"
MONTH = r"(1[0-2]|0?[1-9])"
# a number with a unit is not a year: "2030세대", "2000명" (분석 #3); "2023년 세대수" is unaffected, 년 comes first
NOT_UNIT = r"(?!\s*(?:명|가구|원|개|호|건|세대|세|%|㎡|평|달러|만|천))"
# birth years are breakdown values, not periods (B2): "1993년생", "1990년대생", "1980~1989년생", "1993년 출생자".
# "2000년 이전 출생자" stays a period (B3), "2023년 출생아 수" a year (출생 followed by 아·률·통계 … counts births),
# and "2023년 생산가능인구" a year (생 followed by another word is not the 년생 suffix; a particle or the end is)
BIRTH = re.compile(rf"(?<!\d)(?:{Y}\s*년?\s*{DASH}\s*)?{Y}\s*년\s*(?:대\s*)?"
                   r"(?:생(?=$|[^가-힣]|은|는|이|가|의|을|를|과|와|들|로|도|만|부터|까지)|출생\s*(?:자|한|인구|코호트|세대))")
YEAR = re.compile(rf"(?<!\d)({Y})(?!\d)(?!\s*년대){NOT_UNIT}")
# "A~B", "A년부터 B년", "A년 이후 B년까지" (a closed range, 분석 #5), "A년에서 B년까지"; 년생 is blanked before
RANGE = re.compile(rf"(?<!\d)({Y})\s*년?\s*(?:{DASH}|이후\s*부터|부터|이후|에서)\s*({Y})(?!\d){NOT_UNIT}\s*년?(?:\s*(?:까지|사이))?")
BETWEEN = re.compile(rf"(?<!\d)({Y})\s*년?\s*(?:과|와)\s*({Y})\s*년?\s*사이")  # only with 사이: "2005년과 2025년 비교" is two years
# open ranges and decades are ranges too, not single years: "1995년부터" is 1995 onwards, not 1995 alone
OPEN_FROM = re.compile(rf"(?<!\d)({Y})\s*년?\s*(?:부터|이후)")
OPEN_FROM_DECADE = re.compile(rf"(?<!\d)({D})0\s*년대\s*(?:부터|이후)")  # "2000년대 이후": 2000 onwards (분석 #10)
OPEN_TO = re.compile(rf"(?<!\d)({Y})\s*년?\s*(?:까지|이전)")
DECADE = re.compile(rf"(?<!\d)({D})0\s*년대")
# month spans (분석 #7·#8): "2024년 1월부터 6월까지", "2024년 1~6월", "2023년 12월~2024년 3월", "2015년 3월부터 2020년까지",
# "2023년 2분기" (04~06), "2023년 상반기" (01~06)
MONTH_RANGE = re.compile(rf"(?<!\d)({Y})\s*년\s*{MONTH}\s*월?\s*(?:{DASH}|부터|에서)\s*(?:({Y})\s*년\s*)?{MONTH}\s*월(?:\s*까지)?")
MONTH_TO_YEAR = re.compile(rf"(?<!\d)({Y})\s*년\s*{MONTH}\s*월\s*(?:{DASH}|부터|에서|이후)\s*({Y})\s*년?(?:\s*까지)?")
QUARTER = re.compile(rf"(?<!\d)({Y})\s*년\s*([1-4])\s*분기")
HALF = re.compile(rf"(?<!\d)({Y})\s*년\s*(상|하)\s*반기")
YEAR_MONTH = re.compile(rf"(?<!\d)({Y})\s*년\s*{MONTH}\s*월")
RECENT = re.compile(r"최근|지난\s*\d|올해|작년|지난해|전년")
# --- cycle (step 2, 분석 #13~#15) ---
CYCLES = [  # most specific first: a question that says 월별 and 연간 wants the monthly table
    # not a bare 개월 ("6개월 미만 거주", "6개월간 체류"); "최근 3개월" is monthly data
    ("M", re.compile(r"(?<!개)월(?:별|간|마다)|매월|매달|(?:최근|지난)\s*\d+\s*개월")),
    ("Q", re.compile(r"분기")),
    ("F", re.compile(r"5년\s*(?:마다|주기|단위|간격)|매\s*5년|인구(?:주택)?총조사")),
    # 연간 only in a cycle context: "연간 추이", "연간 시도별 …"; "연간 소득 2천만 원" is an amount
    ("Y", re.compile(r"연간\s*(?:추이|자료|통계|데이터|값)|연간(?=\s+\S+별)|연도별|년도별|년별|연별|매년|해마다")),
]
# --- 시도 (step 3, 분석 #19~#22; C1~C3) ---
# A two-character short name ("부산", "경기") needs a boundary on both sides: not inside 해운대구·세대구성·경제주체, and
# followed only by a 시도 suffix, a particle, a symbol, a space or the end. The character before it may be a particle
# ("서울과부산"). Longer names (서울특별시, 충청도, 수도권) need the front boundary only.
FRONT = r"(?:(?<![가-힣])|(?<=[과와및랑]))"
BACK = (r"(?=$|[^가-힣]|도|시|광역시|특별시|특별자치시|특별자치도|권|의|에|과|와|및|랑|은|는|이|가|을|를|로|부터|까지|만|보다|"
        r"처럼|하고)")
_LONG = sorted((w for w in REGION_WORDS if len(w) >= 3), key=len, reverse=True)
_SHORT = sorted((w for w in REGION_WORDS if len(w) < 3), key=len, reverse=True)
REGION = re.compile(rf"{FRONT}(?:({'|'.join(map(re.escape, _LONG))})|({'|'.join(map(re.escape, _SHORT))}){BACK})")
NOT_GYEONGGI = re.compile(r"경기\s*(?:침체|동향|변동|지수|순환|전망|회복|둔화|부양|대회|종목)")  # the economy, a game
BEFORE_GWANGJU = re.compile(r"경기도?\s*$")  # "경기도 광주시" is a city in 경기도, not 광주광역시
# words that name exactly one 시도, for "the 시도 right before a 시군구" ("서울 중구", "서울시 중구": 중구 sits in
# several 시도); group 1 is the word
_ONE_SIDO = re.compile("(" + "|".join(sorted((re.escape(w) for w, names in REGION_WORDS.items() if len(names) == 1),
                                              key=len, reverse=True)) + r")(?:광역시|특별시|특별자치시|특별자치도|시|도)?\s*$")
# a 시군구 may also follow a short 시도 name with no space ("서울중구"); every short name is two characters
_SHORT_SIDO = "|".join(sorted({old for olds in SIDO.values() for old in olds if len(old) == 2}))
SGG_FRONT = rf"(?:(?<![가-힣])|(?<=[과와및랑])|(?<={_SHORT_SIDO}))"
SIDO_ORDER = list(SIDO)
SIGUNGU: dict[str, list[str]] = load_sigungu()  # 시군구 label -> 시도 names (C3); {} without data/sigungu.json
if not SIGUNGU:
    log.warning("data/sigungu.json not found: 시군구 names in questions are not read as 시도 "
                "(build it with `python -m kosis_rag.shared.vocab --data-dir <provided data>`)")
# --- breakdowns (step 4, 분석 #23~#27; C4·C5) ---
# Only known words (vocab.DIMENSION_WORDS): "월별"·"연도별" are cycles, and an unknown "X별" could drop the right table.
# A word may be written with spaces between its characters ("가구원 수별", "태어난 곳별"). Front boundary: not inside
# another word ("세대구성별" is 세대구성, not 성), but "가구주성별" / "가구주의성별" is 성.
_DIM_WORD = "(?:" + "|".join(r"\s*".join(map(re.escape, w)) for w in sorted(DIMENSION_WORDS, key=len, reverse=True)) + ")"
DIM_FRONT = r"(?:(?<![가-힣])|(?<=가구주)|(?<=가구주의))"
DIMENSION = re.compile(rf"{DIM_FRONT}({_DIM_WORD})\s*별|(?<![가-힣])(남녀)")
DIM_BEFORE = re.compile(rf"{DIM_FRONT}({_DIM_WORD})(?:\s*(?:[·,/]|및)\s*|\s+)$")  # "성 및 ", "시도·" before "연령별"
# --- what is counted (step 5, 분석 #29~#31; C7) ---
# concept -> pattern, in order; each match is blanked before the next pattern runs ("1인 가구" is not also 가구,
# "인구 이동" is 이동자 and not 인구). A word right before 별·상태·자녀… names a breakdown, not what is counted
# ("출생아수별 부인", "혼인상태별 인구", "사망자녀수별 기혼여성"). {n} is the digit of "N인 가구".
MEASURES: list[tuple[str, re.Pattern]] = [
    ("{n}인가구", re.compile(r"(?<!\d)([1-7])\s*인\s*(?:이상\s*)?가구")),
    ("취업자", re.compile(r"취업자|취업\s*인구")),
    ("고령자", re.compile(r"고령자|고령\s*인구|노인\s*인구|노인")),
    ("이동자", re.compile(r"이동자|이동\s*인구|인구\s*이동|전입자|전출자")),
    ("기혼여성", re.compile(r"기혼\s*여성")),
    ("부인", re.compile(r"부인")),
    ("외국인", re.compile(r"외국인")),
    ("주택", re.compile(r"(?<!인구)주택(?!\s*총조사)")),
    ("혼인", re.compile(r"(?:혼인|결혼|초혼|재혼)(?!\s*(?:상태|여부|별|종류))")),
    ("이혼", re.compile(r"이혼(?!\s*(?:상태|여부|별))")),
    ("출생", re.compile(r"출생(?!\s*(?:지|아\s*수\s*별|자녀|순위|별|연도|년도|코호트))|출산(?!\s*(?:순위|자녀|별|기간))|(?<!출)생아")),
    ("사망", re.compile(r"사망(?!\s*(?:자녀|별|원인))")),
    ("인구", re.compile(r"인구(?!\s*(?:총조사|주택|동향|동태|이동))")),
    ("가구", re.compile(r"가구(?!\s*(?:주|원|구성|의\s*크기|특성|규모|유형|형태))")),
]
# C7, two layers: only these concepts filter (words one of which the table's counted thing must hold); every other
# concept goes to TableFilter.soft_measures and only ranks. 혼인상태·출생지 are breakdowns, not these.
HARD_MEASURE_WORDS = {"혼인": ["혼인", "초혼", "재혼"], "이혼": ["이혼"], "출생": ["출생", "출산", "생아"], "사망": ["사망"]}
SOFT_MEASURE_WORDS = {"취업자": ["취업"], "고령자": ["고령", "노인"], "이동자": ["이동"], "부인": ["부인", "기혼여성"],
                      "기혼여성": ["기혼여성", "부인"]}  # others (인구, 가구, 외국인, 주택, 1인가구) are their own word
MAX_RESULTS = 10  # candidates handed on after filtering


class ParsedQuery(BaseModel):
    """What the rules found in the question; empty fields mean the question did not say."""

    years: list[int] = Field(default_factory=list)
    period_start: str | None = None  # "YYYY" or "YYYYMM"
    period_end: str | None = None
    cycle: Literal["Y", "Q", "M", "F"] | None = None
    regions: list[str] = Field(default_factory=list)  # official 시도 names
    recent: bool = False  # "최근 N년", "작년" ...: relative to the latest data, so no year is fixed here
    dimensions: list[str] = Field(default_factory=list)  # breakdown keys asked for (vocab keys), e.g. ["성", "연령"]
    # what is counted, one concept per thing (인구, 가구, 1인가구, 외국인, 출생, 혼인 ...); to_filter decides which
    # of them filter (HARD_MEASURE_WORDS) and which only rank (C7)
    measures: list[str] = Field(default_factory=list)

    def to_filter(self, region_mode: Literal["hard", "soft"] = "hard") -> TableFilter:
        # one year: the table must hold it; two or more: any of them (strict removed gold tables on dev, plans/T2-05.md).
        # A range ("A~B", "A년부터", "B년까지", "2000년대") passes a table holding any point in it. A month period
        # ("2023년 3월", "202401~202406") is judged by its years (B5: yearly tables stay).
        span = {}
        if not self.years and (self.period_start or self.period_end):
            span = {"period_from": int(self.period_start[:4]) if self.period_start else None,
                    "period_to": int(self.period_end[:4]) if self.period_end else None}
        hard = [w for c in self.measures if c in HARD_MEASURE_WORDS for w in HARD_MEASURE_WORDS[c]]
        soft = [w for c in self.measures if c not in HARD_MEASURE_WORDS for w in SOFT_MEASURE_WORDS.get(c, [c])]
        return TableFilter(years=self.years, cycle=self.cycle, regions=self.regions, region_mode=region_mode,
                           mode="strict" if len(self.years) == 1 else "partial", dimensions=self.dimensions,
                           measures=list(dict.fromkeys(hard)), soft_measures=list(dict.fromkeys(soft)), **span)


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
        before = _ONE_SIDO.search(question, 0, m.start())  # "서울 중구", "서울중구": the 시도 right before it
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


def parse_query(question: str, sigungu: dict[str, list[str]] | None = None) -> ParsedQuery:
    """Read the search conditions the question states (years, period range, cycle, 시도, breakdowns, measure) by rule.

    In: question = the user question in Korean; sigungu = 시군구 -> 시도 names (default: data/sigungu.json as loaded
        at import, SIGUNGU; tests pass their own).
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
                       regions=_regions(question, SIGUNGU if sigungu is None else sigungu),
                       recent=bool(RECENT.search(question)), dimensions=_dimensions(question),
                       measures=_measures(question))


class ConditionRetriever(BaseRetriever):
    """Parses each question and applies the conditions to `base`.

    A base with `with_filter` (the Chroma retriever) filters before the vector search; any other base (BM25) is asked
    for its candidates first and filtered after, so give it a large k. At most MAX_RESULTS (10) tables come back.
    When no table meets the conditions they are relaxed one at a time (filters.filter_with_relaxation); the dropped
    ones are in `last_relaxed`.
    """

    base: BaseRetriever
    records: dict[str, TableRecord]
    region_mode: Literal["hard", "soft"] = "hard"
    k: int = MAX_RESULTS
    last_query: ParsedQuery | None = None
    last_removed: dict[str, str] = Field(default_factory=dict)
    last_relaxed: list[str] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        self.last_query = parsed = parse_query(query)
        f = parsed.to_filter(self.region_mode)
        _, self.last_relaxed, _ = filter_with_relaxation(self.records, f)
        f = relax(f, self.last_relaxed)
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
