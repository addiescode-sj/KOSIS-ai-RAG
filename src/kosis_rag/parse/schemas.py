"""What the parse stage reads and writes (T2-41 S2; ParsedQuery from query.py, T2-14 / T2-29).

`ParsedQuery` is what the parser understood; `to_filter` turns it into the registry filter (shared.filters.TableFilter).
Its period and region fields follow evals.schemas.QueryItem (L2), so the evaluation layer can grade them as they are.
ParsedQuery stays a plain BaseModel (it is a SearchState field, and its JSON schema is part of docs/openapi.json); the
node In/Out models are BoundaryModel (strict, extra fields refused).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from kosis_rag.parse.constants import HARD_MEASURE_WORDS, SOFT_MEASURE_WORDS
from kosis_rag.shared.boundary import BoundaryModel
from kosis_rag.shared.filters import TableFilter


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


class ParseInput(BoundaryModel):
    """In of parse_question: the user question."""

    question: str


class ConditionInput(BoundaryModel):
    """In of decide_condition: what parse read (None: the condition is skipped)."""

    query: ParsedQuery | None = None


class ConditionOutput(BoundaryModel):
    """Out of decide_condition: the filter every list is searched under, and the conditions dropped for it."""

    table_filter: TableFilter | None = None  # None when the condition filter is off or the question has no condition
    relaxed: list[str] = Field(default_factory=list)  # dropped so a table could pass (filters.RELAX_ORDER; T2-29)
