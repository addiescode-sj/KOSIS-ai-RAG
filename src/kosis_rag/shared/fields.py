"""Per-table metadata fields (T2-30, KOS-96): what the filters and the vector-store metadata decide on.

Computed from a `TableRecord` with the functions the filters already use, so `tables.jsonl` does not change and the
values match what the filters see. Names follow KOS-96; the T2-28 L0 evaluator scores this model as is.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from kosis_rag.schemas.catalog import TableRecord
from kosis_rag.shared.roles import table_roles
from kosis_rag.shared.vocab import (FEMALE_ONLY, FEMALE_WORDS, HOUSEHOLD_SIZE, MALE_ONLY, MALE_WORDS, MARITAL_WORDS,
                                    SIDO_LABELS, dimension_keys, measure_keys, normalize_dimension)

# T2-41 S1: from filters.py, so fields does not import filters (filters imports fields)
MIN_SIDO = 3  # a class is administrative only with at least this many 시도 labels ("국적(지역)별" has none)
REGION_LEVEL_DIMS = {"시도", "시군구", "읍면동"}  # decided on TableClass.levels when present (T2-25, PR #10)
SEX = re.compile(r"(?:^|\s|의\s?)성$|남녀")  # "성", "가구주의 성"; not "세대구성"
# a table with no administrative class whose name starts with one 시도 is that 시도's table (T2-29, 분석 #22: 33 tables
# such as "서울특별시 총조사인구 총괄(...)"); full names only, followed by a non-Hangul character
TABLE_SIDO = re.compile("^(" + "|".join(sorted((x for x in SIDO_LABELS if len(x) >= 3), key=len, reverse=True))
                        + r")(?![가-힣])")


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


# units accepted from an item name's last bracket, e.g. "남자 (명)"; "(15세 이상)" or "(치매 포함)" are not units
ITEM_UNITS = {"명", "가구", "건", "%", "호", "세", "천명당", "개", "명/㎢", "세대", "만원", "원", "천원", "개월", "㎡", "쌍"}
LAST_BRACKET = re.compile(r"[(（]([^()（）]+)[)）]\s*$")


class TableFields(BaseModel):
    """Fields of one table; an empty value means the source does not say, not that the table lacks it."""

    key: str
    available_years: list[int] = Field(default_factory=list)  # years present in the values
    cycle: list[str] = Field(default_factory=list)  # survey cycle codes, the value codes when the metadata has none
    regions: list[str] | None = None  # official 시도 names; None when the table has no administrative class
    region_level: list[str] = Field(default_factory=list)  # of 시도·시군구·읍면동, the levels the table breaks down to
    measure: str | None = None  # what is counted, from the table name (T2-27)
    measure_unit: str | None = None  # one unit for every value; None when unknown, partial or mixed
    dimensions: list[str] = Field(default_factory=list)  # breakdowns as named by the table
    dimension_keys: list[str] = Field(default_factory=list)  # standard keys (vocab.py); a name without one, normalized
    # T2-30 KOS-109: breakdowns the table holds outside its classes (items, what it counts, labels). Kept apart from
    # dimension_keys so a wrong guess never reads as a real class; the filter owner decides how to use them.
    implied_keys: list[str] = Field(default_factory=list)
    fixed: list[str] = Field(default_factory=list)  # one value for the whole table, e.g. ["성=여성"] (기혼여성인구)
    measure_keys: list[str] = Field(default_factory=list)  # what is counted, one spelling per thing (vocab.measure_keys)


def item_unit(items: list[str]) -> str | None:
    """The unit every item names in its last bracket, or None when any item names none or they differ."""
    units = set()
    for item in items:
        m = LAST_BRACKET.search(item)
        unit = m.group(1).strip() if m else None
        if unit not in ITEM_UNITS:
            return None  # an item without a unit may be a %, a ratio or money: do not guess
        units.add(unit)
    return units.pop() if len(units) == 1 else None


_WORD = re.compile(r"[가-힣]+")
REGION_NAME = re.compile(r"^(?:행정구역|지역|도|시도|대도시)")


def implied(record: TableRecord, measure: str | None, keys: list[str]) -> tuple[list[str], list[str]]:
    """(implied keys, fixed values) for breakdowns that sit outside the classes; keys already in `keys` are skipped."""
    words = {w for item in record.items for w in _WORD.findall(item)}
    text = f"{record.info.table_name} {measure or ''}"  # not items: "15세이상 여자수" is often a base count beside others
    found, fixed = [], []
    if measure and HOUSEHOLD_SIZE.search(measure):
        found.append("가구원수")  # "1인가구" counts households of one size
    if words & MALE_WORDS and words & FEMALE_WORDS:
        found.append("성")  # items split by sex: 남자/여자, 남편/아내
    elif FEMALE_ONLY.search(text) and not (words & MALE_WORDS or MALE_ONLY.search(text)):
        found.append("성")
        fixed.append("성=여성")
    elif MALE_ONLY.search(text) and not (words & FEMALE_WORDS or FEMALE_ONLY.search(text)):
        found.append("성")
        fixed.append("성=남성")
    if len(words & MARITAL_WORDS) >= 2:
        found.append("혼인상태")
    if any("년생" in c.name for c in record.classes):
        found.append("연령")  # birth-year cohorts
    for c in record.classes:  # an area class without level counts ("도별", "지역별"), read by its labels; not 출생지 etc.
        if (not c.levels and dimension_keys(c.name) in (["행정구역"], ["시도"], [])
                and REGION_NAME.match(c.name) and len({SIDO_LABELS[x] for x in c.labels if x in SIDO_LABELS}) >= MIN_SIDO):
            found.append("시도")
    out = [k for k in dict.fromkeys(found) if k not in keys]
    return out, (fixed if "성" in out else [])


def table_fields(record: TableRecord) -> TableFields:
    measure, dims = table_roles(record)
    names = dims + [c.name for c in record.classes]
    regions = table_regions(record)
    # a name with no standard key keeps its normalized spelling as the key (KOS-109 D7), so it can still be matched
    keys = list(dict.fromkeys(k for d in dims for k in (dimension_keys(d) or [normalize_dimension(d)])))
    implied_keys, fixed = implied(record, measure, keys)
    return TableFields(
        key=record.info.key,
        available_years=sorted({int(p[:4]) for p in record.periods if p[:4].isdigit()}),
        cycle=record.cycle or record.prd_se,
        regions=None if regions is None else sorted(regions),
        region_level=[d for d in sorted(REGION_LEVEL_DIMS) if has_dimension(record, d, names)],
        measure=measure,
        measure_unit=record.unit or item_unit(record.items),
        dimensions=dims,
        dimension_keys=keys,
        implied_keys=implied_keys,
        fixed=fixed,
        measure_keys=measure_keys(measure),
    )
