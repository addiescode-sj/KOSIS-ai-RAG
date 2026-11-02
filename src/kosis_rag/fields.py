"""Per-table metadata fields (T2-30, KOS-96): what the filters and the vector-store metadata decide on.

Computed from a `TableRecord` with the functions the filters already use, so `tables.jsonl` does not change and the
values match what the filters see. Names follow KOS-96; the T2-28 L0 evaluator calls `available_years` "years" and
writes a national table's regions as "none" where this model has None.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from kosis_rag.filters import REGION_LEVEL_DIMS, has_dimension, table_regions
from kosis_rag.roles import table_roles
from kosis_rag.schemas import TableRecord
from kosis_rag.vocab import dimension_keys

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
    dimension_keys: list[str] = Field(default_factory=list)  # the same as standard keys (vocab.py); unmapped names drop


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


def table_fields(record: TableRecord) -> TableFields:
    measure, dims = table_roles(record)
    names = dims + [c.name for c in record.classes]
    regions = table_regions(record)
    return TableFields(
        key=record.info.key,
        available_years=sorted({int(p[:4]) for p in record.periods if p[:4].isdigit()}),
        cycle=record.cycle or record.prd_se,
        regions=None if regions is None else sorted(regions),
        region_level=[d for d in sorted(REGION_LEVEL_DIMS) if has_dimension(record, d, names)],
        measure=measure,
        measure_unit=record.unit or item_unit(record.items),
        dimensions=dims,
        dimension_keys=list(dict.fromkeys(k for d in dims for k in dimension_keys(d))),
    )
