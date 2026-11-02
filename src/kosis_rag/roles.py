"""What a table counts and what it is broken down by (T2-27), read from its name.

KOSIS names most tables "A/B/C별 X" (2,090 of 2,364): A, B, C are the breakdowns and X is what is counted. A word
such as 혼인 or 출생 can sit on either side — 혼인 건수 (counted) vs 혼인상태별 인구 (breakdown) — so search needs
the role, not just the word. A name that does not have that shape gets no measure (its items are often breakdown
values such as 계·남자·여자) and its class names as breakdowns.
"""

from __future__ import annotations

import re

from kosis_rag.schemas import TableRecord

# "...-시군구", "… – 읍면동(연도 끝자리 0, 5), 시군구(그 외 연도)": a region-level tail, not part of the subject
LEVEL_TAIL = re.compile(r"\s*[-–]\s*(?:시도|시군구|읍면동|동읍면|전국)[^-–]*$")  # not "-혼인 귀화자(계)"
TRAILING_PAREN = re.compile(r"\s*[(（]([^()（）]*)[)）]\s*$")
# the last "별" that ends a word and is followed by the counted thing ("특별시" has no space after 별)
SPLIT = re.compile(r"^(?P<dims>.*별)\s+(?P<measure>[^/]+)$")
DIM_SEP = re.compile(r"\s*(?:/|,|·|\s및\s)\s*")
INNER_PAREN = re.compile(r"\s*[(（][^()（）]*[)）]")


def _strip_byeol(part: str) -> str:
    part = part.strip()
    return part[:-1].strip() if part.endswith("별") and len(part) > 1 else part


def _dims(text: str) -> list[str]:
    return [_strip_byeol(p) for p in DIM_SEP.split(text) if p.strip()]


def table_roles(record: TableRecord) -> tuple[str | None, list[str]]:
    """(measure, dimensions); measure is None when the name does not say what is counted."""
    name = LEVEL_TAIL.sub("", record.info.table_name).strip() or record.info.table_name.strip()
    paren_dims: list[str] = []
    while (m := TRAILING_PAREN.search(name)) and name[: m.start()].strip():
        if "별" in m.group(1) and "별" not in name[: m.start()]:
            paren_dims = _dims(m.group(1))  # "총조사인구 총괄(시군구/성/5세연령별)": breakdowns in brackets
        name = name[: m.start()].strip()
    if m := SPLIT.match(name):
        return INNER_PAREN.sub("", m.group("measure")).strip(), _dims(m.group("dims"))  # "사망자수(1997~), 사망률"
    if paren_dims:
        return name, paren_dims
    return None, [_strip_byeol(c.name) for c in record.classes]
