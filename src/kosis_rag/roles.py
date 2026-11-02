"""What a table counts and what it is broken down by (T2-27), read from its name.

KOSIS names most tables "A/B/C별 X" (2,090 of 2,364): A, B, C are the breakdowns and X is what is counted. A word
such as 혼인 or 출생 can sit on either side — 혼인 건수 (counted) vs 혼인상태별 인구 (breakdown) — so search needs
the role, not just the word. A name that does not have that shape gets no measure (its items are often breakdown
values such as 계·남자·여자) and its class names as breakdowns.
"""

from __future__ import annotations

import re

from kosis_rag.schemas import TableRecord

# "...-시군구", "… – 읍면동(연도 끝자리 0, 5), 시군구(그 외 연도)", "… / 전국": a region-level tail, not the subject
LEVEL_TAIL = re.compile(  # not "-혼인 귀화자(계)", not "시도/시군구별 혼인종류"
    r"\s*(?:[-–]\s*(?:시도|시군구|읍면동|동읍면|전국)[^-–]*|/\s*(?:시도|시군구|읍면동|전국)\s*)$")
TRAILING_PAREN = re.compile(r"\s*[(（]([^()（）]*)[)）]\s*$")
# the last "별" that ends a word and is followed by the counted thing ("특별시" has no space after 별)
SPLIT = re.compile(r"^(?P<dims>.*별)\s+(?P<measure>[^/]+)$")
DIM_SEP = re.compile(r"\s*(?:/|,|·|\s및\s)\s*")
# "성별/연령/활동제약돌봄유형별인구": the counted word written straight after the last 별 (T2-30)
COUNTED = "인구|가구|가구수|가구원|고령자|이동인구|주택|주택수|사업체|종사자"
SPLIT_JOINED = re.compile(rf"^(?P<dims>.*별)(?P<measure>{COUNTED})$")
# "인구동태건수 및 동태율(출생,사망,혼인,이혼)": a bundle table whose bracket lists what it counts (T2-30)
BUNDLE = re.compile(r"(?:건수|동태율|동향|변동요인|추이)$")
LIST_SEP = re.compile(r"\s*[,·]\s*")
NOT_COUNTED = re.compile(r"별|등$|세|\d|이상|이하|-")  # "(1세별, 5세별)", "(성비, … 등)", "(서울,인천,경기-시군구)"
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
    counted: list[str] = []
    while (m := TRAILING_PAREN.search(name)) and name[: m.start()].strip():
        if "별" in m.group(1) and "별" not in name[: m.start()]:
            paren_dims = _dims(m.group(1))  # "총조사인구 총괄(시군구/성/5세연령별)": breakdowns in brackets
        parts = LIST_SEP.split(m.group(1).strip())
        if BUNDLE.search(name[: m.start()].strip()) and len(parts) > 1 and not any(NOT_COUNTED.search(p) for p in parts):
            counted = parts
        name = name[: m.start()].strip()
    if counted:
        return "·".join(counted), [_strip_byeol(c.name) for c in record.classes]
    if m := SPLIT.match(name) or SPLIT_JOINED.match(name):
        return INNER_PAREN.sub("", m.group("measure")).strip(), _dims(m.group("dims"))  # "사망자수(1997~), 사망률"
    if paren_dims:
        return name, paren_dims
    return None, [_strip_byeol(c.name) for c in record.classes]
