"""What a table counts and what it is broken down by (T2-27), read from its name.

KOSIS names most tables "A/B/C별 X" (2,090 of 2,364): A, B, C are the breakdowns and X is what is counted. A word
such as 혼인 or 출생 can sit on either side — 혼인 건수 (counted) vs 혼인상태별 인구 (breakdown) — so search needs
the role, not just the word. A name that does not have that shape gets no measure (its items are often breakdown
values such as 계·남자·여자) and its class names as breakdowns.
"""

from __future__ import annotations

import re

from kosis_rag.schemas.catalog import TableRecord

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


# T2-42 (KOS-118): a class that stacks several breakdowns as blocks of labels. "특성별" in 119 tables holds e.g.
# 남성·여성 | 사할린·한반도·일본 | 1세대·2세대 | 대한민국·러시아 | 재외동포(F-4) | 1년 미만 | 임대 아파트 | 중학교 ...
# Each block is one breakdown, counted on its own (marginal totals: no cross of two blocks). Blocks are read from
# labels that recur across those 119 tables, not from golden tables or questions.
BUNDLED_CLASSES = {"특성"}
LABEL_BLOCKS: list[tuple[str, re.Pattern]] = [
    ("성", re.compile(r"^(?:남성|여성|남자|여자)$")),
    ("출생지", re.compile(r"^(?:사할린|중앙아시아 국가|기타 러시아 지역|한반도|일본)$")),
    ("이주세대", re.compile(r"^\d세대(?: 이상)?$")),
    ("국적", re.compile(r"^(?:대한민국|러시아)$")),
    ("체류자격", re.compile(r"\(F-\d\)$")),
    ("거주기간", re.compile(r"^\d+년(?:~\d+년)? (?:미만|이상)$")),
    ("주거시설형태", re.compile(r"^(?:입소 시설|임대 아파트|자녀 또는 친인척 집|개인 소유 주택)$")),
    ("교육정도", re.compile(r"^(?:초등학교 이하|중학교|고등학교|전문대학|일반대학교|대학원)$")),
    ("산업", re.compile(r"제조업$")),
    ("조직형태", re.compile(r"^(?:개인사업체|회사 법인|회사 외 법인)$")),
    ("매출액규모", re.compile(r"^(?:5억원 미만|\d+억원~\d+억원 미만|500억원 이상)$")),
    ("종사자규모", re.compile(r"^\d+명(?:~\d+명)? (?:미만|이상)$")),
    ("혼인상태", re.compile(r"^미혼(?:→혼인 변화)?$")),
    ("출산여부", re.compile(r"^(?:미출산(?:→출산 변화)?|1자녀 출산|다자녀 출산)$")),
]
# a class whose labels are what is counted ("기본항목": 출생아수(명), 조사망률(천명당)), not a breakdown
MEASURE_CLASSES = {"기본항목"}
UNIT_LABEL = re.compile(r"[(（][^()（）]+[)）]\s*$")


def label_blocks(labels: list[str]) -> list[str]:
    """Breakdown names a bundled class holds, in label order; labels of no known block (전체, 모름) are skipped.

    Side effects: none.
    """
    found = [name for label in labels for name, pattern in LABEL_BLOCKS if pattern.search(label.strip())]
    return list(dict.fromkeys(found))


def bundled_blocks(record: TableRecord) -> list[list[str]]:
    """Per bundled class of the table, the breakdowns it holds (each counted alone: never two of one list together).

    Side effects: none. Used by filters.lacks_role.
    """
    return [blocks for c in record.classes if _strip_byeol(c.name).replace(" ", "") in BUNDLED_CLASSES
            and (blocks := label_blocks(c.labels))]


def _unbundle(record: TableRecord, dims: list[str]) -> list[str]:
    """dims with a bundled class replaced by its blocks and a class of counted items dropped (T2-42)."""
    by_name = {_strip_byeol(c.name).replace(" ", ""): c for c in record.classes}
    out: list[str] = []
    for d in dims:
        key = d.replace(" ", "")
        c = by_name.get(key)
        if c is not None and key in BUNDLED_CLASSES and (blocks := label_blocks(c.labels)):
            out += blocks
        elif (c is not None and key in MEASURE_CLASSES and c.labels
              and sum(bool(UNIT_LABEL.search(x)) for x in c.labels) * 2 > len(c.labels)):
            continue
        else:
            out.append(d)
    return list(dict.fromkeys(out))


def table_roles(record: TableRecord) -> tuple[str | None, list[str]]:
    """(measure, dimensions); measure is None when the name does not say what is counted."""
    measure, dims = _name_roles(record)
    return measure, _unbundle(record, dims)


def _name_roles(record: TableRecord) -> tuple[str | None, list[str]]:
    """(measure, dimensions) read from the table name, with class names where the name does not list them."""
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
    return _fallback_measure(record, name), [_strip_byeol(c.name) for c in record.classes]


# T2-30 KOS-109 (evaluation note D6): a name that is not "A별 X" may still say what is counted
SUMMARY_TAIL = re.compile(r"\s*(?:현황|추이|분포|구조)$")
ITEM_NOISE = re.compile(r"[-_]\s*계$|\s*[(（][^()（）]*[)）]|\d+세\s*이상\s*")  # "외국인주민-계", "(명)", "6세 이상 "
BREAKDOWN_VALUES = {"계", "합계", "총계", "소계", "남자", "여자", "남", "여", "전체", "기타"}
GENERIC_ITEMS = {"비율", "구성비", "비중", "평균"}
# "성, 혼인상태 및 경제활동상태인구" lists breakdowns, not counted things
BREAKDOWN_HEADS = {"성", "연령", "혼인상태", "교육정도", "경제활동상태", "산업", "직업", "세대구성", "국적"}  # says how, not what: "누적 혼인 비율" has the one item "비율"


def _fallback_measure(record: TableRecord, name: str) -> str | None:
    """The one item a table has ("외국인", "고령친화 용품 제품 관련 종사자 수"), or the word before 현황·추이·분포·구조
    ("… 주택수 추이" -> 주택수). A generic item (비율) gives the name when the name ends with it. None when neither
    holds or the item is a breakdown value such as 계."""
    listed = re.sub(r"\s*등$", "", name.split("/")[-1])  # "시군구/출생아수, 합계출산율": the part after the area
    parts = [p for p in LIST_SEP.split(listed) if p]
    if (len(parts) > 1 and "별" not in name and not any(NOT_COUNTED.search(p) for p in parts)
            and not any(p.split(" 및 ")[0] in BREAKDOWN_HEADS for p in parts)):
        return "·".join(parts)  # "출생아수, 합계출산율, 자연증가 등": the name lists what is counted
    if len(record.items) == 1:
        item = ITEM_NOISE.sub("", record.items[0]).strip()
        if item in GENERIC_ITEMS:  # the name says what; a survey question name ("… 여부") does not, so leave it empty
            return name if name.endswith(item) else None
        if len(item) >= 2 and item not in BREAKDOWN_VALUES:
            return item
    if SUMMARY_TAIL.search(name):
        words = SUMMARY_TAIL.sub("", name).split()
        if words and len(words[-1]) >= 2 and "별" not in words[-1]:
            return words[-1]
    return None
