"""Shared vocabulary (T2-30, KOS-97; T2-29): region, breakdown and unit names, one place for tables and questions.

This module owns the dictionaries (T2-29, C5); `filters` and `query` import them, so nothing here imports those two
(it used to be the other way round, which stopped the parser from using the table-side keys).
- 시도 names (SIDO, SIDO_LABELS) and question region words (REGION_WORDS: a wide name such as 수도권 or 영남 stands
  for several 시도).
- 시군구 -> 시도, read from the KOSIS region codes of one 읍면동 table (code[:2] is the 시도); built with
  `python -m kosis_rag.shared.vocab --data-dir <provided data>` into data/sigungu.json (gitignored like all data/).
- breakdown names -> standard keys (DIMENSION_KEYS, table side): KOSIS writes one breakdown many ways (현거주지 /
  현 거주지 / 현거주지별, 종사상지위 / 종사상의 지위). DIMENSION_WORDS is the question side: the same names plus
  question-only aliases (연령대 -> 연령, 지역별 -> 시도), to the same keys.
- unit spellings: labels only, never a scale (천명 stays 천명; 천명당 is a rate, not 1,000 people).
- question words for one breakdown label, e.g. "1인 가구" is the 가구원수 label "1인".
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

__all__ = ["SIDO", "SIDO_LABELS", "REGION_WORDS", "DIMENSION_KEYS", "DIMENSION_WORDS", "dimension_keys", "unit_key",
           "LABEL_ALIASES", "measure_keys", "normalize_dimension", "build_sigungu", "REGION_LEVELS", "SIDO_ORDER"]

REPO = Path(__file__).resolve().parents[3]
SIGUNGU = REPO / "data" / "sigungu.json"
SIGUNGU_SOURCE = "values/101_DT_1IN1502.parquet"  # 인구, 가구 및 주택 - 읍면동: every 시도 and 시군구 with codes

# 시도 label -> its current official name; old names (직할시, 강원도, 전라북도, 제주도) and bare short labels map to it
SIDO = {
    "서울특별시": ["서울"], "부산광역시": ["부산직할시", "부산"], "대구광역시": ["대구직할시", "대구"],
    "인천광역시": ["인천직할시", "인천"], "광주광역시": ["광주직할시", "광주"], "대전광역시": ["대전직할시", "대전"],
    "울산광역시": ["울산"], "세종특별자치시": ["세종"], "경기도": ["경기"], "강원특별자치도": ["강원도", "강원"],
    "충청북도": ["충북"], "충청남도": ["충남"], "전북특별자치도": ["전라북도", "전북"], "전라남도": ["전남"],
    "경상북도": ["경북"], "경상남도": ["경남"], "제주특별자치도": ["제주도", "제주"],
}
SIDO_ORDER = list(SIDO)  # official 시도 names in display order (from query.py, T2-41 S2)
SIDO_LABELS = {name: name for name in SIDO} | {old: name for name, olds in SIDO.items() for old in olds}
REGION_LEVELS = ["전국", "동읍면부", "시도", "시군구", "읍면동", "하위 지역"]  # display order of TableClass.levels (from metadata.py)
_CAPITAL = ["서울특별시", "인천광역시", "경기도"]
# Question wording -> official 시도 names; a wide name stands for several (C2: 영남·호남·충청권·비수도권 unfold too)
REGION_WORDS: dict[str, list[str]] = {
    "수도권": _CAPITAL, "비수도권": [name for name in SIDO if name not in _CAPITAL],
    "충청도": ["충청북도", "충청남도"], "전라도": ["전북특별자치도", "전라남도"], "경상도": ["경상북도", "경상남도"],
    "영남": ["부산광역시", "대구광역시", "울산광역시", "경상북도", "경상남도"],
    "호남": ["광주광역시", "전북특별자치도", "전라남도"],
    "충청권": ["대전광역시", "세종특별자치시", "충청북도", "충청남도"],
    "강원권": ["강원특별자치도"], "제주권": ["제주특별자치도"],
}
REGION_WORDS["영남권"], REGION_WORDS["호남권"] = REGION_WORDS["영남"], REGION_WORDS["호남"]
for _name, _olds in SIDO.items():
    for _word in [_name, *_olds]:  # the short form also covers "서울시", "부산시" ...
        REGION_WORDS.setdefault(_word, [_name])

# normalized breakdown name -> key. Normalized = no spaces, no trailing 별 or bracket note, no leading 장래.
_KEY_GROUPS: dict[str, list[str]] = {
    "성": ["성", "남녀"],
    "연령": ["연령", "나이", "연령대", "5세연령"],
    "시도": ["시도", "도"],
    "시군구": ["시군구", "구시군"],
    "읍면동": ["읍면동"],
    "행정구역": ["행정구역", "지역"],
    "동읍면부": ["동읍면부", "시부읍부.면부"],
    "혼인상태": ["혼인상태", "배우관계"],
    "교육정도": ["교육정도", "학력"],
    "국적": ["국적", "현재국적", "국적(지역)"],
    "직업": ["직업"],
    "산업": ["산업"],
    "가구원수": ["가구원수", "가구의크기"],
    "출생지": ["출생지"],
    "현거주지": ["현거주지"],
    "1년전거주지": ["1년전거주지"],
    "5년전거주지": ["5년전거주지"],
    "통근통학지": ["통근통학지", "통근학지"],
    "목적지": ["목적지"],
    "거처종류": ["거처의종류", "거처종류"],
    "점유형태": ["점유형태", "거처의점유형태"],
    "세대구성": ["세대구성"],
    "경제활동상태": ["경제활동상태"],
    "종사상지위": ["종사상지위", "종사상의지위"],
    "사용방수": ["사용방수"],
    "이용교통수단": ["이용교통수단"],
    "출생자녀수": ["출생자녀수"],
    "출생아수": ["출생아수"],
    "가구주성": ["가구주의성"],
    "가구주연령": ["가구주의연령", "가구주연령", "가구주연령대"],
    "가구주혼인상태": ["가구주의혼인상태"],
    "가구주교육정도": ["가구주의교육정도"],
    "가구주경제활동상태": ["가구주의경제활동상태"],
    "가구주직업": ["가구주의직업"],
    "가구주종사상지위": ["가구주의종사상지위"],
    "가구주산업": ["가구주의산업"],
    "가구주와의관계": ["가구주와의관계", "가구주와관계"],
    "모연령": ["모의연령"],
    "월세금": ["월세금"],
    "전세금": ["전세금", "전세금규모"],
    "사글세": ["사글세", "사글세규모"],
    "보증금": ["보증금", "보증금규모"],
    "거주기간": ["거주기간"],
    "활동제약유형": ["활동제약유형"],
    "사회활동제약유형": ["사회활동제약유형", "사회활동활동제약유형"],
    "초혼연령": ["초혼연령"],
    "근로장소": ["근로장소", "근무지"],
    "보육상태": ["보육상태"],
    "체류자격": ["체류자격"],
    "생활비원천": ["생활비원천"],
    "소요시간": ["소요시간"],
    "다문화가구구성": ["다문화가구구성"],
    "전공분야": ["전공분야"],
    "주거시설형태": ["주거시설형태"],
    "출생지유형": ["출생지유형"],
    "총출산기간": ["총출산기간"],
    "추가계획자녀수": ["추가계획자녀수", "추가자녀계획수"],
    "자동차보유대수": ["자동차보유대수"],
    "취업상태": ["취업상태"],
    "난방시설": ["난방시설"],
    "출산순위": ["출산순위"],
    "시나리오": ["시나리오", "가정"],
    "입국연도": ["입국연도"],
    "사망자녀수": ["사망자녀수"],
    "월": ["월"],
}
# every key is also its own name, so dimension_keys(key) == [key] (the question side names keys directly)
DIMENSION_KEYS: dict[str, str] = {key: key for key in _KEY_GROUPS} | {
    name: key for key, names in _KEY_GROUPS.items() for name in names}
# Question side (T2-29, C4·C5): "X별" in a question -> key. Table spellings, minus 월 (월별 is a cycle) and the
# "가구주의 X"·"모의 연령" compounds (a question's "가구주 연령별" means 연령; the filter matches 가구주의 연령 by name),
# plus question-only aliases. Keys are written without spaces; the parser allows spaces between the characters.
_QUESTION_SKIP = {"월"}
_QUESTION_ALIASES = {
    "연령층": "연령", "나이대": "연령", "가구규모": "가구원수", "결혼여부": "혼인상태", "혼인여부": "혼인상태",
    "결혼상태": "혼인상태", "태어난곳": "출생지", "지역": "시도", "행정구역": "시도",  # C4: 지역별·행정구역별 -> 시도
}
DIMENSION_WORDS: dict[str, str] = {
    name: key for key, names in _KEY_GROUPS.items()
    if key not in _QUESTION_SKIP and not key.startswith("가구주") and key != "모연령"
    for name in names if name not in _QUESTION_ALIASES
} | _QUESTION_ALIASES
assert set(DIMENSION_WORDS.values()) <= set(_KEY_GROUPS), "question breakdown names need a key"
_NOTE = re.compile(r"\s*[(（][^()（）]*[)）]\s*$")  # "직업(대분류)", "연령(5세)"
_BYEOL = re.compile(r"별$")
_SPLIT_MIN = 1  # shortest key that may start a glued name ("현 거주지 성" -> 현거주지 + 성)


def normalize_dimension(name: str) -> str:
    """One spelling of a breakdown name: no spaces, no trailing 별 or bracket note, no leading 장래."""
    return _normalize(name)


def _normalize(name: str) -> str:
    name = _NOTE.sub("", name.strip())
    name = _BYEOL.sub("", name.strip())
    name = re.sub(r"\s+", "", name)
    return name[2:] if name.startswith("장래") and len(name) > 2 else name


def dimension_keys(name: str) -> list[str]:
    """Standard keys for one breakdown name; [] when unmapped. A name glued from two breakdowns gives both."""
    norm = _normalize(name)
    if norm in DIMENSION_KEYS:
        return [DIMENSION_KEYS[norm]]
    for i in range(len(norm) - _SPLIT_MIN, _SPLIT_MIN - 1, -1):  # longest known head first
        head, tail = norm[:i], norm[i:]
        if head in DIMENSION_KEYS and tail in DIMENSION_KEYS:
            return [DIMENSION_KEYS[head], DIMENSION_KEYS[tail]]
    return []


UNITS = {"명", "가구", "건", "%", "세", "호", "천명", "천명당", "개", "명/㎢", "세대", "만원", "원", "천원", "개월", "㎡", "쌍",
         "년", "배", "인"}
_UNIT_SPELLING = {"퍼센트": "%", "％": "%", "천 명": "천명", "명/km²": "명/㎢", "명/㎞²": "명/㎢"}


def unit_key(unit: str | None) -> str | None:
    """The unit as one spelling, or None when empty or not a known unit. Values are never rescaled."""
    if not unit:
        return None
    unit = _UNIT_SPELLING.get(unit.strip(), unit.strip())
    return unit if unit in UNITS else None


# question words -> (breakdown key, label as the tables write it)
LABEL_ALIASES: dict[str, tuple[str, str]] = {
    **{f"{n}인 가구": ("가구원수", f"{n}인") for n in range(1, 7)},
    **{f"{n}인가구": ("가구원수", f"{n}인") for n in range(1, 7)},
    "7인 이상 가구": ("가구원수", "7인 이상"),
}


# Breakdowns a table holds outside its classes (T2-30 KOS-109, evaluation note D1-D3). Words picked from how often
# they occur as whole words in the 2,364 tables' item names (남자/여자 462 tables each, 남/여 62/61, 남편/아내 24/25,
# 미혼 60, 사별 66, 이혼 74, 유배우 30, 배우자있음 17), not from golden questions.
MALE_WORDS = {"남자", "남", "남편", "남성"}
FEMALE_WORDS = {"여자", "여", "아내", "여성"}
FEMALE_ONLY = re.compile(r"여성|부인|아내|기혼여성|여자수")  # a table about women only: "기혼여성인구", "기혼부인수"
MALE_ONLY = re.compile(r"남성|남편")
MARITAL_WORDS = {"유배우", "미혼", "사별", "이혼", "배우자있음"}
HOUSEHOLD_SIZE = re.compile(r"\d+인\s*가구")  # "1인가구", "2인 가구" in what a table counts


def measure_keys(measure: str | None) -> list[str]:
    """What a table counts, written one way: spaces dropped ("취업 인구" = "취업인구"); a "·" list also gives each
    part ("통근·통학인구" -> 통근, 통학인구; "출생·사망·혼인·이혼" -> each). No meaning is widened (고령자 is not 인구)."""
    if not measure:
        return []
    compact = re.sub(r"\s+", "", measure)
    parts = [p for p in compact.replace("·", "|").replace(",", "|").split("|") if p]
    return list(dict.fromkeys([compact.replace("·", ""), *parts])) if len(parts) > 1 else [compact]


def build_sigungu(parquet: Path) -> dict[str, list[str]]:
    """{시군구 label: [시도, ...]} from a table whose region codes are 2 digits (시도) and 5 digits (시군구)."""
    import pyarrow.parquet as pq

    rows = pq.read_table(parquet, columns=["dim1_code", "dim1_label"]).group_by(["dim1_code", "dim1_label"]) \
        .aggregate([]).to_pylist()
    sido = {r["dim1_code"]: SIDO_LABELS[r["dim1_label"]] for r in rows
            if len(r["dim1_code"] or "") == 2 and r["dim1_label"] in SIDO_LABELS}
    out: dict[str, set[str]] = {}
    for r in rows:
        code, label = r["dim1_code"] or "", r["dim1_label"]
        if len(code) == 5 and code[:2] in sido:
            out.setdefault(label, set()).add(sido[code[:2]])  # 중구 sits under several 시도
    return {label: sorted(names) for label, names in sorted(out.items())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build data/sigungu.json (시군구 -> 시도) from the provided data")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=SIGUNGU)
    args = parser.parse_args(argv)
    mapping = build_sigungu(args.data_dir / SIGUNGU_SOURCE)
    args.out.write_text(json.dumps(mapping, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"wrote {args.out}: {len(mapping)} 시군구")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
