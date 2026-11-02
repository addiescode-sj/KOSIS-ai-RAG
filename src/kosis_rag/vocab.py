"""Shared vocabulary (T2-30, KOS-97): region, breakdown and unit names, one place for tables and questions.

Region words stay where the filters and the parser defined them (filters.SIDO, query.REGION_WORDS) and are
re-exported here as the same objects, so existing imports keep working. New here:
- 시군구 -> 시도, read from the KOSIS region codes of one 읍면동 table (code[:2] is the 시도); built with
  `python -m kosis_rag.vocab --data-dir <provided data>` into data/sigungu.json (gitignored like all data/).
- breakdown names -> standard keys: KOSIS writes one breakdown many ways (현거주지 / 현 거주지 / 현거주지별,
  종사상지위 / 종사상의 지위). Keys reuse the question side's names (query.DIMENSION_WORDS) where they exist.
- unit spellings: labels only, never a scale (천명 stays 천명; 천명당 is a rate, not 1,000 people).
- question words for one breakdown label, e.g. "1인 가구" is the 가구원수 label "1인".
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from kosis_rag.filters import SIDO, SIDO_LABELS
from kosis_rag.query import DIMENSION_WORDS, REGION_WORDS

__all__ = ["SIDO", "SIDO_LABELS", "REGION_WORDS", "DIMENSION_KEYS", "dimension_keys", "unit_key", "LABEL_ALIASES",
           "build_sigungu", "load_sigungu"]

REPO = Path(__file__).resolve().parents[2]
SIGUNGU = REPO / "data" / "sigungu.json"
SIGUNGU_SOURCE = "values/101_DT_1IN1502.parquet"  # 인구, 가구 및 주택 - 읍면동: every 시도 and 시군구 with codes

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
DIMENSION_KEYS: dict[str, str] = {name: key for key, names in _KEY_GROUPS.items() for name in names}
assert set(DIMENSION_WORDS.values()) <= set(_KEY_GROUPS), "question breakdown names need a key"
_NOTE = re.compile(r"\s*[(（][^()（）]*[)）]\s*$")  # "직업(대분류)", "연령(5세)"
_BYEOL = re.compile(r"별$")
_SPLIT_MIN = 1  # shortest key that may start a glued name ("현 거주지 성" -> 현거주지 + 성)


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


def load_sigungu(path: Path = SIGUNGU) -> dict[str, list[str]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


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
