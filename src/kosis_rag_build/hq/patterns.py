"""Patterns of the question rules (T2-39; T2-41 S6a: from hq/rules.py). Used by hq/rules.py only."""

from __future__ import annotations

import re

from kosis_rag.shared.vocab import SIDO_LABELS

YEAR = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)(?!\s*년대)")
DECADE = re.compile(r"(?<!\d)(19\d|20\d)0\s*년대")
NUMBER = re.compile(r"(?<![\d.,])\d+(?:,\d{3}(?!\d))*(?:\.\d+)?")  # "3,000", "12.5", "2023"; not "2015, 2020" as one
# a number followed by one of these is a value ("3만 명", "12.5%", "500가구"); 세·살·대 (age), 월·분기, 인 (household
# size: "1인가구"), 개월, 년 are not values
VALUE_UNIT = re.compile(r"\s*(?:만|천|억|조|명|%|퍼센트|프로|가구|건|원|호|쌍|세대|배|채|㎡)")
SIDO_NAME = re.compile(
    r"(?<![가-힣])(" + "|".join(sorted(map(re.escape, SIDO_LABELS), key=len, reverse=True)) + r")"
    r"(?=$|[^가-힣]|시|도|특별|광역|은|는|이|가|의|에|과|와|도|별|로|을|를|만|및|지역|내|권)")
HOMONYM = {"광주": "경기도"}  # a bare short name that is also a 시군구 of this 시도 (경기도 광주시)
DEICTIC = re.compile(r"(?<![가-힣])(?:이|위|해당|본|아래)\s*(?:통계)?\s*표(?=$|[^가-힣]|에|의|는|를|로|가|와|과|만)")
_SPACE = re.compile(r"[\s·,.?!~\-()]+")
