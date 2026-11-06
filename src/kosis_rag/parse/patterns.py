"""Question patterns of the rule parser (T2-14, T2-29; T2-41 S2: from query.py). Regexes only, no logic.

Two-digit years ("23년") are not read: "20년간" and "23년" cannot be told apart by rule (T2-29, B1).
The word lists for 시도 and breakdowns live in shared.vocab (T2-29, C5); this module only has the question patterns.
"""

from __future__ import annotations

import re

from kosis_rag.shared.vocab import DIMENSION_WORDS, REGION_WORDS, SIDO

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
ONE_SIDO = re.compile("(" + "|".join(sorted((re.escape(w) for w, names in REGION_WORDS.items() if len(names) == 1),
                                              key=len, reverse=True)) + r")(?:광역시|특별시|특별자치시|특별자치도|시|도)?\s*$")
# a 시군구 may also follow a short 시도 name with no space ("서울중구"); every short name is two characters
_SHORT_SIDO = "|".join(sorted({old for olds in SIDO.values() for old in olds if len(old) == 2}))
SGG_FRONT = rf"(?:(?<![가-힣])|(?<=[과와및랑])|(?<={_SHORT_SIDO}))"
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
