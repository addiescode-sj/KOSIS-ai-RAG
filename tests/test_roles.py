"""T2-27: what a table counts and what it is broken down by, read from its name (names made up in KOSIS style)."""

import pytest

from kosis_rag.schemas.catalog import TableClass, TableRecord
from kosis_rag.schemas.common import TableInfo
from kosis_rag.shared.roles import table_roles

offline = pytest.mark.offline


def _record(name, classes=()):
    info = TableInfo(
        key="101_DT_X", org_id="101", tbl_id="DT_X", survey_name="인구총조사", organization="국가데이터처",
        table_name=name, period="년", source="api", coverage="A_정상", rows=1, numeric_rows=1, symbol_rows=0,
        parquet="values/101_DT_X.parquet", url="http://kosis.kr/x", menu_path="인구",
    )
    return TableRecord(info=info, classes=[TableClass(name=c, labels=["계"], n_labels=1) for c in classes])


@offline
@pytest.mark.parametrize("name, measure, dims", [
    ("시도/시군구/월별 혼인", "혼인", ["시도", "시군구", "월"]),
    ("성, 연령 및 혼인상태별 인구 - 시군구", "인구", ["성", "연령", "혼인상태"]),
    ("현 거주지별/성별/연령별/출생지별 인구-시도", "인구", ["현 거주지", "성", "연령", "출생지"]),
    ("시도/연령/출생아수별 부인(15세 이상)", "부인", ["시도", "연령", "출생아수"]),
    ("시도/성/연령별 북한출생 인구", "북한출생 인구", ["시도", "성", "연령"]),
    ("시군구/성/연령(5세)별 사망자수(1997~), 사망률(1998~)", "사망자수, 사망률", ["시군구", "성", "연령(5세)"]),
    ("총조사인구 총괄(시군구/성/5세연령별)", "총조사인구 총괄", ["시군구", "성", "5세연령"]),
    ("서울특별시 인구 현황", "인구", ["행정구역", "성"]),  # T2-30 KOS-109: the word before 현황
    ("국적별 한국국적을 취득한 자-혼인 귀화자(계)", "한국국적을 취득한 자-혼인 귀화자", ["국적"]),  # "특별시" is not a breakdown; falls back to class names
    # T2-30: bundle tables name what they count in brackets; a counted word may follow 별 with no space
    ("시도/인구동태건수 및 동태율(출생,사망,혼인,이혼)", "출생·사망·혼인·이혼", ["행정구역", "성"]),
    ("장래 인구변동요인(출생, 사망, 국제이동) / 전국", "출생·사망·국제이동", ["행정구역", "성"]),
    ("성별/연령/활동제약돌봄유형별인구-시도", "인구", ["성", "연령", "활동제약돌봄유형"]),
    ("시도/시군구별 혼인종류", "혼인종류", ["시도", "시군구"]),  # "/시군구별 …" is not a region tail
    ("주요 인구지표(성비,인구성장률,인구구조,부양비 등) / 전국", None, ["행정구역", "성"]),  # "… 등" is an example list
    ("성 및 연령별 추계인구(1세별, 5세별) / 전국", "추계인구", ["성", "연령"]),
])
def test_name_gives_what_is_counted_and_the_breakdowns(name, measure, dims):
    assert table_roles(_record(name, classes=("행정구역별", "성별"))) == (measure, dims)


@offline
def test_same_word_on_either_side_of_the_name():
    counted, _ = table_roles(_record("시도/성/연령별 혼인율"))
    breakdown, dims = table_roles(_record("성별/연령별/혼인상태별 인구"))
    assert "혼인" in counted and "혼인" not in breakdown and "혼인상태" in dims


@offline
def test_bundle_table_passes_the_counted_filter_for_each_listed_word():
    from kosis_rag.query import parse_query
    from kosis_rag.shared.filters import TableFilter, lacks_role

    bundle = _record("시도/인구동태건수 및 동태율(출생,사망,혼인,이혼)", classes=("행정구역별",))
    for question in ("2024년 서울 출생아 수", "2024년 부산 사망자 수", "2024년 혼인 건수", "2024년 이혼 건수"):
        assert lacks_role(bundle, parse_query(question).to_filter()) is None, question
    assert lacks_role(bundle, TableFilter(measures=["인구"])) == "세는 대상 출생·사망·혼인·이혼 (요청 인구)"


@offline
@pytest.mark.parametrize("name, items, measure", [  # T2-30 KOS-109: names that are not "A별 X"
    ("출생아수, 합계출산율, 자연증가 등", ["인구동태건수 및 동태율 추이"], "출생아수·합계출산율·자연증가"),
    ("2010년 행정구역 기준 주택수 추이(2000~2010)", ["단독주택", "아파트"], "주택수"),
    ("한국국적을 가지지 않은 자-기타 외국인(남)", ["외국인"], "외국인"),
    ("누적 혼인 비율", ["비율"], "누적 혼인 비율"),  # a generic item takes the name when the name ends with it
    ("건강 문제로 인한 일상생활 제약 여부", ["비율"], None),  # a survey question names no counted thing
    ("성, 혼인상태 및 경제활동상태인구", ["계"], None),  # lists breakdowns, not counted things
    ("한국국적을 가지지 않은 자 - 기타(유학생 등)", ["기타"], None),  # 기타 is a breakdown value
])
def test_names_without_the_usual_shape(name, items, measure):
    record = _record(name)
    record.items = items
    assert table_roles(record)[0] == measure
