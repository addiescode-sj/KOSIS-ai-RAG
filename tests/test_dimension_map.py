"""T2-42 (KOS-118): breakdown names that had no standard key, bundled "특성별" classes and their marginal blocks.

Records are made up in the style of KOSIS tables (labels as the 119 "특성별" tables write them).
"""

import pytest

from kosis_rag.schemas.catalog import TableClass, TableRecord
from kosis_rag.schemas.common import TableInfo
from kosis_rag.shared.filters import TableFilter, crosses_blocks, lacks_role
from kosis_rag.shared.roles import bundled_blocks, label_blocks, table_roles
from kosis_rag.shared.vocab import DIMENSION_WORDS, dimension_keys

offline = pytest.mark.offline

SAKHALIN_LABELS = ["전체", "남성", "여성", "사할린", "한반도", "1세대", "2세대", "4세대 이상", "모름", "대한민국", "러시아",
                   "재외동포(F-4)", "영주(F-5)", "1년 미만", "10년~20년 미만", "20년 이상", "임대 아파트", "중학교", "대학원"]


def _record(name, classes):
    info = TableInfo(
        key="182_DT_X", org_id="182", tbl_id="DT_X", survey_name="영주귀국사할린동포실태조사", organization="x",
        table_name=name, period="년", source="api", coverage="A_정상", rows=1, numeric_rows=1, symbol_rows=0,
        parquet="values/182_DT_X.parquet", url="http://kosis.kr/x", menu_path="인구",
    )
    return TableRecord(info=info, classes=[TableClass(name=n, labels=list(ls), n_labels=len(ls)) for n, ls in classes])


@offline
@pytest.mark.parametrize("name, keys", [
    ("전출지별", ["전출지"]),
    ("출입국 항구", ["항구"]),
    ("한국인 남편의 혼인종류", ["혼인종류"]),
    ("거주지", ["현거주지"]),
    ("1년전 거주지 종류", ["1년전거주지"]),
    ("성별 육아휴직 사용 여부", ["성"]),  # the part before 별 names the breakdown, the rest what is counted
    ("소득수준별 육아휴직 사용 여부", ["소득수준"]),
    ("한국 정부 지원 정책 인지도", []),  # a survey question is not a breakdown: still unmapped
])
def test_new_names_get_a_key(name, keys):
    assert dimension_keys(name) == keys


@offline
def test_new_keys_are_question_words_and_household_relation_too():
    for word in ("전출지", "소득수준", "이주세대", "체류자격", "가구주와의관계"):
        assert word in DIMENSION_WORDS
    assert "가구주의연령" not in DIMENSION_WORDS  # other 가구주 compounds stay table-side (T2-29)


@offline
def test_label_blocks_in_label_order_and_unknown_labels_skipped():
    assert label_blocks(SAKHALIN_LABELS) == ["성", "출생지", "이주세대", "국적", "체류자격", "거주기간", "주거시설형태",
                                             "교육정도"]
    assert label_blocks(["전체", "모름", "잘 모르겠다"]) == []


@offline
def test_bundled_class_becomes_its_blocks_and_counted_items_class_drops():
    survey = _record("한국어 의사소통 정도", [("특성별", SAKHALIN_LABELS), ("한국어 의사소통 정도", ["사례수", "매우 잘할 수 있다"])])
    _, dims = table_roles(survey)
    assert dims[:3] == ["성", "출생지", "이주세대"] and "특성" not in dims
    assert "한국어 의사소통 정도" in dims  # survey answers are left as they are (plan: not dropped in option B)
    vital = _record("시도별 기본항목", [("시도별", ["전국", "서울특별시"]), ("기본항목별", ["출생아수(명)", "조사망률(천명당)"])])
    assert table_roles(vital)[1] == ["시도"]
    plain = _record("특성별 비율", [("특성별", ["계", "기타"])])  # no known block: the name stays
    assert table_roles(plain)[1] == ["특성"]


@offline
def test_two_blocks_of_one_bundled_class_are_not_a_cross():
    t = _record("한국어 의사소통 정도", [("특성별", SAKHALIN_LABELS)])
    _, dims = table_roles(t)
    assert bundled_blocks(t)[0][:2] == ["성", "출생지"]
    assert crosses_blocks(t, ["성", "이주세대"], dims) is not None
    assert lacks_role(t, TableFilter(dimensions=["성", "이주세대"])) is not None
    assert lacks_role(t, TableFilter(dimensions=["체류자격"])) is None
    # 성 held by a real class of its own: 성 × 이주세대 cells may exist, so the table stays
    crossed = _record("성/특성별 응답", [("성별", ["계", "남자", "여자"]), ("특성별", SAKHALIN_LABELS)])
    assert crosses_blocks(crossed, ["성", "이주세대"], table_roles(crossed)[1]) is None
