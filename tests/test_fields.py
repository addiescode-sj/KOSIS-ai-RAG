"""T2-30: per-table fields (no data files; records made up for the tests)."""

import pytest

from kosis_ai.schemas import TableInfo
from kosis_rag.fields import item_unit, table_fields
from kosis_rag.schemas import TableClass, TableRecord

offline = pytest.mark.offline
SIDO = ["서울특별시", "부산광역시", "대구광역시", "인천광역시"]


def _record(name, classes, items=(), unit=None, cycle=("Y",), periods=("2016", "2020", "2025")):
    info = TableInfo(
        key="101_DT_X", org_id="101", tbl_id="DT_X", survey_name="인구총조사", organization="국가데이터처",
        table_name=name, period="년", source="api", coverage="A_정상", rows=1, numeric_rows=1, symbol_rows=0,
        parquet="values/101_DT_X.parquet", url="http://kosis.kr/101_DT_X", menu_path="인구",
    )
    return TableRecord(info=info, classes=classes, items=list(items), unit=unit, cycle=list(cycle),
                       periods=list(periods))


@offline
def test_fields_of_a_mentor_shaped_table():
    region = TableClass(name="행정구역별(읍면동)", labels=["전국", *SIDO], n_labels=5,
                        levels={"전국": 1, "시도": 17, "시군구": 240, "읍면동": 3247})
    record = _record("성, 연령 및 이동자유형별 인구(1세이상) - 읍면동(연도 끝자리 0, 5), 시군구(그 외 연도)",
                     [region, TableClass(name="성별", labels=["계"], n_labels=1)], unit="명")
    f = table_fields(record)
    assert f.available_years == [2016, 2020, 2025] and f.cycle == ["Y"]
    assert f.regions == sorted(SIDO) and f.region_level == ["시군구", "시도", "읍면동"]
    assert (f.measure, f.measure_unit) == ("인구", "명")
    assert f.dimensions == ["성", "연령", "이동자유형"]


@offline
def test_national_table_has_no_regions():
    f = table_fields(_record("성별 혼인 건수", [TableClass(name="성별", labels=["계"], n_labels=1)]))
    assert f.regions is None and f.region_level == []


@offline
@pytest.mark.parametrize("items, unit", [
    (["남자 (명)", "여자(명)"], "명"),
    (["가구_계 (가구)", "1인가구 (가구)"], "가구"),
    (["인구 (명)", "성비"], None),  # 성비 has no unit: it may be a ratio, so do not guess
    (["인구 (명)", "가구 (가구)"], None),  # mixed
    (["취업자 (15세 이상)"], None),  # not a unit
    ([], None),
])
def test_item_unit_only_when_every_item_agrees(items, unit):
    assert item_unit(items) == unit


@offline
def test_table_unit_wins_over_item_names():
    record = _record("성별 인구", [], items=["인구 (가구)"], unit="명")
    assert table_fields(record).measure_unit == "명"
