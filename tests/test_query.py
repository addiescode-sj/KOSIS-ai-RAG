"""T2-14: rule parser and the 시도 filter (no model, no server; questions made up for the tests)."""

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from kosis_ai.schemas import TableInfo
from kosis_rag.filters import TableFilter, filter_tables, table_regions
from kosis_rag.query import ConditionRetriever, parse_query
from kosis_rag.schemas import TableClass, TableRecord

offline = pytest.mark.offline


@offline
@pytest.mark.parametrize("question, fields", [
    ("2023년 서울시 혼인 건수", {"period_start": "2023", "period_end": "2023", "cycle": None, "regions": ["서울특별시"]}),
    ("2015~2020년 경북과 대구 인구 추이", {"period_start": "2015", "period_end": "2020", "cycle": None,
                                     "regions": ["대구광역시", "경상북도"]}),
    ("2024년 3월 월별 출생아 수", {"period_start": "202403", "period_end": "202403", "cycle": "M", "regions": []}),
    ("최근 20년간 분기별 혼인", {"period_start": None, "period_end": None, "cycle": "Q", "regions": []}),
    ("1990년대 충청도 인구", {"period_start": None, "period_end": None, "cycle": None,
                         "regions": ["충청북도", "충청남도"]}),
    ("5년마다 조사하는 강원 가구 수", {"period_start": None, "period_end": None, "cycle": "F",
                                "regions": ["강원특별자치도"]}),
])
def test_parser_reads_periods_cycle_and_sido(question, fields):
    assert parse_query(question).model_dump(include=set(fields)) == fields


@offline
def test_one_year_is_strict_and_two_years_partial():
    assert parse_query("2020년 인구").to_filter().mode == "strict"
    f = parse_query("2005년과 2025년 고령인구 비교").to_filter()
    assert f.years == [2005, 2025] and f.mode == "partial"
    assert parse_query("최근 3개월 사망자 수").recent


def _record(key, labels):
    info = TableInfo(
        key=key, org_id="101", tbl_id=key.split("_", 1)[1], survey_name="인구총조사", organization="국가데이터처",
        table_name=f"표 {key}", period="년", source="api", coverage="A_정상", rows=1, numeric_rows=1, symbol_rows=0,
        parquet=f"values/{key}.parquet", url=f"http://kosis.kr/{key}", menu_path="인구",
    )
    classes = [TableClass(name="행정구역별", labels=labels, n_labels=len(labels))] if labels else []
    return TableRecord(info=info, cycle=["Y"], prd_se=["Y"], periods=["2020"], start=2020, end=2020, classes=classes)


RECORDS = {
    "101_DT_SEOUL": _record("101_DT_SEOUL", ["전국", "서울특별시", "부산광역시", "강원도"]),
    "101_DT_BUSAN": _record("101_DT_BUSAN", ["전국", "부산광역시", "대구광역시", "인천광역시"]),
    "101_DT_NATION": _record("101_DT_NATION", []),  # no administrative class: never dropped by region
    "101_DT_OLD": _record("101_DT_OLD", ["부산직할시", "대구직할시", "서울특별시"]),
}


@offline
def test_region_filter_keeps_national_tables_and_merges_old_names():
    assert table_regions(RECORDS["101_DT_NATION"]) is None
    assert table_regions(RECORDS["101_DT_SEOUL"]) == {"서울특별시", "부산광역시", "강원특별자치도"}
    keys, removed = filter_tables(RECORDS, TableFilter(regions=["강원특별자치도"]))
    assert keys == {"101_DT_SEOUL", "101_DT_NATION"}
    assert removed["101_DT_BUSAN"].startswith("요청 지역 강원특별자치도 없음")
    keys, _ = filter_tables(RECORDS, TableFilter(regions=["부산광역시"]))
    assert keys == set(RECORDS)  # 부산직할시 counts as 부산광역시


@offline
def test_soft_region_mode_drops_nothing():
    assert TableFilter(regions=["서울특별시"], region_mode="soft").is_empty()
    keys, _ = filter_tables(RECORDS, TableFilter(regions=["서울특별시"], region_mode="soft"))
    assert keys == set(RECORDS)


class _Fixed(BaseRetriever):
    keys: list[str]

    def _get_relevant_documents(self, query, *, run_manager):
        return [Document(page_content="", metadata={"key": k}) for k in self.keys]


@offline
def test_condition_retriever_post_filters_or_reranks_a_plain_retriever():
    base = _Fixed(keys=["101_DT_BUSAN", "101_DT_NATION", "101_DT_SEOUL"])
    hard = ConditionRetriever(base=base, records=RECORDS)
    assert [d.metadata["key"] for d in hard.invoke("서울 인구")] == ["101_DT_NATION", "101_DT_SEOUL"]
    assert set(hard.last_removed) == {"101_DT_BUSAN"}
    soft = ConditionRetriever(base=base, records=RECORDS, region_mode="soft")
    assert [d.metadata["key"] for d in soft.invoke("서울 인구")] == ["101_DT_NATION", "101_DT_SEOUL", "101_DT_BUSAN"]

