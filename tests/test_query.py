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


@offline
@pytest.mark.parametrize("question, dims, measures", [
    ("2023년 시도별 혼인 건수", ["시도"], ["혼인"]),
    ("성별 연령별 미혼 인구", ["성", "연령"], []),  # 미혼 is a breakdown value, not what is counted
    ("남녀로 나눈 출생아 수", ["성"], ["출생", "출산", "생아"]),
    ("태어난 곳별 인구", ["출생지"], []),
    ("월별 이혼율", [], ["이혼"]),  # 월별 is a cycle, not a breakdown
])
def test_parser_reads_breakdowns_and_what_is_counted(question, dims, measures):
    parsed = parse_query(question)
    assert (parsed.dimensions, parsed.measures) == (dims, measures)


def _named(key, name, classes):
    record = _record(key, [])
    record.info = record.info.model_copy(update={"table_name": name})
    record.classes = [TableClass(name=c, labels=["계"], n_labels=1) for c in classes]
    return record


@offline
def test_breakdown_and_counted_filters_keep_tables_that_do_not_say():
    tables = {
        "101_DT_MAR": _named("101_DT_MAR", "시도/성/연령별 혼인율", ["시도별", "성별", "연령별"]),
        "101_DT_WIFE": _named("101_DT_WIFE", "시도/연령/출생아수별 부인", ["시도별", "연령별", "출생아수별"]),
        "101_DT_BIRTH": _named("101_DT_BIRTH", "시도/성별 출생", ["시도별", "성별"]),
        "101_DT_VITAL": _named("101_DT_VITAL", "인구동향(출생,사망,혼인,이혼)", ["행정구역별"]),  # says no measure
    }
    keys, removed = filter_tables(tables, TableFilter(measures=["출생", "출산", "생아"]))
    assert keys == {"101_DT_BIRTH", "101_DT_VITAL"}
    assert removed["101_DT_WIFE"] == "세는 대상 부인 (요청 출생·출산·생아)"
    keys, removed = filter_tables(tables, TableFilter(dimensions=["성", "연령"]))
    assert keys == {"101_DT_MAR"} and removed["101_DT_BIRTH"].startswith("분류 연령 없음")


@offline
def test_condition_retriever_returns_at_most_ten():
    base = _Fixed(keys=["101_DT_NATION"] * 15)
    assert len(ConditionRetriever(base=base, records=RECORDS, k=20).invoke("인구")) == 10
