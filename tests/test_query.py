"""T2-14, T2-29: rule parser and the 시도 filter (no model, no server; questions made up for the tests)."""

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from kosis_rag.legacy.retrievers import ConditionRetriever
from kosis_rag.parse.node import parse_query
from kosis_rag.schemas.catalog import TableClass, TableRecord
from kosis_rag.schemas.common import TableInfo
from kosis_rag.shared.filters import TableFilter, filter_tables, filter_with_relaxation, relax, soft_rank, table_regions

offline = pytest.mark.offline
# a made-up 시군구 -> 시도 dictionary (data/sigungu.json is not in git); 중구 sits in several 시도
SIGUNGU = {"해운대구": ["부산광역시"], "수원시": ["경기도"], "광주시": ["경기도"], "부산진구": ["부산광역시"],
           "중구": ["서울특별시", "부산광역시", "대구광역시"]}


@offline
@pytest.mark.parametrize("question, fields", [
    ("2023년 서울시 혼인 건수", {"period_start": "2023", "period_end": "2023", "cycle": None, "regions": ["서울특별시"]}),
    ("2015~2020년 경북과 대구 인구 추이", {"period_start": "2015", "period_end": "2020", "cycle": None,
                                     "regions": ["대구광역시", "경상북도"]}),
    ("2024년 3월 월별 출생아 수", {"period_start": "202403", "period_end": "202403", "cycle": "M", "regions": []}),
    ("최근 20년간 분기별 혼인", {"period_start": None, "period_end": None, "cycle": "Q", "regions": []}),
    ("1990년대 충청도 인구", {"period_start": "1990", "period_end": "1999", "cycle": None,
                         "regions": ["충청북도", "충청남도"]}),
    ("1995년 이후 5년마다 가구 수 추이", {"period_start": "1995", "period_end": None, "cycle": "F", "regions": []}),
    ("2040년까지 장래 인구", {"period_start": None, "period_end": "2040", "cycle": None, "regions": []}),
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


# --- T2-29 step 1: years and periods (분석 #1~#10; B2·B3·B5) ---

@offline
@pytest.mark.parametrize("question, years, start, end", [
    ("1930년 조선인 인구", [1930], "1930", "1930"),  # #1: the data runs 1925~2072
    ("2052년 추계인구", [2052], "2052", "2052"),
    ("1925~2010년 총조사 인구", [], "1925", "2010"),
    ("1993년생 인구", [], None, None),  # #2, B2: a birth year is a breakdown value
    ("1980~1989년생 혼인율", [], None, None),
    ("1990년대생 혼인율", [], None, None),
    ("1993년 출생자 수", [], None, None),
    ("2000년 이전 출생자 수", [], None, "2000"),  # B3: "이전 + 출생" stays a period
    ("2023년 출생아 수", [2023], "2023", "2023"),  # births in 2023, not people born in 2023
    ("2030세대 혼인율", [], None, None),  # #3: a number with a unit
    ("인구 2000명 이상 읍면동 수", [], None, None),
    ("2023년 세대수", [2023], "2023", "2023"),
    ("2015∼2020년 인구", [], "2015", "2020"),  # #4: U+223C
    ("2015～2020년 인구", [], "2015", "2020"),  # full-width tilde
    ("2015년에서 2020년까지 인구", [], "2015", "2020"),
    ("2015년과 2020년 사이 인구", [], "2015", "2020"),
    ("2005년과 2025년 비교", [2005, 2025], "2005", "2025"),  # no 사이: two years
    ("2015년 이후 2020년까지 인구", [], "2015", "2020"),  # #5: a closed range, the end is kept
    ("2015년 이후부터 2020년까지 인구", [], "2015", "2020"),
    ("2020~2015년 인구", [], "2015", "2020"),  # #6: reversed
    ("2024년 1월부터 6월까지 출생아 수", [], "202401", "202406"),  # #7
    ("2024년 1~6월 출생아 수", [], "202401", "202406"),
    ("2023년 12월부터 2024년 3월까지 혼인 건수", [], "202312", "202403"),
    ("2015년 3월부터 2020년까지 인구", [], "201503", "2020"),
    ("2023년 2분기 출생아 수", [], "202304", "202306"),  # #8
    ("2023년 상반기 혼인 건수", [], "202301", "202306"),
    ("2023년 하반기 혼인 건수", [], "202307", "202312"),
    ("2000년대 이후 인구", [], "2000", None),  # #10
    ("20년간 인구 추이", [], None, None),  # two-digit years are not read (B1)
    ("2023학년도 학생 수", [2023], "2023", "2023"),
    ("2023년 3월 출생아 수", [2023], "202303", "202303"),  # a single month keeps the year (B5)
    ("2023년 생산가능인구", [2023], "2023", "2023"),  # 생 starting another word is not the 년생 suffix
    ("1993년생은 몇 명", [], None, None),
])
def test_parser_reads_years_and_periods(question, years, start, end):
    parsed = parse_query(question)
    assert (parsed.years, parsed.period_start, parsed.period_end) == (years, start, end)


@offline
def test_quarter_sets_the_cycle_and_a_half_year_does_not():
    assert parse_query("2023년 2분기 출생아 수").cycle == "Q"
    assert parse_query("2023년 상반기 혼인 건수").cycle is None


@offline
def test_month_periods_filter_by_year():  # B5: yearly tables stay for a month question
    f = parse_query("2023년 3월 출생아 수").to_filter()
    assert f.years == [2023] and f.mode == "strict" and f.period_from is None
    f = parse_query("2024년 1월부터 6월까지 출생아 수").to_filter()
    assert f.years == [] and (f.period_from, f.period_to) == (2024, 2024)


# --- T2-29 step 2: cycle (분석 #13~#15) ---

@offline
@pytest.mark.parametrize("question, cycle", [
    ("6개월 미만 거주한 외국인 수", None),  # #13: a bare 개월 is a duration, not a monthly table
    ("3개월 이상 체류 외국인", None),
    ("6개월간 체류한 외국인", None),  # 개월간 is not 월간
    ("최근 3개월 사망자 수", "M"),
    ("월간 혼인 건수 추이", "M"),
    ("년별 인구 추이", "Y"),  # #14
    ("매 5년 조사 인구", "F"),
    ("인구총조사 기준 가구 수", "F"),
    ("연간 소득 2천만 원 이상 가구", None),  # #15: an amount, not a cycle
    ("연간 추이로 본 출생아 수", "Y"),
    ("연간 시도별 혼인 건수", "Y"),
])
def test_parser_reads_cycles(question, cycle):
    assert parse_query(question).cycle == cycle


# --- T2-29 step 3: 시도 (분석 #19~#22; C1~C3) ---

@offline
@pytest.mark.parametrize("question, regions", [
    ("세대구성별 가구 수", []),  # #19: 대구 inside 세대구성
    ("해운대구 인구", []),  # 대구 inside 해운대구 (the 시군구 itself needs the dictionary, below)
    ("경기 침체 시기의 혼인 건수", []),  # C1 exception list
    ("경기도 광주시 인구", ["경기도"]),  # 광주시 in 경기도, not 광주광역시
    ("경제주체별 소득", []),  # 제주 inside 경제주체
    ("인천공항 이용객", []),  # a short name followed by another word
    ("서울과부산 인구", ["서울특별시", "부산광역시"]),  # a particle before the short name is allowed
    ("서울·부산 인구", ["서울특별시", "부산광역시"]),
    ("서울,부산", ["서울특별시", "부산광역시"]),
    ("서울의 인구", ["서울특별시"]),
    ("부산에서 태어난 사람", ["부산광역시"]),
    ("서울시 혼인", ["서울특별시"]),
    ("강원도 인구", ["강원특별자치도"]),
    ("전라북도 인구", ["전북특별자치도"]),
    ("영남 지역 인구", ["부산광역시", "대구광역시", "울산광역시", "경상북도", "경상남도"]),  # C2
    ("호남권 혼인 건수", ["광주광역시", "전북특별자치도", "전라남도"]),
    ("충청권 인구", ["대전광역시", "세종특별자치시", "충청북도", "충청남도"]),
    ("수도권 인구", ["서울특별시", "인천광역시", "경기도"]),
])
def test_parser_reads_sido_with_boundaries(question, regions):
    assert parse_query(question, sigungu={}).regions == regions


@offline
def test_non_capital_region_unfolds_to_fourteen_sido():
    regions = parse_query("비수도권 인구").regions
    assert len(regions) == 14 and not {"서울특별시", "인천광역시", "경기도"} & set(regions)


@offline
@pytest.mark.parametrize("question, regions", [
    ("해운대구 인구", ["부산광역시"]),  # C3: a 시군구 in one 시도
    ("해운대구의 인구", ["부산광역시"]),
    ("수원시 가구 수", ["경기도"]),
    ("부산진구 인구", ["부산광역시"]),
    ("해운대구와 수원시 인구", ["부산광역시", "경기도"]),
    ("중구 인구", []),  # in several 시도: only with its 시도 in front
    ("서울 중구 인구", ["서울특별시"]),
    ("서울중구 인구", ["서울특별시"]),
    ("서울특별시 중구 인구", ["서울특별시"]),
    ("부산시 중구 인구", ["부산광역시"]),
    ("경기도 광주시 인구", ["경기도"]),
])
def test_sigungu_names_give_their_sido(question, regions):
    assert parse_query(question, sigungu=SIGUNGU).regions == regions


@offline
def test_sigungu_dictionary_is_passed_in():
    """T2-41 S2 (P2-D9): the dictionary is an argument (it was the module global query.SIGUNGU)."""
    assert parse_query("해운대구 인구", sigungu={"해운대구": ["부산광역시"]}).regions == ["부산광역시"]
    assert parse_query("해운대구 인구", sigungu={}).regions == []  # no data/sigungu.json: the 시군구 is simply not read
    assert parse_query("해운대구 인구").regions == []  # None reads no file


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
    # T2-29 C7: measures hold one concept per counted thing (인구 too); 미혼 is a breakdown value, not what is counted
    ("성별 연령별 미혼 인구", ["성", "연령"], ["인구"]),
    ("남녀로 나눈 출생아 수", ["성"], ["출생"]),  # C7: the concept, not the search words (to_filter expands it)
    ("태어난 곳별 인구", ["출생지"], ["인구"]),
    ("월별 이혼율", [], ["이혼"]),  # 월별 is a cycle, not a breakdown
])
def test_parser_reads_breakdowns_and_what_is_counted(question, dims, measures):
    parsed = parse_query(question)
    assert (parsed.dimensions, parsed.measures) == (dims, measures)


# --- T2-29 step 4: breakdowns (분석 #23~#27; C4·C5) ---

@offline
@pytest.mark.parametrize("question, dims", [
    ("세대구성별 가구 수", ["세대구성"]),  # #23: not 성
    ("가구특성별 가구 수", []),  # 특성 is unmapped, and not 성
    ("유형구성별 가구", []),
    ("성·연령별 인구", ["성", "연령"]),  # #24: a list before 별
    ("성 및 연령별 인구", ["성", "연령"]),
    ("성, 연령별 인구", ["성", "연령"]),
    ("시도·시군구별 혼인", ["시도", "시군구"]),
    ("출생지 성별 인구", ["출생지", "성"]),
    ("연령대별 인구", ["연령"]),  # #25: vocab keys and question aliases
    ("가구원 수별 가구", ["가구원수"]),
    ("경제활동상태별 인구", ["경제활동상태"]),
    ("종사상 지위별 취업자", ["종사상지위"]),
    ("거처 종류별 가구", ["거처종류"]),
    ("혼인 여부별 인구", ["혼인상태"]),
    ("1년 전 거주지별 이동자", ["1년전거주지"]),
    ("출생아수별 부인 수", ["출생아수"]),
    ("가구주 성별 가구", ["성"]),  # 가구주 X별 means X
    ("가구주성별 가구", ["성"]),
    ("가구주의 연령별 가구", ["연령"]),
    ("모의 연령별 출생", ["연령"]),
    ("지역별 인구", ["시도"]),  # #27, C4
    ("행정구역별 인구", ["시도"]),
    ("연도별 인구", []),  # cycles are not breakdowns
    ("월별 출생아 수", []),
])
def test_parser_reads_breakdown_keys(question, dims):
    assert parse_query(question).dimensions == dims


# --- T2-29 step 5: what is counted (분석 #29~#31; C7) ---

@offline
@pytest.mark.parametrize("question, measures", [
    ("출생아수별 부인 수", ["부인"]),  # #31: 출생아수별 is a breakdown
    ("사망자녀수별 기혼여성", ["기혼여성"]),
    ("출생 통계", ["출생"]),  # #30
    ("모의 연령별 출생", ["출생"]),
    ("혼인 신고 건수", ["혼인"]),
    ("재혼 건수", ["혼인"]),
    ("혼인상태별 인구", ["인구"]),  # 혼인 is the breakdown, 인구 is counted
    ("출생지별 인구", ["인구"]),
    ("서울 인구", ["인구"]),  # #29: the wide concepts
    ("1인 가구 수", ["1인가구"]),
    ("외국인 수", ["외국인"]),
    ("이동자 수", ["이동자"]),
    ("고령자 비율", ["고령자"]),
    ("취업자 수", ["취업자"]),
    ("주택 수", ["주택"]),
    ("인구총조사 가구 수", ["가구"]),  # 인구총조사 is a survey, not what is counted
    ("사망자 수", ["사망"]),
    ("성별 혼인 건수와 이혼 건수", ["혼인", "이혼"]),
])
def test_parser_reads_counted_concepts(question, measures):
    assert parse_query(question).measures == measures


@offline
def test_only_vital_concepts_filter_and_the_rest_rank():  # C7, two layers
    f = parse_query("2023년 서울 1인 가구 수").to_filter()
    assert f.measures == [] and f.soft_measures == ["1인가구"]
    f = parse_query("재혼 건수").to_filter()
    assert f.measures == ["혼인", "초혼", "재혼"] and f.soft_measures == []
    f = parse_query("고령자 인구").to_filter()
    assert f.measures == [] and f.soft_measures == ["고령", "노인", "인구"]
    assert parse_query("출생아수별 부인 수").to_filter().measures == []  # the 부인 table is not dropped


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


def _timed(key, periods, cycle, prd_se):
    record = _record(key, [])
    record.periods, record.cycle, record.prd_se = periods, cycle, prd_se
    return record


@offline
def test_open_and_closed_ranges_pass_a_table_with_any_point_inside():
    tables = {
        "101_DT_OLD": _timed("101_DT_OLD", ["1985", "1990", "1995", "2000", "2005", "2010"], ["F"], ["F"]),
        "101_DT_NEW": _timed("101_DT_NEW", [str(y) for y in range(2015, 2026)], ["Y"], ["Y"]),
        "101_DT_EARLY": _timed("101_DT_EARLY", ["1970", "1975", "1980"], ["F"], ["F"]),
    }
    f = parse_query("1995년 이후 5년마다 가구 수 추이").to_filter()
    assert f.years == [] and (f.period_from, f.period_to) == (1995, None) and f.cycle == "F"
    keys, removed = filter_tables(tables, f)
    assert keys == {"101_DT_OLD", "101_DT_NEW"}  # yearly data holds every 5th year; 1995 alone is not required
    assert removed["101_DT_EARLY"] == "요청 기간 1995~ 자료 없음 (실제 1970~1980)"
    keys, _ = filter_tables(tables, parse_query("2016~2019년 인구").to_filter())
    assert keys == {"101_DT_NEW"}  # overlap, not both ends


@offline
def test_yearly_question_keeps_five_year_tables_but_ranks_them_lower():  # T2-29 B6 (was: F tables dropped)
    tables = {"101_DT_F": _timed("101_DT_F", ["2015", "2020"], ["F"], ["F"]),
              "101_DT_Y": _timed("101_DT_Y", ["2015", "2016"], ["Y"], ["Y"]),
              "101_DT_M": _timed("101_DT_M", ["201501", "201502"], ["M"], ["M"])}
    f = parse_query("연도별 인구").to_filter()
    keys, removed = filter_tables(tables, f)
    assert keys == {"101_DT_F", "101_DT_Y"} and removed["101_DT_M"].startswith("주기 M")
    docs = [Document(page_content="", metadata={"key": k}) for k in ("101_DT_F", "101_DT_Y")]
    assert [d.metadata["key"] for d in soft_rank(docs, tables, f)] == ["101_DT_Y", "101_DT_F"]


@offline
def test_table_named_after_one_sido_is_that_sidos_table():  # T2-29 분석 #22
    busan = _named("101_DT_BS", "부산광역시 총조사인구 총괄(읍면동/성/연령별)", [])
    assert table_regions(busan) == {"부산광역시"}
    keys, removed = filter_tables({"101_DT_BS": busan}, TableFilter(regions=["서울특별시"]))
    assert keys == set() and removed["101_DT_BS"].startswith("요청 지역 서울특별시 없음")
    keys, _ = filter_tables({"101_DT_BS": busan}, TableFilter(regions=["부산광역시"]))
    assert keys == {"101_DT_BS"}
    assert table_regions(_named("101_DT_X", "부산광역시민 의식 조사", [])) is None  # Hangul follows: not a 시도 head
    assert table_regions(_named("101_DT_Y", "부산 인구", [])) is None  # short names do not count


@offline
def test_soft_measure_words_only_rank():  # C7
    tables = {
        "101_DT_WIFE": _named("101_DT_WIFE", "시도/연령/출생아수별 부인", ["시도별", "연령별", "출생아수별"]),
        "101_DT_POP": _named("101_DT_POP", "시도/성별 인구", ["시도별", "성별"]),
        "101_DT_MAR": _named("101_DT_MAR", "시도/성/연령별 혼인율", ["시도별", "성별", "연령별"]),
    }
    base = _Fixed(keys=["101_DT_WIFE", "101_DT_POP", "101_DT_MAR"])
    docs = ConditionRetriever(base=base, records=tables).invoke("인구")
    assert [d.metadata["key"] for d in docs] == ["101_DT_POP", "101_DT_WIFE", "101_DT_MAR"]  # nothing dropped


# --- T2-29 step 6: relaxation when no table passes (분석 #32; D1~D3) ---

@offline
def test_condition_retriever_relaxes_a_region_no_table_holds():
    records = {k: RECORDS[k] for k in ("101_DT_SEOUL", "101_DT_BUSAN")}  # both have 시도 rows, neither 제주
    base = _Fixed(keys=["101_DT_BUSAN", "101_DT_SEOUL"])
    hard = ConditionRetriever(base=base, records=records)
    assert [d.metadata["key"] for d in hard.invoke("2020년 제주 인구")] == ["101_DT_BUSAN", "101_DT_SEOUL"]
    assert hard.last_relaxed == ["regions"] and hard.last_removed == {}
    assert [d.metadata["key"] for d in hard.invoke("2020년 서울 인구")] == ["101_DT_SEOUL"]
    assert hard.last_relaxed == [] and set(hard.last_removed) == {"101_DT_BUSAN"}
    assert hard.invoke("1990년 서울 인구") == [] and hard.last_relaxed == []  # years are never relaxed (D2)


@offline
def test_relaxation_drops_conditions_in_order_and_keeps_the_years():
    tables = {
        "101_DT_MAR": _named("101_DT_MAR", "시도/성/연령별 혼인율", ["시도별", "성별", "연령별"]),
        "101_DT_BIRTH": _named("101_DT_BIRTH", "시도/성별 출생", ["시도별", "성별"]),
    }
    keys, relaxed, _ = filter_with_relaxation(tables, TableFilter(years=[2020], measures=["사망"], dimensions=["성"]))
    assert relaxed == ["measures"] and keys == set(tables)  # measures first; the breakdown still holds
    keys, relaxed, _ = filter_with_relaxation(tables, TableFilter(measures=["사망"], dimensions=["국적"]))
    assert relaxed == ["measures", "dimensions"] and keys == set(tables)
    keys, relaxed, removed = filter_with_relaxation(tables, TableFilter(years=[1990], measures=["사망"]))
    assert keys == set() and relaxed == [] and set(removed) == set(tables)  # D2: a missing year refuses as it is
    f = TableFilter(regions=["제주특별자치도"])
    keys, relaxed, _ = filter_with_relaxation({k: RECORDS[k] for k in ("101_DT_SEOUL", "101_DT_BUSAN")}, f)
    assert relaxed == ["regions"] and keys == {"101_DT_SEOUL", "101_DT_BUSAN"}
    assert relax(f, relaxed).region_mode == "soft" and f.region_mode == "hard"  # f itself is left alone
    with pytest.raises(ValueError):
        relax(f, ["years"])
