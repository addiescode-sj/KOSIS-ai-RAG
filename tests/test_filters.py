"""T2-05: the registry filter and how the Chroma retriever applies it (fake collection and embedder; no model, no server)."""

import pytest

from kosis_ai.schemas import TableInfo
from kosis_rag.chain import search
from kosis_rag.filters import TableFilter, filter_tables, why_excluded
from kosis_rag.loader import to_document
from kosis_rag.retriever import ChromaTableRetriever
from kosis_rag.schemas import TableRecord

offline = pytest.mark.offline


def _record(key, periods, cycle=("Y",), prd_se=("Y",), org="국가데이터처"):
    info = TableInfo(
        key=key, org_id="101", tbl_id=key.split("_", 1)[1], survey_name="인구총조사", organization=org,
        table_name=f"표 {key}", period="년", source="api", coverage="A_정상", rows=1, numeric_rows=1, symbol_rows=0,
        parquet=f"values/{key}.parquet", url=f"http://kosis.kr/{key}", menu_path="인구",
    )
    years = [int(p[:4]) for p in periods]
    return TableRecord(info=info, cycle=list(cycle), prd_se=list(prd_se), periods=list(periods),
                       start=min(years, default=None), end=max(years, default=None))


RECORDS = {
    "101_DT_A": _record("101_DT_A", ["2015", "2020", "2021", "2022", "2023"]),
    # listed range 2015~2025 but only census years are in the values
    "101_DT_C": _record("101_DT_C", ["2015", "2020", "2025"], cycle=("F",), prd_se=("Y",)),
    "101_DT_M": _record("101_DT_M", ["202301", "202302"], cycle=("M", "Y"), prd_se=("M",), org="통계청"),
    "101_DT_E": _record("101_DT_E", []),
}


class FakeCollection:
    """Returns stored keys in a fixed order (by distance), honouring a `key $in` filter like Chroma."""

    def __init__(self, distances):
        self.distances = distances
        self.calls = []

    def query(self, query_embeddings, n_results, include, where=None):
        self.calls.append(where)
        allowed = set(where["key"]["$in"]) if where else set(self.distances)
        hits = sorted((d, k) for k, d in self.distances.items() if k in allowed)[:n_results]
        docs = [to_document(RECORDS[k].info) for _, k in hits]
        return {"ids": [[k for _, k in hits]], "documents": [[d.page_content for d in docs]],
                "metadatas": [[d.metadata for d in docs]], "distances": [[d for d, _ in hits]]}


def _retriever(distances, **kw):
    return ChromaTableRetriever(collection=FakeCollection(distances), embed_query=lambda q: [0.0], records=RECORDS, **kw)


@offline
def test_retriever_keeps_the_search_tables_contract_and_scores_by_one_minus_distance():
    r = _retriever({"101_DT_C": 0.4, "101_DT_A": 0.1, "101_DT_M": 0.7})
    candidates = search(r, "인구", k=2)
    assert [c.key for c in candidates] == ["101_DT_A", "101_DT_C"]
    assert [round(c.score, 3) for c in candidates] == [0.9, 0.6]
    assert candidates[0].url == "http://kosis.kr/101_DT_A"


@offline
def test_year_filter_uses_real_periods_not_the_listed_range():
    # DT_C covers 2015~2025 by range but holds no 2023 value
    keys, removed = filter_tables(RECORDS, TableFilter(years=[2023]))
    assert keys == {"101_DT_A", "101_DT_M"}  # the monthly table holds 202301, 202302
    assert removed["101_DT_C"].startswith("요청 2023년 없음 (실제 2015~2025, 3개 시점, 주기 F)")
    assert removed["101_DT_E"] == "실제 수록 시점 없음"


@offline
def test_partial_mode_keeps_a_table_with_some_of_the_years():
    assert why_excluded(RECORDS["101_DT_C"], TableFilter(years=[2020, 2023], mode="partial")) is None
    assert why_excluded(RECORDS["101_DT_C"], TableFilter(years=[2020, 2023])) is not None


@offline
def test_cycle_matches_metadata_or_values_and_organization_matches_old_name():
    assert why_excluded(RECORDS["101_DT_C"], TableFilter(cycle="F")) is None  # metadata F, values Y
    assert why_excluded(RECORDS["101_DT_C"], TableFilter(cycle="Y")) is None
    assert why_excluded(RECORDS["101_DT_A"], TableFilter(cycle="M")).startswith("주기")
    assert why_excluded(RECORDS["101_DT_M"], TableFilter(organization="국가데이터처")) is None  # 통계청 = 국가데이터처


@offline
def test_filter_goes_to_chroma_as_one_key_in_condition_and_reasons_are_kept():
    r = _retriever({"101_DT_C": 0.1, "101_DT_A": 0.2}).with_filter(TableFilter(years=[2023]))
    docs = r.invoke("2023년 인구")
    assert [d.metadata["key"] for d in docs] == ["101_DT_A"]
    assert r.collection.calls == [{"key": {"$in": ["101_DT_A", "101_DT_M"]}}]
    assert set(r.last_removed) == {"101_DT_C", "101_DT_E"}


@offline
def test_no_table_left_after_filtering_returns_nothing_without_querying():
    r = _retriever({"101_DT_A": 0.1}).with_filter(TableFilter(years=[1990]))
    assert r.invoke("1990년 인구") == []
    assert r.collection.calls == [] and len(r.last_removed) == 4



@offline
def test_filter_without_records_is_an_error_and_no_filter_sends_no_condition():
    r = ChromaTableRetriever(collection=FakeCollection({"101_DT_A": 0.1}), embed_query=lambda q: [0.0])
    r.invoke("인구")
    assert r.collection.calls == [None]
    with pytest.raises(ValueError):
        r.with_filter(TableFilter(years=[2023])).invoke("2023년 인구")
