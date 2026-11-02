"""T2-24, T2-25: the Chroma document and metadata built for one table (no server, no model)."""

import pytest

from kosis_ai.schemas import TableInfo
from kosis_rag.index import LABEL_CAP, load, region_summary, sido_names, to_document, to_metadata
from kosis_rag.loader import _META_FIELDS
from kosis_rag.schemas import TableClass, TableRecord

offline = pytest.mark.offline

INFO = TableInfo(
    key="101_DT_T", org_id="101", tbl_id="DT_T", survey_name="인구총조사", organization="국가데이터처",
    table_name="1인 가구", period="5년 (2015 ~ 2025)", source="api", coverage="A_정상", rows=6, numeric_rows=5,
    symbol_rows=1, parquet="values/101_DT_T.parquet", url="http://kosis.kr/x", menu_path="인구 > 가구",
)


def _record(n_labels=2, levels=None, items=("1인 가구",)):
    labels = ["전국", "서울특별시"] + [f"지역{i}" for i in range(n_labels - 2)]
    return TableRecord(info=INFO, menu_paths=["인구 > 가구"], cycle=["F"], prd_se=["Y"], periods=["2015", "2020"],
                       start=2015, end=2020, items=list(items),
                       classes=[TableClass(name="행정구역별", labels=labels, n_labels=len(labels), levels=levels or {})])


@offline
def test_document_is_metadata_only_with_real_period_and_capped_labels():
    doc = to_document(_record(n_labels=LABEL_CAP + 7))  # no levels: not a region class, labels listed
    assert "수록기간: 2015~2020 (Y)" in doc and "항목: 1인 가구" in doc
    assert "외 7개" in doc and "지역97" in doc and "지역98" not in doc
    assert "값 " not in doc  # no value summary from Parquet


@offline
def test_document_summarizes_region_classes_by_level_and_starts_with_a_summary_line():
    doc = to_document(_record(n_labels=3500, levels={"전국": 1, "시도": 17, "시군구": 250, "읍면동": 3306}))
    assert doc.splitlines()[0] == "1인 가구 | Y 2015~2020 | 지역 단위: 전국·시도(17)·시군구(250)·읍면동(3,306) | 항목: 1인 가구"
    assert "행정구역별: 지역 단위 전국·시도(17)·시군구(250)·읍면동(3,306)" in doc
    assert "지역0" not in doc  # no place names in the embedding text (they stay in search_text)


@offline
def test_sido_names_are_short_and_merge_old_and_new_names():
    labels = ["전국", "서울특별시", "강원도", "강원특별자치도", "종로구", "제주특별자치도"]
    assert sido_names(labels) == ["서울", "강원", "제주"]
    assert region_summary({"전국": 1, "시도": 3, "시군구": 1}, ["서울", "강원", "제주"]) == "전국·시도(3: 서울, 강원, 제주)·시군구(1)"


@offline
def test_document_cleans_item_names_and_keeps_units_on_one_line():
    doc = to_document(_record(items=["혼인건수 (건)", "조혼인율 (천명당)", "가구_계", "혼인건수 (건)"]))
    assert "항목: 혼인건수, 조혼인율, 가구 계\n단위: 건, 천명당" in doc
    assert "지역 단위: 지역 구분 없음 | 항목: 혼인건수, 조혼인율, 가구 계" in doc.splitlines()[0]


@offline
def test_metadata_is_scalar_keeps_candidate_keys_and_drops_none():
    meta = to_metadata(_record())
    assert set(_META_FIELDS) <= set(meta) and meta["period"] == "5년 (2015 ~ 2025)"
    assert "value_summary" not in meta and "rows" not in meta
    assert meta["start"] == 2015 and meta["cycle"] == "F" and "unit" not in meta and "topic" not in meta
    assert all(isinstance(v, (str, int, float, bool)) for v in meta.values())


class _FakeCollection:
    def __init__(self):
        self.rows = {"101_OLD": "stale"}

    def upsert(self, ids, documents, metadatas, embeddings):
        self.rows.update(zip(ids, documents))

    def get(self, include):
        return {"ids": list(self.rows)}

    def delete(self, ids):
        for i in ids:
            del self.rows[i]

    def count(self):
        return len(self.rows)


class _FakeClient:
    def __init__(self):
        self.collection = _FakeCollection()

    def get_or_create_collection(self, name, embedding_function, metadata):
        assert embedding_function is None and metadata["hnsw:space"] == "cosine"
        return self.collection

    def get_max_batch_size(self):
        return 2


@offline
def test_reload_upserts_in_batches_and_drops_tables_no_longer_in_the_input():
    client = _FakeClient()
    ids, docs = ["a", "b", "c"], ["A", "B", "C"]
    assert load(client, ids, docs, [{}] * 3, [[0.0]] * 3, "c") == 3
    assert load(client, ids, docs, [{}] * 3, [[0.0]] * 3, "c") == 3
    assert client.collection.rows == {"a": "A", "b": "B", "c": "C"}
