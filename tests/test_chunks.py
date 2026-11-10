"""T2-43: region chunks of the very large tables (no server, no model; a small Parquet made here)."""

import pytest

from kosis_rag.schemas.common import TableInfo
from kosis_rag.shared.catalog import _META_FIELDS
from kosis_rag_build import chunks as ch

offline = pytest.mark.offline

INFO = TableInfo(
    key="101_DT_T", org_id="101", tbl_id="DT_T", survey_name="주민등록인구현황", organization="행정안전부",
    table_name="행정구역(읍면동)별/5세별 주민등록인구", period="년 (2020 ~ 2022)", source="zip", coverage="A_정상",
    rows=0, numeric_rows=0, symbol_rows=0, parquet="values/101_DT_T.parquet", url="http://kosis.kr/x",
    menu_path="인구 > 주민등록인구현황",
)
REGIONS = {"00": "전국", "11": "서울특별시", "11110": "종로구", "1111051500": "청운효자동", "4272035000": "동면"}
AGES = [("0", "계"), ("5", "5 - 9세"), ("10", "10 - 14세")]
ITEMS = [("T2", "총인구수 (명)"), ("T3", "남자인구수 (명)")]


def _rows(skip=()):
    """Every region has 2020~2022 and 202001; `skip` drops (code, period) pairs."""
    rows = []
    for code, label in REGIONS.items():
        for se, period in (("Y", "2020"), ("Y", "2021"), ("Y", "2022"), ("M", "202001")):
            if (code, period) in skip:
                continue
            for age_code, age in AGES:
                for item_id, item in ITEMS:
                    blank = code == "11110" and period == "2020" and item_id == "T3"
                    rows.append({"dim1_code": code, "dim1_label": label, "prd_se": se, "period": period,
                                 "dim2_name": "5세별", "dim2_code": age_code, "dim2_label": age, "item_id": item_id,
                                 "item_name": item, "value_kind": "blank" if blank else "ok",
                                 "value": None if blank else 1.0})
    return rows


def _parquet(tmp_path, rows):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    path = tmp_path / "t.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


@offline
def test_registry_covers_every_row_once_and_records_real_periods(tmp_path):
    rows = _rows(skip={("1111051500", "2020"), ("4272035000", "2021")})  # a new dong; a dong with a gap
    chunks = ch.region_registry(_parquet(tmp_path, rows), INFO)
    by = {c.code: c for c in chunks}
    assert sorted(by) == sorted(REGIONS) and sum(c.rows for c in chunks) == len(rows)
    assert by["1111051500"].periods == {"M": ["202001"], "Y": ["2021", "2022"]} and not by["1111051500"].missing
    assert by["4272035000"].missing == {"Y": ["2021"]}  # first~last looks continuous, the middle year is not there
    assert by["11110"].rows - by["11110"].value_rows == len(AGES)  # blank values are counted apart
    assert by["00"].other_labels == ["계", "5 - 9세", "10 - 14세"] and by["00"].items == [i for _, i in ITEMS]


@offline
def test_region_path_and_level_follow_the_code_and_skip_a_parent_the_table_lacks():
    assert ch.region_path("1111051500", REGIONS) == (["서울특별시", "종로구", "청운효자동"], "11110")
    assert ch.region_path("4272035000", REGIONS) == (["동면"], None)  # old code: no 시도, no 시군구 in the table
    assert ch.region_path("00", REGIONS) == (["전국"], None)
    assert [ch.level_of(c) for c in REGIONS] == ["전국", "시도", "시군구", "읍면동", "읍면동"]
    with pytest.raises(ValueError):
        ch.level_of("123")


@offline
def test_registry_rejects_repeated_keys_and_two_labels_for_one_code(tmp_path):
    rows = _rows()
    with pytest.raises(ValueError, match="repeat"):
        ch.region_registry(_parquet(tmp_path, rows + rows[:1]), INFO)
    groups = [{"code": "11", "label": name, "prd_se": "Y", "period": "2020", "rows": 1, "value_rows": 1}
              for name in ("서울특별시", "서울시")]
    with pytest.raises(ValueError, match="two labels"):
        ch.chunks_from_groups(INFO, groups, "5세별", ["계"], ["총인구수 (명)"])


@offline
def test_document_has_no_values_and_metadata_finds_the_rows_again(tmp_path):
    rows = _rows(skip={("4272035000", "2021")})
    path = _parquet(tmp_path, rows)
    chunks = {c.code: c for c in ch.region_registry(path, INFO)}
    doc = ch.to_document(chunks["1111051500"], INFO)
    assert doc.startswith("서울특별시 종로구 청운효자동 | 행정구역(읍면동)별/5세별 주민등록인구 | 지역: 서울특별시 > 종로구 > 청운효자동 (읍면동)")
    assert "수록기간: 월 202001~202001, 년 2020~2022" in doc and "항목: 총인구수, 남자인구수 (명)" in doc
    assert "1.0" not in doc
    assert "(중간에 빈 시점 1개)" in ch.to_document(chunks["4272035000"], INFO)

    for code, chunk in chunks.items():
        meta = ch.to_metadata(chunk, INFO)
        assert all(meta[name] == getattr(INFO, name) for name in _META_FIELDS)  # shared.catalog.to_candidate works
        assert ch.read_chunk_rows(path, meta).num_rows == chunk.rows == meta["rows"]
        assert None not in meta.values() and [] not in meta.values()  # Chroma rejects both
    meta = ch.to_metadata(chunks["4272035000"], INFO)
    assert (meta["start_y"], meta["end_y"], meta["n_y"], meta["missing_y"]) == (2020, 2022, 2, ["2021"])
    assert "parent_code" not in meta and chunks["4272035000"].id == "101_DT_T|4272035000"


@offline
def test_collection_name_follows_the_data_and_is_not_listed_as_a_table_build(tmp_path):
    chunks = ch.region_registry(_parquet(tmp_path, _rows()), INFO)
    assert ch.collection_name("a") == ch.collection_name("a") != ch.collection_name("b")
    assert ch.collection_name("a").startswith("kosis_chunks__region1__")
    record = ch.chunk_record("202611101200", "a", chunks, 1024, 90)
    assert "doc_format" not in record and record["n_docs"] == len(REGIONS)  # chroma.list_builds skips it
    assert record["n_rows"] == sum(c.rows for c in chunks)

    out = tmp_path / "chunks" / "regions.jsonl"
    ch.write_registry(out, chunks)
    assert ch.read_registry(out) == chunks
