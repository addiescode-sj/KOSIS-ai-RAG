"""T2-01: tables.jsonl records take periods, items and labels from the values and flag where the sources disagree."""

import csv
import json

import pandas as pd
import pytest

from kosis_rag.metadata import build_records, main
from kosis_rag.schemas import TableRecord

offline = pytest.mark.offline

ROW = {
    "key": "101_DT_T", "org_id": "101", "tbl_id": "DT_T", "survey_name": "인구총조사", "organization": "국가데이터처",
    "table_name": "1인 가구", "period": "5년 (2015 ~ 2025)", "source": "api", "coverage": "A_정상", "missing_periods": "",
    "rows": "4", "numeric_rows": "4", "symbol_rows": "0", "parquet": "values/101_DT_T.parquet",
    "url": "http://kosis.kr/x", "menu_path": "인구 > 가구",
}


def _data_dir(tmp_path, periods=("2015", "2020"), meta=True):
    (tmp_path / "values").mkdir()
    (tmp_path / "reference").mkdir()
    (tmp_path / "table_metadata").mkdir()
    with (tmp_path / "tables.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ROW))
        writer.writeheader()
        writer.writerow(ROW)
    catalog = {"org_id": "101", "tbl_id": "DT_T", "topic": "인구",
               "catalog_periods": [{"frequency_label": "5년", "start": "2015", "end": "2025"}],
               "paths": [{"path": "인구 > 가구"}, {"path": "인구 > 인구총조사 > 가구"}]}
    (tmp_path / "reference" / "population_table_catalog.jsonl").write_text(json.dumps(catalog) + "\n", encoding="utf-8")
    if meta:
        (tmp_path / "table_metadata" / "101_DT_T.json").write_text(
            json.dumps({"tblEngNm": "One-person households", "unitNmKor": "가구", "periodStr": "F",
                        "containPeriod": " 5년 2015 ~ 2025"}), encoding="utf-8")
    # Listed out of code order on purpose; 중구 appears under two parents (서울 11020, 부산 21020)
    regions = [("11010", "종로구"), ("21020", "중구"), ("00", "전국"), ("11020", "중구"), ("11", "서울특별시")]
    cells = [(p, code, label) for p in periods for code, label in regions]
    pd.DataFrame({
        "prd_se": "Y", "period": [p for p, _, _ in cells], "item_name": "1인 가구",
        "dim1_name": "행정구역별", "dim1_code": [c for _, c, _ in cells], "dim1_label": [r for _, _, r in cells],
        "dim2_name": None, "dim2_code": None, "dim2_label": None,
    }).to_parquet(tmp_path / "values" / "101_DT_T.parquet")
    return tmp_path


@offline
def test_record_takes_periods_and_labels_from_the_values(tmp_path):
    (record,) = build_records(_data_dir(tmp_path))
    assert record.periods == ["2015", "2020"] and (record.start, record.end) == (2015, 2020)
    # KOSIS code order: 전국 (00) -> 시도 (11) -> 시군구, so a capped label list keeps the top levels
    assert [(c.name, c.labels, c.n_labels) for c in record.classes] == [
        ("행정구역별", ["전국", "서울특별시", "종로구", "중구"], 4)
    ]
    assert record.items == ["1인 가구"] and record.unit == "가구"
    assert record.menu_paths == ["인구 > 가구", "인구 > 인구총조사 > 가구"]
    assert "서울특별시" in record.search_text and "One-person households" in record.search_text


@offline
def test_flat_region_codes_still_put_the_total_and_sido_first(tmp_path):
    # 법무부 등록외국인 tables: same-length codes that interleave 시도 with their 시군구
    data = _data_dir(tmp_path)
    regions = [("A01", "총계"), ("A02", "서울특별시"), ("A03", "강남구"), ("A04", "부산광역시"), ("A05", "중구")]
    pd.DataFrame({
        "prd_se": "Y", "period": "2020", "item_name": "등록외국인", "dim1_name": "행정구역(시군구)별",
        "dim1_code": [c for c, _ in regions], "dim1_label": [r for _, r in regions],
        "dim2_name": None, "dim2_code": None, "dim2_label": None,
    }).to_parquet(data / "values" / "101_DT_T.parquet")
    (record,) = build_records(data)
    assert record.classes[0].labels == ["총계", "서울특별시", "부산광역시", "강남구", "중구"]


def _write_regions(data, name, regions):
    pd.DataFrame({
        "prd_se": "Y", "period": "2020", "item_name": "인구", "dim1_name": name,
        "dim1_code": [c for c, _ in regions], "dim1_label": [r for _, r in regions],
        "dim2_name": None, "dim2_code": None, "dim2_label": None,
    }).to_parquet(data / "values" / "101_DT_T.parquet")


@offline
def test_region_class_counts_labels_per_level_from_the_codes(tmp_path):
    # T2-25: the embedding document says "전국·시도(3)·시군구(2)·읍면동(1)" instead of listing the places
    data = _data_dir(tmp_path)
    _write_regions(data, "행정구역(읍면동)별", [
        ("00", "전국"), ("11", "서울특별시"), ("21", "부산광역시"), ("22", "대구광역시"),
        ("11010", "종로구"), ("21020", "중구"), ("11020", "중구"), ("1101053", "사직동"),
    ])
    (record,) = build_records(data)
    assert record.classes[0].levels == {"전국": 1, "시도": 3, "시군구": 2, "읍면동": 1}


@offline
def test_class_without_sido_gets_no_levels(tmp_path):
    # 국적(지역)별 matches the region pattern but lists countries; 성씨·본관 never matches it
    data = _data_dir(tmp_path)
    _write_regions(data, "국적(지역)별", [("A", "계"), ("B", "중국"), ("C", "베트남")])
    (record,) = build_records(data)
    assert record.classes[0].levels == {}


@offline
def test_five_year_survey_keeps_its_cycle_and_flags_the_differences(tmp_path):
    (record,) = build_records(_data_dir(tmp_path))
    assert record.cycle == ["F"] and record.prd_se == ["Y"]  # both kept, neither overwrites the other
    assert record.issues == ["end_differs:2025!=2020", "cycle_code_differs:F!=Y"]


@offline
def test_one_period_table_is_not_flagged_for_its_cycle(tmp_path):
    (record,) = build_records(_data_dir(tmp_path, periods=("2015",)))
    assert record.cycle == ["F"] and record.prd_se == ["Y"]
    assert record.issues == ["end_differs:2025!=2015"]


@offline
def test_missing_sources_are_flagged_not_dropped(tmp_path):
    data = _data_dir(tmp_path, meta=False)
    (data / "values" / "101_DT_T.parquet").unlink()
    (record,) = build_records(data)
    assert record.issues == ["no_metadata", "no_values"] and record.periods == []


@offline
def test_cli_writes_one_valid_record_per_table(tmp_path):
    data = _data_dir(tmp_path)
    out = tmp_path / "tables.jsonl"
    assert main(["--data-dir", str(data), "--out", str(out)]) == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert [TableRecord.model_validate_json(line).info.key for line in lines] == ["101_DT_T"]
