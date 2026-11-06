"""T2-41 S1 (P2-D0): the Phase 1 copies match Phase 1 (vendor/phase1 @e248a45).

schemas/common.py (TableInfo, Source, MAX_DIMS) and shared/units.py (meta_unit) are copies, so tables.jsonl and
TableCandidate.to_source() JSON stay what Phase 1 would write. Skipped when vendor/phase1/src/kosis_ai is absent.
(S3 adds ⑤: llm.base.LLMResult fields and STATUS_* values.)
"""

import csv
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "vendor" / "phase1" / "src" / "kosis_ai").is_dir():
    pytest.skip("vendor/phase1/src/kosis_ai is not checked out", allow_module_level=True)

from kosis_ai import normalize as p1_normalize  # noqa: E402
from kosis_ai import schemas as p1  # noqa: E402

from kosis_rag.schemas import common  # noqa: E402
from kosis_rag.shared import units  # noqa: E402

SAMPLE = ROOT / "tests" / "fixtures" / "tables_sample.csv"
# unitNmKor values as the KOSIS table metadata writes them, plus what must not read as a unit
UNIT_NAMES = ["명", "건", "가구", "호", "%", " 천명 ", "명/㎢", "여자백명당남자수", "인구천명당", "천명당", "가구당",
              "15세이상인구천명당 이용자수당", "세대", "개사", "십만원", "월세없음", "15세 이상", "", "퍼센트", None]


def _fields(model) -> dict:
    """Each field's annotation, default, required flag and validation rules (pattern, min_length ...)."""
    return {name: (str(f.annotation), repr(f.default), f.is_required(), [repr(m) for m in f.metadata])
            for name, f in model.model_fields.items()}


@pytest.mark.parametrize("name", ["TableInfo", "Source"])
def test_1_models_have_the_same_fields_types_defaults_and_rules(name: str):
    ours, theirs = getattr(common, name), getattr(p1, name)
    assert list(ours.model_fields) == list(theirs.model_fields)  # same names in the same order
    assert _fields(ours) == _fields(theirs)
    assert ours.model_config == theirs.model_config  # extra="forbid"
    assert ours.model_json_schema() == theirs.model_json_schema()


def test_1_validation_refuses_the_same_inputs():
    good = {"org_id": "101", "tbl_id": "DT_A", "survey_nm": "조사", "tbl_nm": "표"}
    for bad in ({**good, "org_id": "1"}, {**good, "tbl_id": ""}, {**good, "extra": 1}, {k: good[k] for k in ("org_id",)}):
        with pytest.raises(ValueError):
            p1.Source(**bad)
        with pytest.raises(ValueError):
            common.Source(**bad)
    assert common.Source(**good).citation() == p1.Source(**good).citation()


def test_2_catalog_rows_dump_to_the_same_json():
    with SAMPLE.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows
    for row in rows:
        ours, theirs = common.TableInfo.model_validate(row), p1.TableInfo.model_validate(row)
        assert ours.model_dump_json() == theirs.model_dump_json()
        assert ours.to_source().model_dump_json() == theirs.to_source().model_dump_json()


def test_3_max_dims_is_the_same():
    assert common.MAX_DIMS == p1.MAX_DIMS


def test_4_meta_unit_reads_the_same_unit(tmp_path: Path):
    (tmp_path / "table_metadata").mkdir()
    ids = []
    for i, unit in enumerate(UNIT_NAMES):
        table_id = f"101_DT_U{i:02d}"
        meta = {"unitNmKor": unit} if unit is not None else {"tblNm": "단위 없음"}
        (tmp_path / "table_metadata" / f"{table_id}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        ids.append(table_id)
    ids.append("101_DT_MISSING")  # no metadata file
    got = [units.meta_unit(t, tmp_path) for t in ids]
    assert got == [p1_normalize._meta_unit(t, tmp_path) for t in ids]
    assert got[0] == "명" and got[5] == "천명" and got[-1] is None  # the sample reads real units, not only None
    assert units.KNOWN_UNITS == p1_normalize.KNOWN_UNITS
