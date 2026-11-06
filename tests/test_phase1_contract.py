"""T2-41 S1 (P2-D0): the Phase 1 copies match Phase 1 (vendor/phase1 @e248a45).

schemas/common.py (TableInfo, Source, MAX_DIMS) and shared/units.py (meta_unit) are copies, so tables.jsonl and
TableCandidate.to_source() JSON stay what Phase 1 would write. Skipped when vendor/phase1/src/kosis_ai is absent.
S3 adds ⑤: llm/base.py and llm/hcx.py (copies of kosis_chat/llm/model.py) keep LLMResult's fields, the STATUS_*
values, MockClient's replies and HCXClient's request; the guide model defaults are Phase 1's configs/model.yaml.
S5 adds ⑥: the handover key — SearchModule.search()'s picked[0].key is the tables.csv key Phase 1 looks up
(kosis_chat.api.table_path), not tbl_id. No LLM.
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


# --- ⑤ (S3) llm/base.py and llm/hcx.py: copies of Phase 1 kosis_chat/llm/model.py @e248a45 ---

def test_5_llm_result_fields_and_statuses_are_the_same():
    import dataclasses

    from kosis_chat.llm import model as p1_model

    from kosis_rag.llm import base

    ours = [(f.name, str(f.type), repr(f.default)) for f in dataclasses.fields(base.LLMResult)]
    assert ours == [(f.name, str(f.type), repr(f.default)) for f in dataclasses.fields(p1_model.LLMResult)]
    names = sorted(n for n in dir(p1_model) if n.startswith("STATUS_"))
    assert names == sorted(n for n in dir(base) if n.startswith("STATUS_"))
    assert {n: getattr(base, n) for n in names} == {n: getattr(p1_model, n) for n in names}
    assert (base.LLMClient.name, base.MockClient.name) == (p1_model.LLMClient.name, p1_model.MockClient.name)


@pytest.mark.parametrize("response", ["[mock] answer", ["a", "b"], ("x",)])
def test_5_mock_client_replies_the_same(response):
    from kosis_chat.llm import model as p1_model

    from kosis_rag.llm import base

    ours, theirs = base.MockClient(response), p1_model.MockClient(response)
    for _ in range(3):  # the queue runs out on the third call
        a, b = ours.generate("s", "u"), theirs.generate("s", "u")
        assert dataclasses_dict(a) == dataclasses_dict(b)
    assert ours.calls == theirs.calls


def dataclasses_dict(result) -> dict:
    import dataclasses

    return dataclasses.asdict(result)


@pytest.mark.parametrize("model, thinking, max_tokens", [("HCX-007", "none", 512), ("HCX-007", "low", 5120),
                                                         ("HCX-DASH-002", None, 1024)])
def test_5_hcx_client_sends_the_same_request(monkeypatch, model, thinking, max_tokens):
    from kosis_chat.config import load_settings
    from kosis_chat.llm import model as p1_model

    from kosis_rag.llm.config import LLMConfig
    from kosis_rag.llm.hcx import HCXClient

    monkeypatch.setenv("HCX_API_KEY", "nv-dummy")
    theirs = p1_model.HCXClient(load_settings(backend="hcx", model_name=model, thinking_level=thinking,
                                              max_tokens=max_tokens, temperature=0.0))
    ours = HCXClient(LLMConfig(provider="hcx", api_key="nv-dummy", model=model, thinking=thinking or "none",
                               max_tokens=max_tokens, temperature=0.0))
    assert ours.url == theirs.url and ours.is_reasoning == theirs.is_reasoning
    assert json.dumps(ours._body("sys", "질문")) == json.dumps(theirs._body("sys", "질문"))
    assert ours.timeout == theirs.settings.timeout_seconds
    assert (ours.RETRY_DELAYS, ours.MAX_RETRY_WAIT, ours.RETRY_HTTP_CODES, ours.CONTEXT_EXCEEDED_CODE) == (
        theirs.RETRY_DELAYS, theirs.MAX_RETRY_WAIT, theirs.RETRY_HTTP_CODES, theirs.CONTEXT_EXCEEDED_CODE)


def test_5_guide_model_defaults_are_phase1_model_yaml():
    """chain.clova_chat read Phase 1 load_settings() (configs/model.yaml) until S3; the guide defaults (S4:
    backend/app/config.py, the first key variable) must match."""
    from kosis_chat.config import DEFAULT_BASE_URL, load_settings

    from backend.app import config
    from kosis_rag.llm import hcx

    s = load_settings()
    assert (config.GUIDE_MODEL, config.GUIDE_TEMPERATURE, config.GUIDE_MAX_TOKENS, config.GUIDE_TIMEOUT,
            config.GUIDE_KEY_ENV[0]) == (s.model_name, s.temperature, s.max_tokens, s.timeout_seconds, s.api_key_env)
    assert hcx.DEFAULT_BASE_URL == DEFAULT_BASE_URL == s.base_url


# --- ⑥ (S5) the handover key: what Phase 2 hands Phase 1 is picked[0].key (external review 11/05 P1-1) ---

def test_6_picked_key_is_the_tables_csv_key_phase1_looks_up(tmp_path: Path, monkeypatch):
    pytest.importorskip("langgraph")
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever

    from kosis_chat.api import table_path
    from kosis_rag.adapters.loader import load_resources
    from kosis_rag.api import SearchModule
    from kosis_rag.schemas.catalog import TableRecord
    from kosis_rag.schemas.io import SearchInput
    from kosis_rag.schemas.resources import IndexSpec, SearchOptions
    from kosis_rag.shared.catalog import load_catalog, to_document

    infos = load_catalog(SAMPLE)
    docs = [to_document(i) for i in infos]

    class Listed(BaseRetriever):  # the dense list: the sample tables in file order, catalog metadata
        def _get_relevant_documents(self, query, *, run_manager):
            return [Document(page_content=d.page_content, id=d.id, metadata={**d.metadata, "score": 0.9}) for d in docs]

    records = [TableRecord(info=i, cycle=["Y"], prd_se=["Y"], periods=["2020"], start=2020, end=2020,
                           menu_paths=[i.menu_path]) for i in infos]
    resources = load_resources(spec=IndexSpec(), options=SearchOptions(cond_filter=False, stage2="none"),
                               records=records, dense=Listed())
    out = SearchModule.from_resources(resources).search(SearchInput(question="시군구별 주민등록세대수"))
    assert out.route == "found" and out.picked
    top = out.picked[0]

    with SAMPLE.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    keys = {r["key"] for r in rows}
    assert top.key in keys and top.key == f"{top.org_id}_{top.tbl_id}"  # "<org>_<tbl_id>", the key column's form

    # a Phase 1 work root holding the sample tables.csv (its parquet paths) and the picked table's file
    data = tmp_path / "data"
    (data / "values").mkdir(parents=True)
    with (data / "tables.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    parquet = next(r["parquet"] for r in rows if r["key"] == top.key)
    (data / parquet).write_bytes(b"")
    monkeypatch.setenv("KOSIS_WORKSPACE", str(tmp_path))
    assert table_path(top.key) == data / parquet
    with pytest.raises(LookupError):
        table_path(top.tbl_id)  # tbl_id alone ("DT_...") is not a tables.csv key
