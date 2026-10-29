"""scripts/stat_baseline.py: golden questions through summarize_stat give a run folder eval.grade_run grades.

Made-up table (vendor/phase1/tests/conftest.py, 가상 데이터) and a mock model; no provided data or API key needed.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from eval import grade_run
from eval.scorer import read_jsonl
from kosis_chat import api
from kosis_chat.config import load_settings
from kosis_chat.data.loader import sha256_of
from kosis_chat.llm.model import MockClient

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


phase1_tables = _load("phase1_tables", ROOT / "vendor" / "phase1" / "tests" / "conftest.py")
stat_baseline = _load("stat_baseline", ROOT / "scripts" / "stat_baseline.py")
TABLE = "data/values/101_FAKE_TBL_002.parquet"


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    phase1_tables.write_table(tmp_path / "data", "FAKE_TBL_002", phase1_tables.household_rows(), {
        "survey_name": "가상 총조사", "organization": "가상 기관", "table_name": "가상 가구원수별 가구",
        "period": "년 (2015 ~ 2025)", "url": "https://example.invalid/fake-table-002"})
    monkeypatch.setenv("KOSIS_WORKSPACE", str(tmp_path))
    sha = sha256_of(tmp_path / TABLE)
    fact = {"id": "f1", "value": 3102, "unit": "가구", "period": "2023", "source_ref": "FAKE_TBL_002|T1|11|2023"}
    items = [
        {"id": "Q-1", "kind": "answerable", "question": "2023년 서울특별시 1인 가구 수는?", "table": TABLE,
         "table_sha256": sha, "required_facts": [fact]},
        {"id": "Q-2", "kind": "not_answerable", "question": "2030년 서울 1인 가구 수는?", "table": TABLE,
         "table_sha256": sha, "allowed_reasons": ["no_period"]},
        {"id": "Q-3", "kind": "answerable", "question": "2023년 부산광역시 1인 가구 수는?", "table": TABLE,
         "table_sha256": sha, "required_facts": [fact | {"value": 4102, "source_ref": "FAKE_TBL_002|T1|21|2023"}]},
    ]
    golden = tmp_path / "eval" / "golden" / "golden.jsonl"
    golden.parent.mkdir(parents=True)
    golden.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
    api._cached.cache_clear()
    yield tmp_path
    api._cached.cache_clear()


def test_run_folder_is_graded_by_phase1_grade_run(workspace, monkeypatch):
    real = api.summarize_stat

    def summarize(question, table_id, **kw):
        if "부산" in question:
            raise RuntimeError(f"{table_id}: the model call failed (api_error: down)")
        return real(question, table_id, **kw)

    monkeypatch.setattr(api, "summarize_stat", summarize)
    client = MockClient([json.dumps({"answer": "서울은 {f1}입니다."}, ensure_ascii=False)])
    run = stat_baseline.run(workspace / "eval" / "golden" / "golden.jsonl", "t", load_settings(backend="mock"),
                            client=client)
    assert run.parent == workspace / "eval" / "results"
    assert (run / "golden.jsonl").read_bytes() == (workspace / "eval" / "golden" / "golden.jsonl").read_bytes()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert (manifest["status"], manifest["pipeline"], manifest["failed_calls"]) == ("finished", "stat", 1)
    rows = {r["question_id"]: r for r in read_jsonl(run / "results.jsonl")}
    assert rows["Q-1"]["table_key"] == "101_FAKE_TBL_002" and rows["Q-1"]["stat_answer"]["outcome"] == "answered"
    assert rows["Q-3"]["stat_answer"] is None and "model call failed" in rows["Q-3"]["call_error"]

    assert grade_run.main(["--run", str(run)]) == 0
    graded = next((run / "grades").iterdir())
    verdicts = {g["question_id"]: g for g in read_jsonl(graded / "grades.jsonl")}
    assert verdicts["Q-1"]["verdict"] == verdicts["Q-2"]["verdict"] == "PASS"
    assert verdicts["Q-3"]["verdict"] != "PASS" and "run_failure" in verdicts["Q-3"]["error_types"]
    assert (graded / "report.md").exists()


def test_limit_keeps_only_the_asked_questions_in_the_run_golden(workspace):
    run = stat_baseline.run(workspace / "eval" / "golden" / "golden.jsonl", "t", load_settings(backend="mock"),
                            limit=1, client=MockClient([json.dumps({"answer": "서울은 {f1}입니다."}, ensure_ascii=False)]))
    assert [json.loads(line)["id"] for line in (run / "golden.jsonl").read_text(encoding="utf-8").splitlines()] == ["Q-1"]
    assert [r["question_id"] for r in read_jsonl(run / "results.jsonl")] == ["Q-1"]


def test_a_key_that_reaches_another_file_stops_before_any_call(workspace):
    (workspace / "data" / "tables.csv").write_text(
        (workspace / "data" / "tables.csv").read_text(encoding="utf-8-sig").replace(
            "values/101_FAKE_TBL_002.parquet", "values/other.parquet"), encoding="utf-8-sig")
    (workspace / "data" / "values" / "other.parquet").write_bytes((workspace / TABLE).read_bytes())
    with pytest.raises(LookupError, match="is not"):
        stat_baseline.run(workspace / "eval" / "golden" / "golden.jsonl", "t", load_settings(backend="mock"))
    assert not (workspace / "eval" / "results").exists()
