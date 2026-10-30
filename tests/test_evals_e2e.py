"""e2e (first part of L4): the retriever's top 1 table goes to summarize_stat, graded like L3; offline with stubs."""

import sys
import types

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from kosis_ai.schemas import Refusal, StatAnswer, StatResult

from evals.runners import e2e
from evals.schemas import CalcItem


class Fixed(BaseRetriever):
    keys: list[str]

    def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        return [Document(page_content="", metadata={"key": k}) for k in self.keys]


class Client:
    model = "stub"

    def reset(self) -> None:
        self.calls, self.input_tokens, self.output_tokens, self.failures = 1, 10, 2, 0


def refused(reason: str) -> StatAnswer:
    return StatAnswer.model_construct(outcome="refused", result=StatResult(), parts=[], clarify_options=[],
                                      default_applied={}, refusal=Refusal(reason=reason, message="x"))


def stub_pipeline(monkeypatch, used: list[str]) -> None:
    def summarize_stat(question, table_key, client=None, retry_interval=0.0):
        used.append(table_key)
        return refused("no_period")

    monkeypatch.setitem(sys.modules, "kosis_chat.api", types.SimpleNamespace(summarize_stat=summarize_stat))
    monkeypatch.setattr("evals.ops.usage_client", lambda: Client())


def item() -> CalcItem:
    return CalcItem(qid="c1", kind="not_answerable", question="q", table_key="101_DT_GOLD", table_sha256="0",
                    allowed_reasons=["no_period"])


def test_top1_table_is_used_and_located(monkeypatch):
    used: list[str] = []
    stub_pipeline(monkeypatch, used)
    [row] = e2e.run(Fixed(keys=["101_DT_OTHER", "101_DT_GOLD"]), [item()], k=10)
    assert used == ["101_DT_OTHER"] and row["used_table"] == "101_DT_OTHER" and not row["failed"]
    assert row["input_tokens"] == 10 and row["model"] == "stub"
    loc = e2e.locate(item(), row, "PASS")
    assert loc["gold_rank"] == 2 and not loc["right_table"] and loc["cause"] == "pass"
    assert e2e.locate(item(), row, "FAIL")["cause"] == "retrieval"
    right = e2e.run(Fixed(keys=["101_DT_GOLD"]), [item()], k=10)[0]
    assert e2e.locate(item(), right, "FAIL") | {"top_keys": None} == {
        "used_table": "101_DT_GOLD", "right_table": True, "gold_rank": 1, "top_keys": None, "cause": "answer"}


def test_empty_retrieval_is_a_run_failure(monkeypatch):
    used: list[str] = []
    stub_pipeline(monkeypatch, used)
    [row] = e2e.run(Fixed(keys=[]), [item()], k=10)
    assert used == [] and row["failed"] and row["error"] == "no table retrieved" and row["used_table"] is None
    assert e2e.locate(item(), row, "FAIL")["gold_rank"] is None


def test_run_eval_e2e_writes_report(monkeypatch, tmp_path):
    from evals import ops, report_html, run_eval
    from evals.evaluators import calculation
    from evals.runners import retrieval

    stub_pipeline(monkeypatch, [])
    golden = tmp_path / "calculation_golden.jsonl"
    golden.write_text(item().model_dump_json() + "\n", encoding="utf-8")
    monkeypatch.setattr(run_eval, "CALC_GOLDEN", golden)
    monkeypatch.setattr(run_eval, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(run_eval, "ROOT", tmp_path)  # golden path in the manifest is relative to ROOT
    monkeypatch.setattr(ops, "append_history", lambda *a, **kw: None)
    monkeypatch.setattr(report_html, "read_history", lambda: [])
    monkeypatch.setattr(calculation, "read_cells", lambda path: {})
    monkeypatch.setattr(retrieval, "build", lambda name, items, k: Fixed(keys=["101_DT_OTHER", "101_DT_GOLD"]))
    assert run_eval.main(["--layer", "e2e", "--retriever", "bm25-record", "--stage", "t"]) == 0
    [out] = (tmp_path / "reports").iterdir()
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "검색 1위가 정답 표: 0/1" in report and "retrieval 0" in report  # refusal with an allowed reason passes
    assert "L4 종단 간" in (out / "report.html").read_text(encoding="utf-8")
