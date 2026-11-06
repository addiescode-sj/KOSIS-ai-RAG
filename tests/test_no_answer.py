"""T2-09: HCX-007 stage 2 picks (fake client), the no-answer rules in evals, and the service refusal.

No API: the HCX reply comes from Phase 1 MockClient.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from kosis_chat.llm.model import STATUS_API_ERROR, LLMResult, MockClient
from backend.app.respond.legacy import answer
from kosis_rag.gate import texts
from kosis_rag.legacy.rerankers import LLMReranker
from kosis_rag.rerank.hcx import HcxSelector, parse

from evals.evaluators.retrieval import no_answer_rules

pytestmark = pytest.mark.offline


class ListRetriever(BaseRetriever):
    keys: list[str]

    def _get_relevant_documents(self, query, *, run_manager):
        return [Document(page_content=k, metadata={"key": k, "score": 0.9 - i / 10, "tbl_id": k, "table_name": k,
                                                   "organization": "국가데이터처", "survey_name": "s", "period": "년",
                                                   "org_id": "101", "url": ""}, id=k)
                for i, k in enumerate(self.keys)]


def _rerank(reply, keys=("a", "b", "c")):
    client = MockClient(reply)
    select = HcxSelector(text_of={k: f"표 {k}" for k in keys}, thinking="none", client=client)
    return LLMReranker(base=ListRetriever(keys=list(keys)), select=select, name="hcx").invoke("질문"), client


def test_parse_keeps_valid_numbers_and_reason_only_when_empty():
    assert parse('```json\n{"picked": [2, 9, 2, "x", 1], "reason": "no_item"}\n```', 3) == ([2, 1], None)
    assert parse('{"picked": [], "reason": "out_of_scope"}', 3) == ([], "out_of_scope")
    assert parse('{"picked": [], "reason": "made_up"}', 3) == ([], None)
    assert parse('{"picked": [1], "reason": null}\n{"picked": [1], "reason": null}', 3) == ([1], None)
    with pytest.raises(ValueError):
        parse("모르겠습니다", 3)


def test_hcx_pick_maps_numbers_to_input_keys_and_keeps_stage1_top_score():
    docs, client = _rerank('{"picked": [3], "reason": null}')
    assert [d.metadata["key"] for d in docs] == ["c", "a", "b"]
    meta = docs[0].metadata
    assert meta["stage2_picked_n"] == 1 and meta["stage2_input_top_score"] == pytest.approx(0.9)  # a's score, not c's
    assert client.calls[0][1] == "질문: 질문\n\n후보 통계표:\n[1] 표 a\n\n[2] 표 b\n\n[3] 표 c"


def test_hcx_empty_pick_is_no_answer_with_reason_and_failure_is_not():
    docs, _ = _rerank('{"picked": [], "reason": "out_of_scope"}')
    assert docs[0].metadata["stage2_picked_n"] == 0 and docs[0].metadata["stage2_reason"] == "out_of_scope"
    docs, _ = _rerank(LLMResult(text="", status=STATUS_API_ERROR, error="HTTP 500"))
    assert docs[0].metadata["stage2_picked_n"] == 0 and "HTTP 500" in docs[0].metadata["stage2_error"]


def test_unknown_thinking_level_is_refused():
    with pytest.raises(ValueError):
        HcxSelector(text_of={}, thinking="max", client=MockClient())


def test_no_answer_rules_tau_llm_either():
    rows = [  # (answerable, gold reason, tau says no, picked_n, error, llm reason)
        dict(answerable=False, no_answer_reason="out_of_scope", no_answer_pred=False, stage2_picked_n=0,
             stage2_error=None, stage2_reason="out_of_scope"),
        dict(answerable=False, no_answer_reason="no_period", no_answer_pred=True, stage2_picked_n=2,
             stage2_error=None, stage2_reason=None),
        dict(answerable=True, no_answer_reason=None, no_answer_pred=False, stage2_picked_n=0,
             stage2_error="RuntimeError: x", stage2_reason=None),  # failed call: not a refusal
        dict(answerable=True, no_answer_reason=None, no_answer_pred=False, stage2_picked_n=1,
             stage2_error=None, stage2_reason=None),
    ]
    r = no_answer_rules(rows)
    assert (r["tau"]["tp"], r["tau"]["fp"]) == (1, 0) and r["llm"]["tp"] == 1 and r["llm"]["fp"] == 0
    assert r["either"]["recall"] == 1.0 and r["either"]["recall_by_reason"] == {"no_period": [1, 1], "out_of_scope": [1, 1]}
    assert r["llm_reason_match"] == [1, 1] and r["llm_errors"] == 1


@pytest.mark.parametrize("reason, text", [("out_of_scope", texts.REFUSALS["out_of_scope"]),
                                          ("no_item", texts.REFUSAL_TEXT)])  # no_item is unreliable: generic text
def test_service_refuses_on_empty_pick_and_skips_answer_model(reason, text):
    reply = f'{{"picked": [], "reason": "{reason}"}}'
    select = HcxSelector(text_of={"a": "표 a"}, thinking="none", client=MockClient(reply))
    answer_model = MockClient("안내 문장")
    result = answer(LLMReranker(base=ListRetriever(keys=["a"]), select=select), answer_model, "자동차 등록 대수")
    assert result.no_answer and result.no_answer_reason == reason and result.answer == text
    assert answer_model.calls == []
