from pathlib import Path

import pytest

from kosis_chat.llm.model import STATUS_API_ERROR, LLMResult, MockClient
from kosis_rag import chain
from kosis_rag.loader import load_catalog, to_document
from kosis_rag.retriever import BM25TableRetriever
from kosis_rag.tools import build_tools

FIXTURE = Path(__file__).parent / "fixtures" / "tables_sample.csv"

pytestmark = pytest.mark.offline


@pytest.fixture()
def retriever():
    return BM25TableRetriever.from_documents([to_document(t) for t in load_catalog(FIXTURE)])


def test_search_keeps_table_id_and_ranks_by_name(retriever):
    (search, _) = build_tools(retriever, MockClient())
    hits = search.invoke({"question": "시군구별 주민등록세대수", "k": 2})
    assert hits[0]["tbl_id"] == "DT_1B040B3"
    assert hits[0]["score"] > 0 and len(hits) <= 2


def test_no_token_overlap_refuses_without_calling_model(retriever):
    client = MockClient()
    result = chain.answer(retriever, client, "bitcoin price")
    assert result.no_answer and result.answer_origin == "refusal" and client.calls == []


def test_model_no_match_is_a_refusal(retriever):
    result = chain.answer(retriever, MockClient(chain.NO_MATCH), "주민등록세대수 알려줘")
    assert result.no_answer and result.candidates


@pytest.mark.parametrize("reply", [
    "DT_9999999 표를 보세요.",  # table id not in candidates
    "[1] DT_1B040B3 는 1980년 자료가 있습니다.",  # year outside every candidate period
    LLMResult(text="", status=STATUS_API_ERROR, error="boom"),  # failed call
])
def test_unsupported_or_failed_reply_falls_back_to_template(retriever, reply):
    result = chain.answer(retriever, MockClient([reply]), "주민등록세대수")
    assert result.answer_origin == "template"
    assert "KOSIS / 주민등록인구현황 / 행정구역(시군구)별 주민등록세대수" in result.answer


def test_supported_reply_is_kept_and_cited_by_code(retriever):
    reply = "[1] DT_1B040B3 표가 2020년 시군구별 세대수를 담고 있습니다."
    result = chain.answer(retriever, MockClient([reply]), "2020년 주민등록세대수")
    assert result.answer_origin == "llm" and result.answer.startswith(reply)
    assert "출처:" in result.answer
