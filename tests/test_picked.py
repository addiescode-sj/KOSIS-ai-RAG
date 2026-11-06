"""T2-39: the tables an answer shows and cites are the ones the guide reply recommends, not the top 3 by rank.

Candidates are built in code; the guide LLM is Phase 1 MockClient; no model, no server.
"""

import pytest

from fastapi.testclient import TestClient
from kosis_chat.llm.model import MockClient
from langchain_core.documents import Document

from kosis_rag import chain
from kosis_rag.schemas.io import TableCandidate
from kosis_rag.shared.catalog import to_candidate

pytestmark = pytest.mark.offline


def _cand(i):
    return TableCandidate(key=f"101_DT_{i}", org_id="101", tbl_id=f"DT_{i}", table_name=f"표{i}", survey_name="조사",
                          organization="국가데이터처", period="년 (2015 ~ 2023)", url=f"http://kosis.kr/{i}")


CANDS = [_cand(c) for c in "ABCDE"]
DOCS = [Document(page_content=c.key, metadata={**c.model_dump(), "score": 1.0}, id=c.key) for c in CANDS]


def _ids(tables):
    return [c.tbl_id for c in tables]


# --- 1. picked_from_reply ---

@pytest.mark.parametrize("reply, expected", [
    ("[2] DT_B 표가 맞습니다.", ["DT_B"]),
    ("[3] 표와 [1] 표를 보세요.", ["DT_C", "DT_A"]),  # mention order, not rank order
    ("[1, 3] 표가 관련 있습니다.", ["DT_A", "DT_C"]),
    ("[1~3] 표가 관련 있습니다.", ["DT_A", "DT_B", "DT_C"]),
    ("[1]~[3] 표가 관련 있습니다.", ["DT_A", "DT_B", "DT_C"]),
    ("［2］ 표가 맞습니다.", ["DT_B"]),  # full-width brackets
    ("표DT_D를 보세요.", ["DT_D"]),  # id right after a Korean letter
    ("[2] DT_A 표가 맞습니다.", ["DT_A"]),  # number and id disagree: the id is the data
    ("[1] DT_A 표, 다시 말해 DT_A 표입니다.", ["DT_A"]),  # no repeats
    ("DT_ZZZ와 [2]", ["DT_B"]),  # an id outside the candidates is not picked (the check rejects the reply anyway)
    ("2023년 자료가 있는 표를 보세요.", []),  # names no table
])
def test_picked_from_reply(reply, expected):
    assert _ids(chain.picked_from_reply(reply, CANDS)) == expected


def test_a_number_beyond_the_candidates_fails_the_check_and_never_streams():
    assert chain.unsupported_claims("[7] 표가 맞습니다.", "q", CANDS) == ["[7]"]
    assert chain.unsupported_claims("[2023년] 기준", "q", CANDS) == []  # a bracketed year is not a table number
    assert chain.streamable_prefix("[7] 표가 맞습니다. ", "q", CANDS) is None


# --- 2. the answer cites exactly what it recommends ---

def _answer(reply):
    return chain.guide_answer("출생아 수", DOCS, MockClient([reply]))


def _cited(result):
    return result.answer.split("출처:\n", 1)[1].splitlines() if "출처:" in result.answer else []


def test_llm_answer_cites_only_the_tables_it_names():
    result = _answer("[2] DT_B 표가 출생아 수를 담고 있습니다.")
    assert result.answer_origin == "llm" and _ids(result.picked) == ["DT_B"]
    assert _cited(result) == [f"- {CANDS[1].to_source().citation()}"]  # before T2-39: the top 3 (A, B, C)
    assert len(result.candidates) == 5


def test_reply_naming_no_table_falls_back_to_the_template_and_its_top_3():
    result = _answer("관련 표가 있습니다.")
    assert result.answer_origin == "template" and _ids(result.picked) == ["DT_A", "DT_B", "DT_C"]
    assert len(_cited(result)) == 3 and all(c.table_name in result.answer for c in result.picked)


def test_refusals_pick_nothing():
    assert _answer(chain.NO_MATCH).picked == []
    assert chain.refusal_answer("q", DOCS, None).picked == []
    assert chain.refusal_answer("q", [], None).picked == []


def test_picked_keeps_the_candidate_fields():
    assert chain.guide_answer("q", DOCS, MockClient(["[1] DT_A"])).picked[0] == to_candidate(DOCS[0])


# --- 3. k is the server's budget unless a request gives it ---

def test_answer_api_uses_answer_k_unless_the_request_sets_k(tmp_path, monkeypatch):
    from backend.app import main
    from test_api_nodes import _resources

    monkeypatch.setattr(main, "LOG_DIR", tmp_path)
    resources = _resources()
    resources.settings = resources.settings.model_copy(update={"answer_k": 2})
    main.app.dependency_overrides[main.get_runtime] = lambda: main.Runtime(resources, MockClient("[1] DT_A"))
    try:
        client = TestClient(main.app)
        assert len(client.post("/search", json={"question": "출생아 수"}).json()["candidates"]) == 2
        assert len(client.post("/search", json={"question": "출생아 수", "k": 4}).json()["candidates"]) == 4
    finally:
        main.app.dependency_overrides.clear()
