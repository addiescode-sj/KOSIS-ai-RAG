"""T2-38: the guide reply streams only what the final check would accept, and the final result does not change.

A streaming fake chat model (GenericFakeChatModel splits the reply at whitespace) stands in for ChatClovaX; the search
fakes come from test_graph. No model, no server.
"""

import pytest

pytest.importorskip("langgraph")

from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage, AIMessageChunk  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult  # noqa: E402

from backend.app import main  # noqa: E402
from kosis_rag import chain  # noqa: E402
from kosis_rag.gate import texts  # noqa: E402
from kosis_rag.graph import build_graph, run_search, stream_search  # noqa: E402
from kosis_rag.schemas.io import TableCandidate  # noqa: E402
from test_api_nodes import _resources  # noqa: E402

pytestmark = pytest.mark.offline

CANDS = [TableCandidate(key="101_DT_A", org_id="101", tbl_id="DT_A", table_name="출생아수", survey_name="인구동향조사",
                        organization="국가데이터처", period="년 (2020 ~ 2023)", url="u")]


def _fake(reply):
    return GenericFakeChatModel(messages=iter([AIMessage(content=reply)]))


def _streamed(reply, question="출생아 수"):
    """(token pieces, final state) of the streaming search, and the run_search state for the same reply."""
    resources = _resources()
    pieces, final = [], None
    for name, payload in stream_search(build_graph(resources, _fake(reply)), question):
        if name == "token":
            pieces.append(payload["delta"])
        elif name == "result":
            final = payload
    return pieces, final, run_search(build_graph(resources, _fake(reply)), question)


# --- 1. streamable_prefix ---

@pytest.mark.parametrize("text, expected", [
    ("[1] DT_A 표가 맞", "[1] DT_A 표가 "),  # the cut word waits
    ("[1] DT_", "[1] "),  # a cut id waits too ("DT_" alone would look invented)
    ("2020년부터 2023", "2020년부터 "),
    ("NO_MA", ""), ("NO_MATCH", ""), ("NO_MATCH\n", ""), ("  NO_MATCH \n", ""),  # a refusal never shows
    ("[1] DT_ZZZ 표 ", None), ("1999년 자료 ", None),  # invented id / year: stop
    ("표DT_ZZZ를 보세요 ", None),  # an id right after a Korean letter is caught (\b missed it)
])
def test_streamable_prefix(text, expected):
    assert chain.streamable_prefix(text, "출생아 수", CANDS) == expected


def test_unsupported_claims_finds_an_id_after_a_korean_letter():
    assert chain.unsupported_claims("표DT_ZZZ를 보세요", "q", CANDS) == ["DT_ZZZ"]
    assert chain.unsupported_claims("표DT_A를 보세요", "q", CANDS) == []


# --- 2. the streaming graph ---

def test_stream_sends_the_checked_reply_and_ends_with_the_run_search_result():
    reply = "[1] DT_A 표가 출생아 수를 담고 있습니다."
    pieces, final, expected = _streamed(reply)
    assert len(pieces) > 1 and "".join(pieces) == "[1] DT_A 표가 출생아 수를 담고 "  # the last word comes with the result
    assert final.result == expected.result and final.result.answer_origin == "llm"


def test_no_match_streams_nothing_and_refuses():
    pieces, final, expected = _streamed(texts.NO_MATCH + "\n")
    assert pieces == [] and final.result == expected.result and final.result.no_answer


def test_an_invented_id_stops_the_stream_before_it_and_falls_back_to_the_template():
    pieces, final, expected = _streamed("[1] DT_A 표와 DT_ZZZ 표가 맞습니다.")
    assert "".join(pieces) == "[1] DT_A 표와 "
    assert not chain.unsupported_claims("".join(pieces), "출생아 수", CANDS)
    assert final.result == expected.result and final.result.answer_origin == "template"


def test_a_stream_that_fails_midway_is_a_status_and_the_template():
    class Broken(BaseChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kw):
            raise AssertionError("not used")

        def _stream(self, messages, stop=None, run_manager=None, **kw):
            yield ChatGenerationChunk(message=AIMessageChunk(content="[1] DT_A "))
            raise TimeoutError("cut")

        @property
        def _llm_type(self):
            return "broken"

    result = chain.guide_answer("출생아 수", _resources().dense.invoke("q")[:1], Broken(), on_text=lambda d: None)
    assert result.answer_origin == "template" and result.llm_status == "api_error"


def test_streamed_chunks_sum_their_usage():
    class Counting(BaseChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kw):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="x"))])

        def _stream(self, messages, stop=None, run_manager=None, **kw):
            for text, n in (("a ", 2), ("b", 3)):
                usage = {"input_tokens": 0, "output_tokens": n, "total_tokens": n}
                yield ChatGenerationChunk(message=AIMessageChunk(content=text, usage_metadata=usage))

        @property
        def _llm_type(self):
            return "counting"

    seen = []
    result = chain.llm_runnable(Counting(), seen.append).invoke(chain.PROMPT.invoke({"question": "q", "context": "c"}))
    assert result.text == "a b" and result.usage["output_tokens"] == 5
    assert seen[0] == "a " and set(seen[1:]) == {"a b"}  # LangChain may end with an empty chunk: same text again


# --- 3. the API ---

def test_sse_sends_token_events_between_gate_and_result_and_the_node_api_still_answers(tmp_path, monkeypatch):
    reply = "[1] DT_A 표가 출생아 수를 담고 있습니다."
    monkeypatch.setattr(main, "LOG_DIR", tmp_path)
    main.app.dependency_overrides[main.get_runtime] = lambda: main.Runtime(_resources(), _fake(reply))
    try:
        client = TestClient(main.app)
        with client.stream("POST", "/search/stream", json={"question": "출생아 수"}) as r:
            events = [line.split(": ", 1)[1] for line in r.iter_lines() if line.startswith("event: ")]
        assert events.index("gate") < events.index("token") and events[-2:] == ["guide", "result"]
        # outside a graph there is no stream writer: /nodes/answer must still answer
        main.app.dependency_overrides[main.get_runtime] = lambda: main.Runtime(_resources(), _fake(reply))
        state = client.post("/nodes/parse", json={"question": "출생아 수"}).json()
        for name in ["retrieve", "rerank1", "rerank2", "gate", "answer"]:
            state = client.post(f"/nodes/{name}", json=state).json()
        assert state["result"]["answer_origin"] == "llm"
    finally:
        main.app.dependency_overrides.clear()
