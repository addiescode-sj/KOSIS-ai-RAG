"""T2-37: the node API gives what the graph gives, and a client cannot invent a table.

Same fakes as test_graph (list retriever, fake stage 1 / stage 2, Phase 1 MockClient); no Chroma, no LLM.
"""

import json

import pytest

pytest.importorskip("langgraph")

from fastapi.testclient import TestClient  # noqa: E402

from backend.app import main  # noqa: E402
from kosis_chat.llm.model import MockClient  # noqa: E402
from kosis_rag.graph import build_graph, run_search  # noqa: E402
from kosis_rag.pipeline import load_resources  # noqa: E402
from test_graph import KEYS, RECORDS, CatalogRetriever, _pick, _settings, _stage1  # noqa: E402

pytestmark = pytest.mark.offline

REPLY = "[1] DT_A 표가 출생아 수를 담고 있습니다."


def _resources(**kw):
    parts = {"stage1_score": _stage1, "select": _pick("101_DT_A"), **kw}
    return load_resources(_settings(), records=RECORDS, dense=CatalogRetriever(keys=KEYS), **parts)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "LOG_DIR", tmp_path)
    main.app.dependency_overrides[main.get_runtime] = lambda: main.Runtime(_resources(), MockClient(REPLY))
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def _expected(question, **kw):
    return run_search(build_graph(_resources(**kw), MockClient(REPLY)), question)


def _chain(client, question):
    state = client.post("/nodes/parse", json={"question": question}).json()
    for name in client.get("/nodes").json()["nodes"][1:]:
        r = client.post(f"/nodes/{name}", json=state)
        assert r.status_code == 200, r.text
        state = r.json()
    return state


@pytest.mark.parametrize("question, select, answer_node", [("출생아 수", _pick("101_DT_A"), "guide"),
                                                           ("자동차 등록 대수", _pick(reason="out_of_scope"), "refuse")])
def test_node_chain_equals_search_equals_stream_equals_graph(client, question, select, answer_node):
    main.app.dependency_overrides[main.get_runtime] = lambda: main.Runtime(_resources(select=select), MockClient(REPLY))
    expected = _expected(question, select=select).result.model_dump(mode="json")

    assert client.get("/nodes").json()["nodes"] == ["parse", "retrieve", "rerank1", "rerank2", "gate", "answer"]
    assert _chain(client, question)["result"] == expected
    assert client.post("/search", json={"question": question}).json() == expected

    with client.stream("POST", "/search/stream", json={"question": question}) as r:
        events = [line.split(": ", 1)[1] for line in r.iter_lines() if line.startswith("event: ")]
    events[2:4] = sorted(events[2:4])  # T2-39: the graph's parallel list nodes finish in any order
    assert events == ["parse", "condition", "retrieve_bm25", "retrieve_dense", "fuse", "rerank1", "rerank2", "gate",
                      answer_node, "result"]


def test_stream_result_event_holds_the_search_result_and_each_node_its_own_trace(client):
    r = client.post("/search/stream", json={"question": "출생아 수"})
    blocks = [b for b in r.text.split("\n\n") if b.strip()]
    parsed = [(b.split("\n")[0].removeprefix("event: "), json.loads(b.split("\n")[1].removeprefix("data: "))) for b in blocks]
    assert parsed[-1] == ("result", _expected("출생아 수").result.model_dump(mode="json"))
    assert all(set(data) >= {"seconds"} for _, data in parsed[:-1])  # one node's entry, not the cumulative trace
    assert "parse" not in parsed[1][1]


def test_search_logs_each_node_after_the_response(client, tmp_path):
    client.post("/search", json={"question": "출생아 수"})
    [line] = [json.loads(x) for f in tmp_path.glob("*.jsonl") for x in f.read_text(encoding="utf-8").splitlines()]
    assert set(line["seconds"]) == {"parse", "condition", "retrieve_dense", "retrieve_bm25", "fuse", "rerank1",
                                    "rerank2", "gate", "guide"}  # the graph's nodes (T2-39 split of retrieve)
    assert line["route"] == "guide" and line["top"][0] == "101_DT_A"


def test_node_api_refuses_an_invented_table_and_rewrites_a_forged_citation(client):
    state = _chain(client, "출생아 수")
    state.pop("result")
    fake = {**state["docs"][0], "id": "999_FAKE", "metadata": {**state["docs"][0]["metadata"], "key": "999_FAKE"}}
    assert client.post("/nodes/gate", json={**state, "docs": [fake]}).status_code == 422

    forged = {**state["docs"][0], "metadata": {**state["docs"][0]["metadata"], "url": "http://evil", "table_name": "가짜"}}
    out = client.post("/nodes/answer", json={**state, "docs": [forged]}).json()
    assert out["docs"][0]["metadata"]["url"] == "http://kosis.kr/101_DT_A"
    assert out["result"]["candidates"][0]["table_name"] == "출생아수" and "evil" not in out["result"]["answer"]


def test_node_api_errors(client):
    state = client.post("/nodes/parse", json={"question": "출생아 수"}).json()
    assert client.post("/nodes/answer", json=state).status_code == 422  # no route yet
    assert client.post("/nodes/answer", json={**state, "route": "guide"}).status_code == 422  # guide with no docs
    main.app.dependency_overrides[main.get_runtime] = lambda: main.Runtime(_resources(select=None), MockClient(REPLY))
    assert client.post("/nodes/rerank2", json=state).status_code == 409


def test_openapi_lists_every_node_endpoint():
    paths = main.app.openapi()["paths"]
    assert {f"/nodes/{n}" for n in ("parse", "retrieve", "rerank1", "rerank2", "gate", "answer")} <= set(paths)
    assert {"/nodes", "/search", "/search/stream", "/health"} <= set(paths)


def test_a_missing_llm_key_is_a_503_not_a_500(client):
    from kosis_rag.llm.config import LLMKeyMissing

    def no_key():
        raise LLMKeyMissing("LLMConfig.api_key is required for provider='hcx'")

    main.app.dependency_overrides[main.get_runtime] = no_key
    r = client.post("/search", json={"question": "출생아 수"})
    assert r.status_code == 503 and "LLM key missing" in r.json()["detail"]
