"""T2-41 S5: the backend's answer graph (backend/app/respond) is the module's search graph + refuse | guide.

1. Node names and order: the module's node table, then refuse, guide; the module graph ends at gate.
2. The two outputs agree: SearchOutput.candidates == SearchResult.candidates; on refuse the answer is
   SearchOutput.refusal_text.
3. The module's nodes run on the API state unchanged: the module run and the backend run differ in gate's route
   only ("found" -> "guide", gate_to_api).
4. Node API: the body /nodes/gate returns goes to /nodes/answer as it is, and the result is /search's; /nodes/gate
   and the graph rename the route through the same gate_to_api.
(openapi: the docs/openapi.json diff after scripts/export_openapi.py; SSE event order: tests/test_api_nodes.py.)
Fakes of tests/test_graph.py; no model, no server, no LLM.
"""

import pytest

pytest.importorskip("langgraph")

from fastapi.testclient import TestClient  # noqa: E402

from backend.app import main  # noqa: E402
from backend.app.respond import graph as respond_graph  # noqa: E402
from backend.app.respond.graph import api_nodes, build_answer_graph, gate_to_api, run_answer  # noqa: E402
from backend.app.respond.state import SearchState as ApiState  # noqa: E402
from kosis_chat.llm.model import MockClient  # noqa: E402
from kosis_rag.graph.search import build_graph, graph_nodes, run_search  # noqa: E402
from kosis_rag.schemas.state import SearchState as ModuleState  # noqa: E402
from tests.parity_cases import CASES, _clean, case_resources, module_output, run_case  # noqa: E402
from tests.test_graph import KEYS, RECORDS, CatalogRetriever, _load, _pick, _stage1  # noqa: E402
from tests.test_graph_runs import _hq_resources  # noqa: E402

pytestmark = pytest.mark.offline

REPLY = "[1] DT_A 표가 출생아 수를 담고 있습니다."
IDS = [c.name for c in CASES]


def _resources(**kw):
    parts = {"stage1_score": _stage1, "select": _pick("101_DT_A"), **kw}
    return _load(records=RECORDS, dense=CatalogRetriever(keys=KEYS), **parts)


def _plain_trace(trace):
    """The trace without timings (seconds, *_seconds: they differ between two runs)."""
    return _clean(trace)


# --- 1. node names and order ---

@pytest.mark.parametrize("resources", [_resources(), _resources(stage1_score=None, select=None),
                                       _hq_resources(bm25=True, stage1_score=_stage1)],
                         ids=["both", "none", "hq"])
def test_answer_graph_nodes_are_the_module_table_then_refuse_guide(resources):
    table = list(graph_nodes(resources))
    assert table[-1] == "gate" and not {"refuse", "guide"} & set(table)
    answer = [n for n in build_answer_graph(resources, MockClient()).get_graph().nodes if not n.startswith("__")]
    assert answer == [*table, "refuse", "guide"]
    module = [n for n in build_graph(resources).get_graph().nodes if not n.startswith("__")]
    assert module == table


# --- 2. module output == backend result ---

@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_module_candidates_are_the_backend_candidates(case):
    out, result = module_output(case), run_case(case).result
    assert out.candidates == result.candidates
    if out.route == "refuse":
        assert result.no_answer and result.answer_origin == "refusal"
        assert result.answer == out.refusal_text and result.no_answer_reason == out.refusal_reason
        assert result.picked == out.picked == []
    else:
        assert out.refusal_text is None and run_case(case).route == "guide"


# --- 3. the module's nodes on the API state: only the route name differs ---

@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_module_run_and_backend_run_differ_in_the_route_name_only(case):
    module = run_search(build_graph(case_resources(case), k=case.k), case.question)
    api = run_case(case)
    assert isinstance(module, ModuleState) and isinstance(api, ApiState)
    assert module.route == {"guide": "found", "refuse": "refuse"}[api.route]
    for field in ("question", "query", "table_filter", "relaxed", "removed", "refusal_reason", "query_vecs"):
        assert getattr(module, field) == getattr(api, field), field
    assert [(d.id, d.page_content, _clean(d.metadata)) for d in module.docs] == [
        (d.id, d.page_content, _clean(d.metadata)) for d in api.docs]
    assert set(module.runs) == set(api.runs)
    module_trace = _plain_trace(module.trace)
    api_trace = {name: entry for name, entry in _plain_trace(api.trace).items() if name not in ("refuse", "guide")}
    gate_update = gate_to_api({"route": module.route, "trace": {"gate": module_trace["gate"]}})
    assert gate_update["route"] == api.route
    assert {**module_trace, "gate": gate_update["trace"]["gate"]} == api_trace
    assert list(module.trace) == [n for n in api.trace if n not in ("refuse", "guide")]


def test_gate_to_api_renames_found_only():
    found = {"route": "found", "refusal_reason": None, "trace": {"parse": {"seconds": 0.1}, "gate": {"route": "found"}}}
    assert gate_to_api(found) == {"route": "guide", "refusal_reason": None,
                                  "trace": {"parse": {"seconds": 0.1}, "gate": {"route": "guide"}}}
    refused = {"route": "refuse", "refusal_reason": "no_item", "trace": {"gate": {"route": "refuse"}}}
    assert gate_to_api(refused) is refused
    assert found["route"] == "found"  # the module's update is not changed in place


def test_api_nodes_keep_the_api_state_type():
    state = ApiState(question="출생아 수")
    for name, fn in api_nodes(_resources(), MockClient(REPLY)).items():
        if name in ("refuse", "guide") and name != state.route:
            continue
        state = state.model_copy(update=fn(state))
        assert isinstance(state, ApiState), name
    assert state.route == "guide" and state.result.answer_origin == "llm"


# --- 4. node API: /nodes/gate's body -> /nodes/answer, one gate_to_api for both entry points ---

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "LOG_DIR", tmp_path)
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


@pytest.mark.parametrize("select, route", [(_pick("101_DT_A"), "guide"), (_pick(reason="out_of_scope"), "refuse")])
def test_the_gate_body_goes_to_answer_as_it_is_and_equals_search(client, select, route):
    main.app.dependency_overrides[main.get_runtime] = lambda: main.Runtime(_resources(select=select),
                                                                           client=MockClient(REPLY))
    question = "출생아 수"
    state = client.post("/nodes/parse", json={"question": question}).json()
    for name in ("retrieve", "rerank1", "rerank2", "gate"):
        r = client.post(f"/nodes/{name}", json=state)
        assert r.status_code == 200, (name, r.text)
        state = r.json()
    assert state["route"] in {"guide", "refuse"} and state["route"] == route  # the raw value, no normalizing
    assert state["trace"]["gate"]["route"] == route
    r = client.post("/nodes/answer", json=state)
    assert r.status_code == 200, r.text
    assert r.json()["result"] == client.post("/search", json={"question": question}).json()


def test_nodes_gate_and_the_graph_use_the_same_gate_to_api(client, monkeypatch):
    seen = []
    real = respond_graph.gate_to_api

    def counting(update):
        seen.append(update["route"])
        return real(update)

    monkeypatch.setattr(respond_graph, "gate_to_api", counting)
    main.app.dependency_overrides[main.get_runtime] = lambda: main.Runtime(_resources(), client=MockClient(REPLY))
    state = client.post("/nodes/parse", json={"question": "출생아 수"}).json()
    for name in ("retrieve", "rerank1", "rerank2", "gate"):
        state = client.post(f"/nodes/{name}", json=state).json()
    assert seen == ["found"] and state["route"] == "guide"
    assert client.post("/search", json={"question": "출생아 수"}).json()["answer_origin"] == "llm"
    assert seen == ["found", "found"]
    state = run_answer(build_answer_graph(_resources(), MockClient(REPLY)), "출생아 수")
    assert seen == ["found"] * 3 and state.route == "guide"


def test_without_a_guide_key_the_stream_starts_and_ends_with_an_error_event(client):
    from kosis_rag.llm.config import LLMConfig

    no_key = LLMConfig(provider="clova_chat", model="HCX-DASH-002")
    main.app.dependency_overrides[main.get_runtime] = lambda: main.Runtime(_resources(), guide_llm=no_key)
    r = client.post("/search/stream", json={"question": "출생아 수"})
    events = [line.removeprefix("event: ") for line in r.text.splitlines() if line.startswith("event: ")]
    assert r.status_code == 200 and events[-2:] == ["gate", "error"] and "LLMKeyMissing" in r.text
    assert client.post("/search", json={"question": "출생아 수"}).status_code == 503
