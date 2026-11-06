"""T2-41 S4: the graph is assembled from a node table (graph.search.graph_nodes): names and order are the old
graph_nodes keys, the edges are unchanged, and schemas.state imports without langgraph.

T2-41 S5: the module's table ends at gate (no client); the backend's answer graph (backend.app.respond.graph) adds
refuse | guide with the old edges, and its guide client is made from the LLMConfig when guide first runs.
"""

import os
import subprocess
import sys

import pytest

pytest.importorskip("langgraph")

from backend.app.respond.graph import api_nodes, build_answer_graph, run_answer  # noqa: E402
from backend.app.respond.node import GuideClient  # noqa: E402
from kosis_rag.graph.nodes_api import make_nodes  # noqa: E402
from kosis_rag.graph.search import build_graph, graph_nodes  # noqa: E402
from kosis_rag.llm import providers  # noqa: E402
from kosis_rag.llm.base import MockClient  # noqa: E402
from kosis_rag.llm.config import LLMConfig, LLMKeyMissing  # noqa: E402
from tests.test_graph import KEYS, RECORDS, CatalogRetriever, _load, _pick, _stage1  # noqa: E402
from tests.test_graph_runs import _hq_resources  # noqa: E402

pytestmark = pytest.mark.offline


def _resources(**kw):
    return _load(records=RECORDS, dense=CatalogRetriever(keys=KEYS), **kw)


def _answer_graph_nodes(resources):
    return [n for n in build_answer_graph(resources, MockClient()).get_graph().nodes if not n.startswith("__")]


def test_node_table_order_is_the_old_graph_nodes_order():
    full = graph_nodes(_resources(stage1_score=_stage1, select=_pick()))
    assert list(full) == ["parse", "condition", "retrieve_dense", "retrieve_bm25", "fuse", "rerank1", "rerank2",
                          "gate"]
    assert list(graph_nodes(_resources())) == ["parse", "condition", "retrieve_dense", "retrieve_bm25", "fuse", "gate"]
    hq = graph_nodes(_hq_resources(bm25=True, stage1_score=_stage1))
    assert list(hq) == ["parse", "condition", "retrieve_dense", "retrieve_bm25", "retrieve_hq", "retrieve_hq_bm25",
                        "fuse", "rerank1", "gate"]
    # the backend's answer graph: the same table, then refuse, guide (the pre-S5 graph_nodes order)
    assert _answer_graph_nodes(_resources(stage1_score=_stage1, select=_pick())) == [*full, "refuse", "guide"]
    assert _answer_graph_nodes(_hq_resources(bm25=True, stage1_score=_stage1)) == [*hq, "refuse", "guide"]


SEARCH_EDGES = {("__start__", "parse"), ("parse", "condition"), ("condition", "retrieve_dense"),
                ("condition", "retrieve_bm25"), ("retrieve_dense", "fuse"), ("retrieve_bm25", "fuse"),
                ("fuse", "rerank1"), ("rerank1", "rerank2"), ("rerank2", "gate")}


def test_edges_are_unchanged():
    resources = _resources(stage1_score=_stage1, select=_pick())
    answer = build_answer_graph(resources, MockClient()).get_graph()
    assert {(e.source, e.target) for e in answer.edges} == SEARCH_EDGES | {
        ("gate", "refuse"), ("gate", "guide"), ("refuse", "__end__"), ("guide", "__end__")}
    module = build_graph(resources).get_graph()  # T2-41 S5: the module's search ends at gate
    assert {(e.source, e.target) for e in module.edges} == SEARCH_EDGES | {("gate", "__end__")}


def test_node_api_set_is_unchanged():
    resources = _resources(stage1_score=_stage1, select=_pick())
    assert list(make_nodes(resources)) == ["parse", "retrieve", "rerank1", "rerank2", "gate"]
    assert list(api_nodes(resources, MockClient())) == ["parse", "retrieve", "rerank1", "rerank2", "gate", "refuse",
                                                        "guide"]


def test_the_guide_client_is_built_from_its_config_once_when_guide_first_runs(monkeypatch):
    made = []

    def fake_clova_chat(config, **kw):
        made.append(config)
        return MockClient("[1] DT_A 표입니다.")

    monkeypatch.setitem(providers.PROVIDERS, "clova_chat", fake_clova_chat)
    guide = GuideClient(LLMConfig(provider="clova_chat", model="HCX-DASH-002", api_key="k"))
    graph = build_answer_graph(_resources(select=_pick("101_DT_A")), guide)
    assert made == []
    assert run_answer(graph, "출생아 수").result.answer_origin == "llm"
    run_answer(graph, "사망자 수")
    assert len(made) == 1 and made[0].model == "HCX-DASH-002"
    refused = build_answer_graph(_resources(select=_pick(reason="out_of_scope")), guide)
    assert run_answer(refused, "자동차").route == "refuse" and len(made) == 1  # refuse never needs it


def test_without_a_guide_key_guide_raises_on_entry_and_refuse_answers():
    no_key = LLMConfig(provider="clova_chat", model="HCX-DASH-002")
    with pytest.raises(LLMKeyMissing, match="guide LLM key not set"):
        run_answer(build_answer_graph(_resources(select=_pick("101_DT_A")), no_key), "출생아 수")
    with pytest.raises(LLMKeyMissing):
        run_answer(build_answer_graph(_resources(select=_pick("101_DT_A")), None), "출생아 수")
    refused = run_answer(build_answer_graph(_resources(select=_pick(reason="out_of_scope")), no_key), "자동차")
    assert refused.route == "refuse" and refused.result.no_answer


def test_state_imports_without_langgraph():
    code = ("import sys\n"
            "sys.modules['langgraph'] = None\n"  # any langgraph import now fails
            "import kosis_rag.schemas.state, kosis_rag.schemas.resources, kosis_rag.schemas.io\n"
            "print('ok')\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8",
                         env={**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)})
    assert out.stdout.strip() == "ok", out.stderr
