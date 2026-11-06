"""T2-41 S4: the graph is assembled from a node table (graph.search.graph_nodes): names and order are the old
graph_nodes keys, the edges are unchanged, and schemas.state imports without langgraph.
"""

import os
import subprocess
import sys

import pytest

pytest.importorskip("langgraph")

from kosis_rag.graph.nodes_api import make_nodes  # noqa: E402
from kosis_rag.graph.search import build_graph, graph_nodes  # noqa: E402
from kosis_rag.llm.base import MockClient  # noqa: E402
from tests.test_graph import KEYS, RECORDS, CatalogRetriever, _load, _pick, _stage1  # noqa: E402
from tests.test_graph_runs import _hq_resources  # noqa: E402

pytestmark = pytest.mark.offline


def _resources(**kw):
    return _load(records=RECORDS, dense=CatalogRetriever(keys=KEYS), **kw)


def test_node_table_order_is_the_old_graph_nodes_order():
    full = graph_nodes(_resources(stage1_score=_stage1, select=_pick()), MockClient())
    assert list(full) == ["parse", "condition", "retrieve_dense", "retrieve_bm25", "fuse", "rerank1", "rerank2",
                          "gate", "refuse", "guide"]
    assert list(graph_nodes(_resources(), MockClient())) == ["parse", "condition", "retrieve_dense", "retrieve_bm25",
                                                             "fuse", "gate", "refuse", "guide"]
    hq = graph_nodes(_hq_resources(bm25=True, stage1_score=_stage1), MockClient())
    assert list(hq) == ["parse", "condition", "retrieve_dense", "retrieve_bm25", "retrieve_hq", "retrieve_hq_bm25",
                        "fuse", "rerank1", "gate", "refuse", "guide"]


def test_edges_are_unchanged():
    graph = build_graph(_resources(stage1_score=_stage1, select=_pick()), MockClient()).get_graph()
    edges = {(e.source, e.target) for e in graph.edges}
    assert edges == {("__start__", "parse"), ("parse", "condition"), ("condition", "retrieve_dense"),
                     ("condition", "retrieve_bm25"), ("retrieve_dense", "fuse"), ("retrieve_bm25", "fuse"),
                     ("fuse", "rerank1"), ("rerank1", "rerank2"), ("rerank2", "gate"), ("gate", "refuse"),
                     ("gate", "guide"), ("refuse", "__end__"), ("guide", "__end__")}


def test_node_api_set_is_unchanged():
    nodes = make_nodes(_resources(stage1_score=_stage1, select=_pick()), MockClient())
    assert list(nodes) == ["parse", "retrieve", "rerank1", "rerank2", "gate", "refuse", "guide"]


def test_a_client_factory_is_called_once_when_guide_first_runs():
    from kosis_rag.graph.search import run_search

    made = []

    def factory():
        made.append(1)
        return MockClient("[1] DT_A 표입니다.")

    graph = build_graph(_resources(select=_pick("101_DT_A")), factory)
    assert made == []
    assert run_search(graph, "출생아 수").result.answer_origin == "llm"
    run_search(graph, "사망자 수")
    assert made == [1]
    refused = build_graph(_resources(select=_pick(reason="out_of_scope")), factory)
    assert run_search(refused, "자동차").route == "refuse" and made == [1]  # refuse never calls it


def test_state_imports_without_langgraph():
    code = ("import sys\n"
            "sys.modules['langgraph'] = None\n"  # any langgraph import now fails
            "import kosis_rag.schemas.state, kosis_rag.schemas.resources, kosis_rag.schemas.io\n"
            "print('ok')\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8",
                         env={**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)})
    assert out.stdout.strip() == "ok", out.stderr
