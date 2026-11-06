"""The API's answer graph (T2-41 S5): the module's search nodes, gate's route renamed, then refuse | guide.

                                         +-> retrieve_dense ---+
    START -> parse -> condition -> [embed] +-> retrieve_bm25 ----+-> fuse -> [rerank1] -> [rerank2] -> gate
                                         +-> [retrieve_hq] ----+       -> refuse | guide -> END
                                         +-> [retrieve_hq_bm25]+

The search nodes and their edges are the module's (kosis_rag.graph.search.graph_nodes / add_search_edges), run on
the API state (state.SearchState). The module's gate ends its search with its own route names; the API's route is
"guide" | "refuse" (docs/openapi.json). gate_to_api is the one place that renames it, used by both entry points:
build_answer_graph (/search, /search/stream) and api_nodes (the node API, /nodes/gate), so the body /nodes/gate
returns can be sent to /nodes/answer as it is.
"""

from __future__ import annotations

from typing import Any, Iterator

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph

from kosis_rag.graph.nodes_api import make_nodes
from kosis_rag.graph.search import add_search_edges, graph_nodes
from kosis_rag.llm.base import LLMClient
from kosis_rag.llm.config import LLMConfig
from kosis_rag.schemas.resources import SearchResources

from backend.app.respond.node import GuideClient, Node, answer_nodes, as_guide_client
from backend.app.respond.state import SearchState

GuideLLM = GuideClient | LLMConfig | LLMClient | BaseChatModel | None


def gate_to_api(update: dict) -> dict:
    """The module gate node's update with the API's route: "found" -> "guide" (in route and in gate's trace entry).

    In: update = what the module's gate node returned ({"route", "refusal_reason", "trace"}).
    Out: the same update, route "guide" where the module wrote "found"; a refusal is returned as it is.
    Side effects: none. Node: gate (both in build_answer_graph and in api_nodes).
    """
    if update.get("route") != "found":
        return update
    trace = update["trace"]
    return {**update, "route": "guide", "trace": {**trace, "gate": {**trace["gate"], "route": "guide"}}}


def _api_gate(gate: Node) -> Node:
    """The module's gate node with its update passed through gate_to_api (read at call time)."""

    def run(state: SearchState) -> dict:
        return gate_to_api(gate(state))

    return run


def _with_api_gate(nodes: dict[str, Node]) -> dict[str, Node]:
    return {name: _api_gate(fn) if name == "gate" else fn for name, fn in nodes.items()}


def build_answer_graph(resources: SearchResources, guide_llm: GuideLLM, *, k: int = 5):
    """Compile the API's graph: the module's search nodes up to gate, then refuse | guide (the old build_graph).

    In: resources = adapters.loader.load_resources(spec=..., options=...) (the search options are resources.options);
        guide_llm = the guide LLM: a GuideClient (the Runtime's, shared by every graph), an LLMConfig (built when
        guide first runs), or a ready client (ChatClovaX, MockClient); k = tables shown to gate / refuse / guide.
    Out: a compiled LangGraph graph over state.SearchState. Node names and order: graph_nodes' (parse, condition,
        [embed], retrieve_<list>..., fuse, [rerank1], [rerank2], gate), then refuse, guide.
    Side effects: none when built; running it calls the retrievers, the models and the LLMs. Node: none.
    """
    nodes = {**_with_api_gate(graph_nodes(resources, k=k)), **answer_nodes(as_guide_client(guide_llm), k=k)}
    g = StateGraph(SearchState)
    for name, fn in nodes.items():
        # the module's node functions are annotated with the module's state: run them on the API state
        g.add_node(name, fn, input_schema=SearchState)
    add_search_edges(g, nodes)
    g.add_conditional_edges("gate", lambda state: state.route, {"refuse": "refuse", "guide": "guide"})
    g.add_edge("refuse", END)
    g.add_edge("guide", END)
    return g.compile()


def api_nodes(resources: SearchResources, guide_llm: GuideLLM, *, k: int = 5) -> dict[str, Node]:
    """The node API's nodes, in run order: the module's make_nodes (parse, retrieve, [rerank1], [rerank2], gate with
    gate_to_api), then refuse, guide (answer_nodes).

    In: as build_answer_graph. Out: name -> fn(SearchState) -> update. Side effects: none when made. Node: none.
    """
    return {**_with_api_gate(make_nodes(resources, k=k)), **answer_nodes(as_guide_client(guide_llm), k=k)}


def run_answer(graph, question: str, config: dict | None = None) -> SearchState:
    """Invoke the answer graph and return the final API state (result set by refuse or guide).

    In: graph = build_answer_graph(...); question; config = LangChain run config, passed through as is.
    Side effects: runs every node. Node: none.
    """
    return SearchState.model_validate(graph.invoke({"question": question}, config=config))


def stream_answer(graph, question: str, config: dict | None = None) -> Iterator[tuple[str, Any]]:
    """Run the answer graph and yield each node as it finishes, then the final state (T2-37, the SSE search).

    In: as run_answer.
    Out: ("<node>", that node's own trace entry) per finished node in run order (the parallel retrieve_<list> nodes
        come in the order they finish, T2-39), ("token", {"delta": text}) while guide streams its checked reply
        (T2-38), then ("result", the final SearchState) — the same state run_answer returns.
    Side effects: as run_answer. Node: none.
    """
    final: dict = {}
    for mode, chunk in graph.stream({"question": question}, config=config,
                                    stream_mode=["updates", "values", "custom"]):
        if mode == "custom":
            yield "token", chunk
        elif mode == "updates":
            for name, update in chunk.items():
                yield name, update["trace"][name]
        else:
            final = chunk
    yield "result", SearchState.model_validate(final)
