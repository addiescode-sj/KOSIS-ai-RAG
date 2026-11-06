"""The staged table search as a LangGraph StateGraph (T2-10): one node per stage over one shared state.

                                         +-> retrieve_dense ---+
    START -> parse -> condition -> [embed] +-> retrieve_bm25 ----+-> fuse -> [rerank1] -> [rerank2] -> gate -> END
                                         +-> [retrieve_hq] ----+
                                         +-> [retrieve_hq_bm25]+

Each node calls the same function the old nested retrievers call (T2-41 S2: the parse / retrieve step functions
parse.node.parse_question / decide_condition and retrieve.node.embed / retrieve_list / fuse, over the parts of
legacy.retrievers.hybrid_search: condition_filter, search_list, rrf_fuse; T2-41 S3: the rerank / gate step functions
rerank.node.rerank1 / rerank2 over rerank.stage1.rerank_stage1 / rerank.stage2.rerank_stage2, gate.node.decide_route),
so the graph and legacy.pipeline.build_search(...) rank the same tables. Stages that are off are left out when the
graph is built. The question is parsed once (parse) and its filter
decided once (condition) for every list, where the old path parsed inside the hybrid.

T2-39 split the old `retrieve` node (one hybrid_search call) into parallel list nodes and `fuse`: each retrieve_*
node writes its own ranked list into `runs[name]` (and its filtered-out tables into `run_removed[name]`); the dict
reducer merges the parallel writes, and fuse joins the lists in a fixed order (dense, bm25, hq, hq_bm25) by RRF, so
the result does not depend on which branch finished first. The question-index lists (retrieve_hq,
retrieve_hq_bm25) are added only when the resources hold them (HQ_BUILD); without them the graph returns what the
pre-T2-39 graph returned (tests/test_graph_runs.py). The candidate pool is formed at fuse: tracing (the evals trace module)
sees the retrieve_* nodes as spans without a ranked pool, and fuse as the first search node.

LangGraph is imported only in this package inside kosis_rag (the offline builds have their own package): the old path
works without it. The list nodes run in LangGraph's worker threads. When two or more lists embed the question (dense and
the question index share one query embedder), the `embed` node embeds it once before the branches and each list
searches with that vector, so the embedder is never called from two threads at once and the question is embedded
once. With one embedding list (HQ off) there is no embed node: dense embeds inside its own search, as before.

The node API (backend.app.main, T2-37) keeps its `retrieve` endpoint: graph.nodes_api.make_nodes runs condition,
[embed], the retrieve_<list> nodes and fuse one after another as one `retrieve` node, so a client's chain parse ->
retrieve -> ... -> gate still equals run_search. The graph itself (build_graph, run_search, stream_search) runs
graph_nodes.

T2-41 S4 (from graph.py): graph_nodes assembles a node table (name -> node over one step function; a stage that is off
has no row) and add_search_edges wires it; the state is schemas.state.SearchState, the timing graph.spans.
T2-41 S5: the module graph ends at gate (route found | refuse) and holds no LLM client; the app's answer nodes
(refuse | guide: the guide sentence, its check, the citations and the token stream) are backend/app/respond, which
builds its own graph from graph_nodes and add_search_edges (backend.app.respond.graph.build_answer_graph).
"""

from __future__ import annotations

import time
from typing import Any, Callable, Iterator

from langgraph.graph import END, START, StateGraph

from kosis_rag.gate import node as gate_step
from kosis_rag.gate.node import GateInput
from kosis_rag.graph.spans import timed as _timed
from kosis_rag.parse import node as parse_step  # module access, so a test can patch parse.node.parse_query
from kosis_rag.parse.schemas import ConditionInput, ParseInput
from kosis_rag.rerank import node as rerank_step
from kosis_rag.rerank.schemas import Stage1Input, Stage2Input
from kosis_rag.retrieve import node as retrieve_step
from kosis_rag.retrieve.schemas import EmbedInput, FuseInput, ListInput
from kosis_rag.schemas.resources import SearchResources
from kosis_rag.schemas.state import SearchState


Node = Callable[[SearchState], dict]  # one graph node: current state -> the fields it writes


def graph_nodes(resources: SearchResources, *, k: int = 5) -> dict[str, Node]:
    """The graph's node functions over these resources, in run order; build_graph wires them (T2-39 split).

    In: as build_graph.
    Out: name -> fn(SearchState) -> state update: parse, condition, [embed], retrieve_<list> per list (dense, bm25,
        [hq], [hq_bm25]), fuse, [rerank1], [rerank2], gate. retrieve_hq is present only when
        resources.hq is set, retrieve_hq_bm25 only when resources.hq_bm25 is set (T2-39, HQ_BUILD / HQ_BM25), embed
        only when two or more lists embed the question (an embed_query field: dense and hq), rerank1 only when
        resources.stage1_score is set, rerank2 only when resources.select is set.
    Side effects: none when made; calling a node does what its docstring says. Node: none (it makes them).
    """
    def timed(name: str, state: SearchState, started: float, **extra: Any) -> dict[str, dict[str, Any]]:
        """The trace with this node's entry added (graph.spans.timed over state.trace). Node: every node."""
        return _timed(name, state.trace, started, **extra)

    def parse(state: SearchState) -> dict:
        """Read the question's search conditions once.

        In: state.question (resources.sigungu: the 시군구 dictionary). Out: {"query": ParsedQuery, "trace"}.
        Side effects: none (rules only). Node: parse (step: parse.node.parse_question).
        """
        started = time.perf_counter()
        parsed = parse_step.parse_question(ParseInput(question=state.question), resources)
        return {"query": parsed, "trace": timed("parse", state, started)}

    def condition(state: SearchState) -> dict:
        """Decide once the filter every list is searched under (the old retrieve node's first step).

        In: state.query (its filter is used only when COND_FILTER is on).
        Out: {"table_filter": the filter after relaxation (None when off or no condition), "relaxed": conditions
            dropped because no table met them (filters.RELAX_ORDER; years are never dropped), "trace" (with relaxed)}.
        Side effects: none (rules over the registry); ValueError for a condition without records, as before.
        Node: condition (step: parse.node.decide_condition).
        """
        started = time.perf_counter()
        out = parse_step.decide_condition(ConditionInput(query=state.query), resources)
        return {"table_filter": out.table_filter, "relaxed": out.relaxed,
                "trace": timed("condition", state, started, relaxed=out.relaxed)}

    def retrieve(name: str):
        """The node that searches one list.

        In: name = the list name (rank_<name> metadata, runs key; one of retrieve.node.search_lists).
        Out: the node function. Side effects: none when built. Node: none (it builds retrieve_<name>).
        """
        node = f"retrieve_{name}"

        def run(state: SearchState) -> dict:
            """Search this list under the shared filter.

            In: state.question, state.table_filter, state.query_vecs (this list's question vector from embed, if any:
                the search then uses it instead of calling the embedder).
            Out: {"runs": {name: ranked docs (at most pool)}, "run_removed": {name: key -> why filtered out}, "trace"}.
            Side effects: the retriever's search (Chroma query or BM25 scoring); ValueError as hybrid_search.
            Node: retrieve_dense, retrieve_bm25, retrieve_hq, retrieve_hq_bm25 (step: retrieve.node.retrieve_list).
            """
            started = time.perf_counter()
            out = retrieve_step.retrieve_list(name, ListInput(question=state.question, table_filter=state.table_filter,
                                                              vector=state.query_vecs.get(name)), resources)
            docs, removed = out.docs, out.removed
            return {"runs": {name: docs}, "run_removed": {name: removed},
                    "trace": timed(node, state, started, n_docs=len(docs), n_removed=len(removed),
                                   top_keys=[d.metadata.get("key") for d in docs[:5]])}

        return node, run

    def embed(state: SearchState) -> dict:
        """Embed the question once for every list that embeds it (T2-39, review S4).

        In: state.question. Out: {"query_vecs": list name -> vector (lists sharing one embed_query function share one
            call and one vector), "trace" (calls = embedder calls made)}.
        Side effects: query embedder inference, one call per distinct embedder, in this one thread.
        Node: embed (step: retrieve.node.embed).
        """
        started = time.perf_counter()
        out = retrieve_step.embed(EmbedInput(question=state.question), resources)
        vecs = out.query_vecs
        return {"query_vecs": vecs, "trace": timed("embed", state, started, calls=out.calls, lists=list(vecs))}

    def fuse(state: SearchState) -> dict:
        """Fuse the lists by weighted RRF into the candidate pool (the old retrieve node's last step).

        In: state.runs, state.run_removed (one entry per retrieve_* node).
        Out: {"docs": the fused pool (at most `pool`; metadata as hybrid_search: score from dense, score_from,
            rrf_score, rank_<name>), "removed": key -> why filtered out (lists merged in fuse order), "trace"}.
        Side effects: none. Node: fuse (step: retrieve.node.fuse).
        """
        started = time.perf_counter()
        names = [name for name, _, _ in lists]
        out = retrieve_step.fuse(FuseInput(runs=state.runs, run_removed=state.run_removed), resources)
        docs, removed = out.docs, out.removed
        return {"docs": docs, "removed": removed,
                "trace": timed("fuse", state, started, n_docs=len(docs), n_removed=len(removed),
                               lists=names, relaxed=state.relaxed)}

    def rerank1(state: SearchState) -> dict:
        """Re-order the top rerank_n by the stage 1 cross-encoder.

        In: state.question, state.docs. Out: {"docs": re-ordered with stage1_* metadata, "trace"}.
        Side effects: cross-encoder inference; ValueError for an unknown RERANK_MODE. Node: rerank1 (step:
        rerank.node.rerank1).
        """
        started = time.perf_counter()
        docs = rerank_step.rerank1(Stage1Input(question=state.question, docs=state.docs), resources).docs
        return {"docs": docs, "trace": timed("rerank1", state, started)}

    def rerank2(state: SearchState) -> dict:
        """Let the stage 2 LLM pick among the top stage2_k.

        In: state.question, state.docs.
        Out: {"docs": picked first with stage2_* metadata, "trace": with the top doc's stage2_* fields}.
        Side effects: LLMKeyMissing on entry when resources.stage2_llm has no key (T2-41 S3); one LLM re-ranker call
        (CLOVA / HCX / OpenAI); its errors are stored in stage2_error. Node: rerank2 (step: rerank.node.rerank2).
        """
        started = time.perf_counter()
        docs = rerank_step.rerank2(Stage2Input(question=state.question, docs=state.docs), resources).docs
        stage2 = {key: v for key, v in docs[0].metadata.items() if key.startswith("stage2")} if docs else {}
        return {"docs": docs, "trace": timed("rerank2", state, started, **stage2)}

    def gate(state: SearchState) -> dict:
        """Decide found or refuse on the top k.

        In: state.docs. Out: {"route" (found | refuse), "refusal_reason", "trace" (with route)}. Side effects: none.
        Node: gate (step: gate.node.decide_route).
        """
        started = time.perf_counter()
        out = gate_step.decide_route(GateInput(docs=state.docs[:k]), resources)
        return {"route": out.route, "refusal_reason": out.refusal_reason,
                "trace": timed("gate", state, started, route=out.route)}

    # Lists in fuse order (dense first: its score stays in metadata["score"]): dense and BM25 always, the question
    # index when loaded (T2-39), with HQ_WEIGHT as its RRF weight (retrieve.node.search_lists)
    lists = retrieve_step.search_lists(resources)
    branches = [retrieve(name) for name, _, _ in lists]
    # Lists that embed the question themselves (Chroma dense, HqRetriever: an embed_query field). With two or more,
    # the embed node embeds once and hands each its vector; with one (HQ off) there is no embed node.
    embedders = retrieve_step.embedders(resources)

    # The node table in run order: (name, node, on). A stage that is off has no node; only gate branches at run time.
    table: list[tuple[str, Node, bool]] = [
        ("parse", parse, True),
        ("condition", condition, True),
        ("embed", embed, len(embedders) >= 2),
        *((name, fn, True) for name, fn in branches),
        ("fuse", fuse, True),
        ("rerank1", rerank1, resources.stage1_score is not None),
        ("rerank2", rerank2, resources.select is not None),
        ("gate", gate, True),
    ]
    return {name: fn for name, fn, on in table if on}


def add_search_edges(g: StateGraph, nodes: dict[str, Node]) -> None:
    """Wire graph_nodes' table into g: parse -> condition [-> embed] -> every list at once -> fuse (waits for all),
    then a line through the stages that are on, up to gate. What follows gate is the caller's: build_graph ends it
    (gate -> END), the backend's answer graph branches it (refuse | guide, backend.app.respond.graph).

    In: g = a StateGraph with the nodes already added; nodes = graph_nodes(...) (or a table with the same names).
    Out: None. Side effects: adds the edges to g. Node: none.
    """
    branches = [name for name in nodes if name.startswith("retrieve_")]
    fan_out = "embed" if "embed" in nodes else "condition"  # the node the list branches start from
    stages = ["fuse", *(name for name in ("rerank1", "rerank2") if name in nodes)]
    g.add_edge(START, "parse")
    g.add_edge("parse", "condition")
    if fan_out == "embed":
        g.add_edge("condition", "embed")
    for name in branches:
        g.add_edge(fan_out, name)
    g.add_edge(branches, "fuse")
    for a, b in zip(stages, stages[1:]):
        g.add_edge(a, b)
    g.add_edge(stages[-1], "gate")


def build_graph(resources: SearchResources, *, k: int = 5):
    """Compile the search graph for these resources. Returns the compiled graph (call .invoke({"question": q})).

    In: resources = adapters.loader.load_resources(spec=..., options=...) (records, dense, BM25, stage 1 scorer,
        stage 2 selector, options, spec);
        k = the candidate budget (gate sees docs[:k]; the search stages see the whole pool). No LLM client: the
        graph ends at gate (T2-41 S5).
    Out: a compiled LangGraph graph over SearchState with graph_nodes' nodes: retrieve_hq is added only when
        resources.hq is set, retrieve_hq_bm25 only when resources.hq_bm25 is set (T2-39, HQ_BUILD / HQ_BM25), embed
        only when two or more lists embed the question (an embed_query field: dense and hq), rerank1 only when
        resources.stage1_score is set, rerank2 only when resources.select is set.
    Side effects: none when built; running it calls the retrievers, the models and the LLMs the resources hold.
    Node: none (it builds them).
    """
    nodes = graph_nodes(resources, k=k)
    g = StateGraph(SearchState)
    for name, fn in nodes.items():
        g.add_node(name, fn)
    add_search_edges(g, nodes)
    g.add_edge("gate", END)
    return g.compile()


def run_search(graph, question: str, config: dict | None = None) -> SearchState:
    """Invoke the graph and return the final state as SearchState.

    In: graph = build_graph(...); question = the user question; config = LangChain run config (callbacks, run name),
        passed through as is.
    Out: the final SearchState (route found | refuse set by gate; docs = the whole ranked pool; runs = each list).
    Side effects: runs every node (retrievers, models, LLMs held by the graph's resources).
    Node: none (it runs them).
    """
    return SearchState.model_validate(graph.invoke({"question": question}, config=config))


def stream_search(graph, question: str, config: dict | None = None, *, k: int = 5) -> Iterator[tuple[str, Any]]:
    """Run the graph and yield each node as it finishes, then the module's output (T2-37; T2-41 S5).

    In: as run_search; k = the budget the graph was built with (build_graph's k), for SearchOutput.candidates.
    Out: ("<node>", that node's own trace entry) per finished node in run order (trace is cumulative, so only the
        node's key is sent; the parallel retrieve_<list> nodes come in the order they finish, T2-39), then
        ("output", SearchOutput) — kosis_rag.api.to_output of the final state, what SearchModule.search returns.
    Side effects: as run_search. Node: none (it runs them).
    """
    from kosis_rag.api import to_output  # api imports this module: read the projection when the stream ends

    final: dict = {}
    for mode, chunk in graph.stream({"question": question}, config=config, stream_mode=["updates", "values"]):
        if mode == "updates":
            for name, update in chunk.items():
                yield name, update["trace"][name]
        else:
            final = chunk
    yield "output", to_output(SearchState.model_validate(final), k)
