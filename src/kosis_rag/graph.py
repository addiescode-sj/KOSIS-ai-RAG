"""The staged table search as a LangGraph StateGraph (T2-10): one node per stage over one shared state.

                                         +-> retrieve_dense ---+
    START -> parse -> condition -> [embed] +-> retrieve_bm25 ----+-> fuse -> [rerank1] -> [rerank2] -> gate
                                         +-> [retrieve_hq] ----+       -> refuse | guide -> END
                                         +-> [retrieve_hq_bm25]+

Each node calls the same function the old nested retrievers call (the parts of retriever.hybrid_search:
condition_filter, search_list, rrf_fuse; rerank.rerank_stage1 / rerank_stage2; chain.decide_route / refusal_answer /
guide_answer), so the graph and `chain.answer(build_search(...))` give the same result. Stages that are off are left
out when the graph is built; only `gate` branches at run time. The question is parsed once (parse) and its filter
decided once (condition) for every list, where the old path parsed inside the hybrid.

T2-39 split the old `retrieve` node (one hybrid_search call) into parallel list nodes and `fuse`: each retrieve_*
node writes its own ranked list into `runs[name]` (and its filtered-out tables into `run_removed[name]`); the dict
reducer merges the parallel writes, and fuse joins the lists in a fixed order (dense, bm25, hq, hq_bm25) by RRF, so
the result does not depend on which branch finished first. The question-index lists (retrieve_hq,
retrieve_hq_bm25) are added only when the resources hold them (HQ_BUILD); without them the graph returns what the
pre-T2-39 graph returned (tests/test_graph_runs.py). The candidate pool is formed at fuse: tracing (evals.trace)
sees the retrieve_* nodes as spans without a ranked pool, and fuse as the first search node.

LangGraph is imported only here (and in kosis_rag.hq.graph, the offline question build): the old path works
without it. The list nodes run in LangGraph's worker threads. When two or more lists embed the question (dense and
the question index share one query embedder), the `embed` node embeds it once before the branches and each list
searches with that vector, so the embedder is never called from two threads at once and the question is embedded
once. With one embedding list (HQ off) there is no embed node: dense embeds inside its own search, as before.

The node API (backend.app.main, T2-37) keeps its `retrieve` endpoint: make_nodes runs condition, [embed], the
retrieve_<list> nodes and fuse one after another as one `retrieve` node, so a client's chain parse -> retrieve ->
... -> answer still equals run_search. The graph itself (build_graph, run_search, stream_search) runs graph_nodes.
"""

from __future__ import annotations

import time
from typing import Annotated, Any, Callable, Iterator, Literal

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from kosis_chat.llm.model import LLMClient

from kosis_rag import chain
from kosis_rag import query as query_rules  # module access, so a test can patch query.parse_query
from kosis_rag.filters import TableFilter, filter_with_relaxation, relax
from kosis_rag.loader import to_document
from kosis_rag.pipeline import SearchResources
from kosis_rag.query import ParsedQuery
from kosis_rag.rerank import rerank_stage1, rerank_stage2
from kosis_rag.retriever import condition_filter, rrf_fuse, search_list
from kosis_rag.schemas import SearchResult, TableRecord


def merge(old: dict, new: dict) -> dict:
    """Reducer for the fields parallel nodes write: the union of both dicts, the newer value on a shared key.

    In: old = the field's value; new = one node's update. Out: {**old, **new}. Side effects: none.
    Node: none (LangGraph applies it to runs, run_removed and trace).
    """
    return {**old, **new}


class SearchState(BaseModel):
    """What the nodes share. Every field has one writer, except docs (overwritten by each search stage) and the
    merged fields (runs, run_removed, trace), which each node adds its own key to."""

    question: str
    query: ParsedQuery | None = None  # set by parse
    table_filter: TableFilter | None = None  # set by condition: the filter every list is searched under
    query_vecs: dict[str, list[float]] = Field(default_factory=dict)  # set by embed: list name -> question vector
    relaxed: list[str] = Field(default_factory=list)  # set by condition: conditions dropped so a table could pass (T2-29)
    runs: Annotated[dict[str, list[Document]], merge] = Field(default_factory=dict)  # retrieve_*: list name -> ranked docs
    run_removed: Annotated[dict[str, dict[str, str]], merge] = Field(default_factory=dict)  # retrieve_*: name -> removed
    removed: dict[str, str] = Field(default_factory=dict)  # set by fuse: key -> why filtered out (lists in fuse order)
    docs: list[Document] = Field(default_factory=list)  # overwritten by fuse, rerank1, rerank2
    route: Literal["refuse", "guide"] | None = None  # set by gate
    refusal_reason: str | None = None  # set by gate
    result: SearchResult | None = None  # set by refuse or guide
    trace: Annotated[dict[str, dict[str, Any]], merge] = Field(default_factory=dict)  # node -> {"seconds": float, ...}


Node = Callable[[SearchState], dict]  # one graph node: current state -> the fields it writes


def graph_nodes(resources: SearchResources, client: LLMClient | BaseChatModel, *, k: int = 5) -> dict[str, Node]:
    """The graph's node functions over these resources, in run order; build_graph wires them (T2-39 split).

    In: as build_graph.
    Out: name -> fn(SearchState) -> state update: parse, condition, [embed], retrieve_<list> per list (dense, bm25,
        [hq], [hq_bm25]), fuse, [rerank1], [rerank2], gate, refuse, guide. retrieve_hq is present only when
        resources.hq is set, retrieve_hq_bm25 only when resources.hq_bm25 is set (T2-39, HQ_BUILD / HQ_BM25), embed
        only when two or more lists embed the question (an embed_query field: dense and hq), rerank1 only when
        resources.stage1_score is set, rerank2 only when resources.select is set.
    Side effects: none when made; calling a node does what its docstring says. Node: none (it makes them).
    """
    s = resources.settings

    def timed(name: str, state: SearchState, started: float, **extra: Any) -> dict[str, dict[str, Any]]:
        """The trace with this node's entry added.

        In: name = node name; state = current state; started = perf_counter at node start; extra = fields to record.
        Out: {**state.trace, name: {"seconds": ..., **extra}}. Side effects: none. Node: every node.
        """
        return {**state.trace, name: {"seconds": time.perf_counter() - started, **extra}}

    def parse(state: SearchState) -> dict:
        """Read the question's search conditions once.

        In: state.question. Out: {"query": ParsedQuery, "trace"}. Side effects: none (rules only). Node: parse.
        """
        started = time.perf_counter()
        parsed = query_rules.parse_query(state.question)
        return {"query": parsed, "trace": timed("parse", state, started)}

    def condition(state: SearchState) -> dict:
        """Decide once the filter every list is searched under (the old retrieve node's first step).

        In: state.query (its filter is used only when COND_FILTER is on).
        Out: {"table_filter": the filter after relaxation (None when off or no condition), "relaxed": conditions
            dropped because no table met them (filters.RELAX_ORDER; years are never dropped), "trace" (with relaxed)}.
        Side effects: none (rules over the registry); ValueError for a condition without records, as before.
        Node: condition.
        """
        started = time.perf_counter()
        table_filter = state.query.to_filter() if s.cond_filter and state.query is not None else None
        relaxed: list[str] = []
        if table_filter is not None and not table_filter.is_empty():
            _, relaxed, _ = filter_with_relaxation(resources.records, table_filter)
            table_filter = relax(table_filter, relaxed)
        table_filter = condition_filter(table_filter, resources.records)  # hybrid_search's own step, kept as it was
        return {"table_filter": table_filter, "relaxed": relaxed,
                "trace": timed("condition", state, started, relaxed=relaxed)}

    def retrieve(name: str, retriever: Any):
        """The node that searches one list.

        In: name = the list name (rank_<name> metadata, runs key); retriever = that list's retriever.
        Out: the node function. Side effects: none when built. Node: none (it builds retrieve_<name>).
        """
        node = f"retrieve_{name}"

        def run(state: SearchState) -> dict:
            """Search this list under the shared filter.

            In: state.question, state.table_filter, state.query_vecs (this list's question vector from embed, if any:
                the search then uses it instead of calling the embedder).
            Out: {"runs": {name: ranked docs (at most pool)}, "run_removed": {name: key -> why filtered out}, "trace"}.
            Side effects: the retriever's search (Chroma query or BM25 scoring); ValueError as hybrid_search.
            Node: retrieve_dense, retrieve_bm25, retrieve_hq, retrieve_hq_bm25.
            """
            started = time.perf_counter()
            searcher = retriever
            vec = state.query_vecs.get(name)
            if vec is not None:  # embedded once by the embed node: this copy returns it instead of embedding again
                searcher = retriever.model_copy(update={"embed_query": lambda _question, v=vec: v})
            docs, removed = search_list(state.question, searcher, table_filter=state.table_filter,
                                        records=resources.records, k_each=s.pool)
            return {"runs": {name: docs}, "run_removed": {name: removed},
                    "trace": timed(node, state, started, n_docs=len(docs), n_removed=len(removed),
                                   top_keys=[d.metadata.get("key") for d in docs[:5]])}

        return node, run

    def embed(state: SearchState) -> dict:
        """Embed the question once for every list that embeds it (T2-39, review S4).

        In: state.question. Out: {"query_vecs": list name -> vector (lists sharing one embed_query function share one
            call and one vector), "trace" (calls = embedder calls made)}.
        Side effects: query embedder inference, one call per distinct embedder, in this one thread. Node: embed.
        """
        started = time.perf_counter()
        done: list[tuple[Any, list[float]]] = []  # (embed_query function, its vector)
        vecs: dict[str, list[float]] = {}
        for name, fn in embedders:
            vec = next((v for f, v in done if f is fn), None)
            if vec is None:
                vec = list(fn(state.question))
                done.append((fn, vec))
            vecs[name] = vec
        return {"query_vecs": vecs, "trace": timed("embed", state, started, calls=len(done), lists=list(vecs))}

    def fuse(state: SearchState) -> dict:
        """Fuse the lists by weighted RRF into the candidate pool (the old retrieve node's last step).

        In: state.runs, state.run_removed (one entry per retrieve_* node).
        Out: {"docs": the fused pool (at most `pool`; metadata as hybrid_search: score from dense, score_from,
            rrf_score, rank_<name>), "removed": key -> why filtered out (lists merged in fuse order), "trace"}.
        Side effects: none. Node: fuse.
        """
        started = time.perf_counter()
        names = [name for name, _, _ in lists]
        docs = rrf_fuse([state.runs.get(name, []) for name in names], names, k=s.pool, score_from=0,
                        weights=[w for _, _, w in lists])
        removed: dict[str, str] = {}
        for name in names:
            removed.update(state.run_removed.get(name, {}))
        return {"docs": docs, "removed": removed,
                "trace": timed("fuse", state, started, n_docs=len(docs), n_removed=len(removed),
                               lists=names, relaxed=state.relaxed)}

    def rerank1(state: SearchState) -> dict:
        """Re-order the top rerank_n by the stage 1 cross-encoder.

        In: state.question, state.docs. Out: {"docs": re-ordered with stage1_* metadata, "trace"}.
        Side effects: cross-encoder inference; ValueError for an unknown RERANK_MODE. Node: rerank1.
        """
        started = time.perf_counter()
        docs = rerank_stage1(state.question, state.docs, resources.stage1_score, n=s.rerank_n, mode=s.rerank_mode)
        return {"docs": docs, "trace": timed("rerank1", state, started)}

    def rerank2(state: SearchState) -> dict:
        """Let the stage 2 LLM pick among the top stage2_k.

        In: state.question, state.docs.
        Out: {"docs": picked first with stage2_* metadata, "trace": with the top doc's stage2_* fields}.
        Side effects: one LLM re-ranker call (CLOVA / HCX); its errors are stored in stage2_error. Node: rerank2.
        """
        started = time.perf_counter()
        docs = rerank_stage2(state.question, state.docs, resources.select, k_in=s.stage2_k,
                             name=resources.stage2_name)
        stage2 = {key: v for key, v in docs[0].metadata.items() if key.startswith("stage2")} if docs else {}
        return {"docs": docs, "trace": timed("rerank2", state, started, **stage2)}

    def gate(state: SearchState) -> dict:
        """Decide refuse or guide on the top k.

        In: state.docs. Out: {"route", "refusal_reason", "trace"}. Side effects: none. Node: gate.
        """
        started = time.perf_counter()
        route, reason = chain.decide_route(state.docs[:k])
        return {"route": route, "refusal_reason": reason, "trace": timed("gate", state, started, route=route)}

    def refuse(state: SearchState) -> dict:
        """Build the refusal answer on the top k.

        In: state.question, state.docs, state.refusal_reason. Out: {"result": SearchResult, "trace"}.
        Side effects: none (no LLM call). Node: refuse.
        """
        started = time.perf_counter()
        result = chain.refusal_answer(state.question, state.docs[:k], state.refusal_reason)
        return {"result": result, "trace": timed("refuse", state, started)}

    def guide(state: SearchState) -> dict:
        """Build the guide answer on the top k.

        In: state.question, state.docs. Out: {"result": SearchResult, "trace"}; under graph.stream with "custom" mode
            also {"delta": text} pieces of the reply that already pass the check (chain.streamable_prefix, T2-38).
        Side effects: one guide LLM call (failures come back as a status, then the code template). Node: guide.
        """
        started = time.perf_counter()
        try:  # graph.stream(stream_mode=[..., "custom"]) receives the pieces; invoke ignores them
            writer = get_stream_writer()
        except RuntimeError:  # run outside a graph (the node API's /nodes/answer)
            writer = None
        on_text = (lambda piece: writer({"delta": piece})) if writer else None
        result = chain.guide_answer(state.question, state.docs[:k], client, on_text=on_text)
        return {"result": result, "trace": timed("guide", state, started, origin=result.answer_origin)}

    # Lists in fuse order (dense first: its score stays in metadata["score"]): dense and BM25 always, the question
    # index when loaded (T2-39), with HQ_WEIGHT as its RRF weight
    lists = [("dense", resources.dense, 1.0), ("bm25", resources.bm25, 1.0)]
    if resources.hq is not None:
        lists.append(("hq", resources.hq, s.hq_weight))
    if resources.hq_bm25 is not None:
        lists.append(("hq_bm25", resources.hq_bm25, s.hq_weight))
    branches = [retrieve(name, retriever) for name, retriever, _ in lists]
    # Lists that embed the question themselves (Chroma dense, HqRetriever: an embed_query field). With two or more,
    # the embed node embeds once and hands each its vector; with one (HQ off) there is no embed node.
    embedders = [(name, retriever.embed_query) for name, retriever, _ in lists
                 if callable(getattr(retriever, "embed_query", None))]

    # Nodes: parse and condition, [embed], the lists (run in parallel by build_graph), fuse, the stages that are on
    nodes: dict[str, Node] = {"parse": parse, "condition": condition}
    if len(embedders) >= 2:
        nodes["embed"] = embed
    nodes.update(branches)
    nodes["fuse"] = fuse
    if resources.stage1_score is not None:
        nodes["rerank1"] = rerank1
    if resources.select is not None:
        nodes["rerank2"] = rerank2
    return {**nodes, "gate": gate, "refuse": refuse, "guide": guide}


RETRIEVAL_STEPS = ("condition", "embed", "fuse")  # with every retrieve_<list>: the pre-T2-39 retrieve node's parts
MERGED = ("runs", "run_removed", "trace")  # the SearchState fields with the merge reducer


def is_retrieval_step(name: str) -> bool:
    """True for a graph node that is part of the old `retrieve` node (condition, embed, retrieve_<list>, fuse)."""
    return name in RETRIEVAL_STEPS or name.startswith("retrieve_")


def apply_update(state: SearchState, update: dict) -> SearchState:
    """The state after one node's update, the way the graph applies it (merge on runs, run_removed and trace)."""
    merged = {f: merge(getattr(state, f), update[f]) for f in MERGED if f in update}
    return state.model_copy(update={**update, **merged})


def make_nodes(resources: SearchResources, client: LLMClient | BaseChatModel, *, k: int = 5) -> dict[str, Node]:
    """The node functions the node API exposes, in run order (T2-37): parse, retrieve, [rerank1], [rerank2], gate,
    refuse, guide.

    In: as build_graph.
    Out: name -> fn(SearchState) -> state update. `retrieve` runs the graph's retrieval steps (condition, [embed],
        retrieve_<list>..., fuse; T2-39) one after another in the calling thread and writes what the pre-T2-39
        retrieve node wrote (docs, removed, relaxed) plus the steps' own fields (table_filter, runs, run_removed;
        query_vecs is left empty, the vectors are not sent back) and a trace entry per step and one for `retrieve`.
        It ignores those fields in the incoming state, so a client cannot hand it a list or a vector. The lists are
        fused in fuse's fixed order, so the chain parse -> retrieve -> ... -> answer equals run_search. The other
        nodes are graph_nodes' own.
    Side effects: none when made; calling a node does what its docstring says. Node: none (it makes them).
    """
    nodes = graph_nodes(resources, client, k=k)
    steps = [name for name in nodes if is_retrieval_step(name)]

    def retrieve(state: SearchState) -> dict:
        """condition -> [embed] -> retrieve_<list>... -> fuse as one step (the node API's /nodes/retrieve).

        In: state.question, state.query. Out: {"docs", "removed", "relaxed", "table_filter", "runs", "run_removed",
            "query_vecs" (empty), "trace" (each step's entry and "retrieve")}.
        Side effects: as the steps (Chroma query, BM25 scoring, query embedding). Node: none (the API's retrieve).
        """
        started = time.perf_counter()
        cur = state.model_copy(update={"table_filter": None, "query_vecs": {}, "relaxed": [], "runs": {},
                                       "run_removed": {}, "removed": {}, "docs": []})
        for name in steps:
            cur = apply_update(cur, nodes[name](cur))
        entry = {"seconds": time.perf_counter() - started, "n_docs": len(cur.docs), "n_removed": len(cur.removed),
                 "relaxed": cur.relaxed, "steps": steps}
        return {"docs": cur.docs, "removed": cur.removed, "relaxed": cur.relaxed, "table_filter": cur.table_filter,
                "runs": cur.runs, "run_removed": cur.run_removed, "query_vecs": {},
                "trace": {**cur.trace, "retrieve": entry}}

    api: dict[str, Node] = {"parse": nodes["parse"], "retrieve": retrieve}
    return {**api, **{name: fn for name, fn in nodes.items() if name != "parse" and not is_retrieval_step(name)}}


def build_graph(resources: SearchResources, client: LLMClient | BaseChatModel, *, k: int = 5):
    """Compile the search graph for these resources. Returns the compiled graph (call .invoke({"question": q})).

    In: resources = load_resources(...) (records, dense, BM25, stage 1 scorer, stage 2 selector, settings);
        client = the guide LLM (ChatClovaX, or a Phase 1 client such as MockClient); k = tables shown to the answer
        (gate / refuse / guide see docs[:k]; the search stages see the whole pool), same default as chain.answer.
    Out: a compiled LangGraph graph over SearchState with graph_nodes' nodes: retrieve_hq is added only when
        resources.hq is set, retrieve_hq_bm25 only when resources.hq_bm25 is set (T2-39, HQ_BUILD / HQ_BM25), embed
        only when two or more lists embed the question (an embed_query field: dense and hq), rerank1 only when
        resources.stage1_score is set, rerank2 only when resources.select is set.
    Side effects: none when built; running it calls the retrievers, the models and the LLMs the resources hold.
    Node: none (it builds them).
    """
    nodes = graph_nodes(resources, client, k=k)
    g = StateGraph(SearchState)
    for name, fn in nodes.items():
        g.add_node(name, fn)

    # Edges: parse -> condition [-> embed] -> every list at once -> fuse (waits for all), a line through the stages,
    # one branch at the gate
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
    g.add_conditional_edges("gate", lambda state: state.route, {"refuse": "refuse", "guide": "guide"})
    g.add_edge("refuse", END)
    g.add_edge("guide", END)
    return g.compile()


def run_search(graph, question: str, config: dict | None = None) -> SearchState:
    """Invoke the graph and return the final state as SearchState.

    In: graph = build_graph(...); question = the user question; config = LangChain run config (callbacks, run name),
        passed through as is.
    Out: the final SearchState (result set by refuse or guide; docs = the whole ranked pool; runs = each list).
    Side effects: runs every node (retrievers, models, LLMs held by the graph's resources).
    Node: none (it runs them).
    """
    return SearchState.model_validate(graph.invoke({"question": question}, config=config))


def stream_search(graph, question: str, config: dict | None = None) -> Iterator[tuple[str, Any]]:
    """Run the graph and yield each node as it finishes, then the final state (T2-37, the SSE search).

    In: as run_search.
    Out: ("<node>", that node's own trace entry) per finished node in run order (trace is cumulative, so only the
        node's key is sent; the parallel retrieve_<list> nodes come in the order they finish, T2-39),
        ("token", {"delta": text}) while guide streams its checked reply (T2-38), then
        ("result", the final SearchState) — the same state run_search returns.
    Side effects: as run_search. Node: none (it runs them).
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


def ground_docs(docs: list[Document], records: dict[str, TableRecord]) -> list[Document]:
    """Re-attach the catalog fields of docs sent from outside (the node API) from the server's records (T2-37).

    In: docs = documents from a request; records = key -> TableRecord (resources.records).
    Out: the same docs in the same order with id and the catalog metadata (to_document's: key, ids, names, period,
        url) taken from the record; search metadata (score, ranks, stage*) and page_content are kept, since
        the nodes read text and scores by key and pass page_content through. So a client may reorder or re-score
        tables but cannot invent a table or its citation.
    Side effects: raises KeyError listing the keys that are not in records. Node: none (the API calls it).
    """
    keys = [d.metadata.get("key") or d.id for d in docs]
    unknown = [key for key in keys if key not in records]
    if unknown:
        raise KeyError(f"unknown table keys: {unknown}")
    out = []
    for d, key in zip(docs, keys):
        catalog = to_document(records[key].info).metadata  # the catalog fields only
        out.append(Document(page_content=d.page_content, id=key, metadata={**d.metadata, **catalog}))
    return out
