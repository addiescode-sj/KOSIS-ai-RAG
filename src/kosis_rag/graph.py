"""The staged table search as a LangGraph StateGraph (T2-10): one node per stage over one shared state.

    START -> parse -> retrieve -> [rerank1] -> [rerank2] -> gate -> refuse | guide -> END

Each node calls the same function the old nested retrievers call (retriever.hybrid_search, rerank.rerank_stage1 /
rerank_stage2, chain.decide_route / refusal_answer / guide_answer), so the graph and `chain.answer(build_search(...))`
give the same result. Stages that are off are left out when the graph is built; only `gate` branches at run time.
The question is parsed once (parse) and its filter handed to retrieve, where the old path parsed inside the hybrid.

LangGraph is imported only here: the old path works without it.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from kosis_chat.llm.model import LLMClient

from kosis_rag import chain
from kosis_rag import query as query_rules  # module access, so a test can patch query.parse_query
from kosis_rag.filters import filter_with_relaxation, relax
from kosis_rag.pipeline import SearchResources
from kosis_rag.query import ParsedQuery
from kosis_rag.rerank import rerank_stage1, rerank_stage2
from kosis_rag.retriever import hybrid_search
from kosis_rag.schemas import SearchResult


class SearchState(BaseModel):
    """What the nodes share. Every field has one writer, except docs (overwritten by each search stage) and trace."""

    question: str
    query: ParsedQuery | None = None  # set by parse
    removed: dict[str, str] = Field(default_factory=dict)  # set by retrieve: key -> why filtered out
    relaxed: list[str] = Field(default_factory=list)  # set by retrieve: conditions dropped so a table could pass (T2-29)
    docs: list[Document] = Field(default_factory=list)  # overwritten by retrieve, rerank1, rerank2
    route: Literal["refuse", "guide"] | None = None  # set by gate
    refusal_reason: str | None = None  # set by gate
    result: SearchResult | None = None  # set by refuse or guide
    trace: dict[str, dict[str, Any]] = Field(default_factory=dict)  # node name -> {"seconds": float, ...}


def build_graph(resources: SearchResources, client: LLMClient | BaseChatModel, *, k: int = 5):
    """Compile the search graph for these resources. Returns the compiled graph (call .invoke({"question": q})).

    In: resources = load_resources(...) (records, dense, BM25, stage 1 scorer, stage 2 selector, settings);
        client = the guide LLM (ChatClovaX, or a Phase 1 client such as MockClient); k = tables shown to the answer
        (gate / refuse / guide see docs[:k]; the search stages see the whole pool), same default as chain.answer.
    Out: a compiled LangGraph graph over SearchState. rerank1 is added only when resources.stage1_score is set,
        rerank2 only when resources.select is set.
    Side effects: none when built; running it calls the retrievers, the models and the LLMs the resources hold.
    Node: none (it builds them).
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

    def retrieve(state: SearchState) -> dict:
        """Search dense + BM25 under the question's conditions and fuse them by RRF.

        In: state.question, state.query (its filter is used only when COND_FILTER is on).
        Out: {"docs": the fused pool (at most `pool`), "removed": key -> why filtered out, "relaxed": conditions
            dropped because no table met them (filters.RELAX_ORDER; years are never dropped), "trace" (with relaxed)}.
        Side effects: Chroma query and BM25 scoring; ValueError from hybrid_search as in the old path. Node: retrieve.
        """
        started = time.perf_counter()
        table_filter = state.query.to_filter() if s.cond_filter and state.query is not None else None
        relaxed: list[str] = []
        if table_filter is not None and not table_filter.is_empty():
            _, relaxed, _ = filter_with_relaxation(resources.records, table_filter)
            table_filter = relax(table_filter, relaxed)
        docs, removed = hybrid_search(
            state.question, [resources.dense, resources.bm25], names=["dense", "bm25"], table_filter=table_filter,
            records=resources.records, k=s.pool, k_each=s.pool, score_from=0)
        return {"docs": docs, "removed": removed, "relaxed": relaxed,
                "trace": timed("retrieve", state, started, n_docs=len(docs), n_removed=len(removed), relaxed=relaxed)}

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

        In: state.question, state.docs. Out: {"result": SearchResult, "trace"}.
        Side effects: one guide LLM call (failures come back as a status, then the code template). Node: guide.
        """
        started = time.perf_counter()
        result = chain.guide_answer(state.question, state.docs[:k], client)
        return {"result": result, "trace": timed("guide", state, started, origin=result.answer_origin)}

    # Nodes: the search stages that are on, then the answer
    g = StateGraph(SearchState)
    stages = [("parse", parse), ("retrieve", retrieve)]
    if resources.stage1_score is not None:
        stages.append(("rerank1", rerank1))
    if resources.select is not None:
        stages.append(("rerank2", rerank2))
    for name, fn in [*stages, ("gate", gate), ("refuse", refuse), ("guide", guide)]:
        g.add_node(name, fn)

    # Edges: a straight line through the search stages, one branch at the gate
    g.add_edge(START, "parse")
    for (a, _), (b, _) in zip(stages, stages[1:]):
        g.add_edge(a, b)
    g.add_edge(stages[-1][0], "gate")
    g.add_conditional_edges("gate", lambda state: state.route, {"refuse": "refuse", "guide": "guide"})
    g.add_edge("refuse", END)
    g.add_edge("guide", END)
    return g.compile()


def run_search(graph, question: str, config: dict | None = None) -> SearchState:
    """Invoke the graph and return the final state as SearchState.

    In: graph = build_graph(...); question = the user question; config = LangChain run config (callbacks, run name),
        passed through as is.
    Out: the final SearchState (result set by refuse or guide; docs = the whole ranked pool).
    Side effects: runs every node (retrievers, models, LLMs held by the graph's resources).
    Node: none (it runs them).
    """
    return SearchState.model_validate(graph.invoke({"question": question}, config=config))
