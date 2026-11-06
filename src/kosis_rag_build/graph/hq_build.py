"""Offline question generation as LangGraph graphs (T2-39, plans/T2-39.md Idea 1 and 3; T2-41 S6a: the graph part of the
old hq/graph.py — the node bodies are hq/nodes.py).

Per table (build_table_graph, state HqState):

    START -> load -> prompt -> generate -> validate -> rule_check -> [score] -> END
                                  ^            |  (invalid reply, first time: one re-ask)
                                  +------------+
    generate (API failure) or validate (still invalid) -> END with error

Over all tables (build_build_graph, state HqBuildState):

    START -> generate_all (the table graph per table, thread pool) -> filter_all -> write -> END

Why a thread pool instead of a LangGraph `Send` per table, and how the cache and the filter work: hq/nodes.py.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from kosis_rag.schemas.catalog import TableRecord
from kosis_rag.shared import tabletext as text
from kosis_rag_build.hq.constants import CONCURRENCY, FILTER_FRAC, MIN_KEEP
from kosis_rag_build.hq.nodes import CACHE_DIR, HQ_DIR, PairScore, build_nodes, table_nodes
from kosis_rag_build.schemas import HqBuildState, HqState


def _after_generate(state: HqState) -> str:
    return END if state.error else "validate"


def _after_validate(state: HqState) -> str:
    if state.error:
        return END
    return "rule_check" if state.questions else "generate"


def build_table_graph(records: dict[str, TableRecord], client: Any, *, model: str, cache_dir: Path = CACHE_DIR,
                      scorer: PairScore | None = None, fmt: str = text.DOC_FORMAT):
    """Compile the per-table generation graph.

    In: records = key -> TableRecord; client = an object with generate(system, user, history) returning
        .status/.text/.usage (llm.openai.OpenAIClient; tests pass a fake); model = the model name the cache is kept under;
        cache_dir = cache root; scorer = (key, questions) -> scores (None: no score node); fmt = table text format.
    Out: a compiled graph over HqState; invoke({"key": key}).
    Side effects: none when built; running it calls the LLM (cache misses only), writes the cache, runs the scorer.
    Node: none (it builds them: hq.nodes.table_nodes).
    """
    nodes = table_nodes(records, client, model=model, cache_dir=cache_dir, scorer=scorer, fmt=fmt)
    g = StateGraph(HqState)
    for name, fn in nodes.items():
        g.add_node(name, fn)
    g.add_edge(START, "load")
    g.add_edge("load", "prompt")
    g.add_edge("prompt", "generate")
    g.add_conditional_edges("generate", _after_generate, {"validate": "validate", END: END})
    g.add_conditional_edges("validate", _after_validate, {"rule_check": "rule_check", "generate": "generate", END: END})
    if "score" in nodes:
        g.add_edge("rule_check", "score")
        g.add_edge("score", END)
    else:
        g.add_edge("rule_check", END)
    return g.compile()


def build_build_graph(table_graph, *, out_root: Path = HQ_DIR, concurrency: int = CONCURRENCY,
                      log: Callable[[str], None] = print):
    """Compile the whole-build graph around a compiled table graph.

    In: table_graph = build_table_graph(...); out_root = data/hq; concurrency = worker threads (API concurrency);
        log = progress output. Out: a compiled graph over HqBuildState; invoke({"keys", "tables_sha", "model", ...}).
    Side effects: none when built; running it runs every table graph and writes data/hq/<build_id>/.
    Node: none (it builds them: hq.nodes.build_nodes).
    """
    g = StateGraph(HqBuildState)
    for name, fn in build_nodes(table_graph, out_root=out_root, concurrency=concurrency, log=log).items():
        g.add_node(name, fn)
    g.add_edge(START, "generate_all")
    g.add_edge("generate_all", "filter_all")
    g.add_edge("filter_all", "write")
    g.add_edge("write", END)
    return g.compile()


def run_build(records: list[TableRecord], client: Any, *, model: str, tables_sha256: str,
              limit: int | None = None, frac: float = FILTER_FRAC, min_keep: int = MIN_KEEP,
              scorer: PairScore | None = None, cache_dir: Path = CACHE_DIR, out_root: Path = HQ_DIR,
              concurrency: int = CONCURRENCY, fmt: str = text.DOC_FORMAT,
              log: Callable[[str], None] = print) -> HqBuildState:
    """Build the question set for the first `limit` tables (all when None).

    In: records = tables.jsonl in file order; client, model, scorer, cache_dir, fmt as build_table_graph;
        tables_sha256 = sha256 of tables.jsonl; frac, min_keep as hq.nodes.filter_scores (frac is set to 0 without a
        scorer); out_root = data/hq; concurrency = worker threads; log = progress output.
    Out: the final HqBuildState (build_id, out_dir, totals, results, kept).
    Side effects: those of both graphs (LLM calls on cache misses, cache and build files).
    Node: none (it runs the graphs).
    """
    by_key = {r.info.key: r for r in records}
    keys = [r.info.key for r in records][: limit or None]
    table_graph = build_table_graph(by_key, client, model=model, cache_dir=cache_dir, scorer=scorer, fmt=fmt)
    graph = build_build_graph(table_graph, out_root=out_root, concurrency=concurrency, log=log)
    frac = frac if scorer is not None else 0.0
    return HqBuildState.model_validate(graph.invoke({"keys": keys, "tables_sha": tables_sha256, "model": model,
                                                     "frac": frac, "min_keep": min_keep, "limit": limit}))
