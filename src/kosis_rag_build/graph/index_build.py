"""The offline index builds as LangGraph graphs (T2-41 S6a, P2-D3): what the old CLIs did in one function each.

    build_tables_graph()       TablesBuildState     START -> read_parquet -> build_records -> write_jsonl -> END
    build_sigungu_graph()      SigunguBuildState    START -> read_parquet -> build_sigungu -> write_json -> END
    build_chroma_graph(...)    IndexBuildState      START -> read_records -> build_docs -> embed -> load | write_build -> END
                                                    (write_build when state.export is set: a .npz instead of Chroma)
    build_questions_graph(...) QuestionsIndexState  START -> read_questions -> [embed -> load] -> END
                                                    (state.dry_run ends after read_questions)

The node bodies are in tables.py, sigungu.py, chroma.py and hq/questions_index.py; the model and Chroma client are
arguments (None: opened from the state when the node runs), so tests pass fakes. Run: graph.invoke(State(...)).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph

from kosis_rag_build import chroma, sigungu, tables
from kosis_rag_build.hq import questions_index
from kosis_rag_build.schemas import IndexBuildState, QuestionsIndexState, SigunguBuildState, TablesBuildState


def _line(state_type: type, steps: list[tuple[str, Callable]]):
    """START -> steps in order -> END over state_type, compiled."""
    g = StateGraph(state_type)
    for name, fn in steps:
        g.add_node(name, fn)
    g.add_edge(START, steps[0][0])
    for (a, _), (b, _) in zip(steps, steps[1:]):
        g.add_edge(a, b)
    g.add_edge(steps[-1][0], END)
    return g.compile()


def build_tables_graph():
    """data/tables.jsonl from the provided data. invoke(TablesBuildState(data_dir=, out=)).
    Side effects when run: reads the provided data, writes `out`."""
    return _line(TablesBuildState, [("read_parquet", tables.read_parquet), ("build_records", tables.build_records_node),
                                    ("write_jsonl", tables.write_jsonl)])


def build_sigungu_graph():
    """data/sigungu.json from one 읍면동 table. invoke(SigunguBuildState(parquet=, out=)).
    Side effects when run: reads the Parquet file, writes `out`."""
    return _line(SigunguBuildState, [("read_parquet", sigungu.read_parquet),
                                     ("build_sigungu", sigungu.build_sigungu_node), ("write_json", sigungu.write_json)])


def build_chroma_graph(*, embedder: Any = None, client: Any = None, log: Callable[[str], None] = print):
    """The table index: invoke(IndexBuildState(tables=, model=, ..., index_version=)).

    In: embedder = chroma.DocEmbedder, a function of the state giving one, or None (chroma.load_doc_embedder);
        client = a Chroma client for load (None: adapters.chroma.chroma_client(state.chroma)); log = progress lines.
    Side effects when run: loads the model, embeds every table, writes the collection (or the .npz of state.export).
    """
    g = StateGraph(IndexBuildState)
    g.add_node("read_records", chroma.read_records)
    g.add_node("build_docs", chroma.make_build_docs(log))
    g.add_node("embed", chroma.make_embed(embedder, log))
    g.add_node("load", chroma.make_load(client))
    g.add_node("write_build", chroma.write_build_node)
    g.add_edge(START, "read_records")
    g.add_edge("read_records", "build_docs")
    g.add_edge("build_docs", "embed")
    g.add_conditional_edges("embed", lambda s: "write_build" if s.export else "load",
                            {"write_build": "write_build", "load": "load"})
    g.add_edge("load", END)
    g.add_edge("write_build", END)
    return g.compile()


def build_questions_graph(*, embedder: Any = None, client: Any = None):
    """The question index of an hq build: invoke(QuestionsIndexState(build=, ckpt=, ..., index_version=)).

    In: embedder = models.Embedder or a fake with embed_documents / name / ckpt (None: models.load_embedder);
        client = a Chroma client (None: adapters.chroma.chroma_client(state.chroma)).
    Side effects when run: reads the build, embeds its questions, writes the collection (a dry run: reads only).
    """
    g = StateGraph(QuestionsIndexState)
    g.add_node("read_questions", questions_index.read_questions)
    g.add_node("embed", questions_index.make_embed(embedder))
    g.add_node("load", questions_index.make_load(client))
    g.add_edge(START, "read_questions")
    g.add_conditional_edges("read_questions", lambda s: END if s.dry_run else "embed", {"embed": "embed", END: END})
    g.add_edge("embed", "load")
    g.add_edge("load", END)
    return g.compile()
