"""The node API's node set (T2-37; T2-41 S4: from graph.py): make_nodes, apply_update, ground_docs.

backend.app.main exposes one endpoint per node. Its `retrieve` runs the graph's retrieval steps (condition, [embed],
retrieve_<list>..., fuse) as one node, so a client's chain parse -> retrieve -> ... -> gate equals run_search.
T2-41 S5: the set ends at gate; the backend adds its answer nodes (refuse | guide, backend.app.respond).
Node endpoints are stateless: apply_update gives the state after a node the way the graph applies it, and
ground_docs re-attaches tables sent from outside to the server's records.
"""

from __future__ import annotations

import time

from langchain_core.documents import Document

from kosis_rag.graph.search import Node, graph_nodes
from kosis_rag.schemas.catalog import TableRecord
from kosis_rag.schemas.resources import SearchResources
from kosis_rag.schemas.state import SearchState, merge
from kosis_rag.shared.catalog import to_document

RETRIEVAL_STEPS = ("condition", "embed", "fuse")  # with every retrieve_<list>: the pre-T2-39 retrieve node's parts
MERGED = ("runs", "run_removed", "trace")  # the SearchState fields with the merge reducer


def is_retrieval_step(name: str) -> bool:
    """True for a graph node that is part of the old `retrieve` node (condition, embed, retrieve_<list>, fuse)."""
    return name in RETRIEVAL_STEPS or name.startswith("retrieve_")


def apply_update(state: SearchState, update: dict) -> SearchState:
    """The state after one node's update, the way the graph applies it (merge on runs, run_removed and trace)."""
    merged = {f: merge(getattr(state, f), update[f]) for f in MERGED if f in update}
    return state.model_copy(update={**update, **merged})


def make_nodes(resources: SearchResources, *, k: int = 5) -> dict[str, Node]:
    """The search node functions the node API exposes, in run order (T2-37): parse, retrieve, [rerank1], [rerank2],
    gate (T2-41 S5: the answer nodes refuse / guide are the backend's).

    In: as build_graph.
    Out: name -> fn(SearchState) -> state update. `retrieve` runs the graph's retrieval steps (condition, [embed],
        retrieve_<list>..., fuse; T2-39) one after another in the calling thread and writes what the pre-T2-39
        retrieve node wrote (docs, removed, relaxed) plus the steps' own fields (table_filter, runs, run_removed;
        query_vecs is left empty, the vectors are not sent back) and a trace entry per step and one for `retrieve`.
        It ignores those fields in the incoming state, so a client cannot hand it a list or a vector. The lists are
        fused in fuse's fixed order, so the chain parse -> retrieve -> ... -> gate equals run_search. The other
        nodes are graph_nodes' own.
    Side effects: none when made; calling a node does what its docstring says. Node: none (it makes them).
    """
    nodes = graph_nodes(resources, k=k)
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
