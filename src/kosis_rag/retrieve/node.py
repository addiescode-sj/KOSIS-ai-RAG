"""Retrieve stage step functions (T2-39; T2-41 S2: the bodies of graph.py's embed, retrieve_<list> and fuse nodes).

`resources` is the search resources (schemas.resources.SearchResources): the lists are its dense, bm25, [hq] and
[hq_bm25]; options.pool is each list's cut and the pool size, options.hq_weight the question lists' RRF weight.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kosis_rag.retrieve.fuse import rrf_fuse
from kosis_rag.retrieve.lists import search_list
from kosis_rag.retrieve.schemas import EmbedInput, EmbedOutput, FuseInput, FuseOutput, ListInput, ListOutput


def search_lists(resources: Any) -> list[tuple[str, Any, float]]:
    """The lists in fuse order with their RRF weights (T2-39).

    In: resources. Out: [(name, list, weight)]: dense and bm25 (1.0) always (dense first: its score stays in
        metadata["score"]), hq when resources.hq is set and hq_bm25 when resources.hq_bm25 is set (weight
        options.hq_weight). Side effects: none. Node: none (graph_nodes builds one retrieve_<name> node per list).
    """
    s = resources.options
    lists = [("dense", resources.dense, 1.0), ("bm25", resources.bm25, 1.0)]
    if resources.hq is not None:
        lists.append(("hq", resources.hq, s.hq_weight))
    if resources.hq_bm25 is not None:
        lists.append(("hq_bm25", resources.hq_bm25, s.hq_weight))
    return lists


def embedders(resources: Any) -> list[tuple[str, Callable[[str], list[float]]]]:
    """Lists that embed the question themselves (an embed_query field: dense, hq), as (name, embed_query).

    Side effects: none. Node: none (graph_nodes adds the embed node when there are two or more).
    """
    return [(name, retriever.embed_query) for name, retriever, _ in search_lists(resources)
            if callable(getattr(retriever, "embed_query", None))]


def embed(inp: EmbedInput, resources: Any) -> EmbedOutput:
    """Embed the question once for every list that embeds it (T2-39, review S4).

    In: inp.question. Out: EmbedOutput(query_vecs = list name -> vector (lists sharing one embed_query function share
        one call and one vector), calls = embedder calls made).
    Side effects: query embedder inference, one call per distinct embedder, in this one thread. Node: embed.
    """
    done: list[tuple[Any, list[float]]] = []  # (embed_query function, its vector)
    vecs: dict[str, list[float]] = {}
    for name, fn in embedders(resources):
        vec = next((v for f, v in done if f is fn), None)
        if vec is None:
            vec = list(fn(inp.question))
            done.append((fn, vec))
        vecs[name] = vec
    return EmbedOutput(query_vecs=vecs, calls=len(done))


def retrieve_list(name: str, inp: ListInput, resources: Any) -> ListOutput:
    """Search one list under the shared filter.

    In: name = the list name (dense, bm25, hq, hq_bm25); inp.question, inp.table_filter (from condition),
        inp.vector (this list's question vector from embed, if any: the search uses it instead of the embedder).
    Out: ListOutput(docs = ranked docs (at most options.pool), removed = key -> why filtered out, relaxed, missing).
    Side effects: the list's search (Chroma query or BM25 scoring); ValueError as search_list; KeyError for a name
        that is not one of search_lists(resources). Node: retrieve_dense, retrieve_bm25, retrieve_hq, retrieve_hq_bm25.
    """
    retriever = {n: r for n, r, _ in search_lists(resources)}[name]
    docs, removed, relaxed, missing = search_list(inp.question, retriever, table_filter=inp.table_filter,
                                                  records=resources.records, k_each=resources.options.pool,
                                                  vector=inp.vector)
    return ListOutput(docs=docs, removed=removed, relaxed=relaxed, missing=missing)


def fuse(inp: FuseInput, resources: Any) -> FuseOutput:
    """Fuse the lists by weighted RRF into the candidate pool (the old retrieve node's last step).

    In: inp.runs, inp.run_removed (one entry per retrieve_* node; a list with no entry counts as empty).
    Out: FuseOutput(docs = the fused pool (at most options.pool; metadata as hybrid_search: score from dense,
        score_from, rrf_score, rank_<name>), removed = key -> why filtered out (lists merged in fuse order)).
    Side effects: none. Node: fuse.
    """
    lists = search_lists(resources)
    names = [name for name, _, _ in lists]
    docs = rrf_fuse([inp.runs.get(name, []) for name in names], names, k=resources.options.pool, score_from=0,
                    weights=[w for _, _, w in lists])
    removed: dict[str, str] = {}
    for name in names:
        removed.update(inp.run_removed.get(name, {}))
    return FuseOutput(docs=docs, removed=removed)
