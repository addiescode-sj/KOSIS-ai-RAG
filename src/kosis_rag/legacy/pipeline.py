"""The old staged retriever (T2-06, T2-07) over loaded resources (T2-41 S4: from pipeline.build_search, which took
the old settings object and loaded the resources itself).

    HybridRetriever (dense + BM25, RRF, T2-14 condition) -> [CrossEncoderReranker] -> [LLMReranker]

The question index (T2-39) is graph-only: build_search never uses resources.hq / hq_bm25, so this path searches as
it did before T2-39. To keep it from being loaded at all, build the resources with IndexSpec(hq_build=None).
"""

from __future__ import annotations

from langchain_core.retrievers import BaseRetriever

from kosis_rag.legacy.rerankers import CrossEncoderReranker, LLMReranker
from kosis_rag.legacy.retrievers import HybridRetriever
from kosis_rag.parse.node import parse_query
from kosis_rag.schemas.resources import SearchResources


def build_search(resources: SearchResources) -> BaseRetriever:
    """The nested staged retriever over `resources` (adapters.loader.load_resources(spec=..., options=...)).

    In: resources. Out: HybridRetriever (with the T2-14 condition when options.cond_filter is on), wrapped by
        CrossEncoderReranker when stage 1 is on and by LLMReranker when stage 2 is on.
    Side effects: none when built (the resources are already loaded). Node: not used by the graph.
    """
    r, o = resources, resources.options
    retriever: BaseRetriever = HybridRetriever(
        retrievers=[r.dense, r.bm25], names=["dense", "bm25"], k=o.pool, k_each=o.pool, score_from=0,
        records=r.records, condition=(lambda q: parse_query(q, sigungu=r.sigungu).to_filter()) if o.cond_filter else None)
    if r.stage1_score is not None:
        retriever = CrossEncoderReranker(base=retriever, score=r.stage1_score, n=o.rerank_n, mode=o.rerank_mode)
    if r.select is not None:
        retriever = LLMReranker(base=retriever, select=r.select, k_in=o.stage2_k, name=r.stage2_name)
    return retriever
