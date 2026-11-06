"""Stage 1 / stage 2 re-rankers as LangChain BaseRetrievers (T2-07; T2-41 S3, P2-D18: from rerank.py).

Each wraps the retriever before it, so chain.search / chain.answer / the eval runners take the last one like any
retriever (build_search). Their bodies call the stage functions rerank.stage1.rerank_stage1 and
rerank.stage2.rerank_stage2; the graph calls those directly (rerank.node).
"""

from __future__ import annotations

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from kosis_rag.rerank.stage1 import rerank_stage1
from kosis_rag.rerank.stage2 import Score, Select, rerank_stage2
from kosis_rag.shared.constants import RRF_C


class CrossEncoderReranker(BaseRetriever):
    """Stage 1: re-orders the base top n by a cross-encoder; the rest of the base list follows unchanged.

    Retriever-shaped wrapper for the old path: invokes `base`, then rerank_stage1. Side effects: calls `base` and
    `score`; raises ValueError for an unknown mode before calling `base`. Node: not used by the graph.
    """

    base: BaseRetriever
    score: Score
    n: int = 50
    mode: str = "blend"  # blend | score
    c: int = RRF_C
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        """Base candidates re-ordered by rerank_stage1. Node: not used by the graph."""
        if self.mode not in ("blend", "score"):
            raise ValueError(f"unknown stage 1 mode {self.mode!r}; known: blend, score")
        docs = self.base.invoke(query)
        return rerank_stage1(query, docs, self.score, n=self.n, mode=self.mode, c=self.c)


class LLMReranker(BaseRetriever):
    """Stage 2: an LLM picks among the base top k_in; picked tables go first in the base order, then the rest.

    Retriever-shaped wrapper for the old path: invokes `base`, then rerank_stage2. Side effects: calls `base` and
    `select` (selector errors are swallowed into stage2_error). Node: not used by the graph.
    """

    base: BaseRetriever
    select: Select
    k_in: int = 20
    name: str = "llm"
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        """Base candidates re-ordered by rerank_stage2. Node: not used by the graph."""
        docs = self.base.invoke(query)
        return rerank_stage2(query, docs, self.select, k_in=self.k_in, name=self.name)
