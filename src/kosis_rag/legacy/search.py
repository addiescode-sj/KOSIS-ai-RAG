"""The old retriever-shaped search call (T2-41 S5: from chain.search): a retriever's top k as TableCandidates."""

from __future__ import annotations

from langchain_core.retrievers import BaseRetriever

from kosis_rag.schemas.io import TableCandidate
from kosis_rag.shared.catalog import to_candidate


def search(retriever: BaseRetriever, question: str, k: int = 5) -> list[TableCandidate]:
    """The top k tables a retriever gives for the question.

    In: retriever = a staged search (legacy.pipeline.build_search) or any table retriever; question; k.
    Out: to_candidate of the first k documents, in rank order. Side effects: runs the retriever. Node: none (the
    search_tables tool and the eval harness call it).
    """
    return [to_candidate(d) for d in retriever.invoke(question)[:k]]
