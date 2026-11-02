"""Two re-ranking stages after the hybrid search (T2-07). Each is a LangChain BaseRetriever wrapping the one before,
so chain.search / chain.answer / the eval runners take the last one like any retriever.

    hybrid top 50 (T2-06)
      -> CrossEncoderReranker (stage 1): fine-tuned bge-reranker-v2-m3 scores (question, table passage) for the top n.
         mode "blend" = RRF of the re-ranker order and the input order (lab plan 43 dev choice), "score" = re-ranker only.
      -> LLMReranker (stage 2): an LLM re-ranker sees the top k_in table keys and returns the ones it picks; picked
         tables go first in their stage-1 order, then the rest. Keys it invents are dropped (table ids come from the
         input only). A failed call keeps the stage-1 order and records the error.

Stage 2 is pluggable: LLM_RERANKERS maps a name to a factory (text_of, options) -> select(question, keys), where select
returns the picked keys (a list, or an object with .keys, .calls and .usage). "clova" is the first entry; another LLM
re-ranker is one more entry with the same signature.

metadata["score"] is left as the hybrid gave it (the dense cosine the no-answer threshold is calibrated on, T2-09).
The stages add their own fields: input_rank, stage1_score, stage1_rank, stage1_seconds; stage2, stage2_picked,
stage2_picked_n (0 = the LLM picked nothing, a no-answer signal for T2-09), stage2_error, stage2_calls,
stage2_input_tokens, stage2_output_tokens, stage2_seconds.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

RRF_C = 60

Score = Callable[[str, list[str]], list[float]]  # (question, table keys) -> one score per key
Select = Callable[[str, list[str]], Any]  # (question, table keys) -> picked keys, or an object with .keys


def blend_order(scores: list[float], c: int = RRF_C) -> list[int]:
    """Input positions in blended order: 1/(c + re-ranker rank) + 1/(c + input rank); ties keep the input order."""
    by_score = sorted(range(len(scores)), key=lambda i: -scores[i])
    rerank_rank = {i: r for r, i in enumerate(by_score, 1)}
    return sorted(range(len(scores)), key=lambda i: -(1 / (c + rerank_rank[i]) + 1 / (c + i + 1)))


def passage_scorer(scorer: Any, text_of: dict[str, str], query_text: Callable[[str], str] | None = None) -> Score:
    """Score function over table keys from a model.CrossEncoderScorer and the n2 passages it was trained on."""

    def score(question: str, keys: list[str]) -> list[float]:
        q = query_text(question) if query_text else question
        return scorer.score(q, [text_of[k] for k in keys])

    return score


class CrossEncoderReranker(BaseRetriever):
    """Stage 1: re-orders the base top n by a cross-encoder; the rest of the base list follows unchanged."""

    base: BaseRetriever
    score: Score
    n: int = 50
    mode: str = "blend"  # blend | score
    c: int = RRF_C
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        if self.mode not in ("blend", "score"):
            raise ValueError(f"unknown stage 1 mode {self.mode!r}; known: blend, score")
        docs = self.base.invoke(query)
        head, tail = docs[: self.n], docs[self.n:]
        if not head:
            return docs
        started = time.perf_counter()
        sc = self.score(query, [d.metadata["key"] for d in head])
        seconds = time.perf_counter() - started
        by_score = sorted(range(len(head)), key=lambda i: -sc[i])
        rerank_rank = {i: r for r, i in enumerate(by_score, 1)}
        order = by_score if self.mode == "score" else blend_order(sc, self.c)
        return [Document(page_content=head[i].page_content, id=head[i].id, metadata={
            **head[i].metadata, "input_rank": i + 1, "stage1_score": sc[i], "stage1_rank": rerank_rank[i],
            "stage1_seconds": seconds}) for i in order] + tail


class LLMReranker(BaseRetriever):
    """Stage 2: an LLM picks among the base top k_in; picked tables go first in the base order, then the rest."""

    base: BaseRetriever
    select: Select
    k_in: int = 20
    name: str = "llm"
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        docs = self.base.invoke(query)
        keys = [d.metadata["key"] for d in docs[: self.k_in]]
        started = time.perf_counter()
        calls, usage, error = 0, {}, None
        try:
            out = self.select(query, keys) if keys else []
            picked_keys = list(getattr(out, "keys", out))
            calls, usage = getattr(out, "calls", 1 if keys else 0), dict(getattr(out, "usage", {}) or {})
            picked = set(picked_keys) & set(keys)  # an invented key never reaches the answer
        except Exception as exc:  # the answer still gets the stage-1 order
            picked, error = set(), f"{type(exc).__name__}: {exc}"
        meta = {"stage2": self.name, "stage2_picked_n": len(picked), "stage2_error": error, "stage2_calls": calls,
                "stage2_input_tokens": usage.get("promptTokens", usage.get("input_tokens")),
                "stage2_output_tokens": usage.get("completionTokens", usage.get("output_tokens")),
                "stage2_seconds": time.perf_counter() - started}
        first = [d for d in docs if d.metadata["key"] in picked]
        rest = [d for d in docs if d.metadata["key"] not in picked]
        return [Document(page_content=d.page_content, id=d.id,
                         metadata={**d.metadata, **meta, "stage2_picked": d.metadata["key"] in picked})
                for d in first + rest]


def clova(text_of: dict[str, str], options: dict) -> Select:
    from kosis_rag.clova import ClovaReranker

    return ClovaReranker(text_of=text_of, **options)


LLM_RERANKERS: dict[str, Callable[[dict[str, str], dict], Select]] = {
    "clova": clova,
}
