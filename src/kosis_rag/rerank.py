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
stage2_picked_n (0 = the LLM picked nothing, a no-answer signal for T2-09), stage2_reason (why nothing was picked,
when the re-ranker says), stage2_input_top_score (metadata["score"] of the stage 1 top table, so a score threshold
judges the same table with or without stage 2), stage2_error, stage2_calls, stage2_input_tokens,
stage2_output_tokens, stage2_seconds.
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
    """Input positions in blended order: 1/(c + re-ranker rank) + 1/(c + input rank); ties keep the input order.

    Does: mixes the cross-encoder order with the hybrid order for stage 1 "blend" mode.
    In: scores = one cross-encoder score per input position (input = hybrid order); c = RRF constant.
    Out: input positions (0-based) in blended order.
    Side effects: none.
    Node: rerank1 (through rerank_stage1).
    """
    by_score = sorted(range(len(scores)), key=lambda i: -scores[i])
    rerank_rank = {i: r for r, i in enumerate(by_score, 1)}
    return sorted(range(len(scores)), key=lambda i: -(1 / (c + rerank_rank[i]) + 1 / (c + i + 1)))


def passage_scorer(scorer: Any, text_of: dict[str, str], query_text: Callable[[str], str] | None = None) -> Score:
    """Score function over table keys from a model.CrossEncoderScorer and the n2 passages it was trained on.

    Does: turns a loaded cross-encoder into the stage 1 `Score` function over table keys.
    In: scorer = model.CrossEncoderScorer (or any object with .score(question, passages)); text_of = table key ->
        passage in the format the model was trained on; query_text = surface rules applied to the question (optional).
    Out: score(question, keys) -> one float per key, in key order.
    Side effects: none when built; each call runs the cross-encoder (GPU/CPU inference). Raises KeyError for a key
        without a passage.
    Node: rerank1 (built by load_resources, called inside rerank_stage1).
    """

    def score(question: str, keys: list[str]) -> list[float]:
        q = query_text(question) if query_text else question
        return scorer.score(q, [text_of[k] for k in keys])

    return score


def rerank_stage1(query: str, docs: list[Document], score: Score, *, n: int = 50, mode: str = "blend",
                  c: int = RRF_C) -> list[Document]:
    """Re-order the top n candidates by the fine-tuned cross-encoder; the tail follows unchanged.

    In: query = the user question; docs = candidates in current rank order (hybrid order, metadata["key"] set);
        score = (question, keys) -> one cross-encoder score per key; n = how many from the top are re-scored;
        mode = "blend" (RRF of re-ranker and input order) or "score" (re-ranker order only); c = RRF constant.
    Out: the re-ordered top n, then the rest as given. Each re-ordered doc gets metadata input_rank, stage1_score,
         stage1_rank and stage1_seconds; metadata["score"] is left as it was. Empty docs come back as they are.
    Side effects: calls `score` (cross-encoder inference in production); raises ValueError for an unknown mode.
    Node: rerank1.
    """
    if mode not in ("blend", "score"):
        raise ValueError(f"unknown stage 1 mode {mode!r}; known: blend, score")
    head, tail = docs[:n], docs[n:]
    if not head:
        return docs
    # 1. score the head
    started = time.perf_counter()
    sc = score(query, [d.metadata["key"] for d in head])
    seconds = time.perf_counter() - started
    # 2. order by score alone or blended with the input order
    by_score = sorted(range(len(head)), key=lambda i: -sc[i])
    rerank_rank = {i: r for r, i in enumerate(by_score, 1)}
    order = by_score if mode == "score" else blend_order(sc, c)
    return [Document(page_content=head[i].page_content, id=head[i].id, metadata={
        **head[i].metadata, "input_rank": i + 1, "stage1_score": sc[i], "stage1_rank": rerank_rank[i],
        "stage1_seconds": seconds}) for i in order] + tail


def rerank_stage2(query: str, docs: list[Document], select: Select, *, k_in: int = 20,
                  name: str = "llm") -> list[Document]:
    """Let an LLM pick the tables that answer the question among the top k_in candidates.

    In: query = the user question; docs = candidates in current rank order (metadata["key"] set);
        select = (question, keys) -> picked keys, or an object with .keys/.calls/.usage/.reason;
        k_in = how many from the top the LLM sees; name = the re-ranker's name for metadata["stage2"].
    Out: picked docs first (in input order), then the rest; every doc gets stage2, stage2_picked, stage2_picked_n,
         stage2_reason, stage2_input_top_score, stage2_error, stage2_calls, stage2_input_tokens,
         stage2_output_tokens and stage2_seconds. Keys the LLM invents are dropped.
    Side effects: calls `select` (an LLM API in production); its exceptions are caught and stored in stage2_error,
        and the input order is kept.
    Node: rerank2.
    """
    keys = [d.metadata["key"] for d in docs[:k_in]]
    # 1. ask the LLM; a failure keeps the input order
    started = time.perf_counter()
    calls, usage, reason, error = 0, {}, None, None
    try:
        out = select(query, keys) if keys else []
        picked_keys = list(getattr(out, "keys", out))
        calls, usage = getattr(out, "calls", 1 if keys else 0), dict(getattr(out, "usage", {}) or {})
        reason = getattr(out, "reason", None)
        picked = set(picked_keys) & set(keys)  # an invented key never reaches the answer
    except Exception as exc:  # the answer still gets the stage-1 order
        picked, error = set(), f"{type(exc).__name__}: {exc}"
    # 2. record what stage 2 did on every doc
    meta = {"stage2": name, "stage2_picked_n": len(picked), "stage2_reason": reason,
            "stage2_input_top_score": docs[0].metadata.get("score") if docs else None,
            "stage2_error": error, "stage2_calls": calls,
            "stage2_input_tokens": usage.get("promptTokens", usage.get("input_tokens")),
            "stage2_output_tokens": usage.get("completionTokens", usage.get("output_tokens")),
            "stage2_seconds": time.perf_counter() - started}
    # 3. picked first in input order, then the rest
    first = [d for d in docs if d.metadata["key"] in picked]
    rest = [d for d in docs if d.metadata["key"] not in picked]
    return [Document(page_content=d.page_content, id=d.id,
                     metadata={**d.metadata, **meta, "stage2_picked": d.metadata["key"] in picked})
            for d in first + rest]


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


def clova(text_of: dict[str, str], options: dict) -> Select:
    from kosis_rag.clova import ClovaReranker

    return ClovaReranker(text_of=text_of, **options)


def hcx(text_of: dict[str, str], options: dict) -> Select:
    from kosis_rag.hcx import HcxSelector

    return HcxSelector(text_of=text_of, **options)


def gpt(text_of: dict[str, str], options: dict) -> Select:
    from kosis_rag.hcx import HcxSelector, OpenAIClient

    return HcxSelector(text_of=text_of, client=OpenAIClient(**options))


LLM_RERANKERS: dict[str, Callable[[dict[str, str], dict], Select]] = {
    "clova": clova,
    "hcx": hcx,
    "gpt": gpt,
}
