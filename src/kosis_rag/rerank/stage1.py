"""Stage 1 re-ranking: a fine-tuned cross-encoder over the top n candidates (T2-07; T2-41 S3: from rerank.py).

mode "blend" = RRF of the re-ranker order and the input order (lab plan 43 dev choice), "score" = re-ranker only.
metadata["score"] is left as the hybrid gave it; stage 1 adds input_rank, stage1_score, stage1_rank, stage1_seconds.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from langchain_core.documents import Document

from kosis_rag.rerank.stage2 import Score
from kosis_rag.shared.constants import RRF_C


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
