"""Stage 2 re-ranking: an LLM picks among the top k_in candidates (T2-07, T2-09; T2-41 S3: from rerank.py).

Stage 2 is pluggable: rerank.selectors.LLM_RERANKERS maps a name to a factory (text_of, config) -> select(question,
keys), where select returns the picked keys (a list, or an object with .keys, .calls, .usage and .reason).

metadata["score"] is left as the hybrid gave it (the dense cosine the no-answer threshold is calibrated on, T2-09).
Stage 2 adds stage2, stage2_picked, stage2_picked_n (0 = the LLM picked nothing, a no-answer signal for T2-09),
stage2_reason (why nothing was picked, when the re-ranker says), stage2_input_top_score (metadata["score"] of the
stage 1 top table, so a score threshold judges the same table with or without stage 2), stage2_error, stage2_calls,
stage2_input_tokens, stage2_output_tokens, stage2_seconds.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from langchain_core.documents import Document

Score = Callable[[str, list[str]], list[float]]  # (question, table keys) -> one score per key
Select = Callable[[str, list[str]], Any]  # (question, table keys) -> picked keys, or an object with .keys


def rerank_stage2(query: str, docs: list[Document], select: Select, *, k_in: int = 20,
                  name: str = "llm") -> list[Document]:
    """Let an LLM pick the tables that answer the question among the top k_in candidates.

    In: query = the user question; docs = candidates in current rank order (metadata["key"] set);
        select = (question, keys) -> picked keys, or an object with .keys/.calls/.usage/.reason;
        k_in = how many from the top the LLM sees; name = the re-ranker's name for metadata["stage2"].
    Out: picked docs first (in the order the LLM gave), then the rest in input order; every doc gets stage2,
         stage2_picked, stage2_picked_n, stage2_reason, stage2_input_top_score, stage2_error, stage2_calls,
         stage2_input_tokens, stage2_output_tokens and stage2_seconds. Keys the LLM invents are dropped.
    Side effects: calls `select` (an LLM API in production); its exceptions are caught and stored in stage2_error,
        and the input order is kept.
    Node: rerank2.
    """
    keys = [d.metadata["key"] for d in docs[:k_in]]
    # 1. ask the LLM; a failure keeps the input order
    started = time.perf_counter()
    calls, usage, reason, error = 0, {}, None, None
    picked_keys: list[str] = []
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
    # 3. picked first in the LLM's order (its rank-1 choice stays first), then the rest in input order
    by_key = {d.metadata["key"]: d for d in docs}
    first = [by_key[k] for k in dict.fromkeys(picked_keys) if k in picked]
    rest = [d for d in docs if d.metadata["key"] not in picked]
    return [Document(page_content=d.page_content, id=d.id,
                     metadata={**d.metadata, **meta, "stage2_picked": d.metadata["key"] in picked})
            for d in first + rest]
