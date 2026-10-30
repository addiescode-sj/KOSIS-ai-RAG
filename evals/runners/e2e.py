"""L4 end to end, first part (plans/T2-03.md): the retriever's top 1 table -> summarize_stat -> the L3 grader.

Same questions and grader as L3 (calculation_golden.jsonl, evaluators/calculation.py), but the table comes from
search instead of the golden set (L3 is the oracle run). The difference between the two runs is what retrieval
costs. Grading is against the golden table's facts and cells, so an answer read from another table fails unless
its numbers are the same. Any BaseRetriever works (runners/retrieval.build), as long as metadata["key"] holds the
table key.
"""

from __future__ import annotations

import time

from langchain_core.retrievers import BaseRetriever

from evals.schemas import CalcItem


def run(retriever: BaseRetriever, items: list[CalcItem], k: int, interval: float = 0.0) -> list[dict]:
    """Rows shaped like the L3 runner's, plus the ranked keys and the table used; failures are recorded, not raised."""
    from kosis_chat.api import summarize_stat  # imported here: offline commands never load the pipeline

    from evals.ops import usage_client

    client = usage_client()
    rows = []
    for n, item in enumerate(items):
        if n and interval:
            time.sleep(interval)
        client.reset()
        started = time.perf_counter()
        keys: list[str] = []
        answer, error = None, None
        try:
            keys = [d.metadata["key"] for d in retriever.invoke(item.question)[:k]]
            if keys:
                answer = summarize_stat(item.question, keys[0], client=client, retry_interval=interval)
            else:
                error = "no table retrieved"
        except Exception as exc:  # a failing question is recorded, the run goes on
            error = f"{type(exc).__name__}: {exc}"
        rows.append({"qid": item.qid, "answer": answer, "error": error, "keys": keys,
                     "used_table": keys[0] if keys else None,
                     "latency_ms": (time.perf_counter() - started) * 1000,
                     "llm_calls": client.calls, "input_tokens": client.input_tokens,
                     "output_tokens": client.output_tokens, "model": client.model,
                     "failed": error is not None, "retried": client.failures > 0})
    return rows


def locate(item: CalcItem, row: dict, verdict: str) -> dict:
    """Where the golden table ranked and why a row did not pass: retrieval (another table at top 1) or answer."""
    keys = row["keys"]
    right = row["used_table"] == item.table_key
    return {"used_table": row["used_table"], "right_table": right,
            "gold_rank": keys.index(item.table_key) + 1 if item.table_key in keys else None,
            "top_keys": keys[:5], "cause": "pass" if verdict == "PASS" else "answer" if right else "retrieval"}
