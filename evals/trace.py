"""Stage-level tracing for L3 / e2e runs (plans/T2-15.md): which span a question failed in, and why; Jaeger export.

One question = one trace:

    eval.question            qid, kind, layer, verdict, failure.stage / failure.span / failure.reason
    ├─ retrieve              e2e only: retriever, top_keys, gold_rank, right_table
    └─ tool.summarize_stat   table_key, outcome, numbers_origin
       └─ judge, analyze.*, compute, compose, verify, template, fallback_chatbot   (Phase 1 spans, T1-22)

The Phase 1 pipeline already records OTel-shaped span dicts in Answer.decision_metadata["spans"]; summarize() keeps
them (summarize_stat drops them), blame() names the failing stage with deterministic rules, and export() replays
the dicts as OTel spans over OTLP when OTEL_EXPORTER_OTLP_ENDPOINT is set (docker compose --profile obs up jaeger).
Tokens and cost stay in evals.ops (ops.json); the gen_ai.* span attributes are only the per-call record.
"""

from __future__ import annotations

import json
import os
import time

from evals.evaluators.calculation import FACT_ERRORS

JAEGER_UI = os.environ.get("JAEGER_UI_URL", "http://localhost:16686")
STAGES = {  # stage -> label, in blame order
    "retrieval": "검색 실패",
    "tool_call": "도구 호출 실패",
    "tool_skipped": "계산 모듈 건너뜀",
    "compute": "계산 거절·실패",
    "understanding": "질문 이해 실패",
    "compose": "문장 생성 실패",
    "unknown": "미분류",
}
JUDGE_ERRORS = {"false_refusal", "missed_refusal", "assumed_condition", "wrong_reason", "disallowed_default",
                "unstated_default", "missing_clarification"}
PLAN_ERRORS = FACT_ERRORS | {"missing_aux"}


def summarize(question: str, table_key: str, client, retry_interval: float = 0.0):
    """summarize_stat that also returns the pipeline's spans: (StatAnswer | None, spans, error)."""
    from kosis_chat import api
    from kosis_chat.contract import to_stat_answer

    # ponytail: private _build, since summarize_stat returns no spans; drop when Phase 1 exposes them (KOS-95 follow-up)
    bot = api._build(api.table_path(table_key), None, client, retry_interval)
    answer = bot.ask(question)
    spans = (answer.decision_metadata or {}).get("spans") or []
    stat = to_stat_answer(question, answer, org_id=bot.org_id)
    return stat, spans, None if stat else f"RuntimeError: {table_key}: the model call failed ({answer.status}: {answer.error})"


def _failure(stage: str, span: str | None, reason: str, others: set[str]) -> dict:
    extra = f" (같이 나온 오류: {', '.join(sorted(others))})" if others else ""
    return {"stage": stage, "label": STAGES[stage], "span": span, "reason": reason + extra}


def blame(row: dict) -> dict | None:
    """First matching rule (plans/T2-15.md table); None for a PASS row. row: verdict, kind, errors, spans, ...;
    e2e rows also carry right_table, used_table, gold_rank, error."""
    if row["verdict"] == "PASS":
        return None
    errors, spans = set(row["errors"]), row.get("spans") or []
    names = [s["name"] for s in spans]
    analyze = [s for s in spans if s["name"].startswith("analyze.")]
    last = lambda pred: next((s["name"] for s in reversed(spans) if pred(s)), None)  # noqa: E731

    if row.get("right_table") is False:
        rank = row.get("gold_rank")
        return _failure("retrieval", "retrieve", f"검색 1위 {row.get('used_table') or '없음'} ≠ 정답 표 "
                        f"(정답 순위 {rank or '밖'})", errors)
    if "run_failure" in errors:
        span = last(lambda s: s["status"] == "error") or "tool.summarize_stat"
        return _failure("tool_call", span, row.get("error") or "답 없음", errors - {"run_failure"})
    if "fallback_chatbot" in names:
        tried = ", ".join(f"{s['name']}={s['status']}" for s in analyze) or "analyze 없음"
        return _failure("tool_skipped", "fallback_chatbot",
                        f"계산 모듈(compute)을 거치지 않고 LLM이 숫자를 썼다 — 계획을 못 만든 앞 단계: {tried}", errors)
    compute = next((s for s in spans if s["name"] == "compute"), None)
    if compute and row["kind"] in ("answerable", "default_allowed") and (
            compute["status"] == "error" or compute["attributes"].get("outcome") == "refused"):
        return _failure("compute", "compute", f"계산 모듈이 거절함 (reason={compute['attributes'].get('reason')})", errors)
    if hit := errors & JUDGE_ERRORS:
        judge = next((s for s in spans if s["name"] == "judge"), None)
        span = "judge" if judge is None or judge["attributes"].get("final") or not analyze else analyze[-1]["name"]
        outcome = judge["attributes"].get("outcome") if judge else None
        return _failure("understanding", span, f"판정 오류 {', '.join(sorted(hit))} (judge outcome={outcome})", errors - hit)
    if hit := errors & PLAN_ERRORS:
        span = analyze[-1]["name"] if analyze else "tool.summarize_stat"
        return _failure("understanding", span, f"계획이 고른 셀이 틀림: {', '.join(sorted(hit))}", errors - hit)
    if "unsupported_claim" in errors:
        return _failure("compose", last(lambda s: s["name"] == "compose") or "tool.summarize_stat",
                        "근거 없는 숫자를 문장에 씀 (unsupported_claim)", errors - {"unsupported_claim"})
    return _failure("unknown", None, "규칙에 안 맞음", errors)


def _attr(v):
    return v if isinstance(v, (str, bool, int, float)) else json.dumps(v, ensure_ascii=False, default=str)


def export(rows: list[dict], layer: str, retriever: str | None) -> bool:
    """Replay each row as one trace over OTLP and set row["trace_id"]; no endpoint -> nothing sent, False.

    Spans are laid out back to back ending now (rows hold durations, not wall-clock times), so durations and order
    are real while absolute times are not.
    """
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return False
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import Status, StatusCode

    provider = TracerProvider(resource=Resource.create({"service.name": "kosis-evals"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    tracer = provider.get_tracer("evals.trace")
    ms = 1_000_000
    t = time.time_ns() - int(sum(r.get("latency_ms", 0) for r in rows) * ms)
    for r in rows:
        f = r.get("failure") or {}
        end = t + int(r.get("latency_ms", 0) * ms)
        root = tracer.start_span("eval.question", start_time=t, attributes={
            "qid": r["qid"], "kind": r["kind"], "layer": layer, "verdict": r["verdict"], "errors": r["errors"],
            **{f"failure.{k}": v for k, v in f.items() if v is not None and k != "label"}})
        ctx = trace.set_span_in_context(root)
        tool_start = t
        if "right_table" in r:  # e2e: the retrieval step
            tool_start = t + int(r.get("retrieve_ms", 0) * ms)
            s = tracer.start_span("retrieve", context=ctx, start_time=t, attributes={
                "retriever": retriever or "", "top_keys": r.get("top_keys") or [],
                "gold_rank": r.get("gold_rank") or 0, "right_table": r["right_table"]})
            if f.get("span") == "retrieve":
                s.set_status(Status(StatusCode.ERROR, f["reason"]))
            s.end(end_time=tool_start)
        answer = r.get("answer") or {}
        tool = tracer.start_span("tool.summarize_stat", context=ctx, start_time=tool_start, attributes={
            "table_key": r.get("used_table") or r.get("table_key") or "", "outcome": r.get("outcome") or "",
            "numbers_origin": answer.get("numbers_origin") or ""})
        tctx = trace.set_span_in_context(tool)
        for sp in r.get("spans") or []:
            start = tool_start + int(sp["start_ms"] * ms)
            child = tracer.start_span(sp["name"], context=tctx, start_time=start,
                                      attributes={k: _attr(v) for k, v in sp["attributes"].items()})
            if sp["status"] == "error" or f.get("span") == sp["name"]:
                child.set_status(Status(StatusCode.ERROR, f["reason"] if f.get("span") == sp["name"] else "error"))
            child.end(end_time=start + int(sp["duration_ms"] * ms))
        if f.get("stage") == "tool_call":
            tool.set_status(Status(StatusCode.ERROR, f["reason"]))
        tool.end(end_time=end)
        if f:
            root.set_status(Status(StatusCode.ERROR, f"{f['label']}: {f['reason']}"))
        root.end(end_time=end)
        r["trace_id"] = format(root.get_span_context().trace_id, "032x")
        t = end
    provider.shutdown()  # flushes the batch
    return True
