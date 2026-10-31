"""evals.trace: one blame rule per failure stage, the rule order, and the OTLP replay (in-memory exporter)."""

import pytest

from evals import trace


def span(name, status="ok", **attributes):
    return {"name": name, "start_ms": 0.0, "duration_ms": 1.0, "status": status, "attributes": attributes}


RULES = [  # (row fields, spans, expected stage, expected span)
    ({"right_table": False, "used_table": "T_OTHER", "gold_rank": None}, [span("judge")], "retrieval", "retrieve"),
    ({"errors": ["run_failure"], "error": "RuntimeError: x"}, [span("judge"), span("compose", "error")], "tool_call", "compose"),
    ({"errors": ["wrong_value"]}, [span("judge"), span("analyze.rules", "error"), span("fallback_chatbot")], "tool_skipped", "fallback_chatbot"),
    ({"errors": ["false_refusal"]}, [span("judge"), span("analyze.rules"), span("compute", outcome="refused", reason="no_period")], "compute", "compute"),
    ({"errors": ["false_refusal"]}, [span("judge", outcome="refused", final=True)], "understanding", "judge"),
    ({"errors": ["wrong_unit"]}, [span("judge"), span("analyze.rules"), span("analyze.llm_fill"), span("compute")], "understanding", "analyze.llm_fill"),
    ({"errors": ["unsupported_claim"]}, [span("judge"), span("analyze.rules"), span("compute"), span("compose")], "compose", "compose"),
    ({"errors": ["something_new"]}, [span("judge")], "unknown", None),
]


@pytest.mark.parametrize("fields, spans, stage, where", RULES, ids=[r[2] + "-" + str(i) for i, r in enumerate(RULES)])
def test_each_rule(fields, spans, stage, where):
    row = {"qid": "q", "kind": "answerable", "verdict": "FAIL", "errors": [], "spans": spans, **fields}
    f = trace.blame(row)
    assert (f["stage"], f["span"]) == (stage, where) and f["label"] == trace.STAGES[stage]


def test_pass_is_not_blamed_and_order_wins():
    assert trace.blame({"verdict": "PASS", "errors": [], "kind": "answerable"}) is None
    # several errors: the first rule decides, the rest stay in the reason
    row = {"qid": "q", "kind": "answerable", "verdict": "FAIL", "errors": ["unsupported_claim", "wrong_dims"],
           "spans": [span("judge"), span("analyze.rules"), span("compute"), span("compose")]}
    f = trace.blame(row)
    assert f["stage"] == "understanding" and "unsupported_claim" in f["reason"]
    # a refusal the compute module made is the compute stage, not understanding
    row = {**row, "errors": ["false_refusal"], "spans": [span("judge"), span("compute", outcome="refused")]}
    assert trace.blame(row)["stage"] == "compute"
    # verify errors before a good retry are normal flow: an answered row is never tool_call
    row = {**row, "errors": ["wrong_value"], "spans": [span("judge"), span("analyze.rules"), span("compute"), span("verify", "error")]}
    assert trace.blame(row)["stage"] == "understanding"


def test_export_replays_one_trace_per_question(monkeypatch):
    from opentelemetry.exporter.otlp.proto.grpc import trace_exporter
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    assert trace.export([], "l3", None) is False  # no endpoint, nothing sent
    memory = InMemorySpanExporter()
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    monkeypatch.setattr(trace_exporter, "OTLPSpanExporter", lambda: memory)
    row = {"qid": "q", "kind": "answerable", "verdict": "FAIL", "errors": ["wrong_unit"], "latency_ms": 10.0,
           "right_table": True, "retrieve_ms": 2.0, "used_table": "T", "gold_rank": 1, "top_keys": ["T"],
           "spans": [span("judge"), span("analyze.rules", detail={"a": 1}), span("compute")]}
    row["failure"] = trace.blame(row)
    assert trace.export([row], "e2e", "bm25") is True
    spans = {s.name: s for s in memory.get_finished_spans()}
    assert set(spans) == {"eval.question", "retrieve", "tool.summarize_stat", "judge", "analyze.rules", "compute"}
    assert {s.context.trace_id for s in spans.values()} == {int(row["trace_id"], 16)}
    assert spans["analyze.rules"].parent.span_id == spans["tool.summarize_stat"].context.span_id
    assert not spans["analyze.rules"].status.is_ok and spans["eval.question"].attributes["failure.stage"] == "understanding"


def test_every_grader_error_has_info():
    import re
    from pathlib import Path

    from evals.evaluators import calculation

    added = set(re.findall(r'errors\.add\("(\w+)"', Path(calculation.__file__).read_text(encoding="utf-8")))
    assert added | set().union(*calculation.FAIL.values()) <= calculation.ERROR_INFO.keys()
