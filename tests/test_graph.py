"""T2-10: the stage functions give what the old retriever classes give, and the LangGraph search gives what
chain.answer(build_search(...)) gives.

No model, no server, no API: the dense side is a list retriever, stage 1 / stage 2 are fake functions, the guide LLM
is Phase 1 MockClient.
"""

from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from langchain_core.documents import Document  # noqa: E402
from langchain_core.retrievers import BaseRetriever  # noqa: E402

from kosis_ai.schemas import TableInfo  # noqa: E402
from kosis_chat.llm.model import MockClient  # noqa: E402
from kosis_rag import chain  # noqa: E402
from kosis_rag import query as query_rules  # noqa: E402
from kosis_rag.filters import TableFilter, filter_tables  # noqa: E402
from kosis_rag.graph import build_graph, run_search  # noqa: E402
from kosis_rag.hcx import HcxResult  # noqa: E402
from kosis_rag.loader import to_document  # noqa: E402
from kosis_rag.pipeline import SearchSettings, build_search, load_resources  # noqa: E402
from kosis_rag.rerank import CrossEncoderReranker, LLMReranker, rerank_stage1, rerank_stage2  # noqa: E402
from kosis_rag.retriever import HybridRetriever, hybrid_search  # noqa: E402
from kosis_rag.schemas import TableRecord  # noqa: E402

pytestmark = pytest.mark.offline


def _record(key, name, periods):
    info = TableInfo(
        key=key, org_id="101", tbl_id=key.split("_", 1)[1], survey_name="인구동향조사", organization="국가데이터처",
        table_name=name, period="년", source="api", coverage="A_정상", rows=1, numeric_rows=1, symbol_rows=0,
        parquet=f"values/{key}.parquet", url=f"http://kosis.kr/{key}", menu_path=f"인구 > {name}",
    )
    return TableRecord(info=info, cycle=["Y"], prd_se=["Y"], periods=list(periods), start=int(periods[0][:4]),
                       end=int(periods[-1][:4]), menu_paths=[f"인구 > {name}"])


RECORDS = [_record("101_DT_A", "출생아수", ("2020", "2023")), _record("101_DT_B", "사망자수", ("2023",)),
           _record("101_DT_C", "혼인건수", ("2010",)), _record("101_DT_D", "이혼건수", ("2023",))]
BY_KEY = {r.info.key: r for r in RECORDS}
KEYS = list(BY_KEY)


class CatalogRetriever(BaseRetriever):
    """Dense stand-in: fixed keys in order, catalog metadata (so answers can cite), `with_filter` like Chroma."""

    keys: list[str]
    records: dict = {}
    last_removed: dict = {}
    allowed: list[str] | None = None

    def with_filter(self, f, records):
        passed, removed = filter_tables(records, f)
        return self.model_copy(update={"allowed": sorted(passed), "last_removed": removed})

    def _get_relevant_documents(self, query, *, run_manager):
        out = []
        for i, key in enumerate(self.keys):
            if self.allowed is None or key in self.allowed:
                doc = to_document(BY_KEY[key].info)
                out.append(Document(page_content=key, metadata={**doc.metadata, "score": 0.9 - i / 10}, id=key))
        return out


class PlainRetriever(BaseRetriever):
    """No `with_filter` (like BM25): filtered after the search."""

    keys: list[str]

    def _get_relevant_documents(self, query, *, run_manager):
        return [Document(page_content=k, metadata={"key": k, "score": 1.0}, id=k) for k in self.keys]


def _stage1(q, keys):
    return [{"101_DT_A": 0.1, "101_DT_B": 0.5, "101_DT_C": 0.9, "101_DT_D": 0.3}[k] for k in keys]


def _pick(*keys, reason=None):
    return lambda q, ks: HcxResult(keys=list(keys), calls=1, usage={"promptTokens": 7}, reason=reason)


def _boom(q, ks):
    raise TimeoutError("slow")


def _plain(docs):
    """Documents without the timing fields (they differ between two runs of the same code)."""
    return [(d.id, d.page_content, {k: v for k, v in d.metadata.items() if not k.endswith("_seconds")}) for d in docs]


# --- 1. extracted functions == the old retriever classes ---

def test_hybrid_search_equals_hybrid_retriever_with_and_without_filter():
    dense, sparse = CatalogRetriever(keys=["101_DT_C", "101_DT_A", "101_DT_B"]), PlainRetriever(keys=["101_DT_B", "101_DT_D", "101_DT_C"])
    for f in (None, TableFilter(years=[2023])):
        old = HybridRetriever(retrievers=[dense, sparse], names=["dense", "bm25"], records=BY_KEY,
                              condition=(lambda q, f=f: f) if f else None)
        docs, removed = hybrid_search("q", [dense, sparse], names=["dense", "bm25"], table_filter=f, records=BY_KEY)
        assert docs == old.invoke("q") and removed == old.last_removed
    assert set(removed) == {"101_DT_C"}


def test_hybrid_search_refuses_a_filter_without_records():
    with pytest.raises(ValueError):
        hybrid_search("q", [PlainRetriever(keys=["101_DT_A"])], table_filter=TableFilter(years=[2023]))


@pytest.mark.parametrize("mode", ["blend", "score"])
def test_rerank_stage1_equals_cross_encoder_reranker(mode):
    base = CatalogRetriever(keys=KEYS)
    old = CrossEncoderReranker(base=base, score=_stage1, n=3, mode=mode).invoke("q")
    assert _plain(rerank_stage1("q", base.invoke("q"), _stage1, n=3, mode=mode)) == _plain(old)
    with pytest.raises(ValueError):
        rerank_stage1("q", base.invoke("q"), _stage1, mode="other")


@pytest.mark.parametrize("select", [
    _pick("101_DT_C", "101_DT_B"),  # normal pick
    lambda q, ks: ["101_DT_D", "zzz"],  # invented key dropped
    _pick(reason="out_of_scope"),  # empty pick with a reason
    _boom,  # selector raises
])
def test_rerank_stage2_equals_llm_reranker(select):
    base = CatalogRetriever(keys=KEYS)
    old = LLMReranker(base=base, select=select, k_in=3, name="hcx").invoke("q")
    assert _plain(rerank_stage2("q", base.invoke("q"), select, k_in=3, name="hcx")) == _plain(old)


# --- 2. graph == chain.answer(build_search(...)) ---

def _settings(**kw):
    return SearchSettings(tables=Path("unused.jsonl"), **{"cond_filter": False, "stage2": "none", **kw})


def _both(question, reply="", *, settings=None, dense_keys=KEYS, stage1=_stage1, select=_pick("101_DT_A")):
    """(old result, old client, graph state, graph client) for the same fakes."""
    settings = settings or _settings()
    parts = dict(records=RECORDS, stage1_score=stage1, select=select)
    old_client, new_client = MockClient(reply), MockClient(reply)
    old = chain.answer(build_search(settings, dense=CatalogRetriever(keys=dense_keys), **parts), old_client, question)
    resources = load_resources(settings, dense=CatalogRetriever(keys=dense_keys), **parts)
    state = run_search(build_graph(resources, new_client), question)
    return old, old_client, state, new_client


@pytest.mark.parametrize("case, question, reply, kw", [
    ("a: guide answer", "출생아 수", "[1] DT_A 표가 출생아 수를 담고 있습니다.", {}),
    ("e: LLM says NO_MATCH", "출생아 수", chain.NO_MATCH, {}),
    ("f: unknown table id -> template", "출생아 수", "DT_ZZZ 표를 보세요.", {}),
    ("d: stage 2 raised -> guide", "출생아 수", "[1] DT_C 표입니다.", {"select": _boom}),
    ("g: stage 1 off", "출생아 수", "[1] DT_A 표입니다.", {"stage1": None}),
    ("h: stage 2 off", "출생아 수", "[1] DT_C 표입니다.", {"select": None}),
    ("i: both off", "출생아 수", "[1] DT_A 표입니다.", {"stage1": None, "select": None}),
])
def test_graph_answer_equals_chain_answer(case, question, reply, kw):
    old, old_client, state, new_client = _both(question, reply, **kw)
    assert state.result == old, case
    assert len(new_client.calls) == len(old_client.calls) == 1


def test_graph_refuses_without_candidates_and_without_calling_the_llm():  # b
    old, old_client, state, new_client = _both("bitcoin price", "안내", dense_keys=[], select=None)
    assert state.result == old and old.no_answer and not old.candidates
    assert state.route == "refuse" and old_client.calls == new_client.calls == []


def test_graph_refuses_on_empty_stage2_pick_with_reason():  # c
    old, old_client, state, new_client = _both("자동차 등록 대수", "안내", select=_pick(reason="out_of_scope"))
    assert state.result == old and old.no_answer_reason == "out_of_scope" and old.answer == chain.REFUSALS["out_of_scope"]
    assert state.refusal_reason == "out_of_scope" and old_client.calls == new_client.calls == []


def test_graph_condition_filter_removes_a_table():  # j
    old, _, state, _ = _both("2023년 인구", "[1] DT_A 표입니다.", settings=_settings(cond_filter=True))
    assert state.result == old
    assert set(state.removed) == {"101_DT_C"} and "101_DT_C" not in [d.metadata["key"] for d in state.docs]
    assert state.query.years == [2023]


def test_graph_relaxes_a_condition_no_table_meets_and_records_it():  # T2-29 D1·D3
    # no record has a 성 class: the breakdown is dropped, the year stays (C has no 2023), both paths agree
    old, _, state, _ = _both("2023년 성별 인구", "[1] DT_A 표입니다.", settings=_settings(cond_filter=True))
    assert state.result == old and not old.no_answer
    assert state.query.dimensions == ["성"] and state.relaxed == ["dimensions"]
    assert state.trace["retrieve"]["relaxed"] == ["dimensions"] and set(state.removed) == {"101_DT_C"}


def test_graph_still_refuses_a_year_no_table_holds():  # T2-29 D2
    old, _, state, _ = _both("1990년 인구", "안내", settings=_settings(cond_filter=True), select=None)
    assert state.result == old and old.no_answer and not old.candidates
    assert state.route == "refuse" and state.relaxed == [] and len(state.removed) == len(KEYS)


def test_graph_keeps_the_whole_pool_and_records_each_node():
    _, _, state, _ = _both("출생아 수", "[1] DT_A 표입니다.")
    assert len(state.docs) == len(KEYS)  # only the answer sees docs[:k]
    assert set(state.trace) == {"parse", "retrieve", "rerank1", "rerank2", "gate", "guide"}
    assert state.trace["rerank2"]["stage2_picked_n"] == 1


# --- 3. graph shape ---

def _nodes(**kw):
    resources = load_resources(_settings(), records=RECORDS, dense=CatalogRetriever(keys=KEYS), **kw)
    return set(build_graph(resources, MockClient()).get_graph().nodes) - {"__start__", "__end__"}


def test_graph_has_only_the_stages_that_are_on():
    assert _nodes(stage1_score=_stage1, select=_pick()) == {"parse", "retrieve", "rerank1", "rerank2", "gate", "refuse", "guide"}
    assert _nodes() == {"parse", "retrieve", "gate", "refuse", "guide"}


# --- evals L1 runner: graph:<preset> gives the rows the old preset gives ---

def test_eval_runner_graph_preset_matches_the_old_preset(monkeypatch):
    import kosis_rag.pipeline as pipeline
    from evals.runners import retrieval as runner
    from evals.schemas import RetrievalItem

    settings = _settings(cond_filter=True)
    fakes = dict(records=RECORDS, dense=CatalogRetriever(keys=KEYS), stage1_score=_stage1, select=_pick(reason="no_item"))
    monkeypatch.setattr(runner, "_staged_settings", lambda name: settings)  # never reads .env
    # build_search calls pipeline.load_resources too: the patch fills in the fakes for both presets
    monkeypatch.setattr(pipeline, "load_resources", lambda s, **kw: load_resources(s, **{**kw, **fakes}))
    items = [RetrievalItem(qid="q1", split="dev", qtype="concept", source="scenario", question="2023년 출생아 수",
                           required=[["101_DT_A"]])]
    old = runner.run(runner.build("staged-hcx", items, 10), items, 10)
    new = runner.run(runner.build("graph:staged-hcx", items, 10), items, 10)
    drop = ("latency_ms", "search_spans")
    assert [{k: v for k, v in r.items() if k not in drop} for r in new] == [{k: v for k, v in r.items() if k not in drop} for r in old]
    assert new[0]["keys"] and new[0]["stage2_picked_n"] == 0 and new[0]["stage2_reason"] == "no_item"
    # T2-36: only the graph run records node spans
    assert old[0]["search_spans"] == [] and [s["name"] for s in new[0]["search_spans"]][:2] == ["parse", "retrieve"]


# --- evals tracing (T2-36): one span per node from the LangGraph callbacks ---

def test_node_spans_record_each_node_once_with_its_output():
    from evals.trace import NodeSpans

    resources = load_resources(_settings(), records=RECORDS, dense=CatalogRetriever(keys=KEYS), stage1_score=_stage1,
                               select=_pick("101_DT_B"))
    nodes = NodeSpans()
    state = run_search(build_graph(resources, MockClient("[1] DT_B 표입니다.")), "출생아 수", {"callbacks": [nodes]})
    spans = {s["name"]: s for s in nodes.spans}
    # inner runs (channel writes, the gate branch, the guide chain and its LLM) carry the node name too: not spans
    assert [s["name"] for s in nodes.spans] == ["parse", "retrieve", "rerank1", "rerank2", "gate", "guide"]
    assert spans["rerank2"]["keys"] == [d.metadata["key"] for d in state.docs]  # the last search node = the result
    assert spans["rerank2"]["keys"][0] == "101_DT_B" and spans["rerank2"]["attributes"]["stage2_picked_n"] == 1
    assert spans["rerank1"]["keys"][0] == "101_DT_C"  # the stage 1 fake puts DT_C first
    assert spans["gate"]["attributes"]["route"] == "guide" and "keys" not in spans["gate"]
    assert all(s["status"] == "ok" and s["duration_ms"] >= 0 for s in nodes.spans)
    starts = [s["start_ms"] for s in nodes.spans]
    assert starts == sorted(starts)


def test_node_spans_mark_a_failing_node():
    from evals.trace import NodeSpans

    resources = load_resources(_settings(), records=RECORDS, dense=CatalogRetriever(keys=KEYS), stage1_score=_boom)
    nodes = NodeSpans()
    with pytest.raises(TimeoutError):
        run_search(build_graph(resources, MockClient()), "출생아 수", {"callbacks": [nodes]})
    assert [(s["name"], s["status"]) for s in nodes.spans] == [("parse", "ok"), ("retrieve", "ok"), ("rerank1", "error")]
    assert "TimeoutError" in nodes.spans[-1]["attributes"]["error"]


# --- 4. the question is parsed once ---

def test_graph_parses_the_question_once(monkeypatch):
    calls = []
    real = query_rules.parse_query

    def counting(question):
        calls.append(question)
        return real(question)

    monkeypatch.setattr(query_rules, "parse_query", counting)
    resources = load_resources(_settings(cond_filter=True), records=RECORDS, dense=CatalogRetriever(keys=KEYS))
    run_search(build_graph(resources, MockClient("[1] DT_A 표입니다.")), "2023년 인구")
    assert calls == ["2023년 인구"]
