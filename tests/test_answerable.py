"""KOS-193: the answerability classifier between stage 1 and stage 2 (answerable node).

1. judge(): strict threshold on the best logit of the first answer_top docs, metadata on every doc, inputs unchanged.
2. gate refuses a classifier-refused question with reason "not_answerable".
3. The graph with a fake answer scorer: a refused question never calls the stage 2 selector and ends "refuse"; a
   passed one calls it and ends as the graph without the classifier. Without an answer scorer the node table, the
   edges and the node API set are the old ones.
4. SearchOptions.label, the loader's checks, and the card text the classifier was trained on.

No model, no server, no API: fakes only (tests/test_graph_runs.py's).
"""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from kosis_rag.answerable import cards
from kosis_rag.answerable.node import card_scorer, judge
from kosis_rag.answerable.schemas import AnswerInput
from kosis_rag.gate.node import GateInput, decide_route
from kosis_rag.schemas.resources import IndexSpec, SearchOptions

pytestmark = pytest.mark.offline


def _docs(*keys):
    return [Document(page_content=f"표 {k}", id=k, metadata={"key": k, "score": 0.9 - i / 10})
            for i, k in enumerate(keys)]


def _res(logits, theta=0.0, top=5):
    """Stand-in resources: answer_score gives logits[key] and records every call."""
    calls = []

    def score(question, keys):
        calls.append((question, list(keys)))
        return [logits[k] for k in keys]

    return SimpleNamespace(options=SearchOptions(answer_theta=theta, answer_top=top), answer_score=score), calls


# --- 1. judge ---

def test_judge_refuses_only_strictly_below_theta():
    docs = _docs("a", "b")
    res, _ = _res({"a": -1.0, "b": 0.5}, theta=0.5)
    out = judge(AnswerInput(question="q", docs=docs), res)
    assert out.top_score == 0.5 and not out.refused and out.n_scored == 2  # 0.5 is not < 0.5
    res, _ = _res({"a": -1.0, "b": 0.5}, theta=0.5 + 1e-6)
    assert judge(AnswerInput(question="q", docs=docs), res).refused


def test_judge_scores_only_the_first_answer_top_docs():
    docs = _docs("a", "b", "c")
    res, calls = _res({"a": -3.0, "b": -2.0, "c": 9.0}, theta=0.0, top=2)
    out = judge(AnswerInput(question="질문", docs=docs), res)
    assert calls == [("질문", ["a", "b"])]
    assert out.refused and out.top_score == -2.0 and out.n_scored == 2  # c (9.0) is outside the top 2


def test_judge_marks_every_doc_keeps_the_order_and_does_not_mutate_inputs():
    docs = _docs("a", "b", "c")
    before = [dict(d.metadata) for d in docs]
    res, _ = _res({"a": 1.0, "b": 2.0, "c": 3.0}, theta=5.0, top=2)
    out = judge(AnswerInput(question="q", docs=docs), res)
    assert [d.metadata["key"] for d in out.docs] == ["a", "b", "c"]
    assert [d.page_content for d in out.docs] == [d.page_content for d in docs]
    assert [d.id for d in out.docs] == ["a", "b", "c"]
    assert all(d.metadata["answer_refused"] is True and d.metadata["answer_top_score"] == 2.0 for d in out.docs)
    assert [d.metadata.get("answer_score") for d in out.docs] == [1.0, 2.0, None]  # c was not scored
    assert "answer_score" not in out.docs[2].metadata
    assert all(out.docs[i].metadata["score"] == before[i]["score"] for i in range(3))  # search metadata kept
    assert [d.metadata for d in docs] == before and all(o is not d for o, d in zip(out.docs, docs))


def test_judge_with_no_docs_does_not_score():
    res, calls = _res({}, theta=0.0)
    out = judge(AnswerInput(question="q", docs=[]), res)
    assert (out.docs, out.top_score, out.refused, out.n_scored) == ([], None, False, 0) and calls == []


def test_card_scorer_falls_back_to_the_key_text():
    seen = []

    class Fake:
        def score(self, questions, texts):
            seen.append((questions, texts))
            return [float(len(t)) for t in texts]

    score = card_scorer(Fake(), {"a": "카드 a"})
    assert score("q", ["a", "zz"]) == [4.0, 2.0]
    assert seen == [(["q", "q"], ["카드 a", "zz"])]


# --- 2. gate ---

def test_gate_refuses_a_classifier_refusal_with_not_answerable():
    res, _ = _res({"a": -9.0, "b": -9.0}, theta=0.0)
    docs = judge(AnswerInput(question="q", docs=_docs("a", "b")), res).docs
    out = decide_route(GateInput(docs=docs))
    assert (out.route, out.refusal_reason) == ("refuse", "not_answerable")
    res, _ = _res({"a": 1.0, "b": -9.0}, theta=0.0)
    passed = judge(AnswerInput(question="q", docs=_docs("a", "b")), res).docs
    assert decide_route(GateInput(docs=passed)).route == "found"
    # the existing rules are unchanged: empty -> refuse, stage 2 empty pick -> refuse with its reason
    assert decide_route(GateInput(docs=[])).route == "refuse"
    picked_none = [Document(page_content="x", metadata={**passed[0].metadata, "stage2_picked_n": 0,
                                                        "stage2_error": None, "stage2_reason": "no_item"})]
    assert decide_route(GateInput(docs=picked_none)).refusal_reason == "no_item"


# --- 3. graph ---

def _graph_parts():
    pytest.importorskip("langgraph")
    from tests import test_graph_runs as g

    return g


def _counting(select):
    calls = []

    def wrapped(q, keys):
        calls.append((q, list(keys)))
        return select(q, keys)

    return wrapped, calls


REFUSED_Q, PASSED_Q = "자동차 등록 대수", "출생아 수"


def _fake_answer_score(question, keys):
    """-5 for every table on the refused question; on the other, 1.0 for DT_A and -1.0 for the rest."""
    return [-5.0 if question == REFUSED_Q else (1.0 if k.endswith("_DT_A") else -1.0) for k in keys]


def _resources(g, *, answer=True, select=None, **kw):
    settings = g._settings(answer_theta=0.0) if answer else g._settings()
    return g._load(settings, records=g.RECORDS, dense=g.CatalogRetriever(keys=g.KEYS), stage1_score=g._stage1,
                   select=select, answer_score=_fake_answer_score if answer else None, **kw)


def test_a_refused_question_never_calls_stage_2_and_ends_refuse():
    g = _graph_parts()
    from kosis_rag.graph.search import build_graph, run_search

    select, calls = _counting(g._pick(g.B))
    state = run_search(build_graph(_resources(g, select=select)), REFUSED_Q)
    assert calls == []
    assert (state.route, state.refusal_reason) == ("refuse", "not_answerable")
    assert state.trace["rerank2"]["skipped"] == "answer_refused"
    a = state.trace["answerable"]
    assert a["refused"] is True and a["top_score"] == -5.0 and a["theta"] == 0.0 and a["n_scored"] == 4
    assert all(d.metadata["answer_refused"] and "stage2_picked_n" not in d.metadata for d in state.docs)


def test_a_passed_question_calls_stage_2_and_ends_as_without_the_classifier():
    g = _graph_parts()
    from kosis_rag.graph.search import build_graph, run_search

    select, calls = _counting(g._pick(g.B))
    state = run_search(build_graph(_resources(g, select=select)), PASSED_Q)
    old_select, old_calls = _counting(g._pick(g.B))
    old = run_search(build_graph(_resources(g, answer=False, select=old_select)), PASSED_Q)
    assert len(calls) == 1 and calls == old_calls
    assert (state.route, state.refusal_reason) == (old.route, old.refusal_reason) == ("found", None)
    assert [d.metadata["key"] for d in state.docs] == [d.metadata["key"] for d in old.docs]
    assert state.docs[0].metadata["key"] == g.B and state.trace["answerable"]["refused"] is False
    assert "skipped" not in state.trace["rerank2"]
    strip = lambda docs: [{k: v for k, v in d.metadata.items() if not k.startswith("answer_")  # noqa: E731
                           and not k.endswith("_seconds")} for d in docs]
    assert strip(state.docs) == strip(old.docs)


def test_stage_2_empty_pick_still_refuses_after_a_passed_classifier():
    g = _graph_parts()
    from kosis_rag.graph.search import build_graph, run_search

    state = run_search(build_graph(_resources(g, select=g._pick(reason="out_of_scope"))), PASSED_Q)
    assert (state.route, state.refusal_reason) == ("refuse", "out_of_scope")


def test_without_an_answer_scorer_the_table_and_edges_are_the_old_ones():
    g = _graph_parts()
    from kosis_rag.graph.nodes_api import make_nodes
    from kosis_rag.graph.search import build_graph, graph_nodes

    old = _resources(g, answer=False, select=g._pick(g.B))
    assert old.answer_score is None
    assert list(graph_nodes(old)) == ["parse", "condition", "retrieve_dense", "retrieve_bm25", "fuse", "rerank1",
                                      "rerank2", "gate"]
    edges = {("__start__", "parse"), ("parse", "condition"), ("condition", "retrieve_dense"),
             ("condition", "retrieve_bm25"), ("retrieve_dense", "fuse"), ("retrieve_bm25", "fuse"),
             ("fuse", "rerank1"), ("rerank1", "rerank2"), ("rerank2", "gate"), ("gate", "__end__")}
    assert {(e.source, e.target) for e in build_graph(old).get_graph().edges} == edges
    assert list(make_nodes(old)) == ["parse", "retrieve", "rerank1", "rerank2", "gate"]
    # a stand-in resources object without the attribute (older tests) builds the old table too
    stand_in = SimpleNamespace(**{k: v for k, v in vars(old).items() if k != "answer_score"})
    assert list(graph_nodes(stand_in)) == list(graph_nodes(old))

    new = _resources(g, select=g._pick(g.B))
    assert list(graph_nodes(new)) == ["parse", "condition", "retrieve_dense", "retrieve_bm25", "fuse", "rerank1",
                                      "answerable", "rerank2", "gate"]
    new_edges = (edges - {("rerank1", "rerank2")}) | {("rerank1", "answerable"), ("answerable", "rerank2")}
    assert {(e.source, e.target) for e in build_graph(new).get_graph().edges} == new_edges
    assert list(make_nodes(new)) == ["parse", "retrieve", "rerank1", "answerable", "rerank2", "gate"]


def test_the_classifier_without_stage_1_or_stage_2_sits_between_fuse_and_gate():
    g = _graph_parts()
    from kosis_rag.graph.search import build_graph, run_search

    resources = g._load(g._settings(answer_theta=0.0), records=g.RECORDS, dense=g.CatalogRetriever(keys=g.KEYS),
                        answer_score=_fake_answer_score)
    graph = build_graph(resources)
    edges = {(e.source, e.target) for e in graph.get_graph().edges}
    assert {("fuse", "answerable"), ("answerable", "gate")} <= edges
    assert run_search(graph, REFUSED_Q).refusal_reason == "not_answerable"
    assert run_search(graph, PASSED_Q).route == "found"


# --- 4. label, loader, cards ---

def test_label_names_the_classifier_only_when_its_checkpoint_is_set():
    options = SearchOptions(stage2="gpt")
    assert options.label(IndexSpec(rerank_ckpt="r.pt")) == "hybrid+cond -> stage1-blend -> gpt"
    assert options.label(IndexSpec(rerank_ckpt="r.pt", answer_ckpt="a.pt")) == \
        "hybrid+cond -> stage1-blend -> answerable -> gpt"
    assert SearchOptions().label(IndexSpec(answer_ckpt="a.pt")) == "hybrid+cond -> answerable"
    assert SearchOptions().label(IndexSpec()) == "hybrid+cond"


def test_loader_needs_a_theta_with_the_classifier_before_loading_anything():
    from kosis_rag.adapters.loader import load_resources

    with pytest.raises(ValueError, match="ANSWER_THETA"):  # raised before any file or model is opened
        load_resources(spec=IndexSpec(answer_ckpt="no/such/ckpt.pt"), options=SearchOptions())
    g = _graph_parts()
    with pytest.raises(ValueError, match="ANSWER_THETA"):
        g._load(g._settings(), records=g.RECORDS, dense=g.CatalogRetriever(keys=g.KEYS),
                answer_score=_fake_answer_score)
    given = g._load(g._settings(answer_theta=-7.0), records=g.RECORDS, dense=g.CatalogRetriever(keys=g.KEYS),
                    answer_score=_fake_answer_score)
    assert given.answer_score is _fake_answer_score and given.options.answer_theta == -7.0


RECORD = {"info": {"key": "101_DT_X", "table_name": "출생아수", "survey_name": "인구동향조사",
                   "organization": "국가데이터처"},
          "prd_se": ["Y"], "periods": ["2015", "2016", "2017", "2020"], "unit": "명",
          "items": ["출생아수", "합계_출산율"],
          "classes": [{"name": "시도", "levels": {"1": 17}}, {"name": "성별", "labels": ["남자", "여자"]}]}
CARD = ("통계표: 출생아수 | 인구동향조사 (국가데이터처)\n"
        "수록 연도: 2015~2017, 2020 (주기: 연)\n"
        "단위: 명\n"
        "항목: 출생아수, 합계 출산율\n"
        "시도: 1 17개\n"
        "성별: 남자, 여자")


def test_card_text_is_pinned_and_load_cards_reads_raw_records(tmp_path):
    assert cards.CARD_FORMAT == "ans-c1"
    assert cards.card(RECORD) == CARD
    path = tmp_path / "tables.jsonl"
    path.write_text(json.dumps(RECORD, ensure_ascii=False) + "\n\n", encoding="utf-8")
    assert cards.load_cards(path) == {"101_DT_X": CARD}


COPIED = ("CARD_FORMAT", "MAX_ITEMS", "MAX_LABELS", "CYCLE_NAMES", "years_of", "compress_years", "_months", "_clean",
          "_capped", "card")


def test_cards_is_an_exact_copy_of_the_parent_repo_cards():
    """The classifier was trained on scripts/answerability/cards.py's text: the copy's code must be the same."""
    parent = Path(__file__).resolve().parents[3] / "scripts" / "answerability" / "cards.py"
    if not parent.is_file():
        pytest.skip("not inside the parent repo")

    def defs(path):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {(n.name if isinstance(n, ast.FunctionDef) else ast.unparse(n.targets[0])): ast.dump(n)
                for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Assign))}

    mine, theirs = defs(Path(cards.__file__)), defs(parent)
    assert {k: mine[k] for k in COPIED} == {k: theirs[k] for k in COPIED}
