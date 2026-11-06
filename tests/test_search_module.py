"""T2-41 S4: kosis_rag.api.SearchModule — search(SearchInput) -> SearchOutput, stream, search_tables, from_resources.

Two boundaries are checked separately: the app's answer graph (T2-41 S5: run_answer(build_answer_graph(res, client))
.result) still gives the S0 result.json, and the module's SearchModule.search() / to_output give the S0 output.json
(tests/fixtures/parity). Fakes of tests/test_graph.py; no model, no server, no LLM.
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

pytest.importorskip("langgraph")

from kosis_rag.api import SearchModule, to_output  # noqa: E402
from kosis_rag.graph.search import run_search  # noqa: E402
from kosis_rag.schemas.io import SearchInput, SearchOutput  # noqa: E402
from tests.parity_cases import CASES, case_resources, case_texts, dumps, module_output  # noqa: E402
from tests.test_graph import KEYS, RECORDS, CatalogRetriever, _load, _pick, _settings, _stage1  # noqa: E402

pytestmark = pytest.mark.offline
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "parity"


def _module(**kw):
    parts = {"stage1_score": _stage1, "select": _pick("101_DT_A"), **kw}
    return SearchModule.from_resources(_load(records=RECORDS, dense=CatalogRetriever(keys=KEYS), **parts))


def _projected(output: SearchOutput) -> dict:
    return output.model_dump(mode="json", exclude={"spans"})


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_to_output_of_the_module_graph_state_is_the_s0_output_fixture(case):
    module = SearchModule.from_resources(case_resources(case))
    out = to_output(run_search(module.graph(case.k), case.question), case.k)
    assert dumps(_projected(out)) == (FIXTURES / f"{case.name}.output.json").read_text(encoding="utf-8")


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_the_app_answer_graph_still_gives_the_s0_result(case):
    assert case_texts(case)["result"] == (FIXTURES / f"{case.name}.result.json").read_text(encoding="utf-8")


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_search_module_search_is_the_s0_output_fixture(case):
    out = module_output(case)
    assert dumps(_projected(out)) == (FIXTURES / f"{case.name}.output.json").read_text(encoding="utf-8")


def test_spans_follow_the_node_order_and_end_at_gate():
    out = _module().search(SearchInput(question="출생아 수"))
    assert [s["node"] for s in out.spans][:2] == ["parse", "condition"]
    assert [s["node"] for s in out.spans][-4:] == ["fuse", "rerank1", "rerank2", "gate"]
    assert all("seconds" in s for s in out.spans) and not {"refuse", "guide"} & {s["node"] for s in out.spans}
    assert out.spans[-1]["route"] == "found"  # the module's own route name (the API's is "guide")


def test_k_none_is_answer_k_and_schema_version_defaults():
    module = _module(select=None)
    inp = SearchInput(question="출생아 수")
    assert inp.schema_version == "2.0" and inp.k is None
    out = module.search(inp)
    assert out.schema_version == "2.0" and out.route == "found"
    assert module.resources.options.answer_k == 5 and len(out.candidates) == min(5, len(KEYS))
    assert len(module.search(SearchInput(question="출생아 수", k=2)).candidates) == 2
    narrow = SearchModule.from_resources(_load(_settings(answer_k=2), records=RECORDS,
                                               dense=CatalogRetriever(keys=KEYS)))
    assert len(narrow.search(SearchInput(question="출생아 수")).candidates) == 2
    with pytest.raises(ValidationError):
        SearchInput(question="")
    with pytest.raises(ValidationError):
        SearchInput(question="q", k=21)


def test_stream_ends_with_the_search_output():
    module = _module()
    inp = SearchInput(question="출생아 수")
    events = list(module.stream(inp))
    names = [name for name, _ in events]
    assert names[-1] == "output" and names[:2] == ["parse", "condition"] and "gate" in names
    assert not {"refuse", "guide", "token", "result"} & set(names)
    assert _projected(events[-1][1]) == _projected(module.search(inp))


def test_search_tables_is_the_candidates():
    module = _module()
    out = module.search(SearchInput(question="출생아 수", k=3))
    assert module.search_tables("출생아 수", 3) == out.candidates and len(out.candidates) == 3
    assert out.picked[0].key == "101_DT_A"  # the key Phase 1 looks up (tables.csv "101_DT_...")


def test_refusal_carries_the_code_text_and_no_picks():
    out = _module(select=_pick(reason="out_of_scope")).search(SearchInput(question="자동차 등록 대수"))
    assert out.route == "refuse" and out.refusal_reason == "out_of_scope" and out.picked == []
    assert out.refusal_text and out.candidates  # candidates kept on refuse (as the app's refusal answer)


def test_the_graph_is_compiled_once_per_k():
    module = _module()
    assert module.graph(5) is module.graph(5) and module.graph(3) is not module.graph(5)


def test_output_json_is_the_contract():
    out = _module().search(SearchInput(question="출생아 수"))
    data = json.loads(out.model_dump_json())
    assert list(data) == ["schema_version", "question", "route", "refusal_reason", "refusal_text", "picked",
                          "candidates", "spans"]
