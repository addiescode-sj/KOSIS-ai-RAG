"""T2-41 S0: the mock parity cases, frozen as tests/fixtures/parity/*.json and compared byte for byte at every step.

Each case runs the search graph on the fakes of tests/test_graph.py (no model, no server, no API) and gives three
JSON texts:

- result: the backend `SearchResult` (`run_answer(build_answer_graph(...)).result`), what `/search` returns;
- output: the module's `SearchOutput` (T2-41 S5: `SearchModule.search()` itself, without spans; at S0 it was
  projected from the final state);
- state: a summary of the backend graph's final state (removed, relaxed, route, refusal_reason, docs' key and metadata).

Timings never enter a fixture: no spans, no `seconds`, no `*_seconds` metadata (they differ between two runs).
The steps of the refactor change only the lines that build the graph and read the state here; the JSON must stay the
same bytes (tests/test_parity_fixtures.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from kosis_chat.llm.model import MockClient

from backend.app.respond.graph import build_answer_graph, run_answer
from backend.app.respond.state import SearchState
from kosis_rag.adapters.loader import load_resources
from kosis_rag.api import SearchModule
from kosis_rag.gate.texts import NO_MATCH
from kosis_rag.schemas.io import SearchInput, SearchOutput
from kosis_rag.schemas.resources import IndexSpec, SearchResources

from tests.test_graph import RECORDS, CatalogRetriever, KEYS, _boom, _pick, _settings, _stage1

K = 5  # tables shown to gate / refuse / guide (build_answer_graph's default)
# Keys never written to a fixture (timings: they differ between two runs of the same code)
EXCLUDED_KEYS = ("seconds",)
EXCLUDED_SUFFIXES = ("_seconds",)


@dataclass(frozen=True)
class Case:
    """One parity case: a question, the guide LLM's reply and the fakes the resources are built from."""

    name: str
    question: str
    reply: str = ""
    settings: dict[str, Any] = field(default_factory=dict)  # overrides of tests.test_graph._settings
    stage1: Callable | None = _stage1
    select: Callable | None = _pick("101_DT_A")
    dense_keys: tuple[str, ...] = tuple(KEYS)
    k: int = K  # tables shown to gate / refuse / guide; a case below K pins the docs[:k] cut


CASES = (
    Case("found_guide", "출생아 수", "[1] DT_A 표가 출생아 수를 담고 있습니다."),
    Case("refuse_out_of_scope", "자동차 등록 대수", "안내", select=_pick(reason="out_of_scope")),
    Case("refuse_no_item", "출생아 수 성비", "안내", select=_pick(reason="no_item")),
    Case("refuse_no_candidates", "1999년 출생아 수", "안내", settings={"cond_filter": True}),
    Case("stage2_error", "출생아 수", "[1] DT_C 표입니다.", select=_boom),
    Case("stage2_off", "출생아 수", "[1] DT_C 표입니다.", settings={"stage2": "none"}, select=None),
    Case("template_fallback", "출생아 수", "없는 표 [9]"),
    Case("llm_no_match", "출생아 수", NO_MATCH),
    Case("stage1_off", "출생아 수", "[1] DT_A 표입니다.", stage1=None),
    Case("found_k2", "출생아 수", "[1] DT_A 표가 출생아 수를 담고 있습니다.", k=2),
    Case("stage2_off_k2", "출생아 수", "[1] DT_C 표입니다.", settings={"stage2": "none"}, select=None, k=2),
)
CASE_NAMES = tuple(c.name for c in CASES)
KINDS = ("result", "output", "state")


def case_resources(case: Case) -> SearchResources:
    """The resources of one case over fresh fakes."""
    return load_resources(spec=IndexSpec(), options=_settings(**case.settings), records=RECORDS,
                          dense=CatalogRetriever(keys=list(case.dense_keys)),
                          stage1_score=case.stage1, select=case.select)


def run_case(case: Case) -> SearchState:
    """The final state of the backend's answer graph for one case (fresh fakes and a fresh MockClient each time)."""
    return run_answer(build_answer_graph(case_resources(case), MockClient(case.reply), k=case.k), case.question)


def module_output(case: Case) -> SearchOutput:
    """The module's SearchModule.search() output for one case (fresh fakes; the module calls no guide LLM)."""
    return SearchModule.from_resources(case_resources(case)).search(SearchInput(question=case.question, k=case.k))


def _clean(value: Any) -> Any:
    """value with the timing keys dropped from every dict, at any depth."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()
                if k not in EXCLUDED_KEYS and not k.endswith(EXCLUDED_SUFFIXES)}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def summarize_state(state: SearchState) -> dict[str, Any]:
    """The parts of the final state the later steps must keep: query, table_filter, removed, relaxed, route,
    refusal_reason, docs, and trace (node names, order and non-timing extras; seconds dropped)."""
    return {
        "query": state.query.model_dump(mode="json") if state.query is not None else None,
        "table_filter": _jsonable(state.table_filter),
        "trace": {name: _clean(v) for name, v in state.trace.items()},
        "trace_order": list(state.trace),
        "removed": state.removed,
        "relaxed": state.relaxed,
        "route": state.route,
        "refusal_reason": state.refusal_reason,
        "docs": [{"key": d.metadata.get("key", d.id), "metadata": _clean(d.metadata)} for d in state.docs],
    }


def _jsonable(value: Any) -> Any:
    """A pydantic model, dataclass or plain value as JSON-ready data (None stays None)."""
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    import dataclasses
    if dataclasses.is_dataclass(value):
        return json.loads(json.dumps(dataclasses.asdict(value), ensure_ascii=False, default=str))
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def dumps(value: Any) -> str:
    """The fixture text: sorted keys, UTF-8 Korean as is, indent 1, one trailing newline."""
    return json.dumps(_clean(value), sort_keys=True, ensure_ascii=False, indent=1) + "\n"


def case_texts(case: Case) -> dict[str, str]:
    """kind -> fixture text for one case (result, output, state)."""
    state = run_case(case)
    return {
        "result": dumps(state.result.model_dump(mode="json")),
        "output": dumps(module_output(case).model_dump(mode="json", exclude={"spans"})),
        "state": dumps(summarize_state(state)),
    }
