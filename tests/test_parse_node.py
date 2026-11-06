"""T2-41 S2: the parse stage step functions (parse.node) and the injected 시군구 dictionary (P2-D9).

parse used to read data/sigungu.json at import (query.SIGUNGU); now load_resources reads it once into
SearchResources.sigungu and parse_question passes it to parse_query. No model, no server.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from kosis_rag.adapters import files
from kosis_rag.parse.node import decide_condition, parse_query, parse_question
from kosis_rag.parse.schemas import ConditionInput, ConditionOutput, ParsedQuery, ParseInput
from kosis_rag.adapters import loader
from kosis_rag.adapters.loader import load_resources
from kosis_rag.schemas.resources import IndexSpec, SearchOptions
from kosis_rag.retrieve.lists import condition_filter
from kosis_rag.shared.filters import filter_with_relaxation, relax
from tests.test_graph import RECORDS, CatalogRetriever, KEYS

pytestmark = pytest.mark.offline
ROOT = Path(__file__).resolve().parents[1]
SIGUNGU = {"해운대구": ["부산광역시"], "수원시": ["경기도"], "중구": ["서울특별시", "부산광역시"]}
QUESTIONS = ["해운대구와 수원시 인구", "서울 중구 인구", "중구 인구", "2023년 부산 출생아 수", "2023년 성별 인구",
             "1990년 인구", "출생아 수"]


def _resources(sigungu=None, **kw):
    options = SearchOptions(**{"cond_filter": True, "stage2": "none", **kw})
    return load_resources(spec=IndexSpec(), options=options, records=RECORDS, dense=CatalogRetriever(keys=KEYS),
                          sigungu=sigungu)


def test_importing_parse_reads_no_file():
    """Done S2: `import kosis_rag.parse.node` reads no data/sigungu.json and logs nothing (it used to at import)."""
    code = ("import builtins, logging, pathlib\n"
            "seen = []\n"
            "real_open, real_read, real_exists = builtins.open, pathlib.Path.read_text, pathlib.Path.exists\n"
            "builtins.open = lambda f, *a, **k: (seen.append(str(f)), real_open(f, *a, **k))[1]\n"
            "pathlib.Path.read_text = lambda self, *a, **k: (seen.append(str(self)), real_read(self, *a, **k))[1]\n"
            "pathlib.Path.exists = lambda self, *a, **k: (seen.append(str(self)), real_exists(self, *a, **k))[1]\n"
            "logging.basicConfig(level=logging.DEBUG)\n"
            "import kosis_rag.parse.node, kosis_rag.legacy.retrievers\n"
            "print([s for s in seen if 'sigungu' in s])\n")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}  # pytest.ini's pythonpath too
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", cwd=ROOT,
                         env=env, check=True)
    assert out.stdout.strip() == "[]" and "sigungu" not in out.stderr


@pytest.mark.parametrize("question", QUESTIONS)
def test_parse_question_passes_the_resources_dictionary(question):
    """parse_question == parse_query with the injected dictionary (what the old global gave for that file)."""
    for sigungu in ({}, SIGUNGU):
        resources = _resources(sigungu)
        assert parse_question(ParseInput(question=question), resources) == parse_query(question, sigungu=sigungu)


def test_the_dictionary_changes_only_the_regions():
    with_file = parse_question(ParseInput(question="해운대구와 수원시 인구"), _resources(SIGUNGU))
    without = parse_question(ParseInput(question="해운대구와 수원시 인구"), _resources({}))
    assert with_file.regions == ["부산광역시", "경기도"] and without.regions == []
    assert with_file.model_copy(update={"regions": []}) == without


def test_load_resources_reads_the_dictionary_once(monkeypatch, caplog):
    calls = []
    monkeypatch.setattr(files, "load_sigungu", lambda: calls.append(1) or dict(SIGUNGU))
    assert _resources().sigungu == SIGUNGU and calls == [1]
    assert _resources({"수원시": ["경기도"]}).sigungu == {"수원시": ["경기도"]} and calls == [1]  # given: not read
    monkeypatch.setattr(files, "load_sigungu", lambda: {})
    monkeypatch.setattr(loader, "_SIGUNGU_WARNED", [])  # the warning is once per process
    with caplog.at_level("WARNING", logger="kosis_rag.adapters.loader"):
        assert _resources().sigungu == {}
    assert "sigungu.json not found" in caplog.text


@pytest.mark.parametrize("question", QUESTIONS)
@pytest.mark.parametrize("cond", [True, False])
def test_decide_condition_equals_the_old_condition_node(question, cond):
    """The pre-S2 graph condition body: to_filter, relax, then the hybrid's condition_filter step."""
    resources = _resources({}, cond_filter=cond)
    parsed = parse_query(question)
    old = parsed.to_filter() if cond else None
    relaxed = []
    if old is not None and not old.is_empty():
        _, relaxed, _ = filter_with_relaxation(resources.records, old)
        old = relax(old, relaxed)
    old = condition_filter(old, resources.records)
    assert decide_condition(ConditionInput(query=parsed), resources) == ConditionOutput(table_filter=old, relaxed=relaxed)
    assert decide_condition(ConditionInput(query=None), resources) == ConditionOutput()


def test_step_models_are_strict():
    with pytest.raises(ValidationError):
        ParseInput(question="q", k=3)  # unknown field
    with pytest.raises(ValidationError):
        ParseInput(question=1)  # no coercion
    assert ParseInput("q").question == "q"  # positional, in field order
    assert ConditionInput(ParsedQuery(years=[2023])).query.years == [2023]
