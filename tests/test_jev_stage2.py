"""T3-23 Jev: a Jev-style decision model (Clef-Flash over /v1/systemone) as stage 2.

No server, no API: the transport is a fake `send`, the selector gets a fake client.
"""

import pytest
from langchain_core.documents import Document
from pydantic import SecretStr

from kosis_rag.gate.node import GateInput, decide_route
from kosis_rag.llm.config import LLMConfig, LLMKeyMissing
from kosis_rag.llm.systemone import PATH, SystemOneClient
from kosis_rag.rerank import selectors
from kosis_rag.rerank.constants import JEV_MODEL, JEV_THRESHOLD
from kosis_rag.rerank.stage2 import rerank_stage2
from kosis_rag.rerank.systemone import DecisionSelector, cut, jev_prompt, probabilities

offline = pytest.mark.offline
SECRET = "test-key"
TEXT = {k: f"표 {k}" for k in "abcd"}


def _config(**kw):
    return selectors.default_config("jev", api_key=SECRET, base_url="https://x.example/v1", **kw)


class FakeClient:
    def __init__(self, answers, usage=None):
        self.answers, self.usage, self.sent = answers, usage or {"input_tokens": 42, "output_tokens": 0}, []

    def decide(self, state, questions):
        self.sent.append((state, questions))
        return {"model": JEV_MODEL, "answers": self.answers, "usage": self.usage}, 1


def _docs(keys):
    return [Document(page_content=k, metadata={"key": k, "score": 0.9 - i / 10}) for i, k in enumerate(keys)]


@offline
def test_default_config_and_selector_settings():
    c = _config()
    assert (c.provider, c.model, c.base_url, c.decision_mode, c.threshold) == (
        "systemone", "Cloudflare/clef-flash", "https://x.example/v1", None, None)
    s = selectors.LLM_RERANKERS["jev"](TEXT, c)
    assert isinstance(s, DecisionSelector) and (s.mode, s.threshold) == ("noul", JEV_THRESHOLD["noul"])
    s = selectors.LLM_RERANKERS["jev"](TEXT, _config(decision_mode="choice", threshold=0.2))
    assert (s.mode, s.threshold) == ("choice", 0.2)
    with pytest.raises(ValueError):
        DecisionSelector(text_of=TEXT, config=_config(decision_mode="rank"))


@offline
def test_prompts_are_read_from_md_blocks():
    noul, choice = jev_prompt("noul", "v1"), jev_prompt("choice", "v1")
    assert noul.state == "{question}" and "{table}" in noul.instructions and "<!--" not in noul.instructions
    assert "{table}" not in choice.instructions and noul.true is None
    s = DecisionSelector(text_of=TEXT, config=_config(prompt_version="v1"))
    assert s.prompt == noul
    with pytest.raises(FileNotFoundError):
        DecisionSelector(text_of=TEXT, config=_config(prompt_version="v999"))


@offline
def test_cut_orders_by_probability_and_keeps_ties_in_input_order():
    assert cut({"a": 0.2, "b": 0.9, "c": 0.6, "d": 0.6}, 0.5) == ["b", "c", "d"]
    assert cut({"a": 0.2, "b": 0.3}, 0.5) == []
    assert cut({k: 0.9 for k in "abcdefg"}, 0.5) == list("abcde")  # MAX_PICK


@offline
def test_noul_selector_sends_one_question_per_table_and_picks_above_threshold():
    fake = FakeClient({"t1": {"type": "noul", "noul": 0.1}, "t2": {"type": "noul", "noul": 0.8},
                       "t3": {"type": "noul", "noul": 0.7}})
    s = DecisionSelector(text_of=TEXT, config=_config(), client=fake)
    out = s("출생아 수", ["a", "b", "c"])
    state, questions = fake.sent[0]
    assert state == "출생아 수" and list(questions) == ["t1", "t2", "t3"]
    assert questions["t2"]["type"] == "noul" and "표 b" in questions["t2"]["instructions"]
    assert out.keys == ["b", "c"] and out.scores == {"a": 0.1, "b": 0.8, "c": 0.7} and out.usage["input_tokens"] == 42


@offline
def test_choice_selector_reads_option_probabilities_dict_or_list():
    fake = FakeClient({"pick": {"type": "choice", "choice": "t2", "probabilities": {"t1": 0.01, "t2": 0.9, "t3": 0.09}}})
    out = DecisionSelector(text_of=TEXT, config=_config(decision_mode="choice"), client=fake)("q", ["a", "b", "c"])
    assert fake.sent[0][1]["pick"]["criteria"] == {"t1": "표 a", "t2": "표 b", "t3": "표 c"}
    assert out.keys == ["b", "c"]  # 0.09 >= choice default 0.05
    assert probabilities({"answers": {"pick": {"probabilities": [0.5, 0.5]}}}, "choice", 2) == [0.5, 0.5]
    with pytest.raises(ValueError):
        probabilities({"error": "x"}, "noul", 2)
    # other Jev-style servers (K-Decision) may answer noul with a number or another key
    assert probabilities({"answers": {"t1": 0.7, "t2": {"probability": 0.4}, "t3": {}}}, "noul", 3) == [0.7, 0.4, 0.0]


@offline
def test_stage2_records_scores_and_nothing_above_threshold_is_a_refusal():
    low = FakeClient({f"t{i}": {"type": "noul", "noul": 0.1} for i in range(1, 5)})
    docs = rerank_stage2("q", _docs("abcd"), DecisionSelector(text_of=TEXT, config=_config(), client=low), name="jev")
    assert [d.metadata["key"] for d in docs] == list("abcd")
    assert docs[0].metadata["stage2_picked_n"] == 0 and docs[0].metadata["stage2_scores"]["a"] == 0.1
    assert decide_route(GateInput(docs=docs)).route == "refuse"
    high = FakeClient({"t1": {"noul": 0.2}, "t2": {"noul": 0.3}, "t3": {"noul": 0.95}, "t4": {"noul": 0.6}})
    docs = rerank_stage2("q", _docs("abcd"), DecisionSelector(text_of=TEXT, config=_config(), client=high), name="jev")
    assert [d.metadata["key"] for d in docs] == ["c", "d", "a", "b"] and decide_route(GateInput(docs=docs)).route == "found"


@offline
def test_stage2_without_scores_keeps_the_old_metadata():
    docs = rerank_stage2("q", _docs("ab"), lambda q, ks: ["b"])
    assert "stage2_scores" not in docs[0].metadata


@offline
def test_client_retries_server_errors_then_fails_and_needs_url_and_key():
    calls = []

    def send(url, body, key, timeout):
        calls.append((url, body["model"], key))
        return (503, {"error": "busy"}) if len(calls) < 3 else (200, {"answers": {}, "usage": {}})

    client = SystemOneClient(_config(), waits=(0, 0, 0), send=send)
    reply, n = client.decide("q", {"t1": {"type": "noul"}})
    assert n == 3 and calls[0] == ("https://x.example/v1" + PATH, JEV_MODEL, SECRET)
    bad = SystemOneClient(_config(), waits=(0,), send=lambda *a: (401, {"error": "bad key"}))
    with pytest.raises(RuntimeError, match="HTTP 401"):
        bad.decide("q", {})
    with pytest.raises(ValueError):
        SystemOneClient(LLMConfig(provider="systemone", model=JEV_MODEL, api_key=SecretStr(SECRET))).decide("q", {})
    with pytest.raises(LLMKeyMissing):
        SystemOneClient(selectors.default_config("jev", base_url="https://x.example/v1")).decide("q", {})
