"""T2-41 S3 (§0.6, P2-D8): LLM keys come in as LLMConfig arguments; a node that needs the LLM refuses a missing key
when it starts (LLMKeyMissing), a search that needs none runs without a key; the provider table.

No API, no .env: the graph runs on tests/test_graph.py's fakes, every LLMConfig is built here.
"""

import pytest
from pydantic import SecretStr

pytest.importorskip("langgraph")

from kosis_chat.llm.model import MockClient  # noqa: E402
from kosis_rag import chain, pipeline  # noqa: E402
from kosis_rag.graph import build_graph, run_search, stream_search  # noqa: E402
from kosis_rag.llm import providers  # noqa: E402
from kosis_rag.llm.base import MockClient as OurMock  # noqa: E402
from kosis_rag.llm.clova_rerank import ClovaReranker  # noqa: E402
from kosis_rag.llm.config import LLMConfig, LLMKeyMissing, Provider, require_key  # noqa: E402
from kosis_rag.llm.hcx import HCXClient  # noqa: E402
from kosis_rag.llm.openai import OpenAIClient  # noqa: E402
from kosis_rag.pipeline import load_resources  # noqa: E402
from kosis_rag.rerank import selectors  # noqa: E402
from kosis_rag.rerank.hcx import HcxSelector  # noqa: E402

from tests.test_graph import RECORDS, KEYS, CatalogRetriever, _settings, _stage1  # noqa: E402

SECRET = "nv-test-secret-0000"


def _graph(stage2: str, stage2_llm: LLMConfig | None):
    resources = load_resources(_settings(stage2=stage2), records=RECORDS, dense=CatalogRetriever(keys=KEYS),
                               stage1_score=_stage1, stage2_llm=stage2_llm)
    return resources, build_graph(resources, MockClient("[1] DT_A 표입니다."))


def test_keyless_search_without_stage2_runs_to_the_end():
    resources, graph = _graph("none", None)
    assert resources.select is None and resources.stage2_llm is None
    state = run_search(graph, "출생아 수")
    assert state.result is not None and state.route == "guide" and "rerank2" not in state.trace


def test_missing_key_raises_when_rerank2_starts_after_the_rules_ran():
    resources, graph = _graph("hcx", LLMConfig(provider="hcx", model="HCX-007"))
    assert isinstance(resources.select, HcxSelector) and resources.stage2_llm.api_key is None  # made without a key
    seen = []
    with pytest.raises(LLMKeyMissing, match="provider='hcx'"):
        for name, _ in stream_search(graph, "출생아 수"):
            seen.append(name)
    assert seen[:2] == ["parse", "condition"] and {"retrieve_dense", "retrieve_bm25", "fuse", "rerank1"} <= set(seen)
    assert "rerank2" not in seen and "gate" not in seen
    assert issubclass(LLMKeyMissing, RuntimeError)


def test_mock_provider_needs_no_key():
    resources, graph = _graph("hcx", LLMConfig(provider="mock", model="mock"))
    state = run_search(graph, "출생아 수")  # MockClient's "[mock] answer" is no JSON: a stage 2 error, not a refusal
    top = state.docs[0].metadata
    assert top["stage2"] == "hcx" and "no JSON" in top["stage2_error"] and state.route == "guide"


def test_a_given_selector_is_not_checked_for_a_key():
    resources = load_resources(_settings(), records=RECORDS, dense=CatalogRetriever(keys=KEYS),
                               select=lambda q, ks: ["101_DT_B"])
    assert resources.stage2_llm is None and resources.stage2_name == "custom"
    state = run_search(build_graph(resources, MockClient("[1] DT_B 표입니다.")), "출생아 수")
    assert state.docs[0].metadata["key"] == "101_DT_B"


def test_secret_is_not_shown():
    config = LLMConfig(provider="hcx", model="HCX-007", api_key=SECRET)
    for shown in (repr(config), str(config), str(config.model_dump()), config.model_dump_json()):
        assert SECRET not in shown
    assert isinstance(config.api_key, SecretStr) and require_key(config) == SECRET
    assert require_key(None) is None and require_key(LLMConfig(provider="mock", model="m")) is None
    with pytest.raises(LLMKeyMissing):
        require_key(LLMConfig(provider="openai", model="gpt-4o-mini"))


def test_provider_table_covers_every_provider():
    assert set(providers.PROVIDERS) == set(Provider.__args__)
    assert isinstance(providers.build_client(LLMConfig(provider="mock", model="m")), OurMock)
    assert isinstance(providers.build_client(LLMConfig(provider="openai", model="gpt-4o-mini")), OpenAIClient)
    assert isinstance(providers.build_client(LLMConfig(provider="clova_rerank", model="r"), text_of={}), ClovaReranker)
    hcx = providers.build_client(LLMConfig(provider="hcx", model="HCX-007", api_key=SECRET))
    assert isinstance(hcx, HCXClient) and hcx.url == "https://clovastudio.stream.ntruss.com/v3/chat-completions/HCX-007"
    with pytest.raises(LLMKeyMissing):
        providers.build_client(LLMConfig(provider="hcx", model="HCX-007"))
    with pytest.raises(ValueError, match="unknown LLM provider"):
        providers.build_client(LLMConfig.model_construct(provider="nope", model="m", api_key=None))


def test_clova_chat_takes_the_config():
    chat_mod = pytest.importorskip("langchain_naver")  # optional: the guide LLM client (not installed in every env)
    chat = providers.clova_chat(LLMConfig(provider="clova_chat", model="HCX-DASH-002", api_key=SECRET))
    assert (chat.model_name, chat.temperature, chat.max_tokens) == ("HCX-DASH-002", 0.0, 1024)
    given = LLMConfig(provider="clova_chat", model="HCX-DASH-002", api_key=SECRET, max_tokens=64)
    assert chain.clova_chat(given).max_tokens == 64  # a given config is used as is (no .env read)


def test_stage2_defaults_and_selector_settings():
    assert {n: (c := selectors.default_config(n)).provider for n in selectors.LLM_RERANKERS} == {
        "clova": "clova_rerank", "hcx": "hcx", "gpt": "openai"}
    assert selectors.default_config("hcx").model == "HCX-007" and selectors.default_config("gpt").model == "gpt-4o-mini"
    config = selectors.default_config("hcx", api_key=SECRET, thinking="low")
    select = selectors.LLM_RERANKERS["hcx"]({"a": "A"}, config)
    assert select.thinking == "low"
    body = select._client()._body("s", "u")  # the old Settings: HCX-007, maxCompletionTokens by thinking, temperature 0
    assert body["maxCompletionTokens"] == 5120 and body["thinking"] == {"effort": "low"} and body["temperature"] == 0.0
    gpt = selectors.LLM_RERANKERS["gpt"]({"a": "A"}, selectors.default_config("gpt"))
    assert isinstance(gpt.client, OpenAIClient) and (gpt.client.model, gpt.client.max_tokens) == ("gpt-4o-mini", 512)
    assert gpt.client.temperature == 0.0 and gpt.client.seed is None


def test_env_stage2_llm_reads_the_old_variables(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)  # never reads the repo's .env here
    monkeypatch.setenv("HCX_API_KEY", SECRET)
    monkeypatch.setenv("HCX_THINKING", " LOW ")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    hcx = pipeline.env_stage2_llm("hcx")
    assert (hcx.provider, hcx.model, hcx.thinking, require_key(hcx)) == ("hcx", "HCX-007", "low", SECRET)
    assert pipeline.env_stage2_llm("clova").provider == "clova_rerank"
    assert pipeline.env_stage2_llm("gpt").api_key is None  # OPENAI_API_KEY unset: LLMKeyMissing at rerank2


def test_clova_reranker_takes_the_key_from_the_config():
    sent = []
    select = ClovaReranker(text_of={"a": "A"}, waits=(), config=LLMConfig(provider="clova_rerank", model="r",
                                                                         api_key=SECRET, timeout=5.0),
                           send=lambda body, key, timeout: sent.append((key, timeout)) or {
                               "status": {"code": "20000"}, "result": {"citedDocuments": [{"id": "1"}]}})
    assert select("q", ["a"]).keys == ["a"] and sent == [(SECRET, 5.0)]
    assert ClovaReranker(text_of={}).timeout == 60.0  # the old default
    with pytest.raises(LLMKeyMissing):
        ClovaReranker(text_of={"a": "A"}, config=LLMConfig(provider="clova_rerank", model="r"))("q", ["a"])
