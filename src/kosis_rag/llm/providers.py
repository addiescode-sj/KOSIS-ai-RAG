"""Provider table (T2-41 S3, P2-D8): LLMConfig.provider -> the function that builds its client.

    hcx           HCXClient(config)                    (llm/hcx.py, Phase 1 copy; the stage 2 HCX selector)
    openai        OpenAIClient(config)                 (llm/openai.py; stage 2 "gpt", hq question generation)
    clova_rerank  ClovaReranker(config=config, text_of=...)   (llm/clova_rerank.py; stage 2 "clova")
    clova_chat    clova_chat(config) -> ChatClovaX     (the guide LLM; the backend's from S5)
    mock          MockClient()                         (tests and CI only, no key)

A client checks the key when it needs it (HCXClient when made, the others at their first call), so building a
client never reads .env or the environment. Another provider is one more entry with the same signature.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kosis_rag.llm.base import MockClient
from kosis_rag.llm.clova_rerank import ClovaReranker
from kosis_rag.llm.config import LLMConfig
from kosis_rag.llm.hcx import HCXClient
from kosis_rag.llm.openai import OpenAIClient

CLOVA_CHAT_TEMPERATURE = 0.0  # Phase 1 configs/model.yaml defaults of the guide model (tests/test_phase1_contract.py)
CLOVA_CHAT_MAX_TOKENS = 1024
CLOVA_CHAT_TIMEOUT = 60.0


def clova_chat(config: LLMConfig) -> Any:
    """HCX through langchain-naver's ChatClovaX (the guide LLM).

    In: config = provider clova_chat; model, temperature (None: 0), max_tokens (None: 1024), timeout (None: 60 s),
        api_key (None: ChatClovaX falls back to its own CLOVASTUDIO_API_KEY, as before T2-41).
    Out: a ChatClovaX with stream_usage on (streamed chunks carry token usage, T2-38).
    Side effects: none (no call until it is invoked). Node: guide (backend; chain.clova_chat until S5).
    """
    from langchain_naver import ChatClovaX

    key = config.api_key.get_secret_value() if config.api_key is not None else None
    return ChatClovaX(
        model=config.model,
        temperature=CLOVA_CHAT_TEMPERATURE if config.temperature is None else config.temperature,
        max_tokens=CLOVA_CHAT_MAX_TOKENS if config.max_tokens is None else config.max_tokens,
        timeout=CLOVA_CHAT_TIMEOUT if config.timeout is None else config.timeout,
        **({"api_key": key} if key else {}),
        stream_usage=True,  # off by default for a non-OpenAI base URL: streamed chunks would carry no token usage (T2-38)
    )


PROVIDERS: dict[str, Callable[..., Any]] = {
    "hcx": HCXClient,
    "openai": OpenAIClient,
    "clova_rerank": lambda config, **kw: ClovaReranker(config=config, **kw),
    "clova_chat": clova_chat,
    "mock": lambda config, **kw: MockClient(**kw),
}


def build_client(config: LLMConfig, **kwargs: Any) -> Any:
    """The client for config.provider (PROVIDERS).

    In: config = the LLM settings; kwargs = provider-specific arguments (clova_rerank: text_of; mock: response).
    Out: the client. Side effects: HCXClient raises LLMKeyMissing for a config without a key; ValueError for an
        unknown provider. Node: rerank2 (the HCX selector builds its client at the first call).
    """
    if config.provider not in PROVIDERS:
        raise ValueError(f"unknown LLM provider {config.provider!r}; known: {', '.join(PROVIDERS)}")
    return PROVIDERS[config.provider](config, **kwargs)
