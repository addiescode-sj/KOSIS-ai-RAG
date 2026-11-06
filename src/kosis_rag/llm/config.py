"""LLM settings the caller declares and passes in (T2-41 S3, plans/T2-41.md 입출력 계약; §0.6: no .env here).

LLMConfig names the provider (llm/providers.py PROVIDERS), the model and the key. The key may be None when the config
is made: a node that needs the LLM raises LLMKeyMissing when it starts (rerank2; backend guide from S5), so a search
that needs no LLM (stage2 "none") runs without a key. provider "mock" never needs one.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, SecretStr

Provider = Literal["clova_chat", "clova_rerank", "hcx", "openai", "mock"]


class LLMKeyMissing(RuntimeError):
    """An LLM node started with a config that has no api_key (provider other than mock)."""


class LLMConfig(BaseModel):
    """One LLM the module calls: provider, key, model and the request settings.

    None in an optional field means "the client's own default" (HCX: temperature 0, timeout 60 s; the stage 2
    selectors pick max_tokens from `thinking`). The key is a SecretStr: it is not shown by repr() or model_dump().
    """

    provider: Provider
    api_key: SecretStr | None = None  # None: LLMKeyMissing when an LLM node starts (provider != "mock")
    model: str
    thinking: str = "none"  # HCX-007 thinking.effort: none | low | medium | high (was HCX_THINKING)
    temperature: float | None = None
    max_tokens: int | None = None
    seed: int | None = None
    retry_interval: float = 0.0  # seconds between a caller's repeated calls (Phase 1 summarize_stat); the clients here
    # do not use it (HCXClient keeps Phase 1's RETRY_DELAYS)
    timeout: float | None = None  # request timeout in seconds; None = the client's default (clova_chat: 60)


def require_key(config: LLMConfig | None) -> str | None:
    """The config's key as text; raises LLMKeyMissing when a non-mock config has none.

    In: config = the LLM settings (None: nothing to check). Out: the key (None for mock or no config).
    Side effects: raises LLMKeyMissing. Node: rerank2 (on entry) and the clients that send the key.
    """
    if config is None or config.provider == "mock":
        return config.api_key.get_secret_value() if config is not None and config.api_key is not None else None
    if config.api_key is None:
        raise LLMKeyMissing(f"LLMConfig.api_key is required for provider={config.provider!r}")
    return config.api_key.get_secret_value()
