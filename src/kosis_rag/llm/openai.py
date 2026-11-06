"""OpenAIClient: the HCXClient generate() contract over OpenAI chat completions (T2-41 S3: from hcx.py).

The same prompt and parser as the HCX stage 2 selector with another model (T2-09 lab: HCX-007 vs gpt-4o-mini), and
the hq question generator's client (T2-39). The key comes from LLMConfig.api_key (it used to be OPENAI_API_KEY from
this repo's .env, read here); the SDK retries 429/5xx.
"""

from __future__ import annotations

from typing import Any

from kosis_rag.llm.base import STATUS_OK
from kosis_rag.llm.config import LLMConfig, require_key

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.0  # the stage 2 re-ranker's defaults: temperature 0, 512 tokens, no seed
DEFAULT_MAX_TOKENS = 512


class OpenAIClient:
    """generate(system, user, history=None) over OpenAI chat completions in JSON mode.

    The defaults (config fields left None) are the stage 2 re-ranker's (temperature 0, 512 tokens, no seed). The
    question generator of T2-39 (kosis_rag.hq) sets temperature, max_tokens and seed, and passes earlier turns as
    `history` to ask once more.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    @property
    def model(self) -> str:
        return self.config.model or DEFAULT_MODEL

    @property
    def temperature(self) -> float:
        return DEFAULT_TEMPERATURE if self.config.temperature is None else self.config.temperature

    @property
    def max_tokens(self) -> int:
        return DEFAULT_MAX_TOKENS if self.config.max_tokens is None else self.config.max_tokens

    @property
    def seed(self) -> int | None:
        return self.config.seed

    def generate(self, system: str, user: str, history: list[dict[str, str]] | None = None) -> Any:
        """One chat completion in JSON mode.

        In: system = system prompt; user = the last user message; history = earlier turns ({"role", "content"})
            placed between the system prompt and `user` (None: a single turn, as before).
        Out: SimpleNamespace(status, text, attempts, error, usage={"input_tokens", "output_tokens"}).
        Side effects: one OpenAI API call (the SDK retries 429/5xx); its errors are raised, and LLMKeyMissing when
            the config has no key.
        Node: rerank2 (through HcxSelector) and the hq build's generate / validate.
        """
        from types import SimpleNamespace

        from openai import OpenAI

        key = require_key(self.config)
        extra = {"seed": self.seed} if self.seed is not None else {}
        client_args: dict[str, Any] = {"timeout": self.config.timeout} if self.config.timeout is not None else {}
        r = OpenAI(api_key=key, max_retries=5, **client_args).chat.completions.create(
            model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, *(history or []), {"role": "user", "content": user}],
            **extra)
        return SimpleNamespace(status=STATUS_OK, text=r.choices[0].message.content or "", attempts=1, error=None,
                               usage={"input_tokens": r.usage.prompt_tokens, "output_tokens": r.usage.completion_tokens})
