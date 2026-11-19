"""HCX-007 as a stage 2 LLM re-ranker that may pick nothing: the no-answer signal of T2-09 (T2-41 S3: from hcx.py).

The model sees the question and the top k_in table passages (numbered), and answers one JSON line
{"picked": [numbers], "reason": "out_of_scope" | "no_item" | null}. Numbers map back to the input keys, so a table id
can never be invented. An empty pick is the no-answer verdict (rerank_stage2 records stage2_picked_n = 0) and
`reason` says why. Period mismatch is not asked: the T2-14 condition filter decides it in code.

The system prompt is rerank/prompts/stage2-hcx-v3.md (package data, read through importlib.resources; the bytes of
the old hcx.SYSTEM constant, tests/test_prompts_bytes.py). HTTP goes through kosis_rag.llm (the HCXClient copy of
Phase 1: thinking.effort, maxCompletionTokens, retries), built from an LLMConfig at the first call. A failed call or
an unreadable reply raises, so rerank_stage2 keeps the stage 1 order and records the error instead of refusing.

KOS-192 (opt-in, off by default): a selector built with prompt=PROMPT_FILE_RW (stage2-hcx-v5-rw.md) and called with
hint=<rewrite> sees one more user message line `검색용 표현: <rewrite>` under the question. Without both the selector
sends what it sent before (same system prompt, same user message bytes).
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

from kosis_rag.llm.base import STATUS_OK, LLMClient
from kosis_rag.llm.config import LLMConfig, LLMKeyMissing
from kosis_rag.llm.providers import build_client
from kosis_rag.rerank.constants import HINT_LABEL, MAX_PICK, MAX_TOKENS, MODEL, PROMPT_FILE

REASONS = ("out_of_scope", "no_item")


@functools.cache
def system_prompt(name: str = PROMPT_FILE) -> str:
    """The selector's system prompt: rerank/prompts/<name> as UTF-8 text (read once).

    Side effects: reads the package data file (FileNotFoundError when it is missing). Node: rerank2.
    """
    return (resources.files("kosis_rag.rerank") / "prompts" / name).read_text(encoding="utf-8")


def user_message(question: str, passages: list[str], hint: str | None = None) -> str:
    """The selector's user message: the question, then the numbered passages.

    In: question; passages = one table passage per candidate, in rank order; hint = KOS-192 rewrite path only: a
        search phrase shown on its own line `검색용 표현: <hint>` right under the question, its whitespace (line
        breaks included) collapsed to single spaces so it is always one line (None, or nothing but whitespace: no
        line, the message is byte for byte the one without the argument).
    Out: the message text. Side effects: none. Node: rerank2.
    """
    tables = "\n\n".join(f"[{n}] {p}" for n, p in enumerate(passages, 1))
    flat = " ".join(hint.split()) if hint else ""
    head = f"질문: {question}" + (f"\n{HINT_LABEL}: {flat}" if flat else "")
    return f"{head}\n\n후보 통계표:\n{tables}"


def parse(text: str, n: int) -> tuple[list[int], str | None]:
    """Reply -> (picked numbers within 1..n in reply order, reason). Raises ValueError when no JSON object is found."""
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON in reply: {text[:80]!r}")
    data, _ = json.JSONDecoder().raw_decode(text, start)  # the first object; the model sometimes repeats it
    picked = [int(x) for x in data.get("picked") or [] if str(x).isdigit() and 1 <= int(x) <= n]
    picked = list(dict.fromkeys(picked))[:MAX_PICK]
    reason = data.get("reason")
    return picked, (reason if not picked and reason in REASONS else None)


@dataclass
class HcxResult:
    keys: list[str]  # picked tables, most direct first
    calls: int = 1
    usage: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None  # why nothing was picked


@dataclass
class HcxSelector:
    """select(question, keys) -> HcxResult. `text_of` maps a table key to the passage sent for it.

    thinking: HCX-007 thinking.effort (None: config.thinking, or "none" without a config). client: the LLM client
    (tests pass a fake); None = built from `config` at the first call (provider hcx: HCXClient with model
    config.model, max_tokens MAX_TOKENS[thinking] unless config.max_tokens is set, temperature 0 unless set).
    prompt: the system prompt file under rerank/prompts (default PROMPT_FILE; KOS-192: PROMPT_FILE_RW for the rewrite
    path, which also passes `hint` to each call).
    """

    text_of: dict[str, str]
    thinking: str | None = None
    client: LLMClient | None = None  # replaced by a fake in tests
    config: LLMConfig | None = None
    prompt: str = PROMPT_FILE  # KOS-192: rerank/prompts/<file> sent as the system prompt

    def __post_init__(self) -> None:
        if self.thinking is None:
            self.thinking = self.config.thinking if self.config is not None else "none"
        if self.thinking not in MAX_TOKENS:
            raise ValueError(f"unknown HCX_THINKING {self.thinking!r}; known: {', '.join(MAX_TOKENS)}")

    def _client(self) -> Any:
        if self.config is None:
            raise LLMKeyMissing("HcxSelector needs client= or config= (an LLMConfig with api_key)")
        c = self.config
        return build_client(c.model_copy(update={
            "thinking": self.thinking, "max_tokens": MAX_TOKENS[self.thinking] if c.max_tokens is None else c.max_tokens,
            "temperature": 0.0 if c.temperature is None else c.temperature}))

    def __call__(self, question: str, keys: list[str], hint: str | None = None) -> HcxResult:
        """Pick among `keys` for `question`; hint = KOS-192 search phrase line of the user message (None: none, the
        call is the one without the argument). rerank_stage2 calls select(question, keys): a caller that wants a
        hint binds it (functools.partial(selector, hint=...))."""
        if not keys:
            return HcxResult(keys=[], calls=0)
        self.client = self.client or self._client()
        result = self.client.generate(system_prompt(self.prompt),
                                      user_message(question, [self.text_of[k] for k in keys], hint))
        if result.status != STATUS_OK:
            model = getattr(self.client, "model", self.config.model if self.config is not None else MODEL)
            raise RuntimeError(f"{model} call failed ({result.status}): {result.error}")
        picked, reason = parse(result.text, len(keys))
        return HcxResult(keys=[keys[i - 1] for i in picked], calls=result.attempts, usage=dict(result.usage or {}),
                         reason=reason)
