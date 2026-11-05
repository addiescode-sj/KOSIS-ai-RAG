"""HCX-007 as a stage 2 LLM re-ranker that may pick nothing: the no-answer signal of T2-09.

The model sees the question and the top k_in table passages (numbered), and answers one JSON line
{"picked": [numbers], "reason": "out_of_scope" | "no_item" | null}. Numbers map back to the input keys, so a table id
can never be invented. An empty pick is the no-answer verdict (rerank.LLMReranker records stage2_picked_n = 0) and
`reason` says why. Period mismatch is not asked: the T2-14 condition filter decides it in code.

HTTP goes through Phase 1 kosis_chat HCXClient (thinking.effort, maxCompletionTokens, retries). A failed call or an
unreadable reply raises, so LLMReranker keeps the stage 1 order and records the error instead of refusing.

    HCX_THINKING=none | low   # thinking.effort sent to HCX-007 (default none)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kosis_chat.llm.model import STATUS_OK, LLMClient

MODEL = "HCX-007"
REASONS = ("out_of_scope", "no_item")
MAX_PICK = 5
# thinking tokens count toward maxCompletionTokens (Phase 1 configs/model.yaml): leave room when thinking is on
MAX_TOKENS = {"none": 512, "low": 5120, "medium": 5120, "high": 8192}

# v2 (T2-09 dev): v1 asked for tables "that directly hold the number" and refused even rank-1 gold tables
# (lab/T2-09-hcx-no-answer/prompt_v1.txt); v2 asks for the measure only and leaves years and region detail to code.
# v3 clarifies the selector contract and examples while preserving v2's measure-only decision boundary.
PROMPT_VERSION = "v3"
SYSTEM = (
    "당신은 KOSIS 인구통계 통계표 검색의 2차 선택기입니다. 사용자의 질문에 답하는 데 쓸 수 있는 후보 통계표를 고르세요. "
    "질문에 직접 답하거나 질문을 다시 쓰지 마세요.\n\n"
    "# 입력 형식\n"
    "질문: 사용자의 원래 질문\n"
    "후보 통계표:\n"
    "[1] 첫 번째 통계표 설명\n"
    "[2] 두 번째 통계표 설명\n"
    "...\n\n"
    "# 판단 단계\n"
    "1. 질문에서 찾으려는 통계 지표와 세는 대상·항목을 파악하세요.\n"
    "2. 각 후보 통계표 설명에 그 지표가 담겨 있는지 확인하세요. 주제만 비슷하고 지표가 다르면 고르지 마세요. "
    "설명에 없는 지표가 있다고 추측하지 마세요.\n"
    f"3. 지표가 맞는 표를 질문과 가까운 순서로 최대 {MAX_PICK}개 고르세요. "
    "출력에는 입력에 나온 후보 번호만 사용하고 같은 번호를 반복하지 마세요.\n"
    "4. 맞는 표가 없으면 사유를 정하세요. 질문이 인구·가구·출생·사망·혼인·이동 등 인구통계 범위 밖이면 "
    "out_of_scope, 인구통계 질문이지만 후보에 필요한 지표가 없으면 no_item입니다.\n"
    "5. 출력 형식을 확인하세요. 표를 하나라도 골랐으면 reason은 null, 고른 표가 없으면 picked는 []이고 "
    "reason은 두 사유 중 하나입니다.\n\n"
    "# 판단 범위\n"
    "연도·기간, 월·분기·연 주기, 시도·시군구 등 세부 지역 단위가 맞는지는 판단하지 마세요. "
    "이 조건은 별도 코드가 확인합니다. 지표가 맞으면 해당 조건이 달라도 표를 고르세요.\n\n"
    "# 출력 형식\n"
    "설명이나 Markdown 없이 JSON 객체 한 줄만 출력하세요.\n"
    '{"picked": [후보 번호, ...], "reason": null}\n'
    "또는\n"
    '{"picked": [], "reason": "out_of_scope"}\n'
    "또는\n"
    '{"picked": [], "reason": "no_item"}\n\n'
    "# 예시\n"
    "입력:\n"
    "질문: 2025년 출생아 수는?\n"
    "후보 통계표:\n"
    "[1] 출생아 수, 2020~2024년\n"
    "[2] 합계출산율, 2020~2024년\n"
    "출력:\n"
    '{"picked": [1], "reason": null}\n\n'
    "입력:\n"
    "질문: 전입자 수는?\n"
    "후보 통계표:\n"
    "[1] 출생아 수\n"
    "[2] 사망자 수\n"
    "출력:\n"
    '{"picked": [], "reason": "no_item"}\n\n'
    "입력:\n"
    "질문: 자동차 등록 대수는?\n"
    "후보 통계표:\n"
    "[1] 주민등록인구\n"
    "출력:\n"
    '{"picked": [], "reason": "out_of_scope"}'
)


def user_message(question: str, passages: list[str]) -> str:
    tables = "\n\n".join(f"[{n}] {p}" for n, p in enumerate(passages, 1))
    return f"질문: {question}\n\n후보 통계표:\n{tables}"


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
    """select(question, keys) -> HcxResult. `text_of` maps a table key to the passage sent for it."""

    text_of: dict[str, str]
    thinking: str = field(default_factory=lambda: (os.getenv("HCX_THINKING") or "none").strip().lower())
    client: LLMClient | None = None  # replaced by a fake in tests

    def __post_init__(self) -> None:
        if self.thinking not in MAX_TOKENS:
            raise ValueError(f"unknown HCX_THINKING {self.thinking!r}; known: {', '.join(MAX_TOKENS)}")

    def _client(self) -> LLMClient:
        from kosis_chat.config import load_env, load_settings
        from kosis_chat.llm.model import HCXClient

        load_env(Path(__file__).resolve().parents[2] / ".env")  # this repo's HCX_API_KEY, as clova.api_key does
        return HCXClient(load_settings(backend="hcx", model_name=MODEL, thinking_level=self.thinking,
                                       max_tokens=MAX_TOKENS[self.thinking], temperature=0.0))

    def __call__(self, question: str, keys: list[str]) -> HcxResult:
        if not keys:
            return HcxResult(keys=[], calls=0)
        self.client = self.client or self._client()
        result = self.client.generate(SYSTEM, user_message(question, [self.text_of[k] for k in keys]))
        if result.status != STATUS_OK:
            raise RuntimeError(f"{getattr(self.client, 'model', MODEL)} call failed ({result.status}): {result.error}")
        picked, reason = parse(result.text, len(keys))
        return HcxResult(keys=[keys[i - 1] for i in picked], calls=result.attempts, usage=dict(result.usage or {}),
                         reason=reason)


@dataclass
class OpenAIClient:
    """Same generate() contract as HCXClient, over OpenAI chat completions: the same prompt and parser with another
    model (T2-09 lab: HCX-007 vs gpt-4o-mini). OPENAI_API_KEY comes from this repo's .env; the SDK retries 429/5xx.

    The defaults are the stage 2 re-ranker's (temperature 0, 512 tokens, no seed). The question generator of T2-39
    (kosis_rag.hq) sets temperature, max_tokens and seed, and passes earlier turns as `history` to ask once more.
    """

    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = MAX_TOKENS["none"]
    seed: int | None = None

    def generate(self, system: str, user: str, history: list[dict[str, str]] | None = None) -> Any:
        """One chat completion in JSON mode.

        In: system = system prompt; user = the last user message; history = earlier turns ({"role", "content"})
            placed between the system prompt and `user` (None: a single turn, as before).
        Out: SimpleNamespace(status, text, attempts, error, usage={"input_tokens", "output_tokens"}).
        Side effects: one OpenAI API call (the SDK retries 429/5xx); its errors are raised.
        Node: rerank2 (through HcxSelector) and the hq build's generate / validate.
        """
        from types import SimpleNamespace

        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        extra = {"seed": self.seed} if self.seed is not None else {}
        r = OpenAI(max_retries=5).chat.completions.create(
            model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, *(history or []), {"role": "user", "content": user}],
            **extra)
        return SimpleNamespace(status=STATUS_OK, text=r.choices[0].message.content or "", attempts=1, error=None,
                               usage={"input_tokens": r.usage.prompt_tokens, "output_tokens": r.usage.completion_tokens})
