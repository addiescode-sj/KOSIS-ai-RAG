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
PROMPT_VERSION = "v2"
SYSTEM = (
    "당신은 KOSIS 통계표 검색 도우미입니다. 후보 통계표 목록에서 질문에 답하는 데 쓸 수 있는 표를 고르세요.\n"
    "규칙:\n"
    "- 질문이 묻는 통계 지표(세는 대상·항목, 예: 인구, 출생아 수, 전입자 수, 합계출산율)를 담은 표면 고르세요.\n"
    "- 연도·기간, 월·분기 같은 주기, 시도·시군구 같은 세부 지역 단위가 표에 맞는지는 판단하지 마세요. 코드가 따로 확인합니다.\n"
    f"- 맞는 표가 여러 개면 질문에 더 가까운 표부터, 최대 {MAX_PICK}개.\n"
    "- 질문이 묻는 지표를 담은 표가 하나도 없을 때만 picked를 빈 목록으로 두고 reason을 쓰세요: 질문이 인구·가구·출생·사망·"
    '혼인·이동 같은 인구 통계가 아닌 다른 분야면 "out_of_scope", 인구 통계지만 묻는 지표를 담은 표가 없으면 "no_item". '
    "표를 골랐으면 reason은 null.\n"
    '- 다른 말 없이 JSON 한 줄만 출력하세요: {"picked": [번호, ...], "reason": null}'
)


def user_message(question: str, passages: list[str]) -> str:
    tables = "\n\n".join(f"[{n}]\n{p}" for n, p in enumerate(passages, 1))
    return f"후보 통계표:\n{tables}\n\n질문: {question}"


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
            raise RuntimeError(f"HCX-007 call failed ({result.status}): {result.error}")
        picked, reason = parse(result.text, len(keys))
        return HcxResult(keys=[keys[i - 1] for i in picked], calls=result.attempts, usage=dict(result.usage or {}),
                         reason=reason)
