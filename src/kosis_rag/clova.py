"""CLOVA Studio reranker API as a stage 2 LLM re-ranker (T2-07).

`/v1/api-tools/reranker` gets the question and up to K table texts, answers, and lists the documents it cited
(`citedDocuments`). It gives no scores, so stage 2 only says which tables to put first (rerank.LLMReranker).
The question is wrapped as "다음 질문의 수치를 찾을 수 있는 통계표: {q}" (lab plan 42: the plain question made the
model answer the question instead of picking tables). The table text is the n2 passage (kosis_rag.text).

Key: HCX_API_KEY from this repo's .env, read the same way as chain.clova_chat; it is never printed or logged.
Plain urllib, so LangChain callbacks do not see the call: calls, seconds and tokens go into the result instead.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HOST = "https://clovastudio.stream.ntruss.com"
PATH = "/v1/api-tools/reranker"
WRAP = "다음 질문의 수치를 찾을 수 있는 통계표: {q}"
RETRY_CODES = ("429", "5")  # rate limit and server errors are retried; anything else fails at once


def api_key() -> str | None:
    from kosis_chat.config import get_secret, load_env, load_settings

    load_env(Path(__file__).resolve().parents[2] / ".env")
    return get_secret(load_settings().api_key_env)


def post(body: dict, key: str, timeout: float) -> dict:
    """One call; HTTP errors come back as {'http': code} (the body is not kept: it may echo the request)."""
    req = urllib.request.Request(HOST + PATH, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                          "X-NCP-CLOVASTUDIO-REQUEST-ID": uuid.uuid4().hex})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"http": e.code}


@dataclass
class ClovaResult:
    keys: list[str]  # cited tables, in the order the API lists them
    calls: int = 1
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClovaReranker:
    """select(question, keys) -> ClovaResult. `text_of` maps a table key to the passage sent for it."""

    text_of: dict[str, str]
    wrap: bool = True
    max_tokens: int = 512  # answer length the API may use
    waits: tuple[float, ...] = (5, 15, 30)  # seconds before each retry
    timeout: float = 60
    key: str | None = None
    send: Callable[[dict, str, float], dict] = post  # replaced by a fake in tests

    def __call__(self, question: str, keys: list[str]) -> ClovaResult:
        if not keys:
            return ClovaResult(keys=[], calls=0)
        self.key = self.key or api_key()
        if not self.key:
            raise RuntimeError("HCX_API_KEY is not set (.env)")
        body = {"query": WRAP.format(q=question) if self.wrap else question, "maxTokens": self.max_tokens,
                "documents": [{"id": str(n), "doc": self.text_of[k]} for n, k in enumerate(keys, 1)]}
        calls = 0
        for attempt in range(len(self.waits) + 1):
            calls += 1
            res = self.send(body, self.key, self.timeout)
            code = str(res.get("status", {}).get("code", res.get("http", "")))
            if code == "20000":
                result = res.get("result") or {}
                by_id = {str(n): k for n, k in enumerate(keys, 1)}
                cited = [by_id[str(d.get("id", ""))] for d in result.get("citedDocuments") or []
                         if str(d.get("id", "")) in by_id]
                return ClovaResult(keys=list(dict.fromkeys(cited)), calls=calls, usage=dict(result.get("usage") or {}))
            if attempt < len(self.waits) and code.startswith(RETRY_CODES):
                time.sleep(self.waits[attempt])
                continue
            raise RuntimeError(f"CLOVA reranker call failed (code {code}, {calls} calls)")
        raise RuntimeError("unreachable")
