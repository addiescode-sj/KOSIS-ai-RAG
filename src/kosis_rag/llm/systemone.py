"""SystemOneClient: POST <base_url>/systemone, the typed-decision API of Jev-style models (T3-23 Jev; Clef-Flash).

A request is {"model", "state", "questions": {id: {"type": noul | choice | score, "instructions", "criteria"?}}} and
the reply {"model", "answers": {id: {...probabilities}}, "usage": {"input_tokens", ...}}. The server is the Colab
Clef-Flash server (Gemma_server/clef_server.ipynb, behind a cloudflared tunnel) or any server with the same contract.
The key is LLMConfig.api_key (Bearer); it is never printed or logged. Plain urllib, like llm/clova_rerank.py.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from kosis_rag.llm.config import LLMConfig, require_key

PATH = "/systemone"
DEFAULT_TIMEOUT = 120.0
RETRY_CODES = ("429", "5")  # rate limit and server errors are retried; anything else fails at once


def post(url: str, body: dict, key: str, timeout: float) -> tuple[int, dict]:
    """One call -> (HTTP status, JSON body). An HTTP error keeps only a short error text (the body may echo input)."""
    req = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            err = str(json.loads(e.read().decode("utf-8", "replace")).get("error", ""))[:200]
        except Exception:
            err = ""
        return e.code, {"error": err}


@dataclass
class SystemOneClient:
    """decide(state, questions) -> (reply, calls). Raises RuntimeError after the retries, LLMKeyMissing without a key,
    ValueError without config.base_url."""

    config: LLMConfig
    waits: tuple[float, ...] = (2, 5, 15)  # seconds before each retry
    send: Callable[[str, dict, str, float], tuple[int, dict]] = post  # replaced by a fake in tests

    @property
    def model(self) -> str:
        return self.config.model

    def decide(self, state: Any, questions: dict[str, dict]) -> tuple[dict, int]:
        if not self.config.base_url:
            raise ValueError("provider systemone needs LLMConfig.base_url (e.g. https://<tunnel>/v1)")
        key = require_key(self.config)
        url = self.config.base_url.rstrip("/") + PATH
        timeout = self.config.timeout if self.config.timeout is not None else DEFAULT_TIMEOUT
        body = {"model": self.config.model, "state": state, "questions": questions}
        calls = 0
        for attempt in range(len(self.waits) + 1):
            calls += 1
            try:
                code, reply = self.send(url, body, key, timeout)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:  # tunnel hiccup: retry
                code, reply = 503, {"error": f"{type(exc).__name__}: {exc}"}
            if code == 200:
                return reply, calls
            if attempt < len(self.waits) and str(code).startswith(RETRY_CODES):
                time.sleep(self.waits[attempt])
                continue
            raise RuntimeError(f"{self.model} systemone call failed (HTTP {code}, {calls} calls): {reply.get('error')}")
        raise RuntimeError("unreachable")
