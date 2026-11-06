# copied from Phase 1 kosis_chat/llm/model.py@e248a45
"""HCXClient: HyperCLOVA X through the CLOVA Studio v3 chat-completions API (T2-41 S3, P2-D0 copy of Phase 1).

Changed from the source: the constructor takes an LLMConfig (model, api_key, thinking, temperature, max_tokens,
timeout) instead of Phase 1 Settings + an environment key, and a `base_url` argument (Phase 1 Settings.base_url,
same default); a missing key raises LLMKeyMissing (Phase 1: ValueError). The request body, URL, retries, redaction
and response parsing are Phase 1's (tests/test_phase1_contract.py compares the body and URL with Phase 1's client).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import replace
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from kosis_rag.llm.base import (STATUS_API_ERROR, STATUS_INPUT_TOO_LONG, STATUS_TIMEOUT, LLMClient, LLMResult,
                                TraceWriteError, log)
from kosis_rag.llm.config import LLMConfig, require_key

DEFAULT_BASE_URL = "https://clovastudio.stream.ntruss.com"  # Phase 1 config.DEFAULT_BASE_URL
DEFAULT_TEMPERATURE = 0.0  # Phase 1 Settings defaults (configs/model.yaml: temperature 0, max_tokens 1024, timeout 60)
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT = 60.0


class HCXClient(LLMClient):
    """HyperCLOVA X through the CLOVA Studio v3 chat-completions API (non-streaming).

    POST {base_url}/v3/chat-completions/{model} with "Authorization: Bearer <api_key>".
    Reasoning models (HCX-007) take maxCompletionTokens and thinking.effort; the others take maxTokens.
    Rate limits (429) and server errors (5xx) are retried after each delay in RETRY_DELAYS,
    or after the server's Retry-After when that is longer (capped at MAX_RETRY_WAIT).
    """

    name = "hcx"
    REASONING_MODELS = ("HCX-007",)
    RETRY_DELAYS = (5.0, 15.0, 30.0, 60.0)
    MAX_RETRY_WAIT = 120.0
    RETRY_HTTP_CODES = (429, 500, 502, 503, 504)
    CONTEXT_EXCEEDED_CODE = "40003"

    def __init__(
        self, config: LLMConfig, sleep: Callable[[float], None] = time.sleep,
        observer: Callable[[dict[str, Any]], None] | None = None, *, base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        if not config.model or config.model == "TBD":
            raise ValueError("LLMConfig.model (HCX model) is not set")
        key = require_key(config)
        if not key:
            raise ValueError("LLMConfig.api_key is empty")
        self.config = config
        self.url = f"{base_url.rstrip('/')}/v3/chat-completions/{config.model}"
        self._key = key
        self._sleep = sleep
        self.observer = observer

    @property
    def temperature(self) -> float:
        return DEFAULT_TEMPERATURE if self.config.temperature is None else float(self.config.temperature)

    @property
    def max_tokens(self) -> int:
        return DEFAULT_MAX_TOKENS if self.config.max_tokens is None else self.config.max_tokens

    @property
    def timeout(self) -> float:
        return DEFAULT_TIMEOUT if self.config.timeout is None else self.config.timeout

    @property
    def is_reasoning(self) -> bool:
        return self.config.model.upper().startswith(self.REASONING_MODELS)

    def generate(self, system: str, user: str) -> LLMResult:
        result, http_code, retry_after = self._observed_call(system, user, 1)
        attempts = 1
        for retry, delay in enumerate(self.RETRY_DELAYS, start=1):
            if http_code not in self.RETRY_HTTP_CODES:
                break
            wait = min(max(delay, retry_after or 0.0), self.MAX_RETRY_WAIT)
            log.warning("HTTP %s, retry %d/%d after %.0fs", http_code, retry, len(self.RETRY_DELAYS), wait)
            self._emit({"type": "http_backoff", "http_attempt": attempts, "scheduled_wait_seconds": wait})
            self._sleep(wait)
            result, http_code, retry_after = self._observed_call(system, user, attempts + 1)
            attempts += 1
        return replace(result, attempts=attempts)

    def _redact(self, value):
        if isinstance(value, str):
            return value.replace(self._key, "[REDACTED]")
        if isinstance(value, dict):
            return {self._redact(key): self._redact(child) for key, child in value.items()}
        if isinstance(value, list):
            return [self._redact(child) for child in value]
        return value

    def _emit(self, event: dict[str, Any]) -> None:
        if self.observer is not None:
            try:
                self.observer(self._redact(event))
            except TraceWriteError:
                raise
            except Exception as exc:
                raise TraceWriteError("HTTP audit observer failed") from exc

    def _observed_call(self, system: str, user: str, attempt: int):
        endpoint = urlsplit(self.url)
        host = endpoint.hostname or ""
        host = f"[{host}]" if ":" in host else host
        authority = f"{host}:{endpoint.port}" if endpoint.port is not None else host
        safe_url = urlunsplit((endpoint.scheme, authority, endpoint.path, "", ""))
        self._emit({"type": "http_request", "http_attempt": attempt, "method": "POST", "url": safe_url,
                    "body": self._body(system, user), "transport": "hcx_http"})
        started = time.perf_counter()
        try:
            result, http_code, retry_after = self._call(system, user)
        except BaseException as exc:
            self._emit({"type": "http_raised", "http_attempt": attempt, "exception_type": type(exc).__name__})
            raise
        result = replace(result, text=self._redact(result.text), error=self._redact(result.error), usage=self._redact(result.usage))
        self._emit({"type": "http_response", "http_attempt": attempt, "http_status": http_code,
                    "call_status": result.status, "text": result.text, "error": result.error,
                    "usage": result.usage, "latency_seconds": time.perf_counter() - started})
        return result, http_code, retry_after

    def _body(self, system: str, user: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.temperature,
        }
        if self.is_reasoning:
            body["maxCompletionTokens"] = self.max_tokens  # includes thinking tokens
            body["thinking"] = {"effort": self.config.thinking or "none"}
        else:
            body["maxTokens"] = self.max_tokens
        return body

    def _request(self, system: str, user: str) -> urllib.request.Request:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self._key}",
            "X-NCP-CLOVASTUDIO-REQUEST-ID": uuid.uuid4().hex,
        }
        data = json.dumps(self._body(system, user)).encode("utf-8")
        return urllib.request.Request(self.url, data=data, headers=headers, method="POST")

    def _call(self, system: str, user: str) -> tuple[LLMResult, int | None, float | None]:
        """One request -> (result, HTTP error code or None, Retry-After seconds or None)."""
        try:
            with urllib.request.urlopen(self._request(system, user), timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                response_status = getattr(resp, "status", None)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            too_long = self.CONTEXT_EXCEEDED_CODE in detail or "context length" in detail.lower()
            status = STATUS_INPUT_TOO_LONG if too_long else STATUS_API_ERROR
            return LLMResult(text="", status=status, error=f"HTTP {exc.code}: {detail}"), exc.code, _retry_after(exc)
        except TimeoutError:
            return LLMResult(text="", status=STATUS_TIMEOUT, error="request timed out"), None, None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                return LLMResult(text="", status=STATUS_TIMEOUT, error="request timed out"), None, None
            return LLMResult(text="", status=STATUS_API_ERROR, error=f"connection failed: {exc.reason}"), None, None
        except json.JSONDecodeError as exc:
            return LLMResult(text="", status=STATUS_API_ERROR, error=f"invalid JSON response: {exc}"), None, None
        return _parse_response(data), response_status, None


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    try:
        return float(exc.headers.get("Retry-After", ""))
    except (TypeError, ValueError):
        return None


def _parse_response(data: dict[str, Any]) -> LLMResult:
    result = data.get("result") or {}
    if str((data.get("status") or {}).get("code")) != "20000" or not isinstance(result.get("message"), dict):
        return LLMResult(text="", status=STATUS_API_ERROR, error=f"unexpected response: {str(data)[:300]}")
    usage = result.get("usage") or {}
    text = result["message"].get("content") or ""
    if not text:
        # With thinking on, a small maxCompletionTokens can be used up before any answer text.
        return LLMResult(text="", status=STATUS_API_ERROR, usage=usage, error=f"empty response (finishReason={result.get('finishReason')})")
    return LLMResult(text=text, usage=usage)
