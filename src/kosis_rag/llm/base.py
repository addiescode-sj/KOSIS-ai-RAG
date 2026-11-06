# copied from Phase 1 kosis_chat/llm/model.py@e248a45
"""The Phase 1 LLM client interface (T2-41 S3, P2-D0): call statuses, LLMResult, LLMClient and MockClient.

A copy, so kosis_rag does not import Phase 1; tests/test_phase1_contract.py checks that the fields and status
values stay Phase 1's. HCXClient is copied to llm/hcx.py (it takes an LLMConfig instead of Phase 1 Settings).
Changed from the source: the module docstring; HCXClient, build_client and the Settings import are not here.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

# Call status values recorded with every answer and eval result.
STATUS_OK = "ok"
STATUS_FORMAT_ERROR = "format_error"
STATUS_TIMEOUT = "timeout"
STATUS_API_ERROR = "api_error"
STATUS_INPUT_TOO_LONG = "input_too_long"
STATUS_PIPELINE_ERROR = "pipeline_error"  # an exception in our own code while answering one question (eval.run_eval)

log = logging.getLogger("kosis.llm")


@dataclass(frozen=True)
class LLMResult:
    text: str
    status: str = STATUS_OK
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    attempts: int = 1  # HTTP requests made, including retries


class TraceWriteError(RuntimeError):
    """Audit persistence failed; never downgrade this to a model or API error."""


class LLMClient(ABC):
    """Common interface; implementations must not raise for API failures, they return a status."""

    name: str = "base"

    def record_event(self, event: dict[str, Any]) -> None:
        """Optional audit hook; bare clients do not claim transport trace availability."""

    @abstractmethod
    def generate(self, system: str, user: str) -> LLMResult:
        """Return the model's reply for one system + user prompt pair."""


class MockClient(LLMClient):
    """Deterministic client for tests: fixed text, a queue of replies, or a function of the prompt."""

    name = "mock"

    def __init__(
        self,
        response: str | Sequence[LLMResult | str] | Callable[[str, str], LLMResult | str] = "[mock] answer",
    ) -> None:
        self._response = response
        self._queue = list(response) if isinstance(response, (list, tuple)) else None
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> LLMResult:
        self.calls.append((system, user))
        if callable(self._response):
            reply = self._response(system, user)
        elif self._queue is not None:
            if not self._queue:
                return LLMResult(text="", status=STATUS_API_ERROR, error="mock queue exhausted")
            reply = self._queue.pop(0)
        else:
            reply = self._response
        return reply if isinstance(reply, LLMResult) else LLMResult(text=str(reply))
