"""What the rewrite step reads and writes (KOS-192). BoundaryModel: strict, extra fields refused."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from kosis_rag.shared.boundary import BoundaryModel

# used: the model gave a rewrite (the caller may still reject it); none: the model said there is nothing to rewrite;
# rejected: check_rewrite threw it away (set by the caller, with the reason); error: the call or its reply failed
RewriteStatus = Literal["used", "none", "rejected", "error"]


class RewriteInput(BoundaryModel):
    """In of rewrite_question: the user question and, for R2, names of the top stage 1 candidates (empty = R1)."""

    question: str
    table_names: list[str] = Field(default_factory=list)


class RewriteOutput(BoundaryModel):
    """Out of rewrite_question: the rewrite (None unless the model gave one), its status, why it was rejected or
    failed, the call's seconds and tokens."""

    rewrite: str | None = None
    status: RewriteStatus
    reason: str | None = None  # rejected: check_rewrite's reason; error: the error text; else None
    seconds: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
