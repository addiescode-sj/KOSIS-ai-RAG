"""Gate stage step function (T2-41 S3: chain.decide_route as a step): refuse before calling the guide LLM, or not.

The route is "found" or "refuse". Until S5 the graph writes "guide" for "found" (the API state keeps
route refuse | guide); S5 settles the module state on found | refuse.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.documents import Document
from pydantic import Field

from kosis_rag.shared.boundary import BoundaryModel

Route = Literal["found", "refuse"]


class GateInput(BoundaryModel):
    """In of gate: the tables shown to the answer (already cut to k), in rank order, with the search metadata."""

    docs: list[Document] = Field(default_factory=list)


class GateOutput(BoundaryModel):
    """Out of gate: the route and, for a stage 2 refusal, its reason (stage2_reason, possibly None)."""

    route: Route
    refusal_reason: str | None = None


def decide_route(inp: GateInput, resources: Any = None) -> GateOutput:
    """Decide whether the answer refuses before calling the guide LLM.

    In: inp.docs = the tables shown to the answer (cut to k), in rank order; resources = unused (step signature).
    Out: GateOutput. "refuse" when there are no docs, or when stage 2 ran without error and picked nothing (T2-09;
        the reason is its stage2_reason, possibly None); otherwise ("found", None). A stage 2 error is not a
        refusal: the answer keeps the stage 1 order.
    Side effects: none.
    Node: gate.
    """
    if not inp.docs:
        return GateOutput(route="refuse")
    meta = inp.docs[0].metadata
    if meta.get("stage2_picked_n") == 0 and not meta.get("stage2_error"):  # the LLM re-ranker picked nothing (T2-09)
        return GateOutput(route="refuse", refusal_reason=meta.get("stage2_reason"))
    return GateOutput(route="found")
