"""The search graph's shared state (T2-41 S4: from graph.py). No LangGraph import: the reducer is a plain function
LangGraph reads off the Annotated field, so this module imports without langgraph.

T2-41 S5: the module graph ends at gate, so route is found | refuse and there is no `result`; the app's state with
route refuse | guide and the SearchResult is backend/app/respond/state.py.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from kosis_rag.parse.schemas import ParsedQuery
from kosis_rag.shared.filters import TableFilter


def merge(old: dict, new: dict) -> dict:
    """Reducer for the fields parallel nodes write: the union of both dicts, the newer value on a shared key.

    In: old = the field's value; new = one node's update. Out: {**old, **new}. Side effects: none.
    Node: none (LangGraph applies it to runs, run_removed and trace).
    """
    return {**old, **new}


class SearchState(BaseModel):
    """What the nodes share. Every field has one writer, except docs (overwritten by each search stage) and the
    merged fields (runs, run_removed, trace), which each node adds its own key to."""

    question: str
    query: ParsedQuery | None = None  # set by parse
    table_filter: TableFilter | None = None  # set by condition: the filter every list is searched under
    query_vecs: dict[str, list[float]] = Field(default_factory=dict)  # set by embed: list name -> question vector
    relaxed: list[str] = Field(default_factory=list)  # set by condition: conditions dropped so a table could pass (T2-29)
    runs: Annotated[dict[str, list[Document]], merge] = Field(default_factory=dict)  # retrieve_*: list name -> ranked docs
    run_removed: Annotated[dict[str, dict[str, str]], merge] = Field(default_factory=dict)  # retrieve_*: name -> removed
    removed: dict[str, str] = Field(default_factory=dict)  # set by fuse: key -> why filtered out (lists in fuse order)
    docs: list[Document] = Field(default_factory=list)  # overwritten by fuse, rerank1, rerank2
    route: Literal["found", "refuse"] | None = None  # set by gate
    refusal_reason: str | None = None  # set by gate
    trace: Annotated[dict[str, dict[str, Any]], merge] = Field(default_factory=dict)  # node -> {"seconds": float, ...}
