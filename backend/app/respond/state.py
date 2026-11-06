"""The API's search state (T2-41 S5): the module's schemas.state.SearchState fields in the same order, with the app's
route (refuse | guide) and result (the SearchResult refuse / guide write).

/nodes/* take and return it and docs/openapi.json documents it, so its name, field order, types and docstring are
the pre-S5 ones (the openapi diff stays 0). The module's nodes run on it unchanged (they read state attributes and
return updates); only gate's route is renamed, by backend.app.respond.graph.gate_to_api.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from kosis_rag.parse.schemas import ParsedQuery
from kosis_rag.schemas.io import SearchResult
from kosis_rag.schemas.state import merge
from kosis_rag.shared.filters import TableFilter


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
    route: Literal["refuse", "guide"] | None = None  # set by gate (the module's route, renamed by gate_to_api)
    refusal_reason: str | None = None  # set by gate
    result: SearchResult | None = None  # set by refuse or guide
    trace: Annotated[dict[str, dict[str, Any]], merge] = Field(default_factory=dict)  # node -> {"seconds": float, ...}
