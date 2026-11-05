"""I/O of the question generator (T2-39): what the LLM must return, and the per-table state of the generation graph."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from kosis_rag.fields import TableFields
from kosis_rag.schemas import TableRecord

QType = Literal["concept", "condition", "distinct", "keyword"]
TYPES: tuple[str, ...] = ("concept", "condition", "distinct", "keyword")
MAX_QUESTIONS = 12  # the prompt asks for 6; more than this is a malformed reply


class HqQuestion(BaseModel):
    """One hypothetical question: how a person might ask for the table."""

    type: QType
    text: str = Field(min_length=2, max_length=200)


class TableQuestions(BaseModel):
    """The LLM reply for one table (prompt.SYSTEM). key is filled in by the code, not taken from the reply."""

    key: str
    description: str = Field(min_length=2, max_length=400)  # one sentence: what the table shows, broken down by what
    questions: list[HqQuestion] = Field(min_length=1, max_length=MAX_QUESTIONS)


class HqState(BaseModel):
    """One table through the generation graph (hq.graph.build_table_graph); the nodes run in a line and each sets
    the fields named after it below."""

    key: str
    record: TableRecord | None = None  # load
    fields: TableFields | None = None  # load
    doc: str = ""  # load: the table's n2 embedding text (kosis_rag.text.build_doc)
    messages: list[dict[str, str]] = Field(default_factory=list)  # prompt: system + user; validate adds re-ask turns
    prompt_hash: str = ""  # prompt: hash of the fixed template + version (cache folder); the table's own text is
    # keyed separately (graph.user_sha, in the cache file name)
    raw: str | None = None  # generate: the last reply text
    responses: list[dict[str, Any]] = Field(default_factory=list)  # generate: every reply {raw, usage, seconds, cached}
    attempts: int = 0  # generate: replies so far (1 = first answer, 2 = after the one re-ask)
    description: str = ""  # validate
    questions: list[HqQuestion] = Field(default_factory=list)  # validate, then rule_check keeps the passing ones
    positions: list[int] = Field(default_factory=list)  # rule_check: each kept question's place in the reply (its id)
    dropped: dict[str, str] = Field(default_factory=dict)  # rule_check: question text -> reason
    scores: list[float] = Field(default_factory=list)  # score: one per kept question (empty with no scorer)
    error: str | None = None  # generate (API failure) or validate (reply still invalid after the re-ask)
    trace: dict[str, dict[str, Any]] = Field(default_factory=dict)  # node -> {"seconds": float, ...}
