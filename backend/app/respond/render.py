"""What code writes around the guide reply (T2-41 S5: from chain.py): the context list the LLM sees, the template
sentence when the LLM failed, and the citations."""

from __future__ import annotations

from kosis_rag.schemas.io import TableCandidate

from backend.app.respond.texts import CITATION_HEADER, TEMPLATE_OPENING


def format_context(candidates: list[TableCandidate]) -> str:
    """Write the candidate tables as the numbered list the guide prompt shows the LLM.

    In: candidates = tables shown to the answer, in rank order. Out: one line per table,
    "[n] tbl_id | table name | survey | organization | period". Side effects: none. Node: guide.
    """
    return "\n".join(
        f"[{i}] {c.tbl_id} | {c.table_name} | {c.survey_name} | {c.organization} | {c.period}"
        for i, c in enumerate(candidates, 1)
    )


def template_answer(candidates: list[TableCandidate]) -> str:
    """The guide sentence written by code when the LLM failed or its reply did not pass verification.

    In: candidates = the tables shown, in rank order. Out: a fixed opening line and the top 3 tables (name, id,
    organization, period). Side effects: none. Node: guide (through node.guide).
    """
    lines = [TEMPLATE_OPENING]
    lines += [f"[{i}] {c.table_name} ({c.tbl_id}) · {c.organization} · {c.period}" for i, c in enumerate(candidates[:3], 1)]
    return "\n".join(lines)


def with_citations(text: str, tables: list[TableCandidate]) -> str:
    """Append the source lines (KOSIS / survey / table + link) of the tables the sentence recommends.

    In: text = the guide sentence; tables = the tables it points at (picked_from_reply, or the template's top 3).
    Out: text + "출처:" block written by code. Side effects: none. Node: guide (through node.guide).
    """
    return text + f"\n\n{CITATION_HEADER}\n" + "\n".join(f"- {c.to_source().citation()}" for c in tables)
