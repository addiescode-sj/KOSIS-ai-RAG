"""The code check of the guide reply (T2-41 S5: from chain.py): no invented table id, year or [n] is ever shown.

unsupported_claims finds them, picked_from_reply reads which candidates a checked reply recommends, and
streamable_prefix gives the part of a reply still being written that may already be shown (T2-38).
"""

from __future__ import annotations

import re

from backend.app.respond.texts import NO_MATCH
from kosis_rag.schemas.io import TableCandidate

from backend.app.respond.patterns import REF, REF_RANGE, TBL_ID, YEAR


def unsupported_claims(text: str, question: str, candidates: list[TableCandidate]) -> list[str]:
    """Table ids and years in the reply that neither the candidates nor the question contain.

    Does: checks the LLM's guide sentence for invented table ids or years before it is shown.
    In: text = the LLM reply; question = the user question (its years are allowed); candidates = the tables shown.
    Out: the unsupported ids and years, then [n] numbers outside the candidate list, in the order they appear
        (empty = the reply passes).
    Side effects: none.
    Node: guide (through node.guide).
    """
    ranges = [(min(ys), max(ys)) for c in candidates if (ys := [int(y) for y in YEAR.findall(c.period)])]
    asked = set(YEAR.findall(question))
    ids = [t for t in TBL_ID.findall(text) if t not in {c.tbl_id for c in candidates}]
    years = [y for y in YEAR.findall(text) if y not in asked and not any(lo <= int(y) <= hi for lo, hi in ranges)]
    numbers = [f"[{n}]" for _, n in references(text) if not 1 <= n <= len(candidates)]  # e.g. [7] of 5 (T2-39)
    return ids + years + numbers


def references(text: str) -> list[tuple[int, int]]:
    """(position, number) of every [n] reference, ranges expanded ("[1~3]", "[1]~[3]" -> 1, 2, 3)."""

    def span(lo: int, hi: int) -> list[int]:
        return list(range(lo, hi + 1)) if lo <= hi <= lo + 20 else [lo, hi]  # a reversed or huge "range" is two refs

    refs: list[tuple[int, int]] = []
    taken: list[tuple[int, int]] = []
    for m in REF_RANGE.finditer(text):
        refs += [(m.start(), n) for n in span(int(m.group(1)), int(m.group(2)))]
        taken.append(m.span())
    for m in REF.finditer(text):
        if any(a <= m.start() < b for a, b in taken):
            continue
        for part in re.split(r"\s*[,，]\s*", m.group(1)):
            bounds = [int(x) for x in re.split(r"\s*[~\-]\s*", part)]
            refs += [(m.start(), n) for n in span(bounds[0], bounds[-1])]
    return sorted(refs)


def picked_from_reply(text: str, candidates: list[TableCandidate]) -> list[TableCandidate]:
    """The candidate tables a checked guide reply points at, in the order it first names them (T2-39).

    Does: turns the LLM's choice into data, so citations and the shown tables are the ones the reply recommends.
    In: text = a reply that passed unsupported_claims; candidates = the tables shown to the LLM, in rank order ([n] = n-th).
    Out: candidates named by table id (TBL_ID, so "표DT_X" counts) or by [n], without repeats. A [n] right before an id
        ("[2] DT_A") defers to the id when they disagree: the id is the data. Empty when the reply names no table.
    Side effects: none.
    Node: guide (through node.guide).
    """
    by_id = {c.tbl_id: c for c in candidates}
    ids = [(m.start(), m.end(), m.group(0)) for m in TBL_ID.finditer(text)]
    hits: list[tuple[int, TableCandidate]] = [(start, by_id[t]) for start, _, t in ids if t in by_id]
    for pos, n in references(text):
        if not 1 <= n <= len(candidates):
            continue
        close = text.find("]", pos)
        close = text.find("］", pos) if close == -1 else close
        if any(0 <= start - close <= 3 for start, _, _ in ids):  # "[2] DT_A": the id decides
            continue
        hits.append((pos, candidates[n - 1]))
    picked: list[TableCandidate] = []
    for _, c in sorted(hits, key=lambda h: h[0]):
        if c not in picked:
            picked.append(c)
    return picked


def streamable_prefix(text: str, question: str, candidates: list[TableCandidate]) -> str | None:
    """The part of a reply still being written that may already be shown (T2-38).

    Does: lets the guide stream only what would pass the final check, so no invented id or year is ever on screen.
    In: text = the reply so far; question, candidates = as unsupported_claims.
    Out: the text up to its last whitespace (a cut word such as "DT_1B0" or "202" waits); "" while the reply may
        still be NO_MATCH (also "NO_MATCH\n"); None once that part names an unsupported id or year — the reply will
        fall back to the template, so nothing more is sent. Ids and years end at whitespace, so a claim seen in a
        prefix is seen in the whole reply: None stays None as the text grows.
    Side effects: none.
    Node: guide (through node.guide).
    """
    head = text.lstrip()
    if NO_MATCH.startswith(head) or head.startswith(NO_MATCH):
        return ""
    words = re.match(r"[\s\S]*\s", text)
    prefix = words.group(0) if words else ""
    return None if unsupported_claims(prefix, question, candidates) else prefix
