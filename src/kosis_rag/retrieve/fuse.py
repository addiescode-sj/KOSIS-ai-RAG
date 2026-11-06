"""Weighted reciprocal rank fusion of the lists (T2-06, weights T2-39; T2-41 S2: from retriever.py)."""

from __future__ import annotations

from langchain_core.documents import Document

from kosis_rag.shared.constants import RRF_C


def rrf_fuse(lists: list[list[Document]], names: list[str] | None = None, *, k: int = 50, c: int = RRF_C,
             score_from: int = 0, weights: list[float] | None = None) -> list[Document]:
    """Fuse ranked lists by weighted reciprocal rank over table keys (T2-06; weights T2-39).

    In: lists = ranked lists (dense first); names = one label per list for rank_<name> (default r0, r1, ...);
        k = tables returned; c = RRF constant; score_from = index of the list whose score stays in metadata["score"];
        weights = one weight per list (default 1.0 each: RRF = sum of 1 / (c + rank)).
    Out: at most k tables in fused order (ties keep first-seen order); each carries metadata score (the score_from
        list's score, or that list's lowest score for a table only other lists found), score_from (its name or
        "other"), rrf_score and rank_<name> per list it appeared in. The first list a table appears in gives its text.
    Side effects: none.
    Node: fuse (through retrieve.node.fuse; and hybrid_search).
    """
    names = names or [f"r{i}" for i in range(len(lists))]
    weights = weights or [1.0] * len(lists)
    rrf: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    first: dict[str, Document] = {}
    for name, w, docs in zip(names, weights, lists):
        for rank, d in enumerate(docs, 1):
            key = d.metadata["key"]
            rrf[key] = rrf.get(key, 0.0) + w / (c + rank)
            ranks.setdefault(key, {})[name] = rank
            first.setdefault(key, d)

    # keep the score_from list's score (the no-answer threshold is calibrated on it, T2-09)
    anchor = lists[score_from] if score_from < len(lists) else []
    own = {d.metadata["key"]: d.metadata.get("score") for d in anchor}
    floor = min((v for v in own.values() if v is not None), default=None)
    order = sorted(rrf, key=lambda key: -rrf[key])  # stable: ties keep first-seen order
    out = []
    for key in order[:k]:
        d = first[key]
        score, source = (own[key], names[score_from]) if key in own else (floor, "other")
        out.append(Document(page_content=d.page_content, id=d.id, metadata={
            **d.metadata, "score": score, "score_from": source, "rrf_score": rrf[key],
            **{f"rank_{n}": r for n, r in ranks[key].items()}}))
    return out
