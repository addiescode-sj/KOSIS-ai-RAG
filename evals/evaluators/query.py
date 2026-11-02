"""L2 query understanding (T2-14): compare parser fields with the golden fields, field by field.

Only the fields a rule parser answers are graded: period start and end, cycle and 시도 (as a set).
`answerable` stays in the golden for T2-09 and is not graded here.
"""

from __future__ import annotations

from evals.schemas import QueryItem

FIELDS = ("period_start", "period_end", "cycle", "regions")


def score_item(item: QueryItem, parsed: dict) -> dict:
    row = {f: (set(parsed[f]) == set(item.regions)) if f == "regions" else parsed[f] == getattr(item, f) for f in FIELDS}
    row["all"] = all(row[f] for f in FIELDS)
    return row


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    return {"n": n, **{f: sum(r[f] for r in rows) / n if n else 0.0 for f in (*FIELDS, "all")}}
