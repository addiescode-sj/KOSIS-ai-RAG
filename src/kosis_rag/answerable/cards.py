"""The answerability classifier's table card (KOS-193), built from one data/tables.jsonl record.

A COPY of the parent repo's scripts/answerability/cards.py (CARD_FORMAT, MAX_ITEMS, MAX_LABELS, CYCLE_NAMES, years_of,
compress_years, _months, _clean, _capped, card): the classifier was trained on exactly this text, so the function
bodies here must stay identical to that file (a change there is a new CARD_FORMAT and a retrained checkpoint; the
loader refuses a checkpoint whose card_format differs from CARD_FORMAT). load_cards is this package's own.
The card reads raw record fields (periods, prd_se, items, classes, ...), so records are read as JSON dicts, not
through schemas.catalog.TableRecord.
"""

from __future__ import annotations

import json
from pathlib import Path

CARD_FORMAT = "ans-c1"
MAX_ITEMS = 30
MAX_LABELS = 24
CYCLE_NAMES = {"Y": "연", "F": "5년", "M": "월", "Q": "분기", "H": "반기", "IR": "부정기"}


def years_of(periods: list[str]) -> list[int]:
    """Distinct years in a table's periods ("2015", "201507", "2015.07" ...), sorted."""
    out = set()
    for p in periods:
        s = str(p).strip()
        if len(s) >= 4 and s[:4].isdigit():
            out.add(int(s[:4]))
    return sorted(out)


def compress_years(years: list[int]) -> str:
    """[2000, 2005, 2010, 2011, 2012] -> "2000, 2005, 2010~2012" (runs of consecutive years joined with ~)."""
    if not years:
        return "없음"
    parts, start, prev = [], years[0], years[0]
    for y in years[1:] + [None]:
        if y is not None and y == prev + 1:
            prev = y
            continue
        parts.append(str(start) if start == prev else (f"{start}, {prev}" if prev == start + 1 else f"{start}~{prev}"))
        if y is not None:
            start = prev = y
    return ", ".join(parts)


def _months(periods: list[str]) -> str | None:
    """First and last month when the table holds monthly values ("2015.01~2026.09"), else None."""
    ms = sorted(str(p) for p in periods if len(str(p)) == 6 and str(p).isdigit())
    if not ms:
        return None
    return f"{ms[0][:4]}.{ms[0][4:]}~{ms[-1][:4]}.{ms[-1][4:]}"


def _clean(label: str) -> str:
    return " ".join(str(label).replace("_", " ").split())


def _capped(values: list[str], cap: int) -> str:
    shown = [_clean(v) for v in values[:cap]]
    return ", ".join(shown) + (f" 외 {len(values) - cap}개" if len(values) > cap else "")


def card(rec: dict) -> str:
    """The classifier's text for one table record (dict as in data/tables.jsonl)."""
    info = rec.get("info", {})
    survey = info.get("survey_name") or ""
    org = info.get("organization") or ""
    head = f"통계표: {info.get('table_name', '')}"
    if survey or org:
        head += f" | {survey}" + (f" ({org})" if org else "")
    cycles = rec.get("prd_se") or rec.get("cycle") or []
    cyc = ", ".join(CYCLE_NAMES.get(c, c) for c in cycles) or "미상"
    lines = [head, f"수록 연도: {compress_years(years_of(rec.get('periods', [])))} (주기: {cyc})"]
    months = _months(rec.get("periods", []))
    if months:
        lines.append(f"월 자료: {months}")
    if rec.get("unit"):
        lines.append(f"단위: {rec['unit']}")
    items = rec.get("items") or []
    if items:
        lines.append(f"항목: {_capped(items, MAX_ITEMS)}")
    for c in rec.get("classes") or []:
        name = c.get("name", "")
        if c.get("levels"):
            lv = ", ".join(f"{k} {v}개" for k, v in c["levels"].items())
            lines.append(f"{name}: {lv}")
        else:
            lines.append(f"{name}: {_capped(c.get('labels') or [], MAX_LABELS)}")
    return "\n".join(lines)


def load_cards(path: Path) -> dict[str, str]:
    """Table key -> card text for every record of a tables.jsonl file.

    In: path = data/tables.jsonl (one JSON record per non-empty line, info.key set).
    Out: key -> card(record), in file order. Side effects: reads the file; raises on a missing file or an invalid line.
    Node: none directly (adapters.loader.load_resources calls it once; answerable reads what it gives).
    """
    out: dict[str, str] = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[rec["info"]["key"]] = card(rec)
    return out
