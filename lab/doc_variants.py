"""Plan 27: rule-only rebuilds of the per-table search text (no LLM, no golden). Step 1 of 2, doc_eval.py runs them.

Starts from normalize_text.build_doc (n2, labels capped at 60) and never edits it. Per table:
    head      the build_doc lines before 항목 / class lines (title, survey, topic, menu tails, period, name fields)
    R1 unit   "단위: ..." from record["unit"] plus units peeled off item names (L3); no line when there is none
    R2 parts  core = head; children = "통계표: {name}" + the 항목 line, and the same prefix + each class line
    R3 diff   tables sharing a normalized name: "같은 이름 표와 다른 점: ..." (period, name fields, menu tail,
              unit, class names / items / labels no sibling has), at most 8 pieces
    R4 bm25   two BM25 fields: core (head + raw and English names) and labels (items + classes)
Files (gitignored data/lab/docvar/): docvar_stats.md (counts) and r3_samples.md (30 random R3 lines).

    python lab/doc_variants.py
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path

from normalize_text import UNIT_SUFFIX, _menu_tails, _unify, build_doc, load_records, norm_item, norm_label

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "lab" / "docvar"
MAX_DIFF = 8  # pieces on one R3 line


def span(rec: dict) -> str | None:
    if rec.get("start") is None:
        return None
    return str(rec["start"]) if rec["start"] == rec["end"] else f"{rec['start']}~{rec['end']}"


def units(rec: dict) -> list[str]:
    """record unit first, then the units L3 strips from item names ("출생아수(명)" -> 명)."""
    out = [rec["unit"].strip()] if (rec.get("unit") or "").strip() else []
    for it in rec.get("items", []):
        m = UNIT_SUFFIX.search(_unify(it))
        if m and m.group(1) not in out:
            out.append(m.group(1))
    return out


def parts(rec: dict, max_labels: int = 60) -> dict:
    """build_doc split into head / 항목 line / class lines, plus the pieces R1-R4 need."""
    d = build_doc(rec, max_labels=max_labels)
    lines = d["embed_text"].split("\n")
    items = list(dict.fromkeys(i for i in map(norm_item, rec.get("items", [])) if i))
    classes = [(_unify(c["name"]), list(dict.fromkeys(norm_label(l) for l in c["labels"] if l.strip())))
               for c in rec.get("classes", [])]
    n_tail = (1 if items else 0) + len(classes)
    head, tail = lines[: len(lines) - n_tail], lines[len(lines) - n_tail:]
    assert head and head[0].startswith("통계표: "), d["key"]
    item_line = tail[0] if items else None
    class_lines = tail[1:] if items else tail
    info = rec["info"]
    return {
        "key": d["key"], "name": d["fields"]["name"], "fields": d["fields"], "head": head,
        "item_line": item_line, "class_lines": class_lines, "items": items, "classes": classes,
        "units": units(rec), "span": span(rec), "tails": _menu_tails(rec),
        "bm25": d["bm25_text"],
        "bm25_core": " ".join(p for p in [*head, info["table_name"], rec.get("table_name_en") or ""] if p),
        "bm25_labels": " ".join([*items, *(f"{c} {' '.join(ls)}" for c, ls in classes)]),
    }


def diff_lines(ps: list[dict]) -> dict[str, str]:
    """R3: key -> one line naming what this table has that its same-name siblings do not."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for p in ps:
        groups[p["name"]].append(p)
    out = {}
    for sib in groups.values():
        if len(sib) < 2:
            continue
        for p in sib:
            others = [o for o in sib if o is not p]
            got: list[str] = []
            if p["span"] and any(o["span"] != p["span"] for o in others):
                got.append(f"수록기간 {p['span']}")
            for k, v in p["fields"].items():
                if k not in ("name", "rules") and any(o["fields"].get(k) != v for o in others):
                    got.append(f"{k} {v}")
            seen_tails = {t for o in others for t in o["tails"]}
            got += [f"경로 {t.split(' > ')[-1]}" for t in p["tails"] if t not in seen_tails][:2]
            if p["units"] and any(o["units"] != p["units"] for o in others):
                got.append(f"단위 {', '.join(p['units'])}")
            seen_c = {c for o in others for c, _ in o["classes"]}
            got += [f"분류 {c}" for c, _ in p["classes"] if c not in seen_c]
            seen_i = {i for o in others for i in o["items"]}
            got += [i for i in p["items"] if i not in seen_i][:3]
            seen_l = {l for o in others for _, ls in o["classes"] for l in ls}
            got += [l for _, ls in p["classes"] for l in ls if l not in seen_l][:3]
            got = list(dict.fromkeys(got))[:MAX_DIFF]
            if got:
                out[p["key"]] = "같은 이름 표와 다른 점: " + " · ".join(got)
    return out


def added_lines(p: dict, unit: bool, diff: str | None) -> list[str]:
    """The R3 line and the R1 line this table gets under the given switches."""
    return [*([diff] if diff else []), *(["단위: " + ", ".join(p["units"])] if unit and p["units"] else [])]


def texts(p: dict, *, unit: bool = False, diff: str | None = None) -> dict:
    """Single-vector text, core, children and the two BM25 fields for one table under the given switches.
    The R3 line goes right after the title, the R1 line at the end of the head; BM25 gets both appended."""
    head = list(p["head"])
    if diff:
        head.insert(1, diff)
    if unit and p["units"]:
        head.append("단위: " + ", ".join(p["units"]))
    rest = [x for x in [p["item_line"], *p["class_lines"]] if x]
    extra = " ".join(added_lines(p, unit, diff))
    prefix = f"통계표: {p['name']}"
    return {
        "embed": "\n".join(head + rest),
        "core": "\n".join(head),
        "children": [f"{prefix}\n{x}" for x in rest],
        "bm25": " ".join(x for x in [p["bm25"], extra] if x),
        "bm25_core": " ".join(x for x in [p["bm25_core"], extra] if x),
        "bm25_labels": p["bm25_labels"],
    }


def build_all(max_labels: int = 60) -> tuple[list[dict], dict[str, str]]:
    ps = [parts(r, max_labels) for r in load_records()]
    return ps, diff_lines(ps)


def main() -> None:
    ps, diffs = build_all()
    groups: dict[str, int] = defaultdict(int)
    for p in ps:
        groups[p["name"]] += 1
    multi = {n: c for n, c in groups.items() if c > 1}
    n_child = [len([x for x in [p["item_line"], *p["class_lines"]] if x]) for p in ps]
    with_unit = sum(1 for p in ps if p["units"])
    lens = sorted(len(texts(p, unit=True, diff=diffs.get(p["key"]))["embed"]) for p in ps)
    stats = [
        "# Plan 27: document variants", "",
        f"- tables {len(ps)}; same-name groups {len(multi)} covering {sum(multi.values())} tables "
        f"(largest {max(multi.values()) if multi else 0})",
        f"- R1: tables with a unit line {with_unit} (more than one unit {sum(1 for p in ps if len(p['units']) > 1)})",
        f"- R2: children per table median {sorted(n_child)[len(n_child) // 2]}, max {max(n_child)}, total {sum(n_child)}",
        f"- R3: tables with a diff line {len(diffs)} of {sum(multi.values())} in groups "
        f"(no difference found: {sum(multi.values()) - len(diffs)})",
        f"- R1+R3 embed chars: median {lens[len(lens) // 2]}, p90 {lens[int(len(lens) * 0.9)]}, max {lens[-1]}",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "docvar_stats.md").write_text("\n".join(stats) + "\n", encoding="utf-8")
    random.seed(0)
    pick = random.sample(sorted(diffs), min(30, len(diffs)))
    by_key = {p["key"]: p for p in ps}
    with (OUT / "r3_samples.md").open("w", encoding="utf-8") as f:
        for k in pick:
            f.write(f"## {k}\n```\n{texts(by_key[k], unit=True, diff=diffs[k])['core']}\n```\n\n")
    print("\n".join(stats))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
