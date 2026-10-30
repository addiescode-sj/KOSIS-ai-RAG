"""Plan 25: colloquial -> official term lexicons for BM25 query expansion (step 1 of 2, no model, no LLM call).

Sources (docs: phase2-lab/25-colloquial-to-official-lexicon-plan.md)
    D1  data/lab/vocab/lexicon_claude_v1.jsonl   written by a fresh worker from data/lab/vocab/terms_v1.tsv only
    D2  data/lab/synth/src/lexicon.py (LEX)      official -> colloquial, reversed. The synthetic questions were made
                                                 with it, so it is a comparison only and never scored on synth_v0.
Global filters, the same for every entry (no golden is read here):
    an expression of 1 character (spaces removed), equal to its own term, or pointing to MIN_TERMS_DROP or more
    official terms is dropped. "~" and surrounding spaces are trimmed.
Output (gitignored data/lab/vocab/): lexicon_<name>.json  {expression: [official terms]}, review_sample_d1.tsv
(50 random pairs for a person to mark right / wrong), and a coverage report over the 2,364 tables.

    python lab/vocab_expand.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from lab_common import ROOT, docs, load_records

VOCAB = ROOT / "data" / "lab" / "vocab"
D1_FILE = VOCAB / "lexicon_claude_v1.jsonl"
D2_FILE = ROOT / "data" / "lab" / "synth" / "src" / "lexicon.py"
MIN_TERMS_DROP = 4  # plan 25 3.2: one expression -> 4+ official terms is too broad
CONFIDENCE = ("high", "mid", "low")


def clean(expr: str) -> str:
    return re.sub(r"\s+", " ", expr.replace("~", " ")).strip()


def squash(s: str) -> str:
    return re.sub(r"\s+", "", s)


def build(pairs: list[tuple[str, str]]) -> tuple[dict[str, list[str]], dict[str, int]]:
    """(expression, official term) pairs -> {expression: sorted terms} after the global filters."""
    lex: dict[str, set[str]] = defaultdict(set)
    drops = {"1 char": 0, "same as term": 0, f"{MIN_TERMS_DROP}+ terms": 0}
    for expr, term in pairs:
        e = clean(expr)
        if len(squash(e)) < 2:
            drops["1 char"] += 1
        elif squash(e) == squash(term):
            drops["same as term"] += 1
        else:
            lex[e].add(term)
    out = {}
    for e, terms in lex.items():
        if len(terms) >= MIN_TERMS_DROP:
            drops[f"{MIN_TERMS_DROP}+ terms"] += 1
        else:
            out[e] = sorted(terms)
    return dict(sorted(out.items())), drops


def d1_pairs(path: Path, levels: tuple[str, ...]) -> list[tuple[str, str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [(e, r["term"]) for r in rows if r["confidence"] in levels for e in r["colloquial"]]


def d2_pairs(path: Path) -> list[tuple[str, str]]:
    spec = importlib.util.spec_from_file_location("synth_lexicon", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return [(e, term) for term, exprs in mod.LEX.items() for e in exprs]


def sha8(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--d1", type=Path, default=D1_FILE)
    p.add_argument("--d2", type=Path, default=D2_FILE)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    sources = {
        "d1": d1_pairs(args.d1, CONFIDENCE),
        "d1hm": d1_pairs(args.d1, ("high", "mid")),
        "d2": d2_pairs(args.d2),
    }
    sources["d1+d2"] = sources["d1"] + sources["d2"]
    bm25_text = [d["bm25"] for d in docs(load_records(), "norm")]
    lines = ["# Plan 25: lexicons", "", f"D1 {args.d1.name} sha {sha8(args.d1)}, D2 {args.d2.name} sha {sha8(args.d2)}", "",
             "| lexicon | pairs in | expressions kept | official terms | tables per target term (median / max, of 2,364) | dropped |",
             "| --- | --- | --- | --- | --- | --- |"]
    VOCAB.mkdir(parents=True, exist_ok=True)
    for name, pairs in sources.items():
        lex, drops = build(pairs)
        terms = {t for ts in lex.values() for t in ts}
        reach = sorted(sum(t in text for text in bm25_text) for t in terms) or [0]
        (VOCAB / f"lexicon_{name}.json").write_text(json.dumps(lex, ensure_ascii=False, indent=1), encoding="utf-8")
        lines.append(f"| {name} | {len(pairs)} | {len(lex)} | {len(terms)} | {reach[len(reach) // 2]} / {reach[-1]} | "
                     + ", ".join(f"{k} {v}" for k, v in drops.items()) + " |")

    # 50 random D1 pairs for a person: mark ok / wrong / too broad (plan 25 3.2, decision 4)
    lex_d1, _ = build(sources["d1"])
    pairs = sorted((e, t) for e, ts in lex_d1.items() for t in ts)
    sample = random.Random(args.seed).sample(pairs, min(50, len(pairs)))
    with (VOCAB / "review_sample_d1.tsv").open("w", encoding="utf-8", newline="\r\n") as f:
        f.write("expression\tofficial_term\tverdict (ok / wrong / broad)\n")
        for e, t in sample:
            f.write(f"{e}\t{t}\t\n")
    report = "\n".join(lines) + "\n"
    print(report)
    (VOCAB / "lexicon_summary.md").write_text(report, encoding="utf-8")
    print(f"-> {VOCAB}")


if __name__ == "__main__":
    main()
