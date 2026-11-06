"""KOS-88 before/after check with the Phase 1 golden set: does the question's own table come back in the top k?

Six runs on the same questions: {dense, bm25, hybrid} x {raw, norm}.
    dense   bge-m3 collection in chroma/ (build both first with chroma_build.py --text raw / --text norm)
    bm25    in memory, team tokenizer; raw = search_text, norm = bm25_text with norm_query on the question
    hybrid  RRF of the two lists above (c = 60, top 50 each), the EnsembleRetriever formula

Phase 1 questions were written while looking at one of only 10 tables, so they name it more directly than real
users would. Read the numbers as a smoke test of the rules, not as the T2-03 search metric.

    python lab/chroma_eval_p1.py
    python lab/chroma_eval_p1.py --show-changed       # questions whose rank moved between raw and norm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from chroma_common import (
    DEFAULT_MODEL, ROOT, TEXTS, bm25_tokenize, client, docs, encode, load_model, load_records, open_collection,
    query_text, read_jsonl,
)

GOLDEN = ROOT / "eval" / "golden" / "golden.jsonl"
OUT_DIR = ROOT / "data" / "lab"  # gitignored (/data/)
DEPTH, RRF_C, KS = 50, 60, (1, 5, 10)


def rrf(*lists: list[str], c: int = RRF_C) -> list[str]:
    score: dict[str, float] = {}
    for keys in lists:
        for rank, key in enumerate(keys, 1):
            score[key] = score.get(key, 0.0) + 1 / (c + rank)
    return sorted(score, key=score.get, reverse=True)


def rank_of(gold: str, keys: list[str]) -> int | None:
    return keys.index(gold) + 1 if gold in keys else None


def summary(ranks: list[int | None]) -> dict:
    n = len(ranks)
    out = {f"Hit@{k}": sum(r is not None and r <= k for r in ranks) / n for k in KS}
    out["MRR@10"] = sum(1 / r for r in ranks if r and r <= 10) / n
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-labels", type=int, default=60)
    p.add_argument("--show-changed", action="store_true")
    args = p.parse_args()

    items = read_jsonl(GOLDEN)
    records = load_records()
    tokenize = bm25_tokenize()
    store = client()
    model = None
    runs: dict[str, list[int | None]] = {}
    tops: dict[str, list[str]] = {}

    for text in TEXTS:
        ds = docs(records, text, args.max_labels)
        keys = [d["key"] for d in ds]
        bm25 = BM25Okapi([tokenize(d["bm25"]) for d in ds])
        queries = [query_text(it["question"], text) for it in items]
        sparse = []
        for q in queries:
            scores = bm25.get_scores(tokenize(q))
            sparse.append([keys[i] for i in sorted(range(len(keys)), key=lambda i: scores[i], reverse=True)[:DEPTH] if scores[i] > 0])

        col = open_collection(store, args.model, text, args.max_labels)
        if col is None:
            print(f"(skip dense/hybrid {text}: run python lab/chroma_build.py --text {text})")
            dense = None
        else:
            model = model or load_model(args.model, 512)
            vecs = encode(model, args.model, queries, "query")
            res = col.query(query_embeddings=vecs.tolist(), n_results=DEPTH, include=[])
            dense = res["ids"]

        lists = {"bm25": sparse}
        if dense is not None:
            lists |= {"dense": dense, "hybrid": [rrf(d, s) for d, s in zip(dense, sparse)]}
        for channel, per_q in lists.items():
            run = f"{channel}-{text}"
            runs[run] = [rank_of(Path(it["table"]).stem, top) for it, top in zip(items, per_q)]
            tops[run] = [top[0] if top else None for top in per_q]

    kinds = {"answerable": [i for i, it in enumerate(items) if it["kind"] == "answerable"], "all": list(range(len(items)))}
    lines = [f"# KOS-88 before/after, Phase 1 golden ({args.model}, labels > {args.max_labels} summarized)", ""]
    for kind, idx in kinds.items():
        lines += [f"## {kind} (n={len(idx)})", "", "| run | Hit@1 | Hit@5 | Hit@10 | MRR@10 |", "| --- | --- | --- | --- | --- |"]
        for run in sorted(runs):
            s = summary([runs[run][i] for i in idx])
            lines.append(f"| {run} | " + " | ".join(f"{s[m]:.3f}" for m in (*(f"Hit@{k}" for k in KS), "MRR@10")) + " |")
        lines.append("")
    lines += ["## raw -> norm, rank moves (all questions)", ""]
    for channel in ("dense", "bm25", "hybrid"):
        a, b = runs.get(f"{channel}-raw"), runs.get(f"{channel}-norm")
        if a and b:
            ahead = lambda x, y: (x or 999) < (y or 999)  # noqa: E731  (None = not in the top 50)
            better, worse = sum(ahead(y, x) for x, y in zip(a, b)), sum(ahead(x, y) for x, y in zip(a, b))
            lines.append(f"- {channel}: better {better}, worse {worse}, same {len(a) - better - worse}")
    report = "\n".join(lines) + "\n"
    print(report)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "eval_p1_summary.md").write_text(report, encoding="utf-8")
    with (OUT_DIR / "eval_p1_ranks.jsonl").open("w", encoding="utf-8") as f:
        for i, it in enumerate(items):
            row = {"id": it["id"], "kind": it["kind"], "gold": Path(it["table"]).stem, "question": it["question"]}
            row |= {run: runs[run][i] for run in sorted(runs)} | {f"top1 {run}": tops[run][i] for run in sorted(tops)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"-> {OUT_DIR / 'eval_p1_summary.md'}, {OUT_DIR / 'eval_p1_ranks.jsonl'}")

    if args.show_changed:
        for i, it in enumerate(items):
            moved = {c: (runs[f'{c}-raw'][i], runs[f'{c}-norm'][i]) for c in ("dense", "bm25", "hybrid")
                     if f"{c}-raw" in runs and f"{c}-norm" in runs and runs[f'{c}-raw'][i] != runs[f'{c}-norm'][i]}
            if moved:
                print(f"[{it['id']}] " + "  ".join(f"{c} {a}->{b}" for c, (a, b) in moved.items()) + f"  Q: {it['question']}")


if __name__ == "__main__":
    main()
