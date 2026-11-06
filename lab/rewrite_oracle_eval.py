"""Plan 39 (research 38, E3): how far could the RAG Reasoning rewrites lift Hit@k if the best query were picked?

Reads only what plan 35 already saved, so no API and no GPU:
    data/lab/rag_reasoning/<tag>/<qid>.json        the model's own search queries, each with its F0 top 10
    data/lab/rag_reasoning_eval_*_ranks.jsonl      the original question's F0 rank (full depth)
For each answerable retrieval_golden row the candidate queries are the original question plus the rewrites. A rewrite's
gold rank is known only inside its saved top 10, so ranks past 10 count as "not in top 10" (Hit@1/5/10 only).
oracle = the best of the original and every rewrite, picked by looking at the gold: the ceiling a perfect chooser of
rewrites could reach. Also reported: how often rewrites are near-copies of the question, rows where the oracle helps,
rows where the rewrite is worse than the original, and qtype.

    python lab/rewrite_oracle_eval.py
    python lab/rewrite_oracle_eval.py --tag 53c3a1aa --ranks data/lab/rag_reasoning_eval_20261101-002850_ranks.jsonl
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from lab_common import OUT_DIR, RETRIEVAL, ROOT, group_rank, read_jsonl

KS = (1, 5, 10)
FAR = 10_000  # "not in the saved top 10"


def bare(s: str) -> str:
    """Question text without spaces, punctuation and common sentence endings, to spot near-copies."""
    s = re.sub(r"[\s\?\!\.,·~\-]", "", s)
    return re.sub(r"(은|는)?(몇명|얼마|얼마나|뭐|무엇)?(이야|야|이에요|예요|인가요|인가|일까|돼|되나요|이니|니|요|줘|알려줘|보여줘)$", "", s)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default=None, help="folder under data/lab/rag_reasoning (default: the only / latest one)")
    p.add_argument("--ranks", default=None, help="plan 35 ranks file (default: latest rag_reasoning_eval_*_ranks.jsonl)")
    args = p.parse_args()

    base = OUT_DIR / "rag_reasoning"
    tag_dir = base / args.tag if args.tag else max((d for d in base.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime)
    ranks_path = Path(args.ranks) if args.ranks else max(OUT_DIR.glob("rag_reasoning_eval_*_ranks.jsonl"))
    if not ranks_path.is_absolute():
        ranks_path = ROOT / ranks_path
    f0 = {r["qid"]: r.get("F0") for r in read_jsonl(ranks_path)}
    items = [it for it in read_jsonl(RETRIEVAL) if it.get("answerable", True) and it.get("required")]

    rows = []
    for it in items:
        log = json.loads((tag_dir / f"{it['qid']}.json").read_text(encoding="utf-8"))
        qs = log.get("queries", [])
        r0 = f0.get(it["qid"])
        r0 = FAR if r0 is None else r0
        rr = []
        for q in qs:
            r = group_rank(it["required"], q["top"])
            rr.append(FAR if r is None else r)
        copies = [bare(q["query"]) == bare(it["question"]) for q in qs]
        sims = [difflib.SequenceMatcher(None, bare(q["query"]), bare(it["question"])).ratio() for q in qs]
        rows.append({"qid": it["qid"], "qtype": it.get("qtype", "?"), "split": it["split"], "question": it["question"],
                     "queries": [q["query"] for q in qs], "f0": r0, "first": rr[0] if rr else FAR,
                     "best_rw": min(rr) if rr else FAR, "oracle": min([r0, *rr]), "copies": copies, "sims": sims})

    def hit(rs: list[int], k: int) -> float:
        return float(np.mean([r <= k for r in rs])) if rs else 0.0

    def table(ix: list[dict], title: str) -> list[str]:
        out = ["", f"## {title} (n={len(ix)})", "", "| ranking | " + " | ".join(f"Hit@{k}" for k in KS) + " |",
               "| --- | " + " | ".join("---" for _ in KS) + " |"]
        for name, key in (("F0 (original question)", "f0"), ("first rewrite only", "first"),
                          ("best rewrite (oracle among rewrites)", "best_rw"), ("oracle (original + rewrites)", "oracle")):
            out.append(f"| {name} | " + " | ".join(f"{hit([r[key] for r in ix], k):.3f}" for k in KS) + " |")
        return out

    n_q = [len(r["queries"]) for r in rows]
    all_copies = [c for r in rows for c in r["copies"]]
    all_sims = [s for r in rows for s in r["sims"]]
    gain = {k: [r for r in rows if r["oracle"] <= k < r["f0"]] for k in KS}
    worse = [r for r in rows if r["queries"] and r["best_rw"] > r["f0"] and r["f0"] <= 10]
    o1 = hit([r["oracle"] for r in rows], 1)
    if o1 <= 0.45:
        call = "≤ 0.45 → rewrites add little even with a perfect chooser: shelve E1/E2 (plan 39 §4)"
    elif o1 >= 0.50:
        call = "≥ 0.50 → there is room; a way to pick the right query (E1, reranker feedback) is worth testing"
    else:
        call = "between 0.45 and 0.50 → look at the near-copy rate; decide on E3b (5 diverse rewrites per question)"

    rid = time.strftime("%Y%m%d-%H%M%S")
    lines = ["# Plan 39 (E3): ceiling of the RAG Reasoning rewrites", "",
             f"run {rid}; conversations {tag_dir.relative_to(ROOT)}; original ranks {ranks_path.name}",
             "Rewrite ranks are known only inside each query's saved top 10; deeper counts as a miss. "
             "oracle = best of original + rewrites, picked with the gold (a ceiling, not a method).",
             "", "## verdict", "", f"- oracle Hit@1 {o1:.3f} vs F0 {hit([r['f0'] for r in rows], 1):.3f} → {call}"]
    lines += table(rows, "retrieval_golden all (answerable)")
    for sp in ("test", "dev"):
        lines += table([r for r in rows if r["split"] == sp], f"retrieval_golden {sp} (answerable)")
    lines += ["", "## how different are the rewrites", "",
              f"- rewrites per question: mean {np.mean(n_q):.2f}, none {sum(x == 0 for x in n_q)}, max {max(n_q)}",
              f"- near-copies of the question (spaces, punctuation, endings removed): {sum(all_copies)}/{len(all_copies)} "
              f"({np.mean(all_copies) if all_copies else 0:.2f})",
              f"- character similarity to the question: mean {np.mean(all_sims) if all_sims else 0:.2f}, "
              f"≥ 0.8 {np.mean([s >= 0.8 for s in all_sims]) if all_sims else 0:.2f}",
              f"- questions where the oracle beats F0: Hit@1 {len(gain[1])}, Hit@5 {len(gain[5])}, Hit@10 {len(gain[10])}",
              f"- questions where the best rewrite is worse than the original (original in top 10): {len(worse)}"]
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["qtype"]].append(r)
    lines += ["", "## by qtype (Hit@1 / Hit@5)", "", "| qtype | n | F0 | oracle | rewrites / question | near-copy rate |",
              "| --- | --- | --- | --- | --- | --- |"]
    for q, ix in sorted(by.items(), key=lambda kv: -len(kv[1])):
        cp = [c for r in ix for c in r["copies"]]
        lines.append(f"| {q} | {len(ix)} | {hit([r['f0'] for r in ix], 1):.2f} / {hit([r['f0'] for r in ix], 5):.2f} | "
                     f"{hit([r['oracle'] for r in ix], 1):.2f} / {hit([r['oracle'] for r in ix], 5):.2f} | "
                     f"{np.mean([len(r['queries']) for r in ix]):.1f} | {np.mean(cp) if cp else 0:.2f} |")
    fmt = lambda x: "10+" if x >= FAR else str(x)
    lines += ["", "## rows where the oracle beats F0 at Hit@5 or better", "",
              "| qid | qtype | question | rewrites | F0 | best rewrite |", "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        if r["oracle"] < r["f0"] and r["oracle"] <= 5:
            lines.append(f"| {r['qid']} | {r['qtype']} | {r['question']} | {' / '.join(r['queries'])} | "
                         f"{fmt(r['f0'])} | {fmt(r['best_rw'])} |")
    lines += ["", "## rows where the best rewrite is worse than the original", "",
              "| qid | qtype | question | rewrites | F0 | best rewrite |", "| --- | --- | --- | --- | --- | --- |"]
    for r in worse:
        lines.append(f"| {r['qid']} | {r['qtype']} | {r['question']} | {' / '.join(r['queries'])} | "
                     f"{fmt(r['f0'])} | {fmt(r['best_rw'])} |")
    report = "\n".join(lines) + "\n"
    print(report)
    out = OUT_DIR / f"rewrite_oracle_eval_{rid}_summary.md"
    out.write_text(report, encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
