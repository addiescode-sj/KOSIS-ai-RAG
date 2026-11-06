"""Experiment 14: does searching inside topic spaces (with the topic's words removed) beat one big space?

Variants (plan 14):
    V0  one space, as today (embed_eval.py's norm run)
    V1  per-category spaces, texts unchanged            -> effect of splitting alone
    V2  per-category spaces, category words removed     -> the idea under test (documents and question)
    V3  per-category spaces, category mean vector removed from document and question vectors (dense only)
Routers: rule (same word list on the question; none found -> V0 search), emb (2 categories whose mean document
vector is closest to the question), oracle (the gold table's categories: a perfect classifier, upper bound).
Several categories: dense keeps each table's best score, BM25 lists are fused with RRF, hybrid = RRF of the two.

Routing golden (plan 16, rows with q_categories): every question is answerable, alt_tables also count as a hit,
the router is scored against the question's own categories, and two upper bounds run: oracle-t (gold table's
categories, as above) and oracle-q (the question's labelled categories). Results are reported per split.

    python lab/category_eval.py
    python lab/category_eval.py --model nlpai-lab/KURE-v1
    python lab/category_eval.py --golden eval/golden/routing_v0.jsonl [--split dev|test|all]
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from category_db import CATEGORIES, OTHER, build_spaces, route_rule, strip_query, strip_words, table_categories
from chroma_common import DEFAULT_MODEL, bm25_tokenize, docs, encode, load_records, query_text, read_jsonl, source_sha
from chroma_eval_p1 import DEPTH, GOLDEN, KS, OUT_DIR, rank_of, rrf, summary
from embed_eval import doc_vectors, model_for

ORDER = [*CATEGORIES, OTHER]
METRICS = (*(f"Hit@{k}" for k in KS), "MRR@10")
STYLES = ("direct", "paraphrase", "multi")


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def top(scores: np.ndarray, keys: list[str], k: int = DEPTH) -> list[str]:
    return [keys[j] for j in np.argsort(-scores, kind="stable")[:k]]


def merge_dense(hits: list[dict[str, float]]) -> list[str]:
    """Per table, the best score over the searched spaces."""
    best: dict[str, float] = {}
    for h in hits:
        for key, s in h.items():
            best[key] = max(best.get(key, -9.0), s)
    return sorted(best, key=best.get, reverse=True)[:DEPTH]


def best_rank(answers: list[str], keys: list[str]) -> int | None:
    """Best rank over the gold table and its alternatives."""
    ranks = [r for r in (rank_of(a, keys) for a in answers) if r]
    return min(ranks) if ranks else None


def router_lines(routers: dict, items: list[dict], idx: list[int], cats: list[str], table_cats: list[set]) -> list[str]:
    """Routers against the question's own categories (routing golden)."""
    lines = ["| router | hit (any label) | exact set | precision | recall | none found | overlaps table cats |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    n = max(len(idx), 1)
    for rname, routes in routers.items():
        if rname == "oracle-q":
            continue
        hit = exact = none = over = 0
        prec = rec = 0.0
        for i in idx:
            got, want = set(routes[i]), set(items[i]["q_categories"])
            hit += bool(got & want)
            exact += got == want
            none += not got
            over += bool(got & table_cats[i])
            prec += len(got & want) / len(got) if got else 0.0
            rec += len(got & want) / len(want)
        lines.append(f"| {rname} | {hit / n:.3f} | {exact / n:.3f} | {prec / n:.3f} | {rec / n:.3f} | {none / n:.3f} "
                     f"| {over / n:.3f} |")
    lines += ["", "recall per labelled category (questions carrying that label whose route includes it)", "",
              "| category | n | " + " | ".join(r for r in routers if r != "oracle-q") + " |",
              "| --- | --- | " + " | ".join("---" for r in routers if r != "oracle-q") + " |"]
    for c in cats:
        with_c = [i for i in idx if c in items[i]["q_categories"]]
        if not with_c:
            continue
        cells = [f"{sum(c in routes[i] for i in with_c) / len(with_c):.3f}"
                 for r, routes in routers.items() if r != "oracle-q"]
        lines.append(f"| {c} | {len(with_c)} | " + " | ".join(cells) + " |")
    lines += ["", "router hit (any label) by question style", "",
              "| router | " + " | ".join(f"{s} (n={sum(items[i]['style'] == s for i in idx)})" for s in STYLES) + " |",
              "| --- | " + " | ".join("---" for _ in STYLES) + " |"]
    for rname, routes in routers.items():
        if rname == "oracle-q":
            continue
        cells = []
        for s in STYLES:
            sub = [i for i in idx if items[i]["style"] == s]
            cells.append(f"{sum(bool(set(routes[i]) & set(items[i]['q_categories'])) for i in sub) / max(len(sub), 1):.3f}")
        lines.append(f"| {rname} | " + " | ".join(cells) + " |")
    return lines


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-labels", type=int, default=60)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--golden", type=Path, default=GOLDEN)
    p.add_argument("--split", choices=("dev", "test", "all"), default="all", help="routing golden only")
    args = p.parse_args()

    items = read_jsonl(args.golden)
    routing = "q_categories" in items[0]
    if routing and args.split != "all":
        items = [it for it in items if it["split"] == args.split]
    gold = [Path(it["table"]).stem for it in items]
    answers = [[g, *it.get("alt_tables", [])] for g, it in zip(gold, items)]
    records, sha, tokenize = load_records(), source_sha(), bm25_tokenize()
    ds = docs(records, "norm", args.max_labels)
    keys = [d["key"] for d in ds]
    index = {k: i for i, k in enumerate(keys)}
    spaces = build_spaces(records)
    cats = [c for c in ORDER if c in spaces]
    state: dict = {}
    slug = args.model.split("/")[-1]

    # V0 / V1 / V3 share the plain norm vectors; V2 embeds each category's stripped texts once (cached)
    dv = doc_vectors(args.model, "norm", args.max_labels, args.max_tokens, ds, sha, state, args.batch)
    v2 = {}
    for n, c in enumerate(cats):
        sub = [{"key": ds[i]["key"], "embed": strip_words(ds[i]["embed"], c)} for i in spaces[c]]
        v2[c] = doc_vectors(args.model, "norm", args.max_labels, args.max_tokens, sub, sha, state, args.batch,
                            variant=f"-V2c{n:02d}")
    mean = {c: dv[spaces[c]].mean(axis=0) for c in cats}  # V3 centre, also the emb router's category vector
    v3 = {c: unit(dv[spaces[c]] - mean[c]) for c in cats}
    centroid = unit(np.stack([mean[c] for c in cats]))

    print("BM25 indexes ...")
    bm_all = BM25Okapi([tokenize(d["bm25"]) for d in ds])
    bm1 = {c: BM25Okapi([tokenize(ds[i]["bm25"]) for i in spaces[c]]) for c in cats}
    bm2 = {c: BM25Okapi([tokenize(strip_words(ds[i]["bm25"], c)) for i in spaces[c]]) for c in cats}

    model = model_for(args.model, 512, state)
    qn = [query_text(it["question"], "norm") for it in items]
    qv = np.asarray(encode(model, args.model, qn, "query", args.batch), dtype=np.float32)
    table_cats = [table_categories(records[index[g]]) for g in gold]
    routers = {
        "rule": [sorted(route_rule(q) & set(cats)) for q in qn],
        "emb": [[cats[j] for j in np.argsort(-(centroid @ v))[:2]] for v in qv],
        "oracle-t" if routing else "oracle": [sorted(tc) for tc in table_cats],
    }
    if routing:
        routers["oracle-q"] = [sorted(set(it["q_categories"]) & set(cats)) for it in items]
    need = sorted({(i, c) for r in routers.values() for i, cs in enumerate(r) for c in cs})
    stripped = {ic: strip_query(qn[ic[0]], ic[1]) for ic in need}
    sv = dict(zip(need, np.asarray(encode(model, args.model, [stripped[ic] for ic in need], "query", args.batch),
                                   dtype=np.float32)))

    def bm25_rank(bm, idx, q):
        s = np.asarray(bm.get_scores(tokenize(q)))
        return [keys[idx[j]] for j in np.argsort(-s, kind="stable")[:DEPTH] if s[j] > 0]

    v0_dense = [top(dv @ v, keys) for v in qv]
    v0_bm25 = [bm25_rank(bm_all, list(range(len(keys))), q) for q in qn]
    lists: dict[str, list[list[str]]] = {
        f"V0 dense {slug}": v0_dense, "V0 bm25": v0_bm25,
        f"V0 hybrid {slug}": [rrf(a, b) for a, b in zip(v0_dense, v0_bm25)],
    }
    for rname, routes in routers.items():
        for variant in ("V1", "V2", "V3"):
            dense, sparse = [], []
            for i, cs in enumerate(routes):
                if not cs:  # rule router found no category: today's search
                    dense.append(v0_dense[i]); sparse.append(v0_bm25[i]); continue
                hits, bl = [], []
                for c in cs:
                    idx = spaces[c]
                    sub_keys = [keys[j] for j in idx]
                    if variant == "V1":
                        s = dv[idx] @ qv[i]
                    elif variant == "V2":
                        s = v2[c] @ sv[(i, c)]
                    else:
                        s = v3[c] @ unit(qv[i] - mean[c])
                    hits.append(dict(zip(sub_keys, s.tolist())))
                    if variant == "V2":
                        bl.append(bm25_rank(bm2[c], idx, stripped[(i, c)]))
                    else:
                        bl.append(bm25_rank(bm1[c], idx, qn[i]))
                dense.append(merge_dense(hits))
                sparse.append(rrf(*bl)[:DEPTH] if len(bl) > 1 else bl[0])
            lists[f"{variant} {rname} dense {slug}"] = dense
            if variant != "V3":  # V3 changes vectors only; its BM25 would equal V1's
                lists[f"{variant} {rname} bm25"] = sparse
            lists[f"{variant} {rname} hybrid {slug}"] = [rrf(a, b) for a, b in zip(dense, sparse)]

    runs = {name: [best_rank(a, t) for a, t in zip(answers, per_q)] for name, per_q in lists.items()}

    def hit5(run, idx):
        return sum((runs[run][i] or 999) <= 5 for i in idx) / max(len(idx), 1)

    def metric_row(run, idx):
        s = summary([runs[run][i] for i in idx])
        return f"| {run} | " + " | ".join(f"{s[m]:.3f}" for m in METRICS) + " |"

    if routing:
        lines = [f"# Experiment 14 on {args.golden.name} ({args.model}, n2-L{args.max_labels}, "
                 f"docs {args.max_tokens} tokens)", "",
                 f"questions {len(items)}, spaces {len(cats)}, alt answers on "
                 f"{sum(len(a) > 1 for a in answers)} questions", ""]
        groups = {s: [i for i, it in enumerate(items) if it["split"] == s] for s in ("dev", "test")}
        groups = {s: idx for s, idx in groups.items() if idx}
        if len(groups) > 1:
            groups["all"] = list(range(len(items)))
        for gname, idx in groups.items():
            lines += [f"## router, {gname} (n={len(idx)})", ""] + router_lines(routers, items, idx, cats, table_cats)
            lines += ["", f"## retrieval, {gname} (n={len(idx)})", "",
                      "| run | " + " | ".join(METRICS) + " |", "| --- | " + " | ".join("---" for _ in METRICS) + " |"]
            lines += [metric_row(run, idx) for run in runs]
            lines += ["", f"Hit@5 by question style, {gname}", "",
                      "| run | " + " | ".join(STYLES) + " |", "| --- | " + " | ".join("---" for _ in STYLES) + " |"]
            for run in runs:
                cells = [f"{hit5(run, [i for i in idx if items[i]['style'] == s]):.3f}" for s in STYLES]
                lines.append(f"| {run} | " + " | ".join(cells) + " |")
            lines.append("")
        # per labelled category, a few reference runs only (all loaded questions)
        ref = [r for r in ("V0 bm25", f"V0 hybrid {slug}", f"V3 rule dense {slug}", f"V3 rule hybrid {slug}",
                           f"V3 oracle-q hybrid {slug}") if r in runs]
        every = list(range(len(items)))
        lines += [f"## Hit@5 by labelled category (n={len(items)})", "",
                  "| category | n | " + " | ".join(ref) + " |", "| --- | --- | " + " | ".join("---" for _ in ref) + " |"]
        for c in cats:
            with_c = [i for i in every if c in items[i]["q_categories"]]
            if with_c:
                lines.append(f"| {c} | {len(with_c)} | " + " | ".join(f"{hit5(r, with_c):.3f}" for r in ref) + " |")
    else:
        ans = [i for i, it in enumerate(items) if it["kind"] == "answerable"]
        focus = [i for i in ans if gold[i] in ("101_DT_1B8000H", "101_DT_1B8000F")]
        by_table = collections.defaultdict(list)
        for i in ans:
            by_table[gold[i]].append(i)
        lines = [f"# Experiment 14: category spaces ({args.model}, n2-L{args.max_labels}, docs {args.max_tokens} tokens)",
                 "", f"spaces {len(cats)}, documents over all spaces {sum(len(spaces[c]) for c in cats)}", "",
                 "## router (answerable questions)", ""]
        for rname, routes in routers.items():
            found = sum(bool(routes[i]) for i in ans)
            overlap = sum(bool(set(routes[i]) & table_cats[i]) for i in ans)
            lines.append(f"- {rname}: category found {found}/{len(ans)}, overlaps gold table's categories {overlap}/{len(ans)}")
        lines += ["", f"## answerable (n={len(ans)}), plus Hit@5 on the 1B8000H/F questions (n={len(focus)}) and per-table mean", "",
                  "| run | Hit@1 | Hit@5 | Hit@10 | MRR@10 | Hit@5 1B8000H/F | Hit@5 table mean |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for run in runs:
            tmean = sum(hit5(run, v) for v in by_table.values()) / len(by_table)
            lines.append(metric_row(run, ans) + f" {hit5(run, focus):.3f} | {tmean:.3f} |")
        lines += ["", f"## all (n={len(items)})", "", "| run | Hit@1 | Hit@5 | Hit@10 | MRR@10 |", "| --- | --- | --- | --- | --- |"]
        lines += [metric_row(run, range(len(items))) for run in runs]
    report = "\n".join(lines) + "\n"
    print(report)

    tag = "" if args.golden.resolve() == GOLDEN.resolve() else f"_{args.golden.stem}"
    tag += "" if args.split == "all" or not routing else f"_{args.split}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path, ranks_path = OUT_DIR / f"category_eval{tag}_summary.md", OUT_DIR / f"category_eval{tag}_ranks.jsonl"
    summary_path.write_text(report, encoding="utf-8")
    with ranks_path.open("w", encoding="utf-8") as f:
        for i, it in enumerate(items):
            row = {"id": it["id"], "gold": gold[i], "question": it["question"]}
            row |= ({"alt": answers[i][1:], "split": it["split"], "style": it["style"],
                     "q_categories": it["q_categories"]} if routing else {"kind": it["kind"]})
            row |= {"routes": {r: routers[r][i] for r in routers}}
            row |= {run: runs[run][i] for run in runs} | {f"top1 {run}": (lists[run][i] or [None])[0] for run in lists}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"-> {summary_path}, {ranks_path}")


if __name__ == "__main__":
    main()
