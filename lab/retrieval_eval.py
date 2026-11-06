"""Plan 20: the experiment 14 / plan 18 retrieval runs again, on eval/golden/retrieval_golden.jsonl.

Golden: required = [[A, B], [C]] -> any table of a group counts, every group must be found. A question's rank is the
worst over its groups of the best rank inside the group. Rows with answerable = false have no tables; retrieval
metrics use the answerable rows only, and the no-answer rows get score-only metrics (AUROC, rejection at 95% recall).

Nothing is chosen on this golden: router C / k / tau are the plan 18 values (picked on routing_v0 dev), so dev and
test are both plain evaluation. oracle-t = categories of every required table (the golden has no q_categories).
Training sets: seen (all synthetic rows), unseen (rows whose table is any required table dropped).

    python lab/retrieval_eval.py
    python lab/retrieval_eval.py --model nlpai-lab/KURE-v1
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from category_db import build_spaces, route_rule
from category_eval import METRICS, ORDER, merge_dense, top, unit
from chroma_common import DEFAULT_MODEL, ROOT, bm25_tokenize, docs, load_records, query_text, read_jsonl, source_sha
from chroma_eval_p1 import DEPTH, KS, OUT_DIR, rank_of, rrf
from embed_eval import doc_vectors
from synth_router import SYNTH, fit_scores, multi_hot, question_vectors, routes_from

GOLDEN = ROOT / "eval" / "golden" / "retrieval_golden.jsonl"
# plan 18, chosen on routing_v0 dev (data/lab/synth_router_summary.md): (C or k, tau)
CHOSEN = {
    "tfidf seen": (16.0, 0.3), "knn seen": (5, 0.4), "probe seen": (8.0, 0.3),
    "tfidf unseen": (16.0, 0.2), "knn unseen": (5, 0.4), "probe unseen": (8.0, 0.15),
}
QTYPES = ("concept", "exact_name", "sibling", "multi_condition", "period_cycle", "multi_table", "ambiguous")
HARD = ("DT_1B8000F", "DT_1B8000G", "DT_1B8000H", "DT_1B8000I", "INH_1B8000F")  # experiment 14's diluted tables
REASONS = ("out_of_scope", "no_period", "no_item")


def group_rank(groups: list[list[str]], keys: list[str]) -> int | None:
    """Worst over the groups of the best rank inside each group; None if some group is not in the list."""
    worst = 0
    for g in groups:
        ranks = [r for r in (rank_of(t, keys) for t in g) if r]
        if not ranks:
            return None
        worst = max(worst, min(ranks))
    return worst


def any_group_rank(groups: list[list[str]], keys: list[str]) -> int | None:
    ranks = [r for r in (rank_of(t, keys) for g in groups for t in g) if r]
    return min(ranks) if ranks else None


def summary(ranks: list[int | None]) -> dict:
    n = max(len(ranks), 1)
    out = {f"Hit@{k}": sum(r is not None and r <= k for r in ranks) / n for k in KS}
    out["MRR@10"] = sum(1 / r for r in ranks if r and r <= 10) / n
    return out


def auroc(pos: list[float], neg: list[float]) -> float:
    """P(score of an answerable row > score of a no-answer row), ties count half (Mann-Whitney)."""
    if not pos or not neg:
        return float("nan")
    p, n = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float(((p > n).sum() + 0.5 * (p == n).sum()) / (p.size * n.size))


def reject_at(pos: list[float], neg: list[float], recall: float = 0.95) -> float:
    """Share of no-answer rows below the threshold that keeps `recall` of the answerable rows."""
    if not pos or not neg:
        return float("nan")
    thr = float(np.quantile(pos, 1 - recall))
    return sum(s < thr for s in neg) / len(neg)


def size_bucket(groups: list[list[str]]) -> str:
    m = max(len(g) for g in groups)
    return "1" if m == 1 else "2-3" if m <= 3 else "4+"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-labels", type=int, default=60)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--synth", type=Path, default=SYNTH)
    p.add_argument("--golden", type=Path, default=GOLDEN)
    args = p.parse_args()

    items, synth = read_jsonl(args.golden), read_jsonl(args.synth)
    ans = [i for i, it in enumerate(items) if it.get("answerable", True) and it.get("required")]
    noans = [i for i, it in enumerate(items) if i not in set(ans)]
    req = [it.get("required") or [] for it in items]
    records, sha, tokenize = load_records(), source_sha(), bm25_tokenize()
    ds = docs(records, "norm", args.max_labels)
    keys = [d["key"] for d in ds]
    index = {k: i for i, k in enumerate(keys)}
    spaces = build_spaces(records)
    cats = [c for c in ORDER if c in spaces]
    cats_of = {k: set() for k in keys}
    for c in cats:
        for j in spaces[c]:
            cats_of[keys[j]].add(c)
    needed = {t for g in req for grp in g for t in grp}
    missing = sorted(needed - set(index))
    if missing:
        raise SystemExit(f"required tables not in tables.jsonl: {missing[:10]}")
    slug, state = args.model.split("/")[-1], {}

    qn = [query_text(it["question"], "norm") for it in items]
    sn = [query_text(r["question"], "norm") for r in synth]
    print("question vectors ...")
    qv = question_vectors(args.model, qn, args.golden.stem, state, args.batch)
    sv = question_vectors(args.model, sn, args.synth.stem, state, args.batch)
    y_all = multi_hot(synth, cats)
    sets = {"seen": list(range(len(synth))), "unseen": [i for i, r in enumerate(synth) if r["table"] not in needed]}

    # ---- routers: plan 18 settings, nothing tuned here ----
    routers = {"rule": [sorted(route_rule(q) & set(cats)) for q in qn]}
    probe_max = {}
    for name, (hp, tau) in CHOSEN.items():
        kind, sname = name.split()
        rows_i = sets[sname]
        sc = fit_scores(kind, hp, [sn[i] for i in rows_i], sv[rows_i], y_all[rows_i], qn, qv)
        routers[name] = routes_from(sc, cats, tau)
        if kind == "probe":
            probe_max[name] = sc.max(axis=1).tolist()
        print(f"  {name}: {'C' if kind != 'knn' else 'k'}={hp}, tau={tau}")
    oracle = [sorted(set().union(*(cats_of[t] for grp in g for t in grp))) if g else [] for g in req]
    routers["oracle-t"] = oracle

    # ---- retrieval: V0, V1 / V3 per router (dense and hybrid), qknn ----
    dv = doc_vectors(args.model, "norm", args.max_labels, args.max_tokens, ds, sha, state, args.batch)
    mean = {c: dv[spaces[c]].mean(axis=0) for c in cats}
    v3 = {c: unit(dv[spaces[c]] - mean[c]) for c in cats}
    print("BM25 indexes ...")
    bm_all = BM25Okapi([tokenize(d["bm25"]) for d in ds])
    bm1 = {c: BM25Okapi([tokenize(ds[i]["bm25"]) for i in spaces[c]]) for c in cats}

    def bm25_scored(bm, idx, q):
        s = np.asarray(bm.get_scores(tokenize(q)))
        order = [j for j in np.argsort(-s, kind="stable")[:DEPTH] if s[j] > 0]
        return [keys[idx[j]] for j in order], (float(s[order[0]]) if order else 0.0)

    v0_dense = [top(dv @ v, keys) for v in qv]
    v0_sparse = [bm25_scored(bm_all, list(range(len(keys))), q) for q in qn]
    v0_bm25 = [r for r, _ in v0_sparse]
    v0_hybrid = [rrf(a, b) for a, b in zip(v0_dense, v0_bm25)]
    lists = {f"V0 dense {slug}": v0_dense, "V0 BM25": v0_bm25, f"V0 hybrid {slug}": v0_hybrid}
    for rname in routers:
        for variant in ("V1", "V3"):
            dense_out, hybrid_out = [], []
            for i, cs in enumerate(routers[rname]):
                if not cs:
                    dense_out.append(v0_dense[i]); hybrid_out.append(v0_hybrid[i]); continue
                hits, bl = [], []
                for c in cs:
                    idx = spaces[c]
                    s = dv[idx] @ qv[i] if variant == "V1" else v3[c] @ unit(qv[i] - mean[c])
                    hits.append(dict(zip([keys[j] for j in idx], s.tolist())))
                    bl.append(bm25_scored(bm1[c], idx, qn[i])[0])
                sparse = rrf(*bl)[:DEPTH] if len(bl) > 1 else bl[0]
                dense = merge_dense(hits)
                dense_out.append(dense); hybrid_out.append(rrf(dense, sparse))
            lists[f"{variant} {rname} dense {slug}"] = dense_out
            lists[f"{variant} {rname} hybrid {slug}"] = hybrid_out
    for sname in ("seen", "unseen"):
        rows_i = sets[sname]
        sims = qv @ sv[rows_i].T
        qk = []
        for s in sims:
            best: dict[str, float] = {}
            for j in np.argsort(-s)[:500]:
                best.setdefault(synth[rows_i[j]]["table"], float(s[j]))
            qk.append(list(best)[:DEPTH])
        lists[f"qknn {sname} {slug}"] = qk
        lists[f"qknn {sname} + V0 hybrid"] = [rrf(a, b) for a, b in zip(qk, v0_hybrid)]
        lists[f"qknn {sname} + V3 probe {sname}"] = [rrf(a, b) for a, b in
                                                     zip(qk, lists[f"V3 probe {sname} hybrid {slug}"])]
    runs = {name: [group_rank(g, t) if g else None for g, t in zip(req, per_q)] for name, per_q in lists.items()}
    partial = {name: [any_group_rank(g, t) if g else None for g, t in zip(req, per_q)] for name, per_q in lists.items()}

    # ---- report ----
    split_of = {"all": ans, "dev": [i for i in ans if items[i]["split"] == "dev"],
                "test": [i for i in ans if items[i]["split"] == "test"]}
    lines = [f"# Plan 20: retrieval runs on {args.golden.name} ({args.model})", "",
             f"rows {len(items)}: answerable {len(ans)} (dev {len(split_of['dev'])}, test {len(split_of['test'])}), "
             f"no answer {len(noans)}; synthetic seen {len(sets['seen'])}, unseen {len(sets['unseen'])}; "
             "router settings from plan 18, nothing tuned here", ""]

    lines += ["## router vs categories of the required tables (answerable rows)", "",
              "| router | any hit | no category | mean categories |", "| --- | --- | --- | --- |"]
    for rname, rs in routers.items():
        hit = sum(bool(set(rs[i]) & set(oracle[i])) for i in ans) / len(ans)
        empty = sum(not rs[i] for i in ans) / len(ans)
        lines.append(f"| {rname} | {hit:.3f} | {empty:.3f} | {np.mean([len(rs[i]) for i in ans]):.2f} |")

    def metric_table(idx, title):
        out = ["", f"## retrieval, {title} (n={len(idx)})", "",
               "| run | " + " | ".join(METRICS) + " |", "| --- | " + " | ".join("---" for _ in METRICS) + " |"]
        for run in runs:
            s = summary([runs[run][i] for i in idx])
            out.append(f"| {run} | " + " | ".join(f"{s[m]:.3f}" for m in METRICS) + " |")
        return out

    for name, idx in split_of.items():
        lines += metric_table(idx, name)

    # breakdowns: the runs the plan compares, not every router
    key_runs = [f"V0 dense {slug}", "V0 BM25", f"V0 hybrid {slug}"]
    key_runs += [f"{v} {r} {m} {slug}" for r in ("rule", "probe seen", "probe unseen", "oracle-t")
                 for m in ("dense", "hybrid") for v in ("V1", "V3")]
    key_runs += ["qknn seen + V0 hybrid", "qknn seen + V3 probe seen"]

    def hit5_table(title, groups: dict[str, list[int]]):
        out = ["", f"Hit@5 by {title} (all answerable)", "",
               "| run | " + " | ".join(f"{g} ({len(ix)})" for g, ix in groups.items()) + " |",
               "| --- | " + " | ".join("---" for _ in groups) + " |"]
        for run in key_runs:
            out.append(f"| {run} | " + " | ".join(
                f"{sum((ranks[run][i] or 999) <= 5 for i in ix) / max(len(ix), 1):.3f}"
                for g, ix in groups.items() for ranks in [partial if g == "any group" else runs]) + " |")
        return out

    lines += hit5_table("question type", {q: [i for i in ans if items[i]["qtype"] == q] for q in QTYPES})
    lines += hit5_table("largest required group", {b: [i for i in ans if size_bucket(req[i]) == b] for b in ("1", "2-3", "4+")})
    lines += hit5_table("source", {s: [i for i in ans if items[i]["source"] == s]
                                   for s in sorted({items[i]["source"] for i in ans})})
    hard = [i for i in ans if any(h in t for grp in req[i] for t in grp for h in HARD)]
    lines += hit5_table("1B8000 F/G/H/I tables (experiment 14)", {"1B8000": hard, "others": [i for i in ans if i not in set(hard)]})
    multi = [i for i in ans if len(req[i]) > 1]
    lines += hit5_table("multi-table rows, every group vs any group", {"every group": multi, "any group": multi})

    # ---- no-answer rows: score signals only, no threshold is chosen here ----
    def top1_cos(per_q):
        return [float(dv[index[t[0]]] @ qv[i]) if t else -1.0 for i, t in enumerate(per_q)]

    signals = {f"top-1 cosine, {run}": top1_cos(lists[run]) for run in
               (f"V0 dense {slug}", f"V0 hybrid {slug}", f"V1 probe seen hybrid {slug}", f"V3 probe seen hybrid {slug}")}
    signals["top-1 BM25 score, V0"] = [s for _, s in v0_sparse]
    signals |= {f"max probability, {n}": v for n, v in probe_max.items()}
    signals["rule found a category (1/0)"] = [float(bool(r)) for r in routers["rule"]]
    by_reason = {r: [i for i in noans if items[i].get("no_answer_reason") == r] for r in REASONS}
    lines += ["", f"## no-answer rows: does a score tell them apart? (answerable {len(ans)} vs no answer {len(noans)})", "",
              "AUROC 0.5 = no signal. reject@95 = share of no-answer rows below the score that keeps 95% of answerable rows.",
              "", "| signal | AUROC | reject@95 | " + " | ".join(f"AUROC {r} ({len(ix)})" for r, ix in by_reason.items()) + " |",
              "| --- | --- | --- | " + " | ".join("---" for _ in by_reason) + " |"]
    for name, sc in signals.items():
        pos = [sc[i] for i in ans]
        cells = [f"{auroc(pos, [sc[i] for i in ix]):.3f}" for ix in by_reason.values()]
        lines.append(f"| {name} | {auroc(pos, [sc[i] for i in noans]):.3f} | "
                     f"{reject_at(pos, [sc[i] for i in noans]):.3f} | " + " | ".join(cells) + " |")
    lines += ["", "qtype counts: " + ", ".join(f"{k} {v}" for k, v in Counter(it["qtype"] for it in items).items())]
    report = "\n".join(lines) + "\n"
    print(report)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path, ranks_path = OUT_DIR / "retrieval_eval_summary.md", OUT_DIR / "retrieval_eval_ranks.jsonl"
    summary_path.write_text(report, encoding="utf-8")
    with ranks_path.open("w", encoding="utf-8") as f:
        for i, it in enumerate(items):
            row = {k: it.get(k) for k in ("qid", "split", "qtype", "source", "question", "required", "no_answer_reason")}
            row["routes"] = {r: routers[r][i] for r in routers}
            row["signals"] = {n: sc[i] for n, sc in signals.items()}
            row |= {run: runs[run][i] for run in runs} | {f"top3 {run}": lists[run][i][:3] for run in lists}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"-> {summary_path}, {ranks_path}")


if __name__ == "__main__":
    main()
