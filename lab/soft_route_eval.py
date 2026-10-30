"""Plan 22: the router's category probabilities as a soft score over all 2,364 tables (no table is dropped).

s_cat(t) = max over t's categories c of p(c | q). Three ways to mix it into the V0 hybrid (dense + BM25, RRF):
    S1  dense score = cosine + lam * s_cat, then RRF with BM25
    S2  a third RRF list: tables by s_cat (ties by cosine), weight w
    S3  dense score = cosine + mu * sum_c p(c | q) * V3 cosine in c (category mean removed; 0 outside c)
lam / w / mu are chosen on routing_v0 dev only (Hit@5, then MRR@10), then checked unchanged on routing_v0 test
and retrieval_golden. Routers: probe seen / unseen (C=8, plan 18), tfidf seen (C=16), oracle (p = 1 on the
gold categories: q_categories on routing_v0, the required tables' categories on retrieval_golden).

    python lab/soft_route_eval.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from category_db import build_spaces
from category_eval import METRICS, ORDER, best_rank, unit
from chroma_common import DEFAULT_MODEL, bm25_tokenize, docs, load_records, query_text, read_jsonl, source_sha
from chroma_eval_p1 import DEPTH, OUT_DIR, RRF_C
from embed_eval import doc_vectors
from retrieval_eval import GOLDEN, HARD, QTYPES, group_rank, summary 
from synth_router import ROUTING, SYNTH, fit_scores, multi_hot, question_vectors

GRIDS = {"S1": [0.0, 0.01, 0.02, 0.05, 0.1, 0.2], "S2": [0.0, 0.25, 0.5, 1.0], "S3": [0.0, 0.1, 0.25, 0.5]}
ROUTERS = {"probe seen": ("probe", 8.0, "seen"), "probe unseen": ("probe", 8.0, "unseen"),
           "tfidf seen": ("tfidf", 16.0, "seen"), "oracle": None}
B = 5000


def wrrf(lists: list[list[str]], weights: list[float], c: int = RRF_C) -> list[str]:
    score: dict[str, float] = {}
    for keys, w in zip(lists, weights):
        if w == 0:
            continue
        for rank, key in enumerate(keys, 1):
            score[key] = score.get(key, 0.0) + w / (c + rank)
    return sorted(score, key=score.get, reverse=True)[:DEPTH]


def hit5(ranks) -> np.ndarray:
    return np.array([r is not None and r <= 5 for r in ranks], float)


def boot(diff: np.ndarray, groups: list[str] | None, rng) -> tuple[float, float]:
    """95% bootstrap interval of the mean difference; groups -> resample whole clusters (tables)."""
    if groups is None:
        d = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(B)]
    else:
        by: dict[str, list[float]] = {}
        for g, v in zip(groups, diff):
            by.setdefault(g, []).append(v)
        cl = [np.array(v) for v in by.values()]
        d = []
        for _ in range(B):
            pick = rng.integers(0, len(cl), len(cl))
            d.append(np.concatenate([cl[i] for i in pick]).mean())
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-labels", type=int, default=60)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()
    rng = np.random.default_rng(0)

    records, sha, tokenize = load_records(), source_sha(), bm25_tokenize()
    ds = docs(records, "norm", args.max_labels)
    keys = [d["key"] for d in ds]
    index = {k: i for i, k in enumerate(keys)}
    spaces = build_spaces(records)
    cats = [c for c in ORDER if c in spaces]
    M = np.zeros((len(keys), len(cats)), dtype=np.float32)  # table x category membership
    for j, c in enumerate(cats):
        M[spaces[c], j] = 1
    slug, state = args.model.split("/")[-1], {}
    synth = read_jsonl(SYNTH)
    sn = [query_text(r["question"], "norm") for r in synth]
    print("vectors ...")
    sv = question_vectors(args.model, sn, SYNTH.stem, state, args.batch)
    dv = doc_vectors(args.model, "norm", args.max_labels, args.max_tokens, ds, sha, state, args.batch)
    mean = {c: dv[spaces[c]].mean(axis=0) for c in cats}
    v3 = {c: unit(dv[spaces[c]] - mean[c]) for c in cats}
    y_all = multi_hot(synth, cats)
    print("BM25 index ...")
    bm_all = BM25Okapi([tokenize(d["bm25"]) for d in ds])

    # ---- the two goldens: answers, rank function, split, cluster key, oracle categories ----
    rg = read_jsonl(ROUTING)
    rt = [it for it in read_jsonl(GOLDEN) if it.get("answerable", True) and it.get("required")]
    sets = {}
    gold = [Path(it["table"]).stem for it in rg]
    answers = [[g, *it.get("alt_tables", [])] for g, it in zip(gold, rg)]
    sets["routing"] = {
        "items": rg, "tag": ROUTING.stem, "rank": [lambda t, a=a: best_rank(a, t) for a in answers],
        "cluster": gold, "oracle": [set(it["q_categories"]) & set(cats) for it in rg],
        "drop": {g for g, it in zip(gold, rg) if it["split"] == "test"}}
    req_tables = [{t for grp in it["required"] for t in grp} for it in rt]
    sets["retrieval"] = {
        "items": rt, "tag": GOLDEN.stem, "rank": [lambda t, g=it["required"]: group_rank(g, t) for it in rt],
        "cluster": [sorted(it["required"][0])[0] for it in rt],
        "oracle": [{cats[j] for t in ts for j in np.flatnonzero(M[index[t]])} for ts in req_tables],
        "drop": set().union(*req_tables)}

    lists: dict[str, dict[str, list[list[str]]]] = {}
    for name, s in sets.items():
        items = s["items"]
        qn = [query_text(it["question"], "norm") for it in items]
        # retrieval_eval cached the vectors of all 117 rows; the unanswerable ones are not used here
        if name == "retrieval":
            allq = [query_text(it["question"], "norm") for it in read_jsonl(GOLDEN)]
            qv_all = question_vectors(args.model, allq, s["tag"], state, args.batch)
            pos = {q: i for i, q in reversed(list(enumerate(allq)))}
            qv = qv_all[[pos[q] for q in qn]]
        else:
            qv = question_vectors(args.model, qn, s["tag"], state, args.batch)
        probs = {}
        for rname, spec in ROUTERS.items():
            if spec is None:
                probs[rname] = np.array([[float(c in o) for c in cats] for o in s["oracle"]])
                continue
            kind, hp, which = spec
            rows_i = [i for i, r in enumerate(synth) if which == "seen" or r["table"] not in s["drop"]]
            probs[rname] = fit_scores(kind, hp, [sn[i] for i in rows_i], sv[rows_i], y_all[rows_i], qn, qv)
        dense = qv @ dv.T  # (n_q, n_tables)
        bm25 = []
        for q in qn:
            sc = np.asarray(bm_all.get_scores(tokenize(q)))
            bm25.append([keys[j] for j in np.argsort(-sc, kind="stable")[:DEPTH] if sc[j] > 0])
        v3s = np.zeros((len(qn), len(cats), len(keys)), dtype=np.float32)  # V3 cosine per category (0 outside)
        for j, c in enumerate(cats):
            v3s[:, j, spaces[c]] = (unit(qv - mean[c]) @ v3[c].T)

        def topk(scores):
            return [keys[j] for j in np.argsort(-scores, kind="stable")[:DEPTH]]

        out = {"V0 hybrid": [wrrf([topk(dense[i]), bm25[i]], [1, 1]) for i in range(len(qn))]}
        for rname, P in probs.items():
            scat = (P[:, None, :] * M[None, :, :]).max(axis=2)  # (n_q, n_tables)
            for x in GRIDS["S1"]:
                out[f"S1 {rname} {x}"] = [wrrf([topk(dense[i] + x * scat[i]), bm25[i]], [1, 1]) for i in range(len(qn))]
            for x in GRIDS["S2"]:
                out[f"S2 {rname} {x}"] = [wrrf([topk(dense[i]), bm25[i], topk(scat[i] + 1e-3 * dense[i])], [1, 1, x])
                                          for i in range(len(qn))]
            for x in GRIDS["S3"]:
                out[f"S3 {rname} {x}"] = [wrrf([topk(dense[i] + x * (P[i] @ v3s[i])), bm25[i]], [1, 1])
                                          for i in range(len(qn))]
        lists[name] = out
        # failure decomposition: wrong tables in V0's top 5 that share no category with the gold side
        miss, cross, all_cross = 0, [], 0
        for i in range(len(qn)):
            r = s["rank"][i](out["V0 hybrid"][i])
            if r is not None and r <= 5:
                continue
            miss += 1
            gold_t = set(answers[i]) if name == "routing" else req_tables[i]
            gcat = {cats[j] for t in gold_t for j in np.flatnonzero(M[index[t]])}
            wrong = [t for t in out["V0 hybrid"][i][:5] if t not in gold_t]
            flags = [not ({cats[j] for j in np.flatnonzero(M[index[t]])} & gcat) for t in wrong]
            cross += flags
            all_cross += bool(flags) and all(flags)
        s["decomp"] = (miss, float(np.mean(cross)) if cross else float("nan"), all_cross)

    # ---- pick lam / w / mu on routing dev ----
    rgs = sets["routing"]
    dev = [i for i, it in enumerate(rg) if it["split"] == "dev"]
    ranks = {n: {run: [s["rank"][i](l[i]) for i in range(len(l))] for run, l in lists[n].items()} for n, s in sets.items()}
    chosen, curves = {}, {}
    for m in GRIDS:
        for rname in ROUTERS:
            row = []
            for x in GRIDS[m]:
                sm = summary([ranks["routing"][f"{m} {rname} {x}"][i] for i in dev])
                row.append((sm["Hit@5"], sm["MRR@10"], -x, x))
            curves[(m, rname)] = row
            chosen[(m, rname)] = max(row)[3]

    # ---- report ----
    lines = [f"# Plan 22: soft category signal on all tables ({args.model})", "",
             "lam / w / mu chosen on routing_v0 dev (Hit@5, then MRR@10); x = 0 is V0 hybrid", "",
             "## dev curves (Hit@5 / MRR@10 per value)", "", "| method | router | " +
             " | ".join(f"value {k + 1}" for k in range(len(GRIDS["S1"]))) + " | chosen |", "| --- | --- |" + " --- |" * (len(GRIDS["S1"]) + 1)]
    for (m, rname), row in curves.items():
        cells = [f"{h:.3f} / {mrr:.3f}" for h, mrr, _, _ in row] + [""] * (len(GRIDS["S1"]) - len(row))
        edge = " (grid edge)" if chosen[(m, rname)] == GRIDS[m][-1] else ""
        lines.append(f"| {m} | {rname} | " + " | ".join(cells) + f" | {chosen[(m, rname)]}{edge} |")
    lines += ["", "(columns are the grid values in order: S1 " + ", ".join(map(str, GRIDS["S1"])) +
              "; S2 " + ", ".join(map(str, GRIDS["S2"])) + "; S3 " + ", ".join(map(str, GRIDS["S3"])) + ")"]

    def block(n, idx, title):
        s = sets[n]
        base = hit5([ranks[n]["V0 hybrid"][i] for i in idx])
        cl = [s["cluster"][i] for i in idx]
        out = ["", f"## {title} (n={len(idx)})", "",
               "| run | " + " | ".join(METRICS) + " | dHit@5 vs V0 | 95% (question) | 95% (table) | wins / losses |",
               "| --- | " + " | ".join("---" for _ in METRICS) + " | --- | --- | --- | --- |"]
        runs = ["V0 hybrid"] + [f"{m} {r} {chosen[(m, r)]}" for m in GRIDS for r in ROUTERS]
        for run in runs:
            rk = [ranks[n][run][i] for i in idx]
            sm, h = summary(rk), hit5(rk)
            d = h - base
            q, t = boot(d, None, rng), boot(d, cl, rng)
            out.append(f"| {run} | " + " | ".join(f"{sm[k]:.3f}" for k in METRICS) +
                       f" | {d.mean():+.3f} | [{q[0]:+.3f}, {q[1]:+.3f}] | [{t[0]:+.3f}, {t[1]:+.3f}] | "
                       f"{int((d > 0).sum())} / {int((d < 0).sum())} |")
        return out

    lines += block("routing", [i for i, it in enumerate(rg) if it["split"] == "test"], "routing_v0 test")
    for split in ("all", "dev", "test"):
        idx = [i for i, it in enumerate(rt) if split == "all" or it["split"] == split]
        lines += block("retrieval", idx, f"retrieval_golden {split} (answerable)")

    picked = ["V0 hybrid"] + [f"{m} {r} {chosen[(m, r)]}" for m in GRIDS for r in ("probe seen", "oracle")]
    groups = {q: [i for i, it in enumerate(rt) if it["qtype"] == q] for q in QTYPES}
    hard = [i for i, ts in enumerate(req_tables) if any(h in t for t in ts for h in HARD)]
    groups |= {"1B8000": hard, "others": [i for i in range(len(rt)) if i not in set(hard)]}
    lines += ["", "## retrieval_golden Hit@5 by question type / 1B8000 group", "",
              "| run | " + " | ".join(f"{g} ({len(ix)})" for g, ix in groups.items()) + " |",
              "| --- | " + " | ".join("---" for _ in groups) + " |"]
    for run in picked:
        lines.append(f"| {run} | " + " | ".join(
            f"{hit5([ranks['retrieval'][run][i] for i in ix]).mean():.3f}" for ix in groups.values()) + " |")

    lines += ["", "## how much can a category signal fix? (V0 hybrid misses at 5; routing = dev + test)", "",
              "| golden | missed | wrong top-5 tables from another category | missed rows where all wrong top-5 are another category |",
              "| --- | --- | --- | --- |"]
    for n, s in sets.items():
        miss, frac, allc = s["decomp"]
        lines.append(f"| {n} | {miss} | {frac:.3f} | {allc} |")
    report = "\n".join(lines) + "\n"
    print(report)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path, ranks_path = OUT_DIR / "soft_route_summary.md", OUT_DIR / "soft_route_ranks.jsonl"
    summary_path.write_text(report, encoding="utf-8")
    with ranks_path.open("w", encoding="utf-8") as f:
        for n, s in sets.items():
            for i, it in enumerate(s["items"]):
                row = {"golden": n, "id": it.get("id") or it.get("qid"), "split": it["split"], "question": it["question"]}
                row |= {run: ranks[n][run][i] for run in picked}
                row |= {f"top3 {run}": lists[n][run][i][:3] for run in picked}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"-> {summary_path}, {ranks_path}")


if __name__ == "__main__":
    main()
