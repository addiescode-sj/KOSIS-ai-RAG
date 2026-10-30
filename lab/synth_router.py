"""Plan 18: learn the topic router (and question -> table matching) from synthetic questions, check on the routing golden.

Training data: data/lab/synth/synth_v0.jsonl (plan 17). Evaluation: eval/golden/routing_v0.jsonl, tuned on dev only.

Routers (19 topic categories, multi-label):
    tfidf   character 2-4-gram TF-IDF + one-vs-rest logistic regression
    knn     bge-m3 question vectors, label vote of the k most similar synthetic questions
    probe   bge-m3 question vectors (frozen) + one-vs-rest logistic regression (linear probe)
Output rule: the best category always; the second only when its score >= tau. C / k and tau are chosen on dev.

Training sets: seen (all rows), unseen (rows of golden test tables dropped; dev tables are never in synth_v0),
and the same two with template/keyword rows only (-tk), to see the effect of Claude-written rows.
qknn: tables ranked by their most similar synthetic question (query-to-query). Dev tables are not in synth_v0,
so qknn can only hit on test, and only with the seen set.

    python lab/synth_router.py
    python lab/synth_router.py --model nlpai-lab/KURE-v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from category_db import build_spaces, route_rule
from category_eval import METRICS, ORDER, STYLES, best_rank, merge_dense, router_lines, top, unit
from chroma_common import DEFAULT_MODEL, ROOT, bm25_tokenize, docs, encode, load_records, query_text, read_jsonl, source_sha
from chroma_eval_p1 import DEPTH, OUT_DIR, rrf, summary
from embed_eval import CACHE, doc_vectors, model_for

SYNTH = ROOT / "data" / "lab" / "synth" / "synth_v0.jsonl"
ROUTING = ROOT / "eval" / "golden" / "routing_v0.jsonl"
GRID = {"tfidf": [1.0, 4.0, 16.0], "knn": [5, 10, 20, 40], "probe": [0.5, 2.0, 8.0]}
TAUS = [0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 1.01]  # 1.01 = never a second category


def question_vectors(model_name: str, texts: list[str], tag: str, state: dict, batch: int) -> np.ndarray:
    """Query-side vectors for many questions, cached like the document vectors (keyed by the texts' hash)."""
    sha = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()[:8]
    npy = CACHE / f"{model_name.split('/')[-1]}__q-{tag}__{sha}.npy"
    if npy.exists():
        print(f"  cache {npy.name}")
        return np.load(npy)
    vecs = np.asarray(encode(model_for(model_name, 512, state), model_name, texts, "query", batch), dtype=np.float32)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(npy, vecs)
    print(f"  embedded {npy.name}")
    return vecs


def multi_hot(rows: list[dict], cats: list[str]) -> np.ndarray:
    y = np.zeros((len(rows), len(cats)), dtype=int)
    for i, r in enumerate(rows):
        for c in r["q_categories"]:
            if c in cats:
                y[i, cats.index(c)] = 1
    return y


def fit_scores(kind: str, hp, train_text, train_vec, y, test_text, test_vec) -> np.ndarray:
    """Category scores (n_test, n_cats); every category column is learnt (all appear in synth_v0)."""
    if kind == "knn":
        sims = test_vec @ train_vec.T
        nn = np.argsort(-sims, axis=1)[:, :hp]
        w = np.take_along_axis(sims, nn, axis=1).clip(min=0)
        return np.einsum("qk,qkc->qc", w, y[nn]) / np.maximum(w.sum(axis=1, keepdims=True), 1e-9)
    lr = OneVsRestClassifier(LogisticRegression(C=hp, max_iter=2000))
    if kind == "tfidf":
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
        lr.fit(vec.fit_transform(train_text), y)
        return lr.predict_proba(vec.transform(test_text))
    lr.fit(train_vec, y)
    return lr.predict_proba(test_vec)


def routes_from(scores: np.ndarray, cats: list[str], tau: float) -> list[list[str]]:
    out = []
    for s in scores:
        o = np.argsort(-s)
        out.append([cats[o[0]]] + ([cats[o[1]]] if s[o[1]] >= tau else []))
    return out


def hit_rate(routes, items, idx) -> float:
    return sum(bool(set(routes[i]) & set(items[i]["q_categories"])) for i in idx) / max(len(idx), 1)


def f1(routes, items, idx) -> float:
    tot = 0.0
    for i in idx:
        got, want = set(routes[i]), set(items[i]["q_categories"])
        tot += 2 * len(got & want) / (len(got) + len(want))
    return tot / max(len(idx), 1)


def table_cv(kind, hp, tau, rows, text, vec, y, cats, folds=5) -> float:
    """Hit (any label) on synthetic questions, 5 folds split by table: the in-distribution score."""
    tables = sorted({r["table"] for r in rows})
    fold_of = {t: int(hashlib.md5(t.encode()).hexdigest(), 16) % folds for t in tables}
    f = np.array([fold_of[r["table"]] for r in rows])
    hit = 0
    for k in range(folds):
        tr, te = np.where(f != k)[0], np.where(f == k)[0]
        sc = fit_scores(kind, hp, [text[i] for i in tr], vec[tr], y[tr], [text[i] for i in te], vec[te])
        hit += sum(bool(set(r) & set(rows[j]["q_categories"])) for r, j in zip(routes_from(sc, cats, tau), te))
    return hit / len(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-labels", type=int, default=60)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--synth", type=Path, default=SYNTH)
    p.add_argument("--golden", type=Path, default=ROUTING)
    p.add_argument("--no-cv", action="store_true", help="skip the 5-fold score on synthetic data")
    args = p.parse_args()

    items, synth = read_jsonl(args.golden), read_jsonl(args.synth)
    dev = [i for i, it in enumerate(items) if it["split"] == "dev"]
    test = [i for i, it in enumerate(items) if it["split"] == "test"]
    gold = [Path(it["table"]).stem for it in items]
    answers = [[g, *it.get("alt_tables", [])] for g, it in zip(gold, items)]
    records, sha, tokenize = load_records(), source_sha(), bm25_tokenize()
    ds = docs(records, "norm", args.max_labels)
    keys = [d["key"] for d in ds]
    spaces = build_spaces(records)
    cats = [c for c in ORDER if c in spaces]
    index = {k: i for i, k in enumerate(keys)}
    table_cats = [sorted(set(c for c in cats if index[g] in spaces[c])) for g in gold]
    slug, state = args.model.split("/")[-1], {}

    qn = [query_text(it["question"], "norm") for it in items]
    sn = [query_text(r["question"], "norm") for r in synth]
    print("question vectors ...")
    qv = question_vectors(args.model, qn, args.golden.stem, state, args.batch)
    sv = question_vectors(args.model, sn, args.synth.stem, state, args.batch)
    y_all = multi_hot(synth, cats)
    sets = {
        "seen": [i for i in range(len(synth))],
        "unseen": [i for i, r in enumerate(synth) if not r["golden_test"]],
        "seen-tk": [i for i, r in enumerate(synth) if r["generator"] != "hand"],
        "unseen-tk": [i for i, r in enumerate(synth) if r["generator"] != "hand" and not r["golden_test"]],
    }

    # ---- routers: pick C / k on dev hit of the first category, then tau on dev F1 ----
    routers = {"rule": [sorted(route_rule(q) & set(cats)) for q in qn]}
    chosen, cv = {}, {}
    for sname, rows_i in sets.items():
        tr_text, tr_vec, y = [sn[i] for i in rows_i], sv[rows_i], y_all[rows_i]
        for kind in ("tfidf", "knn", "probe"):
            best = None
            for hp in GRID[kind]:
                sc = fit_scores(kind, hp, tr_text, tr_vec, y, qn, qv)
                h = hit_rate(routes_from(sc, cats, 9.0), items, dev)
                if best is None or h > best[0]:
                    best = (h, hp, sc)
            _, hp, sc = best
            tau = max(TAUS, key=lambda t: (f1(routes_from(sc, cats, t), items, dev), -t))
            name = f"{kind} {sname}"
            routers[name] = routes_from(sc, cats, tau)
            chosen[name] = (hp, tau)
            if sname == "seen" and not args.no_cv:
                print(f"  5-fold on synthetic: {name}")
                cv[name] = table_cv(kind, hp, tau, [synth[i] for i in rows_i], tr_text, tr_vec, y, cats)
            print(f"  {name}: {'C' if kind != 'knn' else 'k'}={hp}, tau={tau}, dev hit {hit_rate(routers[name], items, dev):.3f}")
    routers["oracle-q"] = [sorted(set(it["q_categories"]) & set(cats)) for it in items]

    # ---- retrieval: V0 hybrid, V1 / V3 hybrid per router (same as category_eval), qknn ----
    dv = doc_vectors(args.model, "norm", args.max_labels, args.max_tokens, ds, sha, state, args.batch)
    mean = {c: dv[spaces[c]].mean(axis=0) for c in cats}
    v3 = {c: unit(dv[spaces[c]] - mean[c]) for c in cats}
    print("BM25 indexes ...")
    bm_all = BM25Okapi([tokenize(d["bm25"]) for d in ds])
    bm1 = {c: BM25Okapi([tokenize(ds[i]["bm25"]) for i in spaces[c]]) for c in cats}

    def bm25_rank(bm, idx, q):
        s = np.asarray(bm.get_scores(tokenize(q)))
        return [keys[idx[j]] for j in np.argsort(-s, kind="stable")[:DEPTH] if s[j] > 0]

    v0_dense = [top(dv @ v, keys) for v in qv]
    v0_bm25 = [bm25_rank(bm_all, list(range(len(keys))), q) for q in qn]
    v0_hybrid = [rrf(a, b) for a, b in zip(v0_dense, v0_bm25)]
    lists = {f"V0 hybrid {slug}": v0_hybrid}
    retrieval_routers = ["rule", "tfidf seen", "knn seen", "probe seen", "tfidf unseen", "knn unseen", "probe unseen",
                         "oracle-q"]
    for rname in retrieval_routers:
        for variant in ("V1", "V3"):
            out = []
            for i, cs in enumerate(routers[rname]):
                if not cs:
                    out.append(v0_hybrid[i]); continue
                hits, bl = [], []
                for c in cs:
                    idx = spaces[c]
                    s = dv[idx] @ qv[i] if variant == "V1" else v3[c] @ unit(qv[i] - mean[c])
                    hits.append(dict(zip([keys[j] for j in idx], s.tolist())))
                    bl.append(bm25_rank(bm1[c], idx, qn[i]))
                sparse = rrf(*bl)[:DEPTH] if len(bl) > 1 else bl[0]
                out.append(rrf(merge_dense(hits), sparse))
            lists[f"{variant} {rname} hybrid {slug}"] = out
    for sname in ("seen", "unseen"):
        rows_i = sets[sname]
        sims = qv @ sv[rows_i].T
        qk = []
        for s in sims:
            best: dict[str, float] = {}
            for j in np.argsort(-s)[:500]:
                t = synth[rows_i[j]]["table"]
                best.setdefault(t, float(s[j]))
            qk.append(list(best)[:DEPTH])
        lists[f"qknn {sname} {slug}"] = qk
        lists[f"qknn {sname} + V0 hybrid"] = [rrf(a, b) for a, b in zip(qk, v0_hybrid)]
        lists[f"qknn {sname} + V3 probe {sname}"] = [rrf(a, b) for a, b in
                                                     zip(qk, lists[f"V3 probe {sname} hybrid {slug}"])]
    runs = {name: [best_rank(a, t) for a, t in zip(answers, per_q)] for name, per_q in lists.items()}

    # ---- report ----
    lines = [f"# Plan 18: routers learnt from {args.synth.name}, checked on {args.golden.name} ({args.model})", "",
             f"synthetic rows: " + ", ".join(f"{k} {len(v)}" for k, v in sets.items()) +
             f"; golden dev {len(dev)}, test {len(test)}", "",
             "## chosen on dev", "", "| router | C or k | tau | 5-fold hit on synthetic (by table) |", "| --- | --- | --- | --- |"]
    lines += [f"| {n} | {hp} | {tau} | {cv.get(n, float('nan')):.3f} |" for n, (hp, tau) in chosen.items()]
    for gname, idx in (("dev", dev), ("test", test)):
        lines += ["", f"## router, {gname} (n={len(idx)})", ""] + router_lines(routers, items, idx, cats, [set(t) for t in table_cats])
        lines += ["", f"## retrieval, {gname} (n={len(idx)})", "",
                  "| run | " + " | ".join(METRICS) + " |", "| --- | " + " | ".join("---" for _ in METRICS) + " |"]
        for run in runs:
            s = summary([runs[run][i] for i in idx])
            lines.append(f"| {run} | " + " | ".join(f"{s[m]:.3f}" for m in METRICS) + " |")
        lines += ["", f"Hit@5 by question style, {gname}", "",
                  "| run | " + " | ".join(STYLES) + " |", "| --- | " + " | ".join("---" for _ in STYLES) + " |"]
        for run in runs:
            cells = []
            for st in STYLES:
                sub = [i for i in idx if items[i]["style"] == st]
                cells.append(f"{sum((runs[run][i] or 999) <= 5 for i in sub) / max(len(sub), 1):.3f}")
            lines.append(f"| {run} | " + " | ".join(cells) + " |")
    report = "\n".join(lines) + "\n"
    print(report)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path, ranks_path = OUT_DIR / "synth_router_summary.md", OUT_DIR / "synth_router_ranks.jsonl"
    summary_path.write_text(report, encoding="utf-8")
    with ranks_path.open("w", encoding="utf-8") as f:
        for i, it in enumerate(items):
            row = {"id": it["id"], "gold": gold[i], "alt": answers[i][1:], "split": it["split"], "style": it["style"],
                   "question": it["question"], "q_categories": it["q_categories"],
                   "routes": {r: routers[r][i] for r in routers}}
            row |= {run: runs[run][i] for run in runs} | {f"top1 {run}": (lists[run][i] or [None])[0] for run in lists}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"-> {summary_path}, {ranks_path}")


if __name__ == "__main__":
    main()
