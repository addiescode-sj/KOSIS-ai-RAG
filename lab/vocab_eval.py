"""Plan 25: colloquial -> official terms added to the BM25 query only (step 2 of 2, run vocab_expand.py first).

A lexicon expression found in the normalized question (at a word start, longest first, no overlaps) adds its
official terms that the question does not already contain. BM25 score = BM25(question) + w * BM25(added terms);
the question's own words are never removed. Dense stays on the original question (cached vectors), and the
fusion is the V0 hybrid one (dense top-50 + BM25 top-50, RRF c=60). A question with no match is exactly V0.

Chosen on routing_v0 dev only (Hit@5, then MRR@10): lexicon and w. Reported with that choice, nothing re-picked:
routing_v0 test, retrieval_golden all / dev / test (answerable rows). synth_v0 is not used (D2 made it).
Each run writes its own files: data/lab/vocab_eval_<run id>_summary.md and _ranks.jsonl.

    python lab/vocab_eval.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from lab_common import (
    DEFAULT_MODEL, DEPTH, METRICS, OUT_DIR, RETRIEVAL, ROOT, ROUTING, bm25_tokenize, boot, doc_vectors, docs,
    group_rank, load_records, query_text, question_vectors, read_jsonl, rrf, source_sha, summary, top,
)

VOCAB = ROOT / "data" / "lab" / "vocab"
LEXICONS = ("d1", "d1hm", "d2", "d1+d2")
WEIGHTS = (0.25, 0.5, 1.0)
# reporting only (plan 25 1): colloquial groups found in the failure analysis; never used to choose anything
GROUPS = {"출생 구어": r"태어|낳|아이|아기|엄마", "나이 구어": r"나이|노인|어르신|청년|[0-9]0대",
          "이동 구어": r"이사|옮|다른 (지역|시도|곳)|떠난", "성별 구어": r"여성|남성"}


def matcher(lex: dict[str, list[str]]):
    """question -> (matched expressions, added official terms)."""
    exprs = sorted(lex, key=len, reverse=True)
    pats = [(e, re.compile(r"(?<![가-힣A-Za-z0-9])" + re.escape(e))) for e in exprs]

    def run(q: str) -> tuple[list[str], list[str]]:
        used = np.zeros(len(q) + 1, bool)
        hits, added = [], []
        for e, pat in pats:
            for m in pat.finditer(q):
                if used[m.start():m.end()].any():
                    continue
                used[m.start():m.end()] = True
                hits.append(e)
                added += [t for t in lex[e] if t not in q and t not in added]
                break
        return hits, added
    return run


def sha8(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-labels", type=int, default=60)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()

    records, sha, tokenize = load_records(), source_sha(), bm25_tokenize()
    ds = docs(records, "norm", args.max_labels)
    keys = [d["key"] for d in ds]
    state = {}
    dv = doc_vectors(args.model, "norm", args.max_labels, args.max_tokens, ds, sha, state, args.batch)
    bm = BM25Okapi([tokenize(d["bm25"]) for d in ds])
    lexicons = {n: json.loads((VOCAB / f"lexicon_{n}.json").read_text(encoding="utf-8")) for n in LEXICONS}
    match = {n: matcher(lex) for n, lex in lexicons.items()}

    # ---- the two goldens: groups of accepted tables per row ----
    rg = read_jsonl(ROUTING)
    rt = [it for it in read_jsonl(RETRIEVAL) if it.get("answerable", True) and it.get("required")]
    sets = {
        "routing": {"items": rg, "tag": ROUTING.stem, "ids": [it["id"] for it in rg],
                    "groups": [[[Path(it["table"]).stem, *it.get("alt_tables", [])]] for it in rg],
                    "split": [it["split"] for it in rg]},
        "retrieval": {"items": rt, "tag": RETRIEVAL.stem, "ids": [it["qid"] for it in rt],
                      "groups": [it["required"] for it in rt], "split": [it["split"] for it in rt]},
    }
    rt_all = read_jsonl(RETRIEVAL)  # vectors are cached for the whole file
    for name, s in sets.items():
        items = s["items"] if name == "routing" else rt_all
        qn_all = [query_text(it["question"], "norm") for it in items]
        qv_all = question_vectors(args.model, qn_all, s["tag"], state, args.batch)
        keep = list(range(len(items))) if name == "routing" else \
            [i for i, it in enumerate(rt_all) if it.get("answerable", True) and it.get("required")]
        s["qn"], s["qv"] = [qn_all[i] for i in keep], qv_all[keep]
        s["cluster"] = [g[0][0] for g in s["groups"]]

        dense = [top(dv @ v, keys) for v in s["qv"]]
        base = [np.asarray(bm.get_scores(tokenize(q))) for q in s["qn"]]

        def bm25_list(sc: np.ndarray) -> list[str]:
            return [keys[j] for j in np.argsort(-sc, kind="stable")[:DEPTH] if sc[j] > 0]

        runs, s["match"] = {}, {}
        b0 = [bm25_list(sc) for sc in base]
        runs["V0"] = {"hybrid": [rrf(a, b) for a, b in zip(dense, b0)], "bm25": b0}
        for ln in LEXICONS:
            found = [match[ln](q) for q in s["qn"]]
            s["match"][ln] = found
            extra = [np.asarray(bm.get_scores(tokenize(" ".join(add)))) if add else None for _, add in found]
            for w in WEIGHTS:
                bl = [bm25_list(sc + w * ex) if ex is not None else b0[i] for i, (sc, ex) in enumerate(zip(base, extra))]
                runs[f"{ln} w={w}"] = {"hybrid": [rrf(a, b) for a, b in zip(dense, bl)], "bm25": bl}
        s["runs"] = {r: {m: [group_rank(g, t) for g, t in zip(s["groups"], v[m])] for m in v} for r, v in runs.items()}

    # ---- choose lexicon and w on routing dev ----
    rs = sets["routing"]
    dev = [i for i, sp in enumerate(rs["split"]) if sp == "dev"]

    def score(run: str, idx: list[int]) -> tuple[float, float]:
        sm = summary([rs["runs"][run]["hybrid"][i] for i in idx])
        return sm["Hit@5"], sm["MRR@10"]

    candidates = [r for r in rs["runs"] if r != "V0"]
    chosen = max(candidates, key=lambda r: score(r, dev))
    rng = np.random.default_rng(0)

    def block(name: str, idx: list[int], title: str) -> list[str]:
        s = sets[name]
        out = ["", f"## {title} (n={len(idx)})", "",
               "| run | " + " | ".join(METRICS) + " | BM25 Recall@50 | triggered | dHit@5 vs V0 | 95% (question) | 95% (table) | wins / losses |",
               "| --- | " + " | ".join("---" for _ in METRICS) + " | --- | --- | --- | --- | --- | --- |"]
        v0 = np.array([(s["runs"]["V0"]["hybrid"][i] or 999) <= 5 for i in idx], float)
        for run in ["V0", *candidates]:
            sm = summary([s["runs"][run]["hybrid"][i] for i in idx])
            rec = np.mean([s["runs"][run]["bm25"][i] is not None for i in idx])
            ln = run.split()[0]
            trig = np.mean([bool(s["match"][ln][i][1]) for i in idx]) if run != "V0" else 0.0
            h = np.array([(s["runs"][run]["hybrid"][i] or 999) <= 5 for i in idx], float)
            d = h - v0
            if run == "V0":
                ci_q = ci_t = (0.0, 0.0)
            else:
                ci_q = boot(d, None, rng)
                ci_t = boot(d, [s["cluster"][i] for i in idx], rng)
            mark = " **(chosen)**" if run == chosen else ""
            out.append(f"| {run}{mark} | " + " | ".join(f"{sm[m]:.3f}" for m in METRICS)
                       + f" | {rec:.3f} | {trig:.2f} | {d.mean():+.3f} | [{ci_q[0]:+.3f}, {ci_q[1]:+.3f}] | "
                       f"[{ci_t[0]:+.3f}, {ci_t[1]:+.3f}] | {int((d > 0).sum())} / {int((d < 0).sum())} |")
        return out

    def groups_block(idx_by: dict[str, list[int]]) -> list[str]:
        out = ["", f"## Hit@5 by colloquial group, V0 vs chosen ({chosen}); reporting only", "",
               "| golden | group | n | triggered | V0 | chosen |", "| --- | --- | --- | --- | --- | --- |"]
        ln = chosen.split()[0]
        for name, idx in idx_by.items():
            s = sets[name]
            for g, pat in GROUPS.items():
                ix = [i for i in idx if re.search(pat, s["qn"][i])]
                if not ix:
                    continue
                h = lambda run: np.mean([(s["runs"][run]["hybrid"][i] or 999) <= 5 for i in ix])
                trig = np.mean([bool(s["match"][ln][i][1]) for i in ix])
                out.append(f"| {name} | {g} | {len(ix)} | {trig:.2f} | {h('V0'):.3f} | {h(chosen):.3f} |")
        return out

    def triggered_block() -> list[str]:
        out = ["", f"## rows where {chosen.split()[0]} adds terms: Hit@5 V0 vs chosen", "",
               "| golden | split | triggered rows | V0 | chosen | wins / losses |", "| --- | --- | --- | --- | --- | --- |"]
        ln = chosen.split()[0]
        for name, s in sets.items():
            for sp in ("dev", "test"):
                ix = [i for i, x in enumerate(s["split"]) if x == sp and s["match"][ln][i][1]]
                if not ix:
                    continue
                a = np.array([(s["runs"]["V0"]["hybrid"][i] or 999) <= 5 for i in ix], float)
                b = np.array([(s["runs"][chosen]["hybrid"][i] or 999) <= 5 for i in ix], float)
                out.append(f"| {name} | {sp} | {len(ix)} | {a.mean():.3f} | {b.mean():.3f} | "
                           f"{int((b > a).sum())} / {int((b < a).sum())} |")
        return out

    rid = time.strftime("%Y%m%d-%H%M%S")
    hashes = {n: sha8(VOCAB / f"lexicon_{n}.json") for n in LEXICONS}
    lines = [f"# Plan 25: BM25 query expansion with colloquial lexicons ({args.model})", "",
             f"run {rid}; lexicon sha {hashes}; routing {sha8(ROUTING)}, retrieval {sha8(RETRIEVAL)}, tables {sha[:8]}",
             f"chosen on routing_v0 dev (Hit@5, then MRR@10): **{chosen}**; lexicon sizes "
             + ", ".join(f"{n} {len(lexicons[n])}" for n in LEXICONS)]
    idx_r = {sp: [i for i, x in enumerate(rs["split"]) if x == sp] for sp in ("dev", "test")}
    rt_s = sets["retrieval"]["split"]
    idx_t = {"all": list(range(len(rt_s))), "dev": [i for i, x in enumerate(rt_s) if x == "dev"],
             "test": [i for i, x in enumerate(rt_s) if x == "test"]}
    lines += block("routing", idx_r["dev"], "routing_v0 dev (choice made here)")
    lines += block("routing", idx_r["test"], "routing_v0 test")
    for sp, ix in idx_t.items():
        lines += block("retrieval", ix, f"retrieval_golden {sp} (answerable; test was looked at before, exploratory)")
    lines += triggered_block()
    lines += groups_block({"routing": idx_r["test"], "retrieval": idx_t["all"]})
    report = "\n".join(lines) + "\n"
    print(report)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / f"vocab_eval_{rid}_summary.md"
    ranks_path = OUT_DIR / f"vocab_eval_{rid}_ranks.jsonl"
    summary_path.write_text(report, encoding="utf-8")
    with ranks_path.open("w", encoding="utf-8") as f:
        for name, s in sets.items():
            for i, qid in enumerate(s["ids"]):
                row = {"golden": name, "id": qid, "split": s["split"][i], "question": s["items"][i]["question"],
                       "match": {ln: s["match"][ln][i] for ln in LEXICONS}}
                row |= {run: s["runs"][run]["hybrid"][i] for run in s["runs"]}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"-> {summary_path}, {ranks_path}")


if __name__ == "__main__":
    main()
