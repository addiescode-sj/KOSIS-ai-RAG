"""Plan 34: does the CLOVA Studio reranker API pick the right table out of the current search's top K?

The search stays the current one (F0: fine-tuned unseen-e1 dense top-50 + V0 BM25 top-50, RRF c=60). Its top K
tables (K=20) go to /v1/api-tools/reranker with the question. The API gives no scores: it answers and lists the
documents it cited (citedDocuments), or none plus suggestedQueries. Rankings compared:
    F0  the current search
    H1  cited tables first in the order the API lists them, then the rest of F0
    H2  cited tables first in their F0 order, then the rest of F0
H1 vs H2 is the only choice: picked on retrieval_golden dev (Hit@1, then Hit@5, then H2), judged on all answerable
rows against F0 (table-level bootstrap, as in plans 32/33/35). Also reported: by split and qtype, how many tables
are cited, how often nothing is cited on answerable vs no-answer rows, tokens and seconds per question.

Every response is cached in data/lab/hcx_rerank/<tag>/<qid>.json, so a rerun calls nothing and a stopped run
resumes. The key is HCX_API_KEY from the repo root .env (or vendor/phase1/.env); it is never printed.

    python lab/hcx_rerank_eval.py --limit 5        # first 5 questions only, to check the calls
    python lab/hcx_rerank_eval.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
import uuid
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi

from lab_common import (
    DEFAULT_MODEL, DEPTH, METRICS, OUT_DIR, RETRIEVAL, bm25_tokenize, boot, docs, encode, group_rank, load_records,
    query_text, read_jsonl, rrf, source_sha, summary, top,
)
from ft_train import FT_DIR, load_ft
from rag_reasoning_eval import api_key

HOST = "https://clovastudio.stream.ntruss.com"
PATH = "/v1/api-tools/reranker"
TAG_RE = re.compile(r"<doc-?([^>/\s]+)>")  # answer text cites as <docN> / <doc-N>; only used for a cross-check


def post(body: dict, key: str) -> dict:
    """One call; HTTP errors come back as {'http': code, 'body': ...}."""
    req = urllib.request.Request(HOST + PATH, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                          "X-NCP-CLOVASTUDIO-REQUEST-ID": uuid.uuid4().hex})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"http": e.code, "body": e.read().decode("utf-8", "replace")[:500]}


def call(body: dict, key: str, interval: float) -> dict:
    waits = [5, 15, 30, 60, 120]
    for attempt in range(len(waits) + 1):
        time.sleep(interval)
        res = post(body, key)
        code = str(res.get("status", {}).get("code", res.get("http", "")))
        if code == "20000":
            return res
        if attempt < len(waits) and (code.startswith("429") or code.startswith("5") or code == "42901"):
            print(f"    retry after {waits[attempt]}s ({code})")
            time.sleep(waits[attempt])
            continue
        raise RuntimeError(f"reranker call failed ({code}): {str(res)[:300]}")
    raise RuntimeError("unreachable")


def query_vectors(run_dir, qn: list[str], max_tokens: int, batch: int) -> tuple[np.ndarray, np.ndarray]:
    """unseen-e1 document vectors + question vectors: from the fusion_eval caches when they cover all 117 questions."""
    ck = run_dir / "unseen-e1.pt"
    doc_cache, q_cache = ck.with_name("unseen-e1__docs.npy"), ck.with_name("unseen-e1__retrieval.npy")
    assert doc_cache.exists(), f"need {doc_cache} (run lab/fusion_eval.py once first)"
    if q_cache.exists():
        q = np.load(q_cache)
        if len(q) == len(qn):
            return np.load(doc_cache), q
    assert ck.exists(), f"need {ck}"
    model = load_ft(DEFAULT_MODEL, ck, max_tokens)
    model.max_seq_length = 512
    return np.load(doc_cache), np.asarray(encode(model, DEFAULT_MODEL, qn, "query", batch), np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None, help="fine-tuning run under data/lab/ft (default: latest)")
    p.add_argument("--k", type=int, default=20, help="F0 tables sent to the reranker")
    p.add_argument("--api-max-tokens", type=int, default=512, help="reranker maxTokens (answer length)")
    p.add_argument("--interval", type=float, default=1.0, help="seconds between API calls")
    p.add_argument("--limit", type=int, default=0, help="only the first N questions (0 = all 117)")
    p.add_argument("--max-labels", type=int, default=60)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()

    runs = sorted(d for d in FT_DIR.iterdir() if d.is_dir()) if FT_DIR.exists() else []
    run_dir = FT_DIR / args.run if args.run else (runs[-1] if runs else None)
    assert run_dir and run_dir.exists(), "no fine-tuning run under data/lab/ft"

    records, sha, tokenize = load_records(), source_sha(), bm25_tokenize()
    ds = docs(records, "norm", args.max_labels)
    keys = [d["key"] for d in ds]
    text_of = {d["key"]: d["embed"] for d in ds}
    bm = BM25Okapi([tokenize(d["bm25"]) for d in ds])

    all_items = read_jsonl(RETRIEVAL)
    qn_all = [query_text(it["question"], "norm") for it in all_items]
    doc_vec, q_vec = query_vectors(run_dir, qn_all, args.max_tokens, args.batch)
    assert len(doc_vec) == len(keys)
    items = all_items[: args.limit] if args.limit else all_items

    def search(i: int) -> list[str]:
        """F0 for question i: fused list, up to 100 tables."""
        dense = top(doc_vec @ q_vec[i], keys)
        sc = np.asarray(bm.get_scores(tokenize(qn_all[i])))
        bml = [keys[j] for j in np.argsort(-sc, kind="stable")[:DEPTH] if sc[j] > 0]
        return rrf(dense, bml)

    f0 = [search(i) for i in range(len(items))]
    tag = hashlib.sha256(json.dumps([PATH, args.k, args.api_max_tokens, sha[:8], run_dir.name, args.max_labels],
                                    ensure_ascii=False).encode("utf-8")).hexdigest()[:8]
    cache_dir = OUT_DIR / "hcx_rerank" / tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = None

    def rerank(i: int) -> dict:
        """One question: the K candidates (ids = 1..K in F0 order) and the API's answer (cached)."""
        it = items[i]
        path = cache_dir / f"{it['qid']}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        nonlocal key
        key = key or api_key()
        cands = f0[i][: args.k]
        body = {"query": it["question"], "maxTokens": args.api_max_tokens,
                "documents": [{"id": str(n), "doc": text_of[k]} for n, k in enumerate(cands, 1)]}
        t0 = time.perf_counter()
        res = call(body, key, args.interval)["result"]
        log = {"qid": it["qid"], "candidates": cands, "seconds": time.perf_counter() - t0,
               "answer": res.get("result", ""), "cited_ids": [str(d.get("id", "")) for d in res.get("citedDocuments") or []],
               "suggested": res.get("suggestedQueries") or [], "usage": res.get("usage", {})}
        path.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
        return log

    logs = []
    for n in range(len(items)):
        lg = rerank(n)
        by_id = {str(j): k for j, k in enumerate(lg["candidates"], 1)}
        lg["cited"] = list(dict.fromkeys(by_id[c] for c in lg["cited_ids"] if c in by_id))
        lg["unknown_ids"] = [c for c in lg["cited_ids"] if c not in by_id]
        logs.append(lg)
        print(f"  [{n + 1}/{len(items)}] {items[n]['qid']}: {len(lg['cited'])} cited")

    # ---- rankings ----
    ans = [i for i, it in enumerate(items) if it.get("answerable", True) and it.get("required")]
    groups = {i: items[i]["required"] for i in ans}
    lists: dict[str, dict[int, list[str]]] = {m: {} for m in ("F0", "H1", "H2")}
    for i in ans:
        cited = logs[i]["cited"]
        in_f0 = [k for k in f0[i] if k in set(cited)]
        lists["F0"][i] = f0[i]
        lists["H1"][i] = cited + [k for k in f0[i] if k not in set(cited)]
        lists["H2"][i] = in_f0 + [k for k in f0[i] if k not in set(cited)]
    ranks = {m: {i: group_rank(groups[i], lists[m][i]) for i in ans} for m in lists}
    cluster = {i: groups[i][0][0] for i in ans}
    rng = np.random.default_rng(0)

    def hit(m: str, i: int, k: int) -> float:
        r = ranks[m][i]
        return float(r is not None and r <= k)

    split = {sp: [i for i in ans if sp == "all" or items[i]["split"] == sp] for sp in ("all", "test", "dev")}
    dev = split["dev"]
    pick = max(("H2", "H1"), key=lambda m: (np.mean([hit(m, i, 1) for i in dev]) if dev else 0,
                                           np.mean([hit(m, i, 5) for i in dev]) if dev else 0))

    def block(ix: list[int], title: str) -> list[str]:
        out = ["", f"## {title} (n={len(ix)})", "",
               "Δ cells vs F0: mean, 95% question-level, 95% table-level, wins/losses.", "",
               f"| ranking | " + " | ".join(METRICS) + f" | in top {args.k} | ΔHit@1 | ΔHit@5 |",
               "| --- | " + " | ".join("---" for _ in METRICS) + " | --- | --- | --- |"]
        for m in lists:
            sm = summary([ranks[m][i] for i in ix])
            rec = np.mean([ranks[m][i] is not None and ranks[m][i] <= args.k for i in ix]) if ix else 0.0
            cells = []
            for k in (1, 5):
                if m == "F0":
                    cells.append("-")
                    continue
                d = np.array([hit(m, i, k) - hit("F0", i, k) for i in ix])
                q_ = boot(d, None, rng)
                t_ = boot(d, [cluster[i] for i in ix], rng)
                cells.append(f"{d.mean():+.3f} [{q_[0]:+.3f}, {q_[1]:+.3f}] [{t_[0]:+.3f}, {t_[1]:+.3f}] "
                             f"{int((d > 0).sum())}/{int((d < 0).sum())}")
            out.append(f"| {m} | " + " | ".join(f"{sm[x]:.3f}" for x in METRICS) + f" | {rec:.3f} | "
                       + " | ".join(cells) + " |")
        return out

    def verdict() -> list[str]:
        out = ["", "## verdict on retrieval_golden answerable (table-level 95% lower bound > 0 → effect)", "",
               f"- dev choice: **{pick}** (dev Hit@1, then Hit@5; tie → H2)"]
        parts, eff = [], False
        for k in (1, 5):
            d = np.array([hit(pick, i, k) - hit("F0", i, k) for i in ans])
            lo, hi = boot(d, [cluster[i] for i in ans], rng)
            parts.append(f"Hit@{k} {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
            eff = eff or lo > 0
        out.append(f"- {pick} vs F0 → " + "; ".join(parts) + (" → **effect**" if eff else " → no clear effect"))
        return out

    def qtype_block() -> list[str]:
        by: dict[str, list[int]] = defaultdict(list)
        for i in ans:
            by[items[i].get("qtype", "?")].append(i)
        out = ["", "## Hit@1 / Hit@5 by qtype (answerable rows)", "",
               "| qtype | n | " + " | ".join(lists) + " | cited / question |",
               "| --- | --- | " + " | ".join("---" for _ in lists) + " | --- |"]
        for q, ix in sorted(by.items(), key=lambda kv: -len(kv[1])):
            out.append(f"| {q} | {len(ix)} | " + " | ".join(
                f"{np.mean([hit(m, i, 1) for i in ix]):.2f} / {np.mean([hit(m, i, 5) for i in ix]):.2f}" for m in lists)
                + f" | {np.mean([len(logs[i]['cited']) for i in ix]):.1f} |")
        return out

    def cited_block() -> list[str]:
        """Where the gold sits relative to what was cited (answerable rows, gold inside the K candidates)."""
        inside = [i for i in ans if ranks["F0"][i] is not None and ranks["F0"][i] <= args.k]
        all_gold = sum(all(set(grp) & set(logs[i]["cited"]) for grp in groups[i]) for i in inside)
        none = sum(not logs[i]["cited"] for i in inside)
        return ["", f"## citations vs gold (answerable rows whose gold is in F0 top {args.k}: n={len(inside)})", "",
                f"- every gold group cited: {all_gold} / {len(inside)}",
                f"- nothing cited: {none} / {len(inside)}",
                f"- cited ids not among the candidates (ignored): {sum(len(lg['unknown_ids']) for lg in logs)}"]

    def usage_block() -> list[str]:
        noans = [i for i, it in enumerate(items) if not it.get("answerable", True)]

        def stats(ix: list[int]) -> str:
            if not ix:
                return "| - | - | - | - | - |"
            c = [len(logs[i]["cited"]) for i in ix]
            return (f"| {len(ix)} | {np.mean(c):.2f} (max {max(c)}) | {np.mean([x == 0 for x in c]):.2f} | "
                    f"{np.mean([logs[i]['usage'].get('totalTokens', 0) for i in ix]):.0f} | "
                    f"{np.mean([logs[i]['seconds'] for i in ix]):.1f} |")
        return ["", "## usage (seconds as measured when each call was first made)", "",
                "| rows | n | cited tables | nothing cited | tokens / question | seconds / question |",
                "| --- | --- | --- | --- | --- | --- |",
                "| answerable " + stats(ans), "| no answer " + stats(noans)]

    def examples() -> list[str]:
        out = ["", "## changed rows (answerable, F0 rank → H1 / H2)", "", "| qid | qtype | question | cited | F0 | H1 | H2 |",
               "| --- | --- | --- | --- | --- | --- | --- |"]
        for i in ans:
            if ranks["F0"][i] != ranks["H1"][i] or ranks["F0"][i] != ranks["H2"][i]:
                out.append(f"| {items[i]['qid']} | {items[i].get('qtype')} | {items[i]['question']} | "
                           f"{len(logs[i]['cited'])} | {ranks['F0'][i]} | {ranks['H1'][i]} | {ranks['H2'][i]} |")
        return out

    rid = time.strftime("%Y%m%d-%H%M%S")
    lines = ["# Plan 34: CLOVA Studio reranker on top of the current search", "",
             f"run {rid}; responses data/lab/hcx_rerank/{tag}; fine-tuning run {run_dir.name}; tables {sha[:8]}; "
             f"K={args.k}; maxTokens {args.api_max_tokens}",
             "F0 = current search; H1 = cited first in API order, then F0; H2 = cited first in F0 order, then F0."]
    lines += verdict()
    for sp, ix in split.items():
        lines += block(ix, f"retrieval_golden {sp} (answerable)")
    lines += qtype_block()
    lines += cited_block()
    lines += usage_block()
    lines += examples()
    report = "\n".join(lines) + "\n"
    print(report)
    summary_path = OUT_DIR / f"hcx_rerank_eval_{rid}_summary.md"
    ranks_path = OUT_DIR / f"hcx_rerank_eval_{rid}_ranks.jsonl"
    summary_path.write_text(report, encoding="utf-8")
    with ranks_path.open("w", encoding="utf-8") as f:
        for i, it in enumerate(items):
            row = {"qid": it["qid"], "split": it["split"], "qtype": it.get("qtype"), "question": it["question"],
                   "answerable": i in groups, "cited": logs[i]["cited"], "suggested": logs[i]["suggested"]}
            if i in groups:
                row |= {m: ranks[m][i] for m in lists} | {f"top1 {m}": (lists[m][i] or [None])[0] for m in lists}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"-> {summary_path}, {ranks_path}")


if __name__ == "__main__":
    main()
