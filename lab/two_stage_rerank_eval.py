"""Plan 42 B: two-stage re-ranking. Input top 50 -> stage 1 cross-encoder re-ranker -> top K -> stage 2 LLM re-ranker.

Input (--input): soft = the current best, F0 (unseen-e1 + BM25, RRF c=60) then plan 37 rev2 soft year+region
rescoring (beta 0.01, gamma 0.005, top 50); f0 = F0 only.
Stage 1 (--stage1, default BAAI/bge-reranker-v2-m3; a fine-tuned cross-encoder folder works the same) scores
(question, table text) for the input top 50:
    S1a  re-ranker score only
    S1b  RRF of the re-ranker order and the input order (1/(60+r_rerank) + 1/(60+r_input))
S1a vs S1b is chosen on retrieval dev (no API). Stage 2 (--stage2, default clova = CLOVA Studio /v1/api-tools/reranker)
gets the top K and returns the tables it cites (no scores); cited tables go first in stage-1 order, then the rest.
Stage 2 rows (API, 117 questions each):
    input -> stage 2, K 20                    stage 1 chosen -> stage 2, K 10 / K 20
    the same two K 20 rows with the question wrapped: "다음 질문의 수치를 찾을 수 있는 통계표: {question}"
Every row is chosen among on retrieval dev (Hit@1, then Hit@5; the input wins ties) and judged on all answerable
retrieval_golden rows against the input (table-level bootstrap). Also: how often stage 2 cites nothing (answerable vs
no-answer rows), where the gold table sits after each stage, seconds per question.
Caches: stage-1 scores in data/lab/two_stage/<stage1 tag>.json, every stage-2 answer in
data/lab/two_stage/<stage2>/<hash of query + candidates>.json (a rerun calls nothing, a stopped run resumes).
The HCX key comes from the repo root .env (or vendor/phase1/.env) and is never printed.

    python lab/two_stage_rerank_eval.py --limit 5     # first 5 questions only, to check the calls
    python lab/two_stage_rerank_eval.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi

from ft_train import FT_DIR
from hcx_rerank_eval import call
from lab_common import (
    DEPTH, METRICS, OUT_DIR, RETRIEVAL, RRF_C, bm25_tokenize, boot, docs, group_rank, load_records, query_text,
    read_jsonl, source_sha, summary, top,
)
from rag_reasoning_eval import api_key
from rerank_eval import load_reranker
from slot_rerank_eval import make_extractor, meets, region_names, table_facts

BETA, GAMMA, N = 0.01, 0.005, 50  # plan 37 rev2 soft rescoring, fixed
SOFT_SLOTS = ("year", "region")
WRAP = "다음 질문의 수치를 찾을 수 있는 통계표: {q}"
CACHE = OUT_DIR / "two_stage"


def stage2_clova(question: str, cands: list[str], text_of: dict[str, str], args, state: dict) -> dict:
    """CLOVA reranker: candidates get ids 1..K in the order given; returns {'cited': [keys], ...} (cached)."""
    body = {"query": question, "maxTokens": args.api_max_tokens,
            "documents": [{"id": str(n), "doc": text_of[k]} for n, k in enumerate(cands, 1)]}
    h = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    path = CACHE / "clova" / f"{h}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    state["key"] = state.get("key") or api_key()
    t0 = time.perf_counter()
    res = call(body, state["key"], args.interval)["result"]
    ids = [str(d.get("id", "")) for d in res.get("citedDocuments") or []]
    by_id = {str(n): k for n, k in enumerate(cands, 1)}
    log = {"query": question, "candidates": cands, "seconds": time.perf_counter() - t0,
           "cited": list(dict.fromkeys(by_id[i] for i in ids if i in by_id)), "cited_ids": ids,
           "answer": res.get("result", ""), "usage": res.get("usage", {})}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    state["calls"] = state.get("calls", 0) + 1
    return log


STAGE2 = {"clova": stage2_clova}  # a hand-made LLM re-ranker plugs in here with the same signature


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None, help="fine-tuning run under data/lab/ft (default: latest)")
    p.add_argument("--input", choices=("soft", "f0"), default="soft")
    p.add_argument("--stage1", default="BAAI/bge-reranker-v2-m3", help="cross-encoder name or local folder")
    p.add_argument("--stage1-max-length", type=int, default=1024)
    p.add_argument("--stage2", choices=sorted(STAGE2), default="clova")
    p.add_argument("--api-max-tokens", type=int, default=512)
    p.add_argument("--interval", type=float, default=1.0, help="seconds between API calls")
    p.add_argument("--limit", type=int, default=0, help="only the first N questions (0 = all 117)")
    p.add_argument("--max-labels", type=int, default=60)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()

    runs = sorted(d for d in FT_DIR.iterdir() if d.is_dir()) if FT_DIR.exists() else []
    run_dir = FT_DIR / args.run if args.run else (runs[-1] if runs else None)
    assert run_dir and run_dir.exists(), "no fine-tuning run under data/lab/ft"
    cache = {part: run_dir / f"unseen-e1__{part}.npy" for part in ("docs", "retrieval")}
    missing = [str(c) for c in cache.values() if not c.exists()]
    assert not missing, f"need {missing} (run lab/fusion_eval.py once first)"

    records, sha, tokenize = load_records(), source_sha(), bm25_tokenize()
    names = region_names(records)
    slots = make_extractor(names)
    facts = {r["info"]["key"]: table_facts(r, names) for r in records}
    ds = docs(records, "norm", args.max_labels)
    keys = [d["key"] for d in ds]
    text_of = {d["key"]: d["embed"] for d in ds}
    bm = BM25Okapi([tokenize(d["bm25"]) for d in ds])
    doc_vec, q_vec = np.load(cache["docs"]), np.load(cache["retrieval"])
    all_items = read_jsonl(RETRIEVAL)
    assert len(doc_vec) == len(keys) and len(q_vec) == len(all_items), "caches do not match tables / golden"
    items = all_items[: args.limit] if args.limit else all_items
    qn = [query_text(it["question"], "norm") for it in items]

    # ---- input ranking ----
    def first(i: int) -> list[str]:
        dense = top(doc_vec @ q_vec[i], keys)
        sc = np.asarray(bm.get_scores(tokenize(qn[i])))
        bml = [keys[j] for j in np.argsort(-sc, kind="stable")[:DEPTH] if sc[j] > 0]
        s: dict[str, float] = defaultdict(float)
        for lst in (dense, bml):
            for r, k in enumerate(lst, 1):
                s[k] += 1.0 / (RRF_C + r)
        scored = sorted(s.items(), key=lambda kv: -kv[1])
        sl = [(n, v) for n, v in slots(items[i]["question"]).items() if n in SOFT_SLOTS]
        if args.input == "f0" or not sl:
            return [k for k, _ in scored]
        head, tail = scored[:N], scored[N:]
        new = [(k, sc - BETA * sum(not meets(facts[k], n, v) for n, v in sl) + GAMMA * sum(meets(facts[k], n, v) for n, v in sl))
               for k, sc in head]
        new.sort(key=lambda kv: -kv[1])
        return [k for k, _ in new] + [k for k, _ in tail]

    base = [first(i) for i in range(len(items))]

    # ---- stage 1 scores for the input top 50, cached ----
    s1tag = hashlib.sha256(json.dumps([args.stage1, args.stage1_max_length, sha[:8], args.max_labels]).encode()).hexdigest()[:8]
    s1path = CACHE / f"stage1__{args.stage1.replace('/', '_').replace(':', '_')[-60:]}__{s1tag}.json"
    s1: dict[str, dict[str, float]] = json.loads(s1path.read_text(encoding="utf-8")) if s1path.exists() else {}
    need = [(i, k) for i in range(len(items)) for k in base[i][:N] if k not in s1.get(qn[i], {})]
    s1_sec = 0.0
    if need:
        model = load_reranker(args.stage1, args.stage1_max_length)
        t0 = time.perf_counter()
        out = model.predict([(qn[i], text_of[k]) for i, k in need], batch_size=args.batch, show_progress_bar=True,
                            convert_to_numpy=True)
        s1_sec = time.perf_counter() - t0
        for (i, k), v in zip(need, np.asarray(out, np.float64).reshape(-1)):
            s1.setdefault(qn[i], {})[k] = float(v)
        CACHE.mkdir(parents=True, exist_ok=True)
        s1path.write_text(json.dumps(s1, ensure_ascii=False), encoding="utf-8")
        del model
        print(f"  stage 1 scored {len(need)} pairs in {s1_sec:.0f}s -> {s1path.name}")
    else:
        print(f"  stage 1 cache {s1path.name}")

    def stage1(i: int, kind: str) -> list[str]:
        head, tail = base[i][:N], base[i][N:]
        sc = np.array([s1[qn[i]][k] for k in head])
        if kind == "S1a":
            order = np.argsort(-sc, kind="stable")
        else:
            rr = np.empty(len(head))
            rr[np.argsort(-sc, kind="stable")] = np.arange(1, len(head) + 1)
            order = np.argsort(-(1 / (RRF_C + rr) + 1 / (RRF_C + np.arange(1, len(head) + 1))), kind="stable")
        return [head[j] for j in order] + tail

    ans = [i for i, it in enumerate(items) if it.get("answerable", True) and it.get("required")]
    noans = [i for i in range(len(items)) if i not in set(ans)]
    groups = {i: items[i]["required"] for i in ans}
    cluster = {i: groups[i][0][0] for i in ans}
    lists: dict[str, dict[int, list[str]]] = {"input": {i: base[i] for i in range(len(items))}}
    for kind in ("S1a", "S1b"):
        lists[kind] = {i: stage1(i, kind) for i in range(len(items))}

    def ranks_of(m: str) -> dict[int, int | None]:
        return {i: group_rank(groups[i], lists[m][i]) for i in ans}

    ranks = {m: ranks_of(m) for m in lists}

    def hit(m: str, i: int, k: int) -> float:
        r = ranks[m][i]
        return float(r is not None and r <= k)

    split = {sp: [i for i in ans if sp == "all" or items[i]["split"] == sp] for sp in ("all", "test", "dev")}
    dev = split["dev"]

    def dev_key(m: str) -> tuple:
        return (np.mean([hit(m, i, 1) for i in dev]) if dev else 0.0, np.mean([hit(m, i, 5) for i in dev]) if dev else 0.0)

    s1pick = max(("S1b", "S1a"), key=dev_key)  # tie -> S1b (keeps more of the input order)

    # ---- stage 2 ----
    rows = [("input", 20, False), (s1pick, 10, False), (s1pick, 20, False), ("input", 20, True), (s1pick, 20, True)]
    state: dict = {}
    seconds: dict[str, list[float]] = defaultdict(list)
    cited_n: dict[str, dict[int, int]] = defaultdict(dict)
    fn = STAGE2[args.stage2]
    for src, k, wrap in rows:
        m = f"{src} → {args.stage2} K{k}" + (" wrapped" if wrap else "")
        lists[m] = {}
        for i, it in enumerate(items):
            cands = lists[src][i][:k]
            q = WRAP.format(q=it["question"]) if wrap else it["question"]
            log = fn(q, cands, text_of, args, state)
            cited = [c for c in log["cited"] if c in set(cands)]
            rest = [c for c in lists[src][i] if c not in set(cited)]
            lists[m][i] = [c for c in lists[src][i] if c in set(cited)] + rest  # cited first, stage-1 order
            seconds[m].append(log.get("seconds", 0.0))
            cited_n[m][i] = len(cited)
            print(f"  {m} [{i + 1}/{len(items)}] {it['qid']}: {len(cited)} cited")
        ranks[m] = ranks_of(m)

    methods = list(lists)
    pick = max(methods, key=lambda m: (*dev_key(m), m == "input"))
    rng = np.random.default_rng(0)

    def delta(m: str, ix: list[int], k: int) -> str:
        d = np.array([hit(m, i, k) - hit("input", i, k) for i in ix])
        q_ = boot(d, None, rng)
        t_ = boot(d, [cluster[i] for i in ix], rng)
        return (f"{d.mean():+.3f} [{q_[0]:+.3f}, {q_[1]:+.3f}] [{t_[0]:+.3f}, {t_[1]:+.3f}] "
                f"{int((d > 0).sum())}/{int((d < 0).sum())}")

    def block(ix: list[int], title: str) -> list[str]:
        out = ["", f"## {title} (n={len(ix)})", "", "Δ cells vs the input: mean, 95% question-level, 95% table-level, wins/losses.", "",
               "| ranking | " + " | ".join(METRICS) + " | ΔHit@1 | ΔHit@5 | cites nothing (answerable) |",
               "| --- | " + " | ".join("---" for _ in METRICS) + " | --- | --- | --- |"]
        for m in methods:
            sm = summary([ranks[m][i] for i in ix])
            cells = ["-", "-"] if m == "input" else [delta(m, ix, 1), delta(m, ix, 5)]
            zero = f"{np.mean([cited_n[m][i] == 0 for i in ix]):.0%}" if m in cited_n and ix else "-"
            out.append(f"| {m}{' (dev choice)' if m == pick else ''} | " + " | ".join(f"{sm[x]:.3f}" for x in METRICS)
                       + " | " + " | ".join(cells) + f" | {zero} |")
        return out

    def verdict() -> list[str]:
        ix = split["all"]
        out = ["", "## verdict on retrieval_golden answerable (table-level 95% lower bound > 0 → effect)", "",
               f"- stage 1 choice on dev: {s1pick}; overall dev choice: **{pick}** (Hit@1, then Hit@5; the input wins ties)",
               "- retrieval_golden test has been looked at many times → exploratory; confirm on the new check set (plan 40)"]
        if pick == "input":
            return out + ["- nothing beat the input on dev → **no change**"]
        parts, eff = [], False
        for k in (1, 5):
            d = np.array([hit(pick, i, k) - hit("input", i, k) for i in ix])
            lo, hi = boot(d, [cluster[i] for i in ix], rng)
            parts.append(f"Hit@{k} {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
            eff = eff or lo > 0
        return out + [f"- {pick} vs input → " + "; ".join(parts) + (" → **effect**" if eff else " → no clear effect")]

    def stage2_block() -> list[str]:
        out = ["", "## stage 2: citing and cost", "",
               "| ranking | cites nothing: answerable | cites nothing: no-answer | mean cited | seconds / question |",
               "| --- | --- | --- | --- | --- |"]
        for m in cited_n:
            a = np.mean([cited_n[m][i] == 0 for i in ans]) if ans else 0
            b = np.mean([cited_n[m][i] == 0 for i in noans]) if noans else 0
            mc = np.mean(list(cited_n[m].values()))
            out.append(f"| {m} | {a:.0%} | {b:.0%} | {mc:.2f} | {np.mean(seconds[m]):.1f} |")
        return out

    def moves() -> list[str]:
        if pick == "input":
            return []
        src = pick.split(" →")[0]
        out = ["", f"## gold position: input → {src} → {pick} (rows that changed)", "",
               "| qid | split | qtype | question | input | stage 1 | final |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for i in ans:
            a, b, c = ranks["input"][i], ranks[src][i], ranks[pick][i]
            if len({a, b, c}) > 1:
                out.append(f"| {items[i]['qid']} | {items[i]['split']} | {items[i].get('qtype', '?')} | {items[i]['question']} | {a} | {b} | {c} |")
        return out

    rid = time.strftime("%Y%m%d-%H%M%S")
    lines = ["# Plan 42 B: two-stage re-ranking", "",
             f"run {rid}; input {args.input}; stage 1 {args.stage1} (max length {args.stage1_max_length}); stage 2 {args.stage2} "
             f"(maxTokens {args.api_max_tokens}); fine-tuning run {run_dir.name}; tables {sha[:8]}; questions {len(items)}; "
             f"new API calls {state.get('calls', 0)}; stage 1 seconds {s1_sec:.0f}",
             f"cited tables go first in the order of the list they came from; wrapped query = \"{WRAP}\""]
    lines += verdict()
    for sp in ("all", "test", "dev"):
        lines += block(split[sp], f"retrieval_golden {sp} (answerable)")
    lines += stage2_block()
    lines += moves()
    report = "\n".join(lines) + "\n"
    print(report)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"two_stage_rerank_eval_{rid}_summary.md"
    out.write_text(report, encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
