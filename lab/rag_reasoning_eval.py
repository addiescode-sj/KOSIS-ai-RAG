"""Plan 35: does CLOVA Studio RAG Reasoning alone improve which table comes first?

The search itself stays the current one (F0: fine-tuned unseen-e1 dense top-50 + V0 BM25 top-50, RRF c=60). RAG
Reasoning gets that search as a function (search_kosis_tables(query) -> top-10 table texts), may call it several
times with its own queries (split / rewritten), and finally answers citing tables as <doc-N>. Rankings compared:
    F0  the original question, searched once (now)
    Q   RRF over the result lists (top-100 each) of the model's own queries
    Q+  RRF over the original question's list and the model's queries' lists
    C   tables the answer cites, in citation order, first; then Q+
No setting is tuned, so there is no dev choice: all four are judged on retrieval_golden answerable rows against F0
(table-level bootstrap, as in plans 32/33). Also reported: by split and qtype, searches / tokens / seconds per
question, citations, and on the 28 no-answer rows how often nothing is cited.

Every conversation is cached in data/lab/rag_reasoning/<prompt hash>/<qid>.json, so a rerun calls nothing and a
stopped run resumes. The key is HCX_API_KEY from the repo root .env (or vendor/phase1/.env); it is never printed.

    python lab/rag_reasoning_eval.py
    python lab/rag_reasoning_eval.py --limit 5        # first 5 questions only, to check the calls
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from lab_common import (
    DEFAULT_MODEL, DEPTH, KS, METRICS, OUT_DIR, RETRIEVAL, ROOT, bm25_tokenize, boot, docs, encode, group_rank,
    load_records, query_text, read_jsonl, rrf, source_sha, summary, top,
)
from ft_train import FT_DIR, load_ft

HOST = "https://clovastudio.stream.ntruss.com"
PATH = "/v1/api-tools/rag-reasoning"
SHOWN = 10  # tables handed back to the model per search
SYSTEM = (
    "너는 KOSIS 통계표 검색 도우미다. 사용자 질문에 답하는 데 필요한 통계표를 search_kosis_tables 함수로 찾는다. "
    "질문에 조건이 여러 개 있거나 통계표가 둘 이상 필요하면 나눠서 검색한다. 검색 결과에 맞는 표가 없으면 "
    "공식 통계 용어로 바꿔 다시 검색한다. 검색 결과의 문서는 통계표 설명이며 수치는 들어 있지 않으므로 수치를 "
    "지어내지 말고, 어떤 통계표에서 답을 찾을 수 있는지 답하며 근거로 쓴 통계표를 인용한다."
)
TOOL = {"type": "function", "function": {
    "name": "search_kosis_tables",
    "description": "KOSIS 통계표 2,364개 중에서 검색어와 관련된 통계표 상위 10개를 찾는다. 각 결과는 통계표 이름, "
                   "조사, 수록기간, 항목, 분류(지역·성별·연령 등)를 담은 설명이다.",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string", "description": "통계표를 찾을 검색어. 하나의 정보만 담는다."}},
        "required": ["query"]}}}
CITE = re.compile(r"<(?:doc-)?([^>/\s]+)>")  # the API writes <N>..</N> (docs show <doc-N>); only known ids count


def api_key() -> str:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
        load_dotenv(ROOT / "vendor" / "phase1" / ".env", override=False)
    except ImportError:
        pass
    key = os.environ.get("HCX_API_KEY", "")
    assert key, "HCX_API_KEY is not set (repo root .env)"
    return key


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
        raise RuntimeError(f"RAG Reasoning call failed ({code}): {str(res)[:300]}")
    raise RuntimeError("unreachable")


def make_encoder(run: str | None, max_tokens: int, batch: int):
    """unseen-e1 on the GPU for new queries + its cached document vectors."""
    runs = sorted(d for d in FT_DIR.iterdir() if d.is_dir()) if FT_DIR.exists() else []
    run_dir = FT_DIR / run if run else (runs[-1] if runs else None)
    assert run_dir and run_dir.exists(), "no fine-tuning run under data/lab/ft"
    ck = run_dir / "unseen-e1.pt"
    doc_cache = ck.with_name("unseen-e1__docs.npy")
    assert ck.exists() and doc_cache.exists(), f"need {ck} and {doc_cache} (run lab/fusion_eval.py once first)"
    model = load_ft(DEFAULT_MODEL, ck, max_tokens)
    model.max_seq_length = 512

    def enc(texts: list[str]) -> np.ndarray:
        return np.asarray(encode(model, DEFAULT_MODEL, texts, "query", batch), np.float32)

    return np.load(doc_cache), enc, run_dir.name


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None, help="fine-tuning run under data/lab/ft (default: latest)")
    p.add_argument("--max-searches", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--interval", type=float, default=1.0, help="seconds between API calls")
    p.add_argument("--limit", type=int, default=0, help="only the first N questions (0 = all 117)")
    p.add_argument("--max-labels", type=int, default=60)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()

    records, sha, tokenize = load_records(), source_sha(), bm25_tokenize()
    ds = docs(records, "norm", args.max_labels)
    keys = [d["key"] for d in ds]
    text_of = {d["key"]: d["embed"] for d in ds}
    bm = BM25Okapi([tokenize(d["bm25"]) for d in ds])
    doc_vec, enc, run_name = make_encoder(args.run, args.max_tokens, args.batch)
    assert len(doc_vec) == len(keys)

    memo: dict[str, list[str]] = {}

    def search(q: str) -> list[str]:
        """The current search (F0) for any query text: fused list, up to 100 tables."""
        qn = query_text(q, "norm")
        if qn not in memo:
            dense = top(doc_vec @ enc([qn])[0], keys)
            sc = np.asarray(bm.get_scores(tokenize(qn)))
            bml = [keys[j] for j in np.argsort(-sc, kind="stable")[:DEPTH] if sc[j] > 0]
            memo[qn] = rrf(dense, bml)
        return memo[qn]

    items = read_jsonl(RETRIEVAL)
    if args.limit:
        items = items[: args.limit]
    tag = hashlib.sha256(json.dumps([SYSTEM, TOOL, args.seed, args.max_searches, SHOWN, sha[:8], run_name],
                                    ensure_ascii=False).encode("utf-8")).hexdigest()[:8]
    cache_dir = OUT_DIR / "rag_reasoning" / tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = None

    def converse(it: dict) -> dict:
        """One question: the full conversation, every search and the final answer (cached)."""
        path = cache_dir / f"{it['qid']}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        nonlocal key
        key = key or api_key()
        ids: dict[str, str] = {}  # table key -> short id shown to the model
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": it["question"]}]
        log = {"qid": it["qid"], "queries": [], "calls": 0, "tokens": 0, "seconds": 0.0, "answer": "", "thinking": []}
        for step in range(args.max_searches + 1):
            body = {"messages": messages, "tools": [TOOL], "toolChoice": "auto" if step < args.max_searches else "none",
                    "seed": args.seed, "maxTokens": 1024}
            t0 = time.perf_counter()
            res = call(body, key, args.interval)
            log["calls"] += 1
            log["seconds"] += time.perf_counter() - t0
            msg = res["result"]["message"]
            log["tokens"] += int(res["result"].get("usage", {}).get("totalTokens", 0))
            if msg.get("thinkingContent"):
                log["thinking"].append(msg["thinkingContent"])
            calls = msg.get("toolCalls") or []
            if not calls:
                log["answer"] = msg.get("content") or ""
                break
            messages.append({"role": "assistant", "content": msg.get("content") or "", "toolCalls": calls})
            for tc in calls:
                a = tc.get("function", {}).get("arguments", {})
                a = json.loads(a) if isinstance(a, str) else (a or {})
                q = str(a.get("query", "")).strip() or it["question"]
                found = search(q)[:SHOWN]
                log["queries"].append({"query": q, "top": found})
                result = []
                for k in found:
                    ids.setdefault(k, str(len(ids) + 1))
                    result.append({"id": ids[k], "doc": text_of[k]})
                messages.append({"role": "tool", "toolCallId": tc.get("id", ""),
                                 "content": json.dumps({"search_result": result}, ensure_ascii=False)})
        back = {v: k for k, v in ids.items()}
        log["cited"] = list(dict.fromkeys(back[c] for c in CITE.findall(log["answer"]) if c in back))
        log["ids"] = ids
        path.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
        return log

    logs = []
    for n, it in enumerate(items, 1):
        lg = converse(it)
        back = {v: k for k, v in lg["ids"].items()}  # re-read citations, so cached runs follow the current pattern
        lg["cited"] = list(dict.fromkeys(back[c] for c in CITE.findall(lg["answer"]) if c in back))
        logs.append(lg)
        print(f"  [{n}/{len(items)}] {it['qid']}: {len(lg['queries'])} searches, {len(lg['cited'])} cited")

    # ---- rankings ----
    ans = [i for i, it in enumerate(items) if it.get("answerable", True) and it.get("required")]
    groups = {i: items[i]["required"] for i in ans}
    lists: dict[str, dict[int, list[str]]] = {m: {} for m in ("F0", "Q", "Q+", "C")}
    for i in ans:
        f0 = search(items[i]["question"])
        own = [search(q["query"]) for q in logs[i]["queries"]]
        lists["F0"][i] = f0
        lists["Q"][i] = rrf(*own) if own else f0
        lists["Q+"][i] = rrf(f0, *own)
        cited = logs[i]["cited"]
        lists["C"][i] = cited + [k for k in lists["Q+"][i] if k not in cited]
    ranks = {m: {i: group_rank(groups[i], lists[m][i]) for i in ans} for m in lists}
    cluster = {i: groups[i][0][0] for i in ans}
    rng = np.random.default_rng(0)

    def hit(m: str, i: int, k: int) -> float:
        r = ranks[m][i]
        return float(r is not None and r <= k)

    def block(ix: list[int], title: str) -> list[str]:
        out = ["", f"## {title} (n={len(ix)})", "",
               "Δ cells vs F0: mean, 95% question-level, 95% table-level, wins/losses.", "",
               "| ranking | " + " | ".join(METRICS) + " | Recall@100 | ΔHit@1 | ΔHit@5 |",
               "| --- | " + " | ".join("---" for _ in METRICS) + " | --- | --- | --- |"]
        for m in lists:
            sm = summary([ranks[m][i] for i in ix])
            rec = np.mean([ranks[m][i] is not None for i in ix]) if ix else 0.0
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
        out = ["", "## verdict on retrieval_golden answerable (table-level 95% lower bound > 0 → effect, exploratory)", ""]
        for m in ("Q", "Q+", "C"):
            parts, eff = [], False
            for k in (1, 5):
                d = np.array([hit(m, i, k) - hit("F0", i, k) for i in ans])
                lo, hi = boot(d, [cluster[i] for i in ans], rng)
                parts.append(f"Hit@{k} {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
                eff = eff or lo > 0
            out.append(f"- {m} vs F0 → " + "; ".join(parts) + (" → **effect**" if eff else " → no clear effect"))
        return out

    def qtype_block() -> list[str]:
        by: dict[str, list[int]] = defaultdict(list)
        for i in ans:
            by[items[i].get("qtype", "?")].append(i)
        out = ["", "## Hit@1 / Hit@5 by qtype (answerable rows)", "",
               "| qtype | n | " + " | ".join(lists) + " | searches / question |",
               "| --- | --- | " + " | ".join("---" for _ in lists) + " | --- |"]
        for q, ix in sorted(by.items(), key=lambda kv: -len(kv[1])):
            out.append(f"| {q} | {len(ix)} | " + " | ".join(
                f"{np.mean([hit(m, i, 1) for i in ix]):.2f} / {np.mean([hit(m, i, 5) for i in ix]):.2f}" for m in lists)
                + f" | {np.mean([len(logs[i]['queries']) for i in ix]):.1f} |")
        return out

    def usage_block() -> list[str]:
        noans = [i for i, it in enumerate(items) if not it.get("answerable", True)]
        def stats(ix: list[int]) -> str:
            if not ix:
                return "| - | - | - | - | - | - |"
            s = [len(logs[i]["queries"]) for i in ix]
            c = [len(logs[i]["cited"]) for i in ix]
            return (f"| {len(ix)} | {np.mean(s):.2f} (max {max(s)}) | {np.mean(c):.2f} | "
                    f"{np.mean([x == 0 for x in c]):.2f} | {np.mean([logs[i]['tokens'] for i in ix]):.0f} | "
                    f"{np.mean([logs[i]['seconds'] for i in ix]):.1f} |")
        return ["", "## usage (seconds as measured when each conversation was first run)", "",
                "| rows | n | searches / question | cited tables | nothing cited | tokens / question | seconds / question |",
                "| --- | --- | --- | --- | --- | --- | --- |",
                "| answerable " + stats(ans), "| no answer " + stats(noans)]

    def examples() -> list[str]:
        out = ["", "## query examples (first 20 answerable rows)", "", "| qid | question | model's queries | cited | F0 → C rank |",
               "| --- | --- | --- | --- | --- |"]
        for i in ans[:20]:
            qs = " / ".join(q["query"] for q in logs[i]["queries"]) or "(no search)"
            out.append(f"| {items[i]['qid']} | {items[i]['question']} | {qs} | {len(logs[i]['cited'])} | "
                       f"{ranks['F0'][i]} → {ranks['C'][i]} |")
        return out

    split = {sp: [i for i in ans if sp == "all" or items[i]["split"] == sp] for sp in ("all", "test", "dev")}
    rid = time.strftime("%Y%m%d-%H%M%S")
    lines = ["# Plan 35: CLOVA Studio RAG Reasoning on top of the current search", "",
             f"run {rid}; conversations data/lab/rag_reasoning/{tag}; fine-tuning run {run_name}; tables {sha[:8]}; "
             f"seed {args.seed}; up to {args.max_searches} searches, {SHOWN} tables shown per search",
             "F0 = original question once; Q = model's queries only; Q+ = original + model's queries (RRF); "
             "C = cited tables first, then Q+. No tuning, so no dev choice; test was looked at before (exploratory)."]
    lines += verdict()
    for sp, ix in split.items():
        lines += block(ix, f"retrieval_golden {sp} (answerable)")
    lines += qtype_block()
    lines += usage_block()
    lines += examples()
    report = "\n".join(lines) + "\n"
    print(report)
    summary_path = OUT_DIR / f"rag_reasoning_eval_{rid}_summary.md"
    ranks_path = OUT_DIR / f"rag_reasoning_eval_{rid}_ranks.jsonl"
    summary_path.write_text(report, encoding="utf-8")
    with ranks_path.open("w", encoding="utf-8") as f:
        for i, it in enumerate(items):
            row = {"qid": it["qid"], "split": it["split"], "qtype": it.get("qtype"), "question": it["question"],
                   "answerable": i in groups, "queries": [q["query"] for q in logs[i]["queries"]],
                   "cited": logs[i]["cited"]}
            if i in groups:
                row |= {m: ranks[m][i] for m in lists} | {f"top1 {m}": (lists[m][i] or [None])[0] for m in lists}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"-> {summary_path}, {ranks_path}")


if __name__ == "__main__":
    main()
