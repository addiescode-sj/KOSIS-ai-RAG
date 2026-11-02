"""Compare embedding models on dense retrieval alone (the user's request, 10/31): which model finds the gold table.

Models (same table text n2-L60 and the same questions for all; only the embedding model changes):
    bge-m3       BAAI/bge-m3 as downloaded (V0)
    unseen-e1    bge-m3 fine-tuned in plan 29 without the golden tables (the fair fine-tuned model)
    seen-e2      bge-m3 fine-tuned on all tables (reference only: it saw the golden tables' questions)
    kure         nlpai-lab/KURE-v1
    clova-v1     CLOVA Studio embedding v1, clir-emb-dolphin (API, input up to 500 tokens)
    clova-v2     CLOVA Studio embedding v2 (API, input up to 8,192 tokens)
Main numbers are dense top-50 only. Hybrid (dense top-50 + the unchanged V0 BM25 top-50, RRF c=60) is shown beside
them for reference. Reported on retrieval_golden all / test / dev (answerable) and routing_v0 test / dev, with
question- and table-level bootstrap against bge-m3 dense, Hit@5 by qtype, and the API call log.

CLOVA vectors are cached per text in data/lab/emb/clova-<model>__*.jsonl, so a stopped run resumes and a finished one
calls nothing. The key is HCX_API_KEY from the repo root .env (or vendor/phase1/.env); it is never printed.
A v1 text over its limit is cut to 70% and retried (up to 4 times); the number of cut texts is reported.

    python lab/embed_compare.py
    python lab/embed_compare.py --models bge-m3,kure,clova-v2 --workers 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from lab_common import (
    CACHE, DEFAULT_MODEL, DEPTH, METRICS, OUT_DIR, RETRIEVAL, ROOT, ROUTING, bm25_tokenize, boot, doc_vectors, docs,
    encode, group_rank, load_records, query_text, question_vectors, read_jsonl, rrf, source_sha, summary, top,
)
from ft_train import FT_DIR, load_ft

ALL = ("bge-m3", "unseen-e1", "seen-e2", "kure", "clova-v1", "clova-v2")
KURE = "nlpai-lab/KURE-v1"
CLOVA = {"clova-v1": "/v1/api-tools/embedding/clir-emb-dolphin", "clova-v2": "/v1/api-tools/embedding/v2"}
CLOVA_HOST = "https://clovastudio.stream.ntruss.com"


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


def clova_request(path: str, text: str, key: str) -> dict:
    """One call; returns the parsed JSON body (HTTP errors come back as {'http': code, 'body': ...})."""
    req = urllib.request.Request(CLOVA_HOST + path, data=json.dumps({"text": text}).encode("utf-8"), method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                          "X-NCP-CLOVASTUDIO-REQUEST-ID": uuid.uuid4().hex})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:300]
        return {"http": e.code, "body": body}


class Clova:
    def __init__(self, name: str, workers: int):
        self.name, self.path, self.workers = name, CLOVA[name], workers
        self.key = api_key()
        self.cache_path = CACHE / f"{name}__{CLOVA[name].rsplit('/', 1)[-1]}.jsonl"
        self.cache: dict[str, list[float]] = {}
        if self.cache_path.exists():
            with self.cache_path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        self.cache[row["h"]] = row["v"]
        self.lock = threading.Lock()
        self.log = {"calls": 0, "cut": 0, "retries": 0, "failed": 0, "tokens": 0, "seconds": 0.0}

    def one(self, text: str) -> list[float] | None:
        cur, waits = text, [2, 5, 15, 30, 60]
        for attempt in range(12):
            t0 = time.perf_counter()
            res = clova_request(self.path, cur, self.key)
            with self.lock:
                self.log["calls"] += 1
                self.log["seconds"] += time.perf_counter() - t0
            code = str(res.get("status", {}).get("code", res.get("http", "")))
            if code == "20000":
                with self.lock:
                    self.log["tokens"] += int(res["result"].get("inputTokens", 0))
                    self.log["cut"] += cur != text
                return res["result"]["embedding"]
            if code in ("429", "42900", "42901") or code.startswith("5"):
                with self.lock:
                    self.log["retries"] += 1
                time.sleep(waits[min(attempt, len(waits) - 1)])
                continue
            if res.get("http") == 400 or code.startswith("4"):  # most likely too long for this model
                if len(cur) > 50 and attempt < 4:
                    cur = cur[: int(len(cur) * 0.7)]
                    continue
            print(f"  {self.name}: failed ({code}) {str(res)[:200]}")
            break
        with self.lock:
            self.log["failed"] += 1
        return None

    def embed(self, texts: list[str]) -> np.ndarray:
        hs = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
        todo = list(dict.fromkeys(h for h in hs if h not in self.cache))
        by_h = dict(zip(hs, texts))
        if todo:
            print(f"  {self.name}: {len(todo)} texts to embed ({len(texts) - len(todo)} cached)")
            CACHE.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as f, ThreadPoolExecutor(self.workers) as ex:
                for i, (h, v) in enumerate(zip(todo, ex.map(lambda h: self.one(by_h[h]), todo)), 1):
                    if v is not None:
                        self.cache[h] = v
                        f.write(json.dumps({"h": h, "v": [round(x, 6) for x in v]}) + "\n")
                    if i % 200 == 0:
                        f.flush()
                        print(f"    {i}/{len(todo)}")
        missing = [h for h in hs if h not in self.cache]
        assert not missing, f"{self.name}: {len(missing)} texts have no vector (see the failures above); rerun to retry"
        m = np.asarray([self.cache[h] for h in hs], np.float32)
        return m / np.linalg.norm(m, axis=1, keepdims=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=",".join(ALL), help=f"comma separated, from {ALL}")
    p.add_argument("--run", default=None, help="fine-tuning run under data/lab/ft (default: latest)")
    p.add_argument("--workers", type=int, default=4, help="parallel CLOVA API calls")
    p.add_argument("--max-labels", type=int, default=60)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()
    names = [m.strip() for m in args.models.split(",") if m.strip()]
    bad = [m for m in names if m not in ALL]
    assert not bad, f"unknown models {bad}; choose from {ALL}"
    if "bge-m3" not in names:
        names.insert(0, "bge-m3")  # the baseline every delta is taken against

    records, sha, tokenize = load_records(), source_sha(), bm25_tokenize()
    ds = docs(records, "norm", args.max_labels)
    keys = [d["key"] for d in ds]
    doc_texts = [d["embed"] for d in ds]
    bm = BM25Okapi([tokenize(d["bm25"]) for d in ds])

    rg = read_jsonl(ROUTING)
    rt_all = read_jsonl(RETRIEVAL)
    keep = [i for i, it in enumerate(rt_all) if it.get("answerable", True) and it.get("required")]
    rt = [rt_all[i] for i in keep]
    sets = {
        "routing": {"items": rg, "ids": [it["id"] for it in rg], "split": [it["split"] for it in rg],
                    "groups": [[[Path(it["table"]).stem, *it.get("alt_tables", [])]] for it in rg]},
        "retrieval": {"items": rt, "ids": [it["qid"] for it in rt], "split": [it["split"] for it in rt],
                      "groups": [it["required"] for it in rt], "qtype": [it.get("qtype", "?") for it in rt]},
    }
    qn_r = [query_text(it["question"], "norm") for it in rg]
    qn_t_all = [query_text(it["question"], "norm") for it in rt_all]
    for s, qn in ((sets["routing"], qn_r), (sets["retrieval"], [qn_t_all[i] for i in keep])):
        s["cluster"] = [g[0][0] for g in s["groups"]]
        s["bm25"] = [[keys[j] for j in np.argsort(-sc, kind="stable")[:DEPTH] if sc[j] > 0]
                     for sc in (np.asarray(bm.get_scores(tokenize(q))) for q in qn)]

    # ---- vectors per model ----
    vec: dict[str, dict] = {}
    api_log: dict[str, dict] = {}
    run_dir = None
    for name in names:
        state: dict = {}
        t0 = time.perf_counter()
        if name in ("bge-m3", "kure"):
            mid = DEFAULT_MODEL if name == "bge-m3" else KURE
            v = {"docs": doc_vectors(mid, "norm", args.max_labels, args.max_tokens, ds, sha, state, args.batch),
                 "routing": question_vectors(mid, qn_r, ROUTING.stem, state, args.batch),
                 "retrieval": question_vectors(mid, qn_t_all, RETRIEVAL.stem, state, args.batch)}
        elif name in ("unseen-e1", "seen-e2"):
            if run_dir is None:
                runs = sorted(d for d in FT_DIR.iterdir() if d.is_dir()) if FT_DIR.exists() else []
                run_dir = FT_DIR / args.run if args.run else (runs[-1] if runs else None)
                assert run_dir and run_dir.exists(), "no fine-tuning run under data/lab/ft"
            ck = run_dir / f"{name}.pt"
            assert ck.exists(), f"checkpoint not found: {ck}"
            cache = {part: ck.with_name(f"{ck.stem}__{part}.npy") for part in ("docs", "routing", "retrieval")}
            if all(c.exists() for c in cache.values()):
                print(f"  cache {ck.stem}")
                v = {part: np.load(c) for part, c in cache.items()}
            else:
                model = load_ft(DEFAULT_MODEL, ck, args.max_tokens)
                v = {"docs": np.asarray(encode(model, DEFAULT_MODEL, doc_texts, "passage", args.batch), np.float32)}
                model.max_seq_length = 512
                v["routing"] = np.asarray(encode(model, DEFAULT_MODEL, qn_r, "query", args.batch), np.float32)
                v["retrieval"] = np.asarray(encode(model, DEFAULT_MODEL, qn_t_all, "query", args.batch), np.float32)
                for part, c in cache.items():
                    np.save(c, v[part])
                del model
        else:
            cl = Clova(name, args.workers)
            v = {"docs": cl.embed(doc_texts), "routing": cl.embed(qn_r), "retrieval": cl.embed(qn_t_all)}
            api_log[name] = cl.log
        state.clear()
        assert len(v["docs"]) == len(keys), f"{name}: {len(v['docs'])} document vectors for {len(keys)} tables"
        v["retrieval"] = v["retrieval"][keep] if len(v["retrieval"]) == len(rt_all) else v["retrieval"]
        vec[name] = v
        print(f"  {name}: vectors ready in {time.perf_counter() - t0:.0f}s")

    ranks: dict = {}  # (golden, model, mode) -> group ranks
    tops: dict = {}
    for gname, s in sets.items():
        for name in names:
            sc = vec[name]["docs"] @ vec[name][gname].T
            dense = [top(sc[:, i], keys) for i in range(sc.shape[1])]
            hyb = [rrf(d, b) for d, b in zip(dense, s["bm25"])]
            ranks[gname, name, "dense"] = [group_rank(g, t) for g, t in zip(s["groups"], dense)]
            ranks[gname, name, "hybrid"] = [group_rank(g, t) for g, t in zip(s["groups"], hyb)]
            tops[gname, name] = [d[0] for d in dense]

    rng = np.random.default_rng(0)

    def idx(gname: str, split: str) -> list[int]:
        return [i for i, x in enumerate(sets[gname]["split"]) if split == "all" or x == split]

    def hit(gname: str, name: str, mode: str, i: int, k: int) -> float:
        r = ranks[gname, name, mode][i]
        return float(r is not None and r <= k)

    def delta(gname: str, ix: list[int], name: str, k: int) -> str:
        if name == "bge-m3":
            return "-"
        d = np.array([hit(gname, name, "dense", i, k) - hit(gname, "bge-m3", "dense", i, k) for i in ix])
        lo, hi = boot(d, [sets[gname]["cluster"][i] for i in ix], rng)
        return f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] {int((d > 0).sum())}/{int((d < 0).sum())}"

    def block(gname: str, ix: list[int], title: str) -> list[str]:
        out = ["", f"## {title} (n={len(ix)})", "",
               "Dense columns are the main result. Δ = vs bge-m3 dense: mean, 95% table-level, wins/losses.", "",
               "| model | " + " | ".join(f"dense {m}" for m in METRICS) + " | dense Recall@50 | ΔHit@1 | ΔHit@5 "
               "| hybrid Hit@1 | hybrid Hit@5 |",
               "| --- | " + " | ".join("---" for _ in METRICS) + " | --- | --- | --- | --- | --- |"]
        for name in names:
            sm = summary([ranks[gname, name, "dense"][i] for i in ix])
            hy = summary([ranks[gname, name, "hybrid"][i] for i in ix])
            rec = np.mean([ranks[gname, name, "dense"][i] is not None for i in ix])
            out.append(f"| {name} | " + " | ".join(f"{sm[m]:.3f}" for m in METRICS) + f" | {rec:.3f} | "
                       f"{delta(gname, ix, name, 1)} | {delta(gname, ix, name, 5)} | {hy['Hit@1']:.3f} | "
                       f"{hy['Hit@5']:.3f} |")
        return out

    def qtype_block() -> list[str]:
        s = sets["retrieval"]
        out = ["", "## retrieval_golden dense Hit@1 / Hit@5 by qtype (all answerable rows)", "",
               "| qtype | n | " + " | ".join(names) + " |", "| --- | --- | " + " | ".join("---" for _ in names) + " |"]
        by: dict[str, list[int]] = defaultdict(list)
        for i, q in enumerate(s["qtype"]):
            by[q].append(i)
        for q, ix in sorted(by.items(), key=lambda kv: -len(kv[1])):
            out.append(f"| {q} | {len(ix)} | " + " | ".join(
                f"{np.mean([hit('retrieval', n, 'dense', i, 1) for i in ix]):.2f} / "
                f"{np.mean([hit('retrieval', n, 'dense', i, 5) for i in ix]):.2f}" for n in names) + " |")
        return out

    def api_block() -> list[str]:
        if not api_log:
            return []
        out = ["", "## CLOVA API calls this run (0 calls = all vectors from cache)", "",
               "| model | calls | retries (429/5xx) | texts cut to fit | failed | input tokens | seconds in calls |",
               "| --- | --- | --- | --- | --- | --- | --- |"]
        for n, lg in api_log.items():
            out.append(f"| {n} | {lg['calls']} | {lg['retries']} | {lg['cut']} | {lg['failed']} | {lg['tokens']} | "
                       f"{lg['seconds']:.0f} |")
        return out

    rid = time.strftime("%Y%m%d-%H%M%S")
    lines = [f"# Embedding model comparison, dense retrieval only", "",
             f"run {rid}; tables {sha[:8]}; text n2-L{args.max_labels}; models: {', '.join(names)}"
             + (f"; fine-tuning run {run_dir.name}" if run_dir else ""),
             "unseen-e1 is the fair fine-tuned model; seen-e2 saw the golden tables in training (reference only). "
             "retrieval_golden test was looked at before (exploratory)."]
    for sp in ("all", "test", "dev"):
        lines += block("retrieval", idx("retrieval", sp), f"retrieval_golden {sp} (answerable)")
    for sp in ("test", "dev"):
        lines += block("routing", idx("routing", sp), f"routing_v0 {sp} (same workflow as the training questions)")
    lines += qtype_block()
    lines += api_block()
    report = "\n".join(lines) + "\n"
    print(report)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / f"embed_compare_{rid}_summary.md"
    ranks_path = OUT_DIR / f"embed_compare_{rid}_ranks.jsonl"
    summary_path.write_text(report, encoding="utf-8")
    with ranks_path.open("w", encoding="utf-8") as f:
        for gname, s in sets.items():
            for i, qid in enumerate(s["ids"]):
                row = {"golden": gname, "id": qid, "split": s["split"][i], "question": s["items"][i]["question"]}
                for name in names:
                    row[f"dense {name}"] = ranks[gname, name, "dense"][i]
                    row[f"top1 {name}"] = tops[gname, name][i]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"-> {summary_path}, {ranks_path}")


if __name__ == "__main__":
    main()
