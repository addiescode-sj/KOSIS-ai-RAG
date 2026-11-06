"""Plan 33: re-rank the first-stage candidates with a cross-encoder (bge-reranker-v2-m3, no training).

First stage = the current setting: dense top-50 + V0 BM25 top-50, RRF c=60 (F0 of fusion_eval), for the fine-tuned
unseen-e1 checkpoint (the fair model) and V0 bge-m3 (reference). The cross-encoder reads (question, table text)
for the top-K candidates and only their order changes; everything below K stays where it was:
    F0  no re-ranking
    R0  re-ranker score only, K in {20, 50}
    R1  RRF of the two orders inside top-K: w_r/(60+rank_rerank) + 1/(60+rank_first), w_r in {1, 2, 4}
    R2  minmax(re-ranker score) + b*minmax(first-stage RRF score) inside top-K, b in {.1, .3, .5}
Choice per first stage on retrieval_golden dev (answerable): Hit@1, Hit@5, MRR@10; settings that lose more than 2pt
routing_v0 dev Hit@5 against F0 are left out; within one question the simpler setting wins (F0 < R0 < R1 < R2).
Reported like fusion_eval, plus re-ranking time per question. Re-ranker scores are cached in data/lab/rerank/
(one file per golden, keyed by re-ranker, text and length), so changing the combinations later costs nothing.

    python lab/rerank_eval.py
    python lab/rerank_eval.py --stages unseen-e1,V0 --reranker BAAI/bge-reranker-v2-m3 --max-length 1024
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from lab_common import (
    DEFAULT_MODEL, DEPTH, METRICS, OUT_DIR, RETRIEVAL, RRF_C, ROUTING, bm25_tokenize, boot, doc_vectors, docs,
    encode, group_rank, load_records, query_text, question_vectors, read_jsonl, rrf, source_sha, summary, top,
)
from ft_train import FT_DIR, load_ft, table_info

GUARD = 0.02  # max routing dev Hit@5 loss against F0
RERANK_DIR = OUT_DIR / "rerank"


def configs() -> list[dict]:
    out = [{"id": "F0", "kind": "none", "k": 0, "cx": 0.0}]
    for k in (20, 50):
        out.append({"id": f"R0 K{k}", "kind": "r0", "k": k, "cx": 1 + (k == 50) * 0.1})
    for k in (20, 50):
        for w in (1, 2, 4):
            out.append({"id": f"R1 K{k} w{w}", "kind": "r1", "k": k, "w": w, "cx": 2 + (k == 50) * 0.1 + w / 100})
    for k in (20, 50):
        for b in (0.1, 0.3, 0.5):
            out.append({"id": f"R2 K{k} b{b:.1f}", "kind": "r2", "k": k, "b": b, "cx": 3 + (k == 50) * 0.1 + b / 10})
    return out


def minmax(x: np.ndarray) -> np.ndarray:
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 0 else np.zeros_like(x)


def rerank(cfg: dict, first: list[str], first_score: dict[str, float], score: dict[str, float]) -> list[str]:
    if cfg["kind"] == "none":
        return first
    head, tail = first[: cfg["k"]], first[cfg["k"]:]
    rs = np.array([score[k] for k in head], np.float64)
    if cfg["kind"] == "r0":
        s = rs
    elif cfg["kind"] == "r1":
        rr = np.empty(len(head))
        rr[np.argsort(-rs, kind="stable")] = np.arange(1, len(head) + 1)
        s = cfg["w"] / (RRF_C + rr) + 1 / (RRF_C + np.arange(1, len(head) + 1))
    else:
        s = minmax(rs) + cfg["b"] * minmax(np.array([first_score[k] for k in head], np.float64))
    order = np.argsort(-s, kind="stable")  # ties keep the first-stage order
    return [head[i] for i in order] + tail


def rrf_scores(*lists: list[str], c: int = RRF_C) -> dict[str, float]:
    score: dict[str, float] = {}
    for keys in lists:
        for rank, key in enumerate(keys, 1):
            score[key] = score.get(key, 0.0) + 1 / (c + rank)
    return score


def load_reranker(name: str, max_length: int):
    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder(name, max_length=max_length, device=device)
    if device == "cuda":  # fp16 like the bi-encoder; skipped if this sentence-transformers version hides the module
        try:
            (getattr(model, "model", None) or model).half()
        except Exception as e:  # noqa: BLE001
            print(f"  fp16 skipped: {e}")
    return model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--run", default=None, help="folder name under data/lab/ft (default: latest)")
    p.add_argument("--stages", default="unseen-e1,V0", help="first stages: checkpoint names in the run and/or V0")
    p.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    p.add_argument("--max-length", type=int, default=1024, help="re-ranker tokens per (question, table) pair")
    p.add_argument("--max-labels", type=int, default=60)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    ft_names = [s for s in stages if s != "V0"]
    run_dir = None
    if ft_names:
        runs_dir = sorted(d for d in FT_DIR.iterdir() if d.is_dir()) if FT_DIR.exists() else []
        run_dir = FT_DIR / args.run if args.run else (runs_dir[-1] if runs_dir else None)
        assert run_dir and run_dir.exists(), "no training run found under data/lab/ft"
        missing = [n for n in ft_names if not (run_dir / f"{n}.pt").exists()]
        assert not missing, f"checkpoints not found in {run_dir}: {missing}"
    kmax = max(c["k"] for c in configs())

    records, sha, tokenize = load_records(), source_sha(), bm25_tokenize()
    ds = docs(records, "norm", args.max_labels)
    keys = [d["key"] for d in ds]
    pos = {k: i for i, k in enumerate(keys)}
    info = table_info(records)
    siblings: dict[str, set[str]] = defaultdict(set)
    for k, v in info.items():
        siblings[v["name"]].add(k)
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
        s["qn"] = qn
        s["cluster"] = [g[0][0] for g in s["groups"]]
        s["bm25"] = [[keys[j] for j in np.argsort(-sc, kind="stable")[:DEPTH] if sc[j] > 0]
                     for sc in (np.asarray(bm.get_scores(tokenize(q))) for q in qn)]

    # ---- first-stage vectors (same caches as ft_eval / fusion_eval) ----
    state: dict = {}
    vec = {}
    if "V0" in stages:
        vec["V0"] = {"docs": doc_vectors(args.model, "norm", args.max_labels, args.max_tokens, ds, sha, state, args.batch),
                     "routing": question_vectors(args.model, qn_r, ROUTING.stem, state, args.batch),
                     "retrieval": question_vectors(args.model, qn_t_all, RETRIEVAL.stem, state, args.batch)[keep]}
    state.clear()
    for name in ft_names:
        ck = run_dir / f"{name}.pt"
        cache = {part: ck.with_name(f"{ck.stem}__{part}.npy") for part in ("docs", "routing", "retrieval")}
        if all(c.exists() for c in cache.values()):
            print(f"  cache {ck.stem}")
            v = {part: np.load(c) for part, c in cache.items()}
        else:
            model = load_ft(args.model, ck, args.max_tokens)
            v = {"docs": np.asarray(encode(model, args.model, [d["embed"] for d in ds], "passage", args.batch), np.float32)}
            model.max_seq_length = 512
            v["routing"] = np.asarray(encode(model, args.model, qn_r, "query", args.batch), np.float32)
            v["retrieval"] = np.asarray(encode(model, args.model, qn_t_all, "query", args.batch), np.float32)
            for part, c in cache.items():
                np.save(c, v[part])
            del model
            print(f"  embedded {ck.stem}")
        assert len(v["docs"]) == len(keys), f"{name}: document vectors do not match the {len(keys)} tables"
        v["retrieval"] = v["retrieval"][keep] if len(v["retrieval"]) == len(rt_all) else v["retrieval"]
        vec[name] = v
    stages = [s for s in stages if s in vec]

    first: dict = {}  # (golden, stage) -> F0 lists
    first_sc: dict = {}  # (golden, stage) -> first-stage RRF scores
    for name, s in sets.items():
        for st in stages:
            sc = vec[st]["docs"] @ vec[st][name].T
            dense = [top(sc[:, i], keys) for i in range(sc.shape[1])]
            first[name, st] = [rrf(d, b) for d, b in zip(dense, s["bm25"])]
            first_sc[name, st] = [rrf_scores(d, b) for d, b in zip(dense, s["bm25"])]
    del vec

    # ---- re-ranker scores for every (question, candidate) pair in any first stage's top-K, cached per golden ----
    tag = f"{args.reranker.split('/')[-1]}__n2-L{args.max_labels}__{sha[:8]}-T{args.max_length}"
    scores: dict[str, list[dict[str, float]]] = {}
    timing = {"pairs": 0, "seconds": 0.0}
    reranker = None
    for name, s in sets.items():
        qsha = hashlib.sha256("\n".join(s["qn"]).encode("utf-8")).hexdigest()[:8]
        path = RERANK_DIR / f"{tag}__{name}__{qsha}.json"
        cached: dict[str, dict[str, float]] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        need = []
        for i in range(len(s["qn"])):
            have = cached.get(str(i), {})
            for st in stages:
                need += [(i, k) for k in first[name, st][i][:kmax] if k not in have]
        need = list(dict.fromkeys(need))
        if need:
            if reranker is None:
                reranker = load_reranker(args.reranker, args.max_length)
            started = time.perf_counter()
            out = reranker.predict([(s["qn"][i], ds[pos[k]]["embed"]) for i, k in need], batch_size=args.batch,
                                   show_progress_bar=True, convert_to_numpy=True)
            timing["seconds"] += time.perf_counter() - started
            timing["pairs"] += len(need)
            for (i, k), v in zip(need, np.asarray(out, np.float64).reshape(-1)):
                cached.setdefault(str(i), {})[k] = float(v)
            RERANK_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
            print(f"  re-ranked {len(need)} pairs for {name} -> {path.name}")
        else:
            print(f"  cache {path.name}")
        scores[name] = [cached.get(str(i), {}) for i in range(len(s["qn"]))]

    # ---- every setting on every first stage ----
    cfgs = configs()
    lists: dict = {}
    ranks: dict = {}
    for name, s in sets.items():
        for st in stages:
            for cfg in cfgs:
                fl = [rerank(cfg, f, fs, sc) for f, fs, sc in zip(first[name, st], first_sc[name, st], scores[name])]
                lists[name, st, cfg["id"]] = fl
                ranks[name, st, cfg["id"]] = [group_rank(g, t) for g, t in zip(s["groups"], fl)]

    def idx(name: str, split: str) -> list[int]:
        return [i for i, x in enumerate(sets[name]["split"]) if split == "all" or x == split]

    t_dev, r_dev = idx("retrieval", "dev"), idx("routing", "dev")
    chosen: dict[str, str] = {}
    sel_rows: dict[str, tuple] = {}
    for st in stages:
        g0 = summary([ranks["routing", st, "F0"][i] for i in r_dev])["Hit@5"]
        rows = []
        for cfg in cfgs:
            r = [ranks["retrieval", st, cfg["id"]][i] for i in t_dev]
            rows.append({"id": cfg["id"], "cx": cfg["cx"], "h1": sum(x is not None and x <= 1 for x in r),
                         "h5": sum(x is not None and x <= 5 for x in r), "mrr": summary(r)["MRR@10"],
                         "rh5": summary([ranks["routing", st, cfg["id"]][i] for i in r_dev])["Hit@5"]})
            rows[-1]["ok"] = rows[-1]["rh5"] >= g0 - GUARD - 1e-9
        ok = [r for r in rows if r["ok"]]
        best = max(ok, key=lambda r: (r["h1"], r["h5"], r["mrr"]))
        near = [r for r in ok if r["h1"] >= best["h1"] - 1 and r["h5"] >= best["h5"] - 1]
        chosen[st] = min(near, key=lambda r: (r["cx"], -r["h1"], -r["h5"], -r["mrr"]))["id"]
        sel_rows[st] = (rows, best["id"])

    rng = np.random.default_rng(0)
    base = stages[0]  # the fair first stage (unseen-e1 by default)
    ref = ("V0", "F0") if "V0" in stages else (base, "F0")

    def hit(name: str, st: str, c: str, i: int, k: int) -> float:
        r = ranks[name, st, c][i]
        return float(r is not None and r <= k)

    shown = [(st, c) for st in stages for c in dict.fromkeys(["F0", "R0 K50", chosen[st]])]

    def label(st: str, c: str) -> str:
        return f"{st} {c}" + (" **(chosen)**" if c == chosen[st] else "")

    def delta(name: str, ix: list[int], a: tuple[str, str], b: tuple[str, str], k: int) -> str:
        if a == b:
            return "-"
        d = np.array([hit(name, *a, i, k) - hit(name, *b, i, k) for i in ix])
        ci_q = boot(d, None, rng)
        ci_t = boot(d, [sets[name]["cluster"][i] for i in ix], rng)
        return (f"{d.mean():+.3f} [{ci_q[0]:+.3f}, {ci_q[1]:+.3f}] [{ci_t[0]:+.3f}, {ci_t[1]:+.3f}] "
                f"{int((d > 0).sum())}/{int((d < 0).sum())}")

    def block(name: str, ix: list[int], title: str) -> list[str]:
        out = ["", f"## {title} (n={len(ix)})", "",
               "Δ cells: mean, 95% question-level, 95% table-level, wins/losses.", "",
               "| first stage / re-rank | " + " | ".join(METRICS) + " | ΔHit@1 vs own F0 | ΔHit@5 vs own F0 "
               f"| ΔHit@1 vs {ref[0]} F0 |",
               "| --- | " + " | ".join("---" for _ in METRICS) + " | --- | --- | --- |"]
        for st, c in shown:
            sm = summary([ranks[name, st, c][i] for i in ix])
            out.append(f"| {label(st, c)} | " + " | ".join(f"{sm[m]:.3f}" for m in METRICS)
                       + f" | {delta(name, ix, (st, c), (st, 'F0'), 1)} | {delta(name, ix, (st, c), (st, 'F0'), 5)}"
                       f" | {delta(name, ix, (st, c), ref, 1)} |")
        return out

    def verdict() -> list[str]:
        ix = idx("retrieval", "all")
        out = ["", "## verdict on retrieval_golden all (table-level 95% lower bound > 0 → effect, exploratory)", ""]
        for st in stages:
            c = chosen[st]
            if c == "F0":
                out.append(f"- {st}: F0 kept (no re-ranking setting beat it on dev by more than one question)")
                continue
            parts, eff = [], False
            for k in (1, 5):
                d = np.array([hit("retrieval", st, c, i, k) - hit("retrieval", st, "F0", i, k) for i in ix])
                lo, hi = boot(d, [sets["retrieval"]["cluster"][i] for i in ix], rng)
                parts.append(f"Hit@{k} {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
                eff = eff or lo > 0
            out.append(f"- {st}: {c} vs F0 → " + "; ".join(parts) + (" → **effect**" if eff else " → no clear effect"))
        return out

    def ceiling() -> list[str]:
        ix = idx("retrieval", "all")
        out = ["", "## where the gold table sits in the first stage (retrieval_golden all; a re-ranker can only "
               "reorder top-K)", "", "| first stage | 1 | 2-5 | 6-10 | 11-20 | 21-50 | 51+ | not found |",
               "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for st in stages:
            r = [ranks["retrieval", st, "F0"][i] for i in ix]
            b = [(1, 1), (2, 5), (6, 10), (11, 20), (21, 50), (51, 10**6)]
            out.append(f"| {st} | " + " | ".join(str(sum(1 for x in r if x and lo <= x <= hi)) for lo, hi in b)
                       + f" | {sum(x is None for x in r)} |")
        return out

    def selection() -> list[str]:
        out = ["", "## choice on retrieval_golden dev (answerable) with the routing_v0 dev guardrail", ""]
        for st in stages:
            rows, best = sel_rows[st]
            out += [f"### {st}: chosen **{chosen[st]}** (best on dev: {best})", "",
                    f"| re-rank | dev Hit@1 (of {len(t_dev)}) | dev Hit@5 | dev MRR@10 | routing dev Hit@5 | guard |",
                    "| --- | --- | --- | --- | --- | --- |"]
            for r in rows:
                mark = " **(chosen)**" if r["id"] == chosen[st] else ""
                out.append(f"| {r['id']}{mark} | {r['h1']} | {r['h5']} | {r['mrr']:.3f} | {r['rh5']:.3f} | "
                           f"{'ok' if r['ok'] else 'out'} |")
            out.append("")
        return out

    def sibling_block() -> list[str]:
        out = ["", "## same-name siblings: rows whose gold table has siblings (reporting only)", "",
               "| golden | n | first stage / re-rank | Hit@5 | sibling above gold |", "| --- | --- | --- | --- | --- |"]
        for name, ix in (("routing", idx("routing", "test")), ("retrieval", idx("retrieval", "all"))):
            s = sets[name]
            rows = []
            for i in ix:
                gold = {k for g in s["groups"][i] for k in g}
                sib = {k for g in gold if g in info for k in siblings[info[g]["name"]]} - gold
                if sib:
                    rows.append((i, gold, sib))
            for st, c in shown:
                conf = 0
                for i, gold, sib in rows:
                    p_ = {k: r for r, k in enumerate(lists[name, st, c][i])}
                    conf += min((p_[k] for k in sib if k in p_), default=10**6) < \
                        min((p_[k] for k in gold if k in p_), default=10**6)
                h = np.mean([hit(name, st, c, i, 5) for i, _, _ in rows]) if rows else 0.0
                out.append(f"| {name} | {len(rows)} | {label(st, c)} | {h:.3f} | {conf / max(len(rows), 1):.3f} |")
        return out

    def qtype_block() -> list[str]:
        s = sets["retrieval"]
        out = ["", "## retrieval_golden Hit@1 / Hit@5 by qtype (all answerable rows; reporting only)", "",
               "| qtype | n | " + " | ".join(f"{st} {c}" for st, c in shown) + " |",
               "| --- | --- | " + " | ".join("---" for _ in shown) + " |"]
        by: dict[str, list[int]] = defaultdict(list)
        for i, q in enumerate(s["qtype"]):
            by[q].append(i)
        for q, ix in sorted(by.items(), key=lambda kv: -len(kv[1])):
            out.append(f"| {q} | {len(ix)} | " + " | ".join(
                f"{np.mean([hit('retrieval', st, c, i, 1) for i in ix]):.2f} / "
                f"{np.mean([hit('retrieval', st, c, i, 5) for i in ix]):.2f}" for st, c in shown) + " |")
        return out

    rid = time.strftime("%Y%m%d-%H%M%S")
    per_pair = timing["seconds"] / timing["pairs"] if timing["pairs"] else None
    speed = (f"re-ranking this run: {timing['pairs']} pairs in {timing['seconds']:.0f}s → about "
             f"{per_pair * 20:.2f}s per question at K=20, {per_pair * 50:.2f}s at K=50 (batch {args.batch})"
             if per_pair else "re-ranking: all scores from cache (no timing this run)")
    lines = [f"# Plan 33: cross-encoder re-ranking of the first-stage top-K ({args.reranker})", "",
             f"eval run {rid}; training run {run_dir.name if run_dir else '-'}; tables {sha[:8]}; "
             f"pair length {args.max_length}; {len(cfgs)} settings",
             "first stage = dense top-50 + BM25 top-50, RRF c=60 (F0). chosen: "
             + ", ".join(f"{st} **{chosen[st]}**" for st in stages),
             "unseen-* is the fair first stage (golden tables held out of its training).", speed]
    lines += verdict()
    lines += ceiling()
    for sp in ("all", "test", "dev"):
        lines += block("retrieval", idx("retrieval", sp),
                       f"retrieval_golden {sp} (answerable; test was looked at before, exploratory"
                       + ("; choice made here" if sp == "dev" else "") + ")")
    lines += block("routing", idx("routing", "test"), "routing_v0 test (same workflow as the training questions)")
    lines += sibling_block()
    lines += qtype_block()
    lines += selection()
    report = "\n".join(lines) + "\n"
    print(report)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / f"rerank_eval_{rid}_summary.md"
    ranks_path = OUT_DIR / f"rerank_eval_{rid}_ranks.jsonl"
    summary_path.write_text(report, encoding="utf-8")
    with ranks_path.open("w", encoding="utf-8") as f:
        for name, s in sets.items():
            for i, qid in enumerate(s["ids"]):
                row = {"golden": name, "id": qid, "split": s["split"][i], "question": s["items"][i]["question"]}
                for st, c in shown:
                    row[f"{st} {c}"] = ranks[name, st, c][i]
                    row[f"top1 {st} {c}"] = lists[name, st, c][i][0] if lists[name, st, c][i] else None
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"-> {summary_path}, {ranks_path}")


if __name__ == "__main__":
    main()
