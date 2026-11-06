"""Plan 27: evaluate the rule-only document rebuilds from doc_variants.py (step 2 of 2).

Runs (fusion is always the V0 hybrid: dense top-50 + BM25 top-50, RRF c=60):
    V0          n2-L60 text, one vector per table, one BM25 field
    R1          + unit line
    R2 max / R2 l=λ   core vector + one child vector per 항목 / class line; table score = max(core, children)
                      or core + λ * max(children), λ in LAMBDAS
    R3          + "같은 이름 표와 다른 점" line on same-name tables
    R4 b=β      BM25 on two fields: core + β * labels, β in BETAS
Each family's setting is chosen on routing_v0 dev only (Hit@5, then MRR@10). Families whose choice beats V0 on dev
are combined into one run R*; nothing else is combined. Reported with those choices: routing_v0 test and
retrieval_golden all / dev / test (answerable rows; test was looked at before, so exploratory).
Extra diagnostics: same-name sibling confusion, Hit@5 by document length quartile, which R2 part wins.
Dense vectors are cached in data/lab/emb/ (new texts are embedded once, on the GPU when there is one).
Each run writes its own data/lab/doc_eval_<run id>_summary.md and _ranks.jsonl.

    python lab/doc_eval.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from category_eval import METRICS, top
from chroma_common import DEFAULT_MODEL, bm25_tokenize, query_text, read_jsonl, source_sha
from chroma_eval_p1 import DEPTH, OUT_DIR, rrf
from doc_variants import build_all, texts
from embed_eval import doc_vectors
from retrieval_eval import GOLDEN as RETRIEVAL, group_rank, summary
from soft_route_eval import boot
from synth_router import ROUTING, question_vectors

LAMBDAS = (0.3, 0.5)
BETAS = (0.3, 0.5, 1.0)
BASE = {"unit": False, "diff": False, "dense": "single", "bm25": "single"}
FAMILIES = {
    "R1": {"R1": {"unit": True}},
    "R2": {"R2 max": {"dense": "max"}, **{f"R2 l={l}": {"dense": l} for l in LAMBDAS}},
    "R3": {"R3": {"diff": True}},
    "R4": {f"R4 b={b}": {"bm25": b} for b in BETAS},
}


def sha8(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def file_sha8(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-labels", type=int, default=60)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()

    ps, diffs = build_all(args.max_labels)
    keys = [x["key"] for x in ps]
    n = len(keys)
    sha, tokenize, state = source_sha(), bm25_tokenize(), {}
    name_of = {x["key"]: x["name"] for x in ps}
    siblings: dict[str, set[str]] = defaultdict(set)
    for x in ps:
        siblings[x["name"]].add(x["key"])
    length = {x["key"]: len(texts(x)["embed"]) for x in ps}

    def table_texts(cfg: dict) -> list[dict]:
        return [texts(x, unit=cfg["unit"], diff=diffs.get(x["key"]) if cfg["diff"] else None) for x in ps]

    def encode(tag: str, ks: list[str], txts: list[str]) -> np.ndarray:
        ds = [{"key": k, "embed": t} for k, t in zip(ks, txts)]
        variant = "" if tag == "V0" else f"-P27{tag}-{sha8(chr(10).join(txts))}"  # V0 reuses the n2-L60 cache
        return doc_vectors(args.model, "norm", args.max_labels, args.max_tokens, ds, sha, state, args.batch, variant)

    dense_cache: dict[tuple, tuple] = {}

    def dense_parts(cfg: dict):
        """(single vectors) or (core vectors, child vectors, owner index, child kind) for this text setting."""
        tt = table_texts(cfg)
        tag = ("V0" if not (cfg["unit"] or cfg["diff"]) else "") + ("u" if cfg["unit"] else "") + ("d" if cfg["diff"] else "")
        if cfg["dense"] == "single":
            k = ("single", cfg["unit"], cfg["diff"])
            if k not in dense_cache:
                dense_cache[k] = (encode(tag or "V0", keys, [t["embed"] for t in tt]),)
            return dense_cache[k]
        k = ("multi", cfg["unit"], cfg["diff"])
        if k not in dense_cache:
            core = encode(f"core{tag}", keys, [t["core"] for t in tt])
            ck, cx, owner, kind = [], [], [], []
            for i, t in enumerate(tt):
                for j, c in enumerate(t["children"]):
                    ck.append(f"{keys[i]}#{j}")
                    cx.append(c)
                    owner.append(i)
                    kind.append(c.split("\n", 1)[1].split(":", 1)[0])
            child = encode("child", ck, cx)  # children never carry R1 / R3 lines
            dense_cache[k] = (core, child, np.array(owner), kind)
        return dense_cache[k]

    bm25_cache: dict[tuple, BM25Okapi] = {}

    def bm25_index(cfg: dict, field: str) -> BM25Okapi:
        k = (field, cfg["unit"], cfg["diff"]) if field != "bm25_labels" else (field,)
        if k not in bm25_cache:
            bm25_cache[k] = BM25Okapi([tokenize(t[field]) for t in table_texts(cfg)])
        return bm25_cache[k]

    # ---- goldens ----
    rg = read_jsonl(ROUTING)
    rt_all = read_jsonl(RETRIEVAL)
    keep = [i for i, it in enumerate(rt_all) if it.get("answerable", True) and it.get("required")]
    rt = [rt_all[i] for i in keep]
    sets = {
        "routing": {"items": rg, "ids": [it["id"] for it in rg], "split": [it["split"] for it in rg],
                    "groups": [[[Path(it["table"]).stem, *it.get("alt_tables", [])]] for it in rg]},
        "retrieval": {"items": rt, "ids": [it["qid"] for it in rt], "split": [it["split"] for it in rt],
                      "groups": [it["required"] for it in rt]},
    }
    qv_r = question_vectors(args.model, [query_text(it["question"], "norm") for it in rg], ROUTING.stem, state, args.batch)
    qv_t = question_vectors(args.model, [query_text(it["question"], "norm") for it in rt_all], RETRIEVAL.stem, state,
                            args.batch)[keep]
    sets["routing"]["qv"], sets["retrieval"]["qv"] = qv_r, qv_t
    for s in sets.values():
        s["qn"] = [query_text(it["question"], "norm") for it in s["items"]]
        s["cluster"] = [g[0][0] for g in s["groups"]]

    def run_config(s: dict, cfg: dict) -> dict:
        parts = dense_parts(cfg)
        if cfg["dense"] == "single":
            scores = parts[0] @ s["qv"].T  # (n, q)
            won = None
        else:
            core, child, owner, kind = parts
            cs, ks = core @ s["qv"].T, child @ s["qv"].T
            best = cs.copy()  # a table without children falls back to its core score
            arg = np.full(cs.shape, -1)
            seen = np.zeros(n, bool)
            for r in range(ks.shape[0]):  # max over each table's children
                o = owner[r]
                if not seen[o]:
                    best[o], arg[o], seen[o] = ks[r], r, True
                else:
                    upd = ks[r] > best[o]
                    best[o] = np.where(upd, ks[r], best[o])
                    arg[o] = np.where(upd, r, arg[o])
            if cfg["dense"] == "max":
                scores = np.maximum(cs, best)
                won = np.where(cs >= best, -1, arg)
            else:
                scores = cs + cfg["dense"] * best
                won = None
            won = (won, kind) if won is not None else None
        dense = [top(scores[:, i], keys) for i in range(scores.shape[1])]
        if cfg["bm25"] == "single":
            bm = bm25_index(cfg, "bm25")
            sparse = [np.asarray(bm.get_scores(tokenize(q))) for q in s["qn"]]
        else:
            bc, bl = bm25_index(cfg, "bm25_core"), bm25_index(cfg, "bm25_labels")
            sparse = [np.asarray(bc.get_scores(tokenize(q))) + cfg["bm25"] * np.asarray(bl.get_scores(tokenize(q)))
                      for q in s["qn"]]
        b = [[keys[j] for j in np.argsort(-sc, kind="stable")[:DEPTH] if sc[j] > 0] for sc in sparse]
        hyb = [rrf(d, x) for d, x in zip(dense, b)]
        out = {"hybrid": hyb, "dense": dense, "bm25": b}
        ranks = {m: [group_rank(g, t) for g, t in zip(s["groups"], v)] for m, v in out.items()}
        top1_part = None
        if won:
            w, kind = won
            top1_part = []
            for i, lst in enumerate(hyb):
                j = keys.index(lst[0])
                top1_part.append("core" if w[j, i] < 0 else kind[w[j, i]])
        return {"ranks": ranks, "hybrid": hyb, "top1_part": top1_part}

    runs = {"V0": BASE}
    for fam in FAMILIES.values():
        for rn, ch in fam.items():
            runs[rn] = {**BASE, **ch}
    res = {name: {rn: run_config(s, cfg) for rn, cfg in runs.items()} for name, s in sets.items()}

    # ---- choose on routing dev ----
    rs = sets["routing"]
    dev = [i for i, sp in enumerate(rs["split"]) if sp == "dev"]

    def dev_score(rn: str) -> tuple[float, float]:
        sm = summary([res["routing"][rn]["ranks"]["hybrid"][i] for i in dev])
        return sm["Hit@5"], sm["MRR@10"]

    chosen = {f: max(fam, key=dev_score) for f, fam in FAMILIES.items()}
    better = [f for f, rn in chosen.items() if dev_score(rn) > dev_score("V0")]
    if len(better) >= 2:
        star = dict(BASE)
        for f in better:
            star |= {k: v for k, v in runs[chosen[f]].items() if v != BASE[k]}
        runs["R*"] = star
        for name, s in sets.items():
            res[name]["R*"] = run_config(s, star)
    order = list(runs)
    rng = np.random.default_rng(0)

    def hit5(name: str, rn: str, i: int) -> bool:
        return (res[name][rn]["ranks"]["hybrid"][i] or 999) <= 5

    def block(name: str, idx: list[int], title: str) -> list[str]:
        s = sets[name]
        out = ["", f"## {title} (n={len(idx)})", "",
               "| run | " + " | ".join(METRICS) + " | dense Recall@50 | BM25 Recall@50 | dHit@5 vs V0 | 95% (question) "
               "| 95% (table) | wins / losses |",
               "| --- | " + " | ".join("---" for _ in METRICS) + " | --- | --- | --- | --- | --- | --- |"]
        v0 = np.array([hit5(name, "V0", i) for i in idx], float)
        for rn in order:
            rk = res[name][rn]["ranks"]
            sm = summary([rk["hybrid"][i] for i in idx])
            rd = np.mean([rk["dense"][i] is not None for i in idx])
            rb = np.mean([rk["bm25"][i] is not None for i in idx])
            d = np.array([hit5(name, rn, i) for i in idx], float) - v0
            ci_q = boot(d, None, rng) if rn != "V0" else (0.0, 0.0)
            ci_t = boot(d, [s["cluster"][i] for i in idx], rng) if rn != "V0" else (0.0, 0.0)
            mark = " **(chosen)**" if rn in chosen.values() else ""
            out.append(f"| {rn}{mark} | " + " | ".join(f"{sm[m]:.3f}" for m in METRICS)
                       + f" | {rd:.3f} | {rb:.3f} | {d.mean():+.3f} | [{ci_q[0]:+.3f}, {ci_q[1]:+.3f}] | "
                       f"[{ci_t[0]:+.3f}, {ci_t[1]:+.3f}] | {int((d > 0).sum())} / {int((d < 0).sum())} |")
        return out

    shown = ["V0", *chosen.values(), *(["R*"] if "R*" in runs else [])]

    def sibling_block(idx_by: dict[str, list[int]]) -> list[str]:
        """Rows whose gold table has same-name siblings: is a sibling ranked above the best gold table?"""
        out = ["", "## same-name siblings: rows whose gold table has siblings (reporting only)", "",
               "| golden | n | run | Hit@5 | sibling above gold |", "| --- | --- | --- | --- | --- |"]
        for name, idx in idx_by.items():
            s = sets[name]
            rows = []
            for i in idx:
                gold = {k for g in s["groups"][i] for k in g}
                sib = {k for g in gold if g in name_of for k in siblings[name_of[g]]} - gold
                if sib:
                    rows.append((i, gold, sib))
            for rn in shown:
                conf = 0
                for i, gold, sib in rows:
                    lst = res[name][rn]["hybrid"][i]
                    pos = {k: r for r, k in enumerate(lst)}
                    g = min((pos[k] for k in gold if k in pos), default=10**6)
                    sb = min((pos[k] for k in sib if k in pos), default=10**6)
                    conf += sb < g
                h = np.mean([hit5(name, rn, i) for i, _, _ in rows]) if rows else 0.0
                out.append(f"| {name} | {len(rows)} | {rn} | {h:.3f} | {conf / max(len(rows), 1):.3f} |")
        return out

    def length_block(idx_by: dict[str, list[int]]) -> list[str]:
        out = ["", "## Hit@5 by V0 text length of the first gold table, quartiles (reporting only)", "",
               "| golden | quartile | n | " + " | ".join(shown) + " |", "| --- | --- | --- | " + " | ".join("---" for _ in shown) + " |"]
        for name, idx in idx_by.items():
            s = sets[name]
            ln = np.array([length.get(s["groups"][i][0][0], 0) for i in idx])
            cuts = np.percentile(ln, [25, 50, 75])
            q = np.searchsorted(cuts, ln, side="right")
            for b in range(4):
                ix = [i for i, qq in zip(idx, q) if qq == b]
                if ix:
                    out.append(f"| {name} | Q{b + 1} | {len(ix)} | "
                               + " | ".join(f"{np.mean([hit5(name, rn, i) for i in ix]):.3f}" for rn in shown) + " |")
        return out

    def part_block() -> list[str]:
        rn = chosen["R2"]
        out = ["", f"## {rn}: part that won for the top-1 table (max form only)", ""]
        for name in sets:
            tp = res[name][rn]["top1_part"] or res[name]["R2 max"]["top1_part"]
            out.append(f"- {name} ({'R2 max' if not res[name][rn]['top1_part'] else rn}): "
                       + ", ".join(f"{k} {v}" for k, v in Counter(tp).most_common(8)))
        return out

    rid = time.strftime("%Y%m%d-%H%M%S")
    idx_r = {sp: [i for i, x in enumerate(rs["split"]) if x == sp] for sp in ("dev", "test")}
    ts = sets["retrieval"]["split"]
    idx_t = {"all": list(range(len(ts))), "dev": [i for i, x in enumerate(ts) if x == "dev"],
             "test": [i for i, x in enumerate(ts) if x == "test"]}
    lines = [f"# Plan 27: rule-only document rebuilds ({args.model})", "",
             f"run {rid}; routing {file_sha8(ROUTING)}, retrieval {file_sha8(RETRIEVAL)}, tables {sha[:8]}; "
             f"R3 lines {len(diffs)}",
             "chosen on routing_v0 dev (Hit@5, then MRR@10): "
             + ", ".join(f"{f} **{rn}**" for f, rn in chosen.items())
             + f"; better than V0 on dev: {', '.join(better) or 'none'}"
             + ("; R* = " + " + ".join(chosen[f] for f in better) if "R*" in runs else "; no R* (fewer than 2 families beat V0)")]
    lines += block("routing", idx_r["dev"], "routing_v0 dev (choice made here)")
    lines += block("routing", idx_r["test"], "routing_v0 test")
    for sp, ix in idx_t.items():
        lines += block("retrieval", ix, f"retrieval_golden {sp} (answerable; test was looked at before, exploratory)")
    rep = {"routing": idx_r["test"], "retrieval": idx_t["all"]}
    lines += sibling_block(rep)
    lines += length_block(rep)
    lines += part_block()
    report = "\n".join(lines) + "\n"
    print(report)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / f"doc_eval_{rid}_summary.md"
    ranks_path = OUT_DIR / f"doc_eval_{rid}_ranks.jsonl"
    summary_path.write_text(report, encoding="utf-8")
    with ranks_path.open("w", encoding="utf-8") as f:
        for name, s in sets.items():
            for i, qid in enumerate(s["ids"]):
                row = {"golden": name, "id": qid, "split": s["split"][i], "question": s["items"][i]["question"]}
                row |= {rn: res[name][rn]["ranks"]["hybrid"][i] for rn in order}
                row |= {f"top1 {rn}": res[name][rn]["hybrid"][i][0] for rn in ("V0", *chosen.values())}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"-> {summary_path}, {ranks_path}")


if __name__ == "__main__":
    main()
