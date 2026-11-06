"""Plan 32: re-tune how dense and BM25 are fused, now that the dense side is fine-tuned.

Models: V0 bge-m3 and the plan 29 checkpoints (default seen-e2, unseen-e1). Candidates stay dense top-50 + V0 BM25
top-50; only the fusion changes:
    F0  RRF c=60, equal weights (the current setting)
    F1  weighted RRF  w_d/(c+r_d) + w_b/(c+r_b), w_d=1, w_b in {.25,.5,.75,1}, c in {10,30,60}
    F2  score fusion over the union of candidates: norm(dense cosine) + a*norm(BM25), norm in {min-max, z},
        a in {0,.1,.2,.3,.5}
    D   dense only (= F1 with w_b=0)
Choice per model on retrieval_golden dev (answerable): Hit@1, then Hit@5, then MRR@10. Settings that lose more than
2pt routing_v0 dev Hit@5 against the same model's F0 are left out. If the next settings are within one question
(Hit@1 and Hit@5 counts), the simpler one wins (F0, then F1 near the default, then F2).
Reported: retrieval_golden all / test / dev and routing_v0 test against each model's F0 and against V0 F0, with
question- and table-level bootstrap; same-name siblings above the gold table; Hit@5 by qtype.
Unseen-e1 is the fair model (the golden tables were held out of its training); seen-e2 is reference only.
Writes data/lab/fusion_eval_<id>_summary.md and _ranks.jsonl. Fine-tuned vectors are read from the caches ft_eval
wrote beside each checkpoint (<ckpt>__docs/routing/retrieval.npy), or embedded and cached if missing.

    python lab/fusion_eval.py
    python lab/fusion_eval.py --run 20261030-054248 --ckpts seen-e2,unseen-e1
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from lab_common import (
    DEFAULT_MODEL, DEPTH, METRICS, OUT_DIR, RETRIEVAL, ROUTING, bm25_tokenize, boot, doc_vectors, docs, encode,
    group_rank, load_records, query_text, question_vectors, read_jsonl, rrf, source_sha, summary, top,
)
from ft_train import FT_DIR, load_ft, table_info

GUARD = 0.02  # max routing dev Hit@5 loss against the same model's F0


def configs() -> list[dict]:
    out = [{"id": "F0", "kind": "rrf", "wb": 1.0, "c": 60, "cx": 0.0},
           {"id": "D", "kind": "dense", "cx": 2.0}]
    for wb in (1.0, 0.75, 0.5, 0.25):
        for c in (60, 30, 10):
            if (wb, c) != (1.0, 60):
                out.append({"id": f"F1 wb{wb:.2f} c{c}", "kind": "rrf", "wb": wb, "c": c,
                            "cx": 1 + (1 - wb) + abs(60 - c) / 100})
    for norm in ("mm", "z"):
        for a in (0.0, 0.1, 0.2, 0.3, 0.5):
            out.append({"id": f"F2 {norm} a{a:.1f}", "kind": "score", "norm": norm, "a": a,
                        "cx": 3 + a + (0.05 if norm == "z" else 0.0)})
    return out


def wrrf(dense: list[str], bm: list[str], wb: float, c: int) -> list[str]:
    score: dict[str, float] = {}
    for w, keys in ((1.0, dense), (wb, bm)):
        for rank, key in enumerate(keys, 1):
            score[key] = score.get(key, 0.0) + w / (c + rank)
    return sorted(score, key=score.get, reverse=True)


def normed(x: np.ndarray, how: str) -> np.ndarray:
    if how == "mm":
        span = x.max() - x.min()
        return (x - x.min()) / span if span > 0 else np.zeros_like(x)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def fuse(cfg: dict, dense: list[str], bm: list[str], dsc: np.ndarray, bsc: np.ndarray, pos: dict[str, int]) -> list[str]:
    if cfg["kind"] == "dense":
        return dense
    if cfg["kind"] == "rrf":
        return wrrf(dense, bm, cfg["wb"], cfg["c"])
    seen = set(dense)
    cand = dense + [k for k in bm if k not in seen]
    ix = [pos[k] for k in cand]
    s = normed(dsc[ix].astype(np.float64), cfg["norm"]) + cfg["a"] * normed(bsc[ix].astype(np.float64), cfg["norm"])
    order = sorted(range(len(cand)), key=lambda i: -s[i])  # stable: ties keep dense-first order
    return [cand[i] for i in order]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--run", default=None, help="folder name under data/lab/ft (default: latest)")
    p.add_argument("--ckpts", default="seen-e2,unseen-e1", help="checkpoint names in the run, comma separated")
    p.add_argument("--max-labels", type=int, default=60)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()
    runs_dir = sorted(d for d in FT_DIR.iterdir() if d.is_dir()) if FT_DIR.exists() else []
    run_dir = FT_DIR / args.run if args.run else (runs_dir[-1] if runs_dir else None)
    assert run_dir and run_dir.exists(), "no training run found under data/lab/ft"
    ckpts = [run_dir / f"{n.strip()}.pt" for n in args.ckpts.split(",") if n.strip()]
    missing = [str(c) for c in ckpts if not c.exists()]
    assert not missing, f"checkpoints not found: {missing}"

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
                    "groups": [[[Path(it["table"]).stem, *it.get("alt_tables", [])]] for it in rg],
                    "qtype": [it.get("qtype", "?") for it in rg]},
        "retrieval": {"items": rt, "ids": [it["qid"] for it in rt], "split": [it["split"] for it in rt],
                      "groups": [it["required"] for it in rt], "qtype": [it.get("qtype", "?") for it in rt]},
    }
    qn_r = [query_text(it["question"], "norm") for it in rg]
    qn_t_all = [query_text(it["question"], "norm") for it in rt_all]
    for s, qn in ((sets["routing"], qn_r), (sets["retrieval"], [qn_t_all[i] for i in keep])):
        s["cluster"] = [g[0][0] for g in s["groups"]]
        s["bsc"] = [np.asarray(bm.get_scores(tokenize(q))) for q in qn]
        s["bm25"] = [[keys[j] for j in np.argsort(-sc, kind="stable")[:DEPTH] if sc[j] > 0] for sc in s["bsc"]]

    # ---- vectors: V0 from the existing caches; checkpoints from the ft_eval caches (embedded if missing) ----
    state: dict = {}
    vec = {"V0": {"docs": doc_vectors(args.model, "norm", args.max_labels, args.max_tokens, ds, sha, state, args.batch),
                  "routing": question_vectors(args.model, qn_r, ROUTING.stem, state, args.batch),
                  "retrieval": question_vectors(args.model, qn_t_all, RETRIEVAL.stem, state, args.batch)[keep]}}
    state.clear()
    for ck in ckpts:
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
        assert len(v["docs"]) == len(keys), f"{ck.stem}: document vectors do not match the {len(keys)} tables"
        v["retrieval"] = v["retrieval"][keep] if len(v["retrieval"]) == len(rt_all) else v["retrieval"]
        vec[ck.stem] = v
    models = list(vec)

    # ---- every fusion setting for every model ----
    cfgs = configs()
    cid = [c["id"] for c in cfgs]
    lists: dict = {}  # (golden, model, config) -> fused key lists
    ranks: dict = {}  # (golden, model, config) -> group ranks
    for name, s in sets.items():
        for mn in models:
            sc = vec[mn]["docs"] @ vec[mn][name].T
            dense = [top(sc[:, i], keys) for i in range(sc.shape[1])]
            for cfg in cfgs:
                fl = [fuse(cfg, d, b, sc[:, i], s["bsc"][i], pos) for i, (d, b) in enumerate(zip(dense, s["bm25"]))]
                lists[name, mn, cfg["id"]] = fl
                ranks[name, mn, cfg["id"]] = [group_rank(g, t) for g, t in zip(s["groups"], fl)]
            ref = [rrf(d, b) for d, b in zip(dense, s["bm25"])]  # F0 must be the old fusion exactly
            assert ref == lists[name, mn, "F0"], f"F0 differs from lab_common.rrf ({name}, {mn})"

    def idx(name: str, split: str) -> list[int]:
        return [i for i, x in enumerate(sets[name]["split"]) if split == "all" or x == split]

    def counts(name: str, mn: str, c: str, ix: list[int]) -> tuple[int, int, float]:
        r = [ranks[name, mn, c][i] for i in ix]
        return (sum(x is not None and x <= 1 for x in r), sum(x is not None and x <= 5 for x in r),
                summary(r)["MRR@10"])

    # ---- choice on retrieval dev, routing dev guardrail ----
    t_dev, r_dev = idx("retrieval", "dev"), idx("routing", "dev")
    chosen: dict[str, str] = {}
    sel_rows: dict[str, list] = {}
    for mn in models:
        g0 = summary([ranks["routing", mn, "F0"][i] for i in r_dev])["Hit@5"]
        rows = []
        for cfg in cfgs:
            h1, h5, mrr = counts("retrieval", mn, cfg["id"], t_dev)
            rh5 = summary([ranks["routing", mn, cfg["id"]][i] for i in r_dev])["Hit@5"]
            rows.append({"id": cfg["id"], "cx": cfg["cx"], "h1": h1, "h5": h5, "mrr": mrr, "rh5": rh5,
                         "ok": rh5 >= g0 - GUARD - 1e-9})
        ok = [r for r in rows if r["ok"]]
        best = max(ok, key=lambda r: (r["h1"], r["h5"], r["mrr"]))
        near = [r for r in ok if r["h1"] >= best["h1"] - 1 and r["h5"] >= best["h5"] - 1]
        chosen[mn] = min(near, key=lambda r: (r["cx"], -r["h1"], -r["h5"], -r["mrr"]))["id"]
        sel_rows[mn] = (rows, best["id"])

    rng = np.random.default_rng(0)

    def hit(name: str, mn: str, c: str, i: int, k: int) -> float:
        r = ranks[name, mn, c][i]
        return float(r is not None and r <= k)

    shown = [(mn, c) for mn in models for c in dict.fromkeys(["F0", "D", chosen[mn]])]

    def label(mn: str, c: str) -> str:
        return f"{mn} {c}" + (" **(chosen)**" if c == chosen[mn] else "")

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
               "| model / fusion | " + " | ".join(METRICS) + " | ΔHit@1 vs own F0 | ΔHit@5 vs own F0 "
               "| ΔHit@5 vs V0 F0 |",
               "| --- | " + " | ".join("---" for _ in METRICS) + " | --- | --- | --- |"]
        for mn, c in shown:
            sm = summary([ranks[name, mn, c][i] for i in ix])
            out.append(f"| {label(mn, c)} | " + " | ".join(f"{sm[m]:.3f}" for m in METRICS)
                       + f" | {delta(name, ix, (mn, c), (mn, 'F0'), 1)} | {delta(name, ix, (mn, c), (mn, 'F0'), 5)}"
                       f" | {delta(name, ix, (mn, c), ('V0', 'F0'), 5)} |")
        return out

    def verdict() -> list[str]:
        ix = idx("retrieval", "all")
        out = ["", "## verdict on retrieval_golden all (table-level 95% lower bound > 0 → effect, exploratory)", ""]
        for mn in models:
            c = chosen[mn]
            if c == "F0":
                out.append(f"- {mn}: F0 kept (nothing on dev beat it by more than one question)")
                continue
            parts, eff = [], False
            for k in (1, 5):
                d = np.array([hit("retrieval", mn, c, i, k) - hit("retrieval", mn, "F0", i, k) for i in ix])
                lo, hi = boot(d, [sets["retrieval"]["cluster"][i] for i in ix], rng)
                parts.append(f"Hit@{k} {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
                eff = eff or lo > 0
            out.append(f"- {mn}: {c} vs F0 → " + "; ".join(parts) + (" → **effect**" if eff else " → no clear effect"))
        return out

    def selection() -> list[str]:
        out = ["", "## choice on retrieval_golden dev (answerable) with the routing_v0 dev guardrail", ""]
        for mn in models:
            rows, best = sel_rows[mn]
            edge = " — best sits on the grid edge (a0.5 or c10); a wider grid may do better" \
                if best.endswith("a0.5") or best.endswith("c10") else ""
            out += [f"### {mn}: chosen **{chosen[mn]}** (best on dev: {best}){edge}", "",
                    f"| fusion | dev Hit@1 (of {len(t_dev)}) | dev Hit@5 | dev MRR@10 | routing dev Hit@5 | guard |",
                    "| --- | --- | --- | --- | --- | --- |"]
            for r in rows:
                mark = " **(chosen)**" if r["id"] == chosen[mn] else ""
                out.append(f"| {r['id']}{mark} | {r['h1']} | {r['h5']} | {r['mrr']:.3f} | {r['rh5']:.3f} | "
                           f"{'ok' if r['ok'] else 'out'} |")
            out.append("")
        return out

    def sibling_block() -> list[str]:
        out = ["", "## same-name siblings: rows whose gold table has siblings (reporting only)", "",
               "| golden | n | model / fusion | Hit@5 | sibling above gold |", "| --- | --- | --- | --- | --- |"]
        for name, ix in (("routing", idx("routing", "test")), ("retrieval", idx("retrieval", "all"))):
            s = sets[name]
            rows = []
            for i in ix:
                gold = {k for g in s["groups"][i] for k in g}
                sib = {k for g in gold if g in info for k in siblings[info[g]["name"]]} - gold
                if sib:
                    rows.append((i, gold, sib))
            for mn, c in shown:
                conf = 0
                for i, gold, sib in rows:
                    p_ = {k: r for r, k in enumerate(lists[name, mn, c][i])}
                    conf += min((p_[k] for k in sib if k in p_), default=10**6) < \
                        min((p_[k] for k in gold if k in p_), default=10**6)
                h = np.mean([hit(name, mn, c, i, 5) for i, _, _ in rows]) if rows else 0.0
                out.append(f"| {name} | {len(rows)} | {label(mn, c)} | {h:.3f} | {conf / max(len(rows), 1):.3f} |")
        return out

    def qtype_block() -> list[str]:
        s = sets["retrieval"]
        out = ["", "## retrieval_golden Hit@5 by qtype (all answerable rows; reporting only)", "",
               "| qtype | n | " + " | ".join(f"{mn} {c}" for mn, c in shown) + " |",
               "| --- | --- | " + " | ".join("---" for _ in shown) + " |"]
        by: dict[str, list[int]] = defaultdict(list)
        for i, q in enumerate(s["qtype"]):
            by[q].append(i)
        for q, ix in sorted(by.items(), key=lambda kv: -len(kv[1])):
            out.append(f"| {q} | {len(ix)} | " + " | ".join(f"{np.mean([hit('retrieval', mn, c, i, 5) for i in ix]):.3f}"
                                                         for mn, c in shown) + " |")
        return out

    rid = time.strftime("%Y%m%d-%H%M%S")
    lines = [f"# Plan 32: dense + BM25 fusion re-tuned per model ({args.model})", "",
             f"eval run {rid}; training run {run_dir.name}; tables {sha[:8]}; {len(cfgs)} fusion settings",
             "chosen: " + ", ".join(f"{mn} **{chosen[mn]}**" for mn in models),
             "unseen-* is the fair model (golden tables held out of training); seen-* is reference only."]
    lines += verdict()
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
    summary_path = OUT_DIR / f"fusion_eval_{rid}_summary.md"
    ranks_path = OUT_DIR / f"fusion_eval_{rid}_ranks.jsonl"
    summary_path.write_text(report, encoding="utf-8")
    with ranks_path.open("w", encoding="utf-8") as f:
        for name, s in sets.items():
            for i, qid in enumerate(s["ids"]):
                row = {"golden": name, "id": qid, "split": s["split"][i], "question": s["items"][i]["question"]}
                for mn, c in shown:
                    row[f"{mn} {c}"] = ranks[name, mn, c][i]
                    row[f"top1 {mn} {c}"] = lists[name, mn, c][i][0] if lists[name, mn, c][i] else None
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"-> {summary_path}, {ranks_path}")


if __name__ == "__main__":
    main()
