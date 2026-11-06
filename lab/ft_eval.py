"""Plan 29: evaluate the fine-tuned bge-m3 checkpoints from ft_train.py (step 2 of 2).

Every checkpoint in data/lab/ft/<run>/ is loaded on top of the base model; documents (V0 n2-L60 passages) and the
golden questions are embedded with it (cached next to the checkpoint). Dense top-50 + the unchanged V0 BM25 top-50
are fused like V0 (RRF c=60), so only the dense side differs. Per regime the epoch is chosen on routing_v0 dev
(hybrid Hit@5, then MRR@10). Reported: routing_v0 test, retrieval_golden all / dev / test (answerable; test was
looked at before, exploratory), hybrid and dense; same-name sibling confusion; retrieval_golden Hit@5 by qtype.
Each run writes data/lab/ft_eval_<run id>_summary.md and _ranks.jsonl.

    python lab/ft_eval.py                 # latest training run
    python lab/ft_eval.py --run 20261030-150000
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--run", default=None, help="folder name under data/lab/ft (default: latest)")
    p.add_argument("--max-labels", type=int, default=60)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()
    runs_dir = sorted(d for d in FT_DIR.iterdir() if d.is_dir()) if FT_DIR.exists() else []
    run_dir = FT_DIR / args.run if args.run else (runs_dir[-1] if runs_dir else None)
    assert run_dir and run_dir.exists(), "no training run found under data/lab/ft"
    ckpts = sorted(run_dir.glob("*.pt"))
    assert ckpts, f"no checkpoints in {run_dir}"

    records, sha, tokenize = load_records(), source_sha(), bm25_tokenize()
    ds = docs(records, "norm", args.max_labels)
    keys = [d["key"] for d in ds]
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

    # ---- vectors: V0 from the existing caches, each checkpoint embedded once and cached beside it ----
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
        v["retrieval"] = v["retrieval"][keep] if len(v["retrieval"]) == len(rt_all) else v["retrieval"]
        vec[ck.stem] = v
    names = list(vec)

    res: dict[str, dict[str, dict]] = {}
    for name, s in sets.items():
        res[name] = {}
        for rn in names:
            sc = vec[rn]["docs"] @ vec[rn][name].T
            dense = [top(sc[:, i], keys) for i in range(sc.shape[1])]
            hyb = [rrf(d, b) for d, b in zip(dense, s["bm25"])]
            res[name][rn] = {"hybrid": hyb, "ranks": {"hybrid": [group_rank(g, t) for g, t in zip(s["groups"], hyb)],
                                                      "dense": [group_rank(g, t) for g, t in zip(s["groups"], dense)],
                                                      "bm25": [group_rank(g, t) for g, t in zip(s["groups"], s["bm25"])]}}

    rs = sets["routing"]
    dev = [i for i, sp in enumerate(rs["split"]) if sp == "dev"]

    def dev_score(rn: str) -> tuple[float, float]:
        sm = summary([res["routing"][rn]["ranks"]["hybrid"][i] for i in dev])
        return sm["Hit@5"], sm["MRR@10"]

    regimes = sorted({rn.split("-e")[0] for rn in names if rn != "V0"})
    chosen = {rg_: max([rn for rn in names if rn.startswith(rg_ + "-e")], key=dev_score) for rg_ in regimes}
    shown = ["V0", *chosen.values()]
    rng = np.random.default_rng(0)

    def hit5(name: str, rn: str, i: int, mode: str = "hybrid") -> bool:
        return (res[name][rn]["ranks"][mode][i] or 999) <= 5

    def block(name: str, idx: list[int], title: str) -> list[str]:
        s = sets[name]
        out = ["", f"## {title} (n={len(idx)})", "",
               "| run | " + " | ".join(METRICS) + " | dense Hit@5 | dense Recall@50 | dHit@5 vs V0 | 95% (question) "
               "| 95% (table) | wins / losses |",
               "| --- | " + " | ".join("---" for _ in METRICS) + " | --- | --- | --- | --- | --- | --- |"]
        v0 = np.array([hit5(name, "V0", i) for i in idx], float)
        for rn in names:
            rk = res[name][rn]["ranks"]
            sm = summary([rk["hybrid"][i] for i in idx])
            dh = np.mean([hit5(name, rn, i, "dense") for i in idx])
            dr = np.mean([rk["dense"][i] is not None for i in idx])
            d = np.array([hit5(name, rn, i) for i in idx], float) - v0
            ci_q = boot(d, None, rng) if rn != "V0" else (0.0, 0.0)
            ci_t = boot(d, [s["cluster"][i] for i in idx], rng) if rn != "V0" else (0.0, 0.0)
            mark = " **(chosen)**" if rn in chosen.values() else ""
            out.append(f"| {rn}{mark} | " + " | ".join(f"{sm[m]:.3f}" for m in METRICS)
                       + f" | {dh:.3f} | {dr:.3f} | {d.mean():+.3f} | [{ci_q[0]:+.3f}, {ci_q[1]:+.3f}] | "
                       f"[{ci_t[0]:+.3f}, {ci_t[1]:+.3f}] | {int((d > 0).sum())} / {int((d < 0).sum())} |")
        return out

    def sibling_block(idx_by: dict[str, list[int]]) -> list[str]:
        out = ["", "## same-name siblings: rows whose gold table has siblings (reporting only)", "",
               "| golden | n | run | Hit@5 | sibling above gold |", "| --- | --- | --- | --- | --- |"]
        for name, idx in idx_by.items():
            s = sets[name]
            rows = []
            for i in idx:
                gold = {k for g in s["groups"][i] for k in g}
                sib = {k for g in gold if g in info for k in siblings[info[g]["name"]]} - gold
                if sib:
                    rows.append((i, gold, sib))
            for rn in shown:
                conf = 0
                for i, gold, sib in rows:
                    pos = {k: r for r, k in enumerate(res[name][rn]["hybrid"][i])}
                    conf += min((pos[k] for k in sib if k in pos), default=10**6) < \
                        min((pos[k] for k in gold if k in pos), default=10**6)
                h = np.mean([hit5(name, rn, i) for i, _, _ in rows]) if rows else 0.0
                out.append(f"| {name} | {len(rows)} | {rn} | {h:.3f} | {conf / max(len(rows), 1):.3f} |")
        return out

    def qtype_block() -> list[str]:
        s = sets["retrieval"]
        out = ["", "## retrieval_golden Hit@5 by qtype (all answerable rows; reporting only)", "",
               "| qtype | n | " + " | ".join(shown) + " |", "| --- | --- | " + " | ".join("---" for _ in shown) + " |"]
        by: dict[str, list[int]] = defaultdict(list)
        for i, q in enumerate(s["qtype"]):
            by[q].append(i)
        for q, ix in sorted(by.items(), key=lambda kv: -len(kv[1])):
            out.append(f"| {q} | {len(ix)} | " + " | ".join(f"{np.mean([hit5('retrieval', rn, i) for i in ix]):.3f}"
                                                         for rn in shown) + " |")
        return out

    rid = time.strftime("%Y%m%d-%H%M%S")
    metas = {ck.stem: json.loads(ck.with_suffix(".json").read_text(encoding="utf-8")) for ck in ckpts
             if ck.with_suffix(".json").exists()}
    idx_r = {sp: [i for i, x in enumerate(rs["split"]) if x == sp] for sp in ("dev", "test")}
    ts = sets["retrieval"]["split"]
    idx_t = {"all": list(range(len(ts))), "dev": [i for i, x in enumerate(ts) if x == "dev"],
             "test": [i for i, x in enumerate(ts) if x == "test"]}
    lines = [f"# Plan 29: fine-tuned bge-m3 dense + V0 BM25 ({args.model})", "",
             f"eval run {rid}; training run {run_dir.name}; tables {sha[:8]}",
             "chosen on routing_v0 dev (hybrid Hit@5, then MRR@10): " + ", ".join(f"{r} **{c}**" for r, c in chosen.items()),
             "", "training: " + "; ".join(f"{k}: rows {m.get('rows')}, loss {m.get('loss_mean', 0):.4f}, "
                                          f"{m.get('seconds')}s" for k, m in metas.items())]
    lines += block("routing", idx_r["dev"], "routing_v0 dev (choice made here)")
    lines += block("routing", idx_r["test"], "routing_v0 test")
    for sp, ix in idx_t.items():
        lines += block("retrieval", ix, f"retrieval_golden {sp} (answerable; test was looked at before, exploratory)")
    lines += sibling_block({"routing": idx_r["test"], "retrieval": idx_t["all"]})
    lines += qtype_block()
    report = "\n".join(lines) + "\n"
    print(report)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / f"ft_eval_{rid}_summary.md"
    ranks_path = OUT_DIR / f"ft_eval_{rid}_ranks.jsonl"
    summary_path.write_text(report, encoding="utf-8")
    with ranks_path.open("w", encoding="utf-8") as f:
        for name, s in sets.items():
            for i, qid in enumerate(s["ids"]):
                row = {"golden": name, "id": qid, "split": s["split"][i], "question": s["items"][i]["question"]}
                row |= {rn: res[name][rn]["ranks"]["hybrid"][i] for rn in names}
                row |= {f"dense {rn}": res[name][rn]["ranks"]["dense"][i] for rn in names}
                row |= {f"top1 {rn}": res[name][rn]["hybrid"][i][0] for rn in shown}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"-> {summary_path}, {ranks_path}")


if __name__ == "__main__":
    main()
