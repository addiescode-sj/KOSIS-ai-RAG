"""Plan 43 step 3 (PC): evaluate the fine-tuned stage-1 re-ranker (no API).

Input: the current best top 50 = F0 (unseen-e1 dense + BM25, RRF c=60) then plan 37 rev2 soft year+region rescoring.
Stage 1 re-scores the input top 50 with each model:
    base      BAAI/bge-reranker-v2-m3 as downloaded
    e1, e2..  the same model with the weights of data/lab/rrft/<run>/unseen-e<n>.pt put back (rerank_ft_train.py)
and two ways to use the score:
    S1a  re-ranker score only
    S1b  RRF of the re-ranker order and the input order (1/(60+r_rerank) + 1/(60+r_input))
Choice (same as plan 37 rev2): pooled dev = retrieval dev 29 + routing dev 104, Hit@1 + Hit@5 summed over questions;
a setting may not lose more than 2pt Hit@5 on either dev set; the input wins ties.
Judged on retrieval_golden answerable (89) against the input, table-level bootstrap 95% (lower bound > 0 → effect).
Also: routing test, every setting on every split, gold moves for the questions the input had at 6..50, qtype Hit@5,
seconds per question. Scores are cached in data/lab/rrft/<run>/stage1__<model>__<tag>.json.

    python lab/rerank_ft_eval.py                 # latest run under data/lab/rrft
    python lab/rerank_ft_eval.py --run 20261101-2300xx --batch 8
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

from ft_train import FT_DIR
from lab_common import (
    DEPTH, METRICS, OUT_DIR, RETRIEVAL, ROUTING, RRF_C, bm25_tokenize, boot, docs, group_rank, load_records,
    query_text, read_jsonl, source_sha, summary, top,
)
from slot_rerank_eval import make_extractor, meets, region_names, table_facts

BASE = "BAAI/bge-reranker-v2-m3"
BETA, GAMMA, N = 0.01, 0.005, 50  # plan 37 rev2 soft rescoring, fixed
SOFT_SLOTS = ("year", "region")
RRFT = OUT_DIR / "rrft"


def load_cross_encoder(ckpt: Path | None, max_length: int):
    """Base re-ranker, optionally with the fine-tuned layers put back; fp16 on GPU after loading."""
    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ce = CrossEncoder(BASE, max_length=max_length, device=device)
    hf = getattr(ce, "model", None) or ce
    if ckpt is not None:
        sd = torch.load(ckpt, map_location="cpu")
        dtype = next(hf.parameters()).dtype
        res = hf.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=False)
        assert not res.unexpected_keys, f"weights do not fit the base model: {res.unexpected_keys[:3]}"
        print(f"  loaded {ckpt.name}: {len(sd)} tensors")
    if device == "cuda":
        hf.half()
    return ce


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None, help="re-ranker run under data/lab/rrft (default: latest)")
    p.add_argument("--emb-run", default=None, help="embedding run under data/lab/ft (default: latest)")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-labels", type=int, default=60)
    p.add_argument("--no-base", action="store_true", help="skip the untrained re-ranker")
    args = p.parse_args()

    runs = sorted(d for d in RRFT.iterdir() if d.is_dir() and list(d.glob("unseen-e*.pt"))) if RRFT.exists() else []
    run_dir = RRFT / args.run if args.run else (runs[-1] if runs else None)
    assert run_dir and run_dir.exists(), "no re-ranker run with unseen-e*.pt under data/lab/rrft"
    ckpts = sorted(run_dir.glob("unseen-e*.pt"), key=lambda x: int(x.stem.split("-e")[-1]))
    models: dict[str, Path | None] = {} if args.no_base else {"base": None}
    models.update({c.stem.split("-")[-1]: c for c in ckpts})  # e1, e2, ...
    meta = {c.stem.split("-")[-1]: json.loads(c.with_suffix(".json").read_text(encoding="utf-8"))
            for c in ckpts if c.with_suffix(".json").exists()}

    eruns = sorted(d for d in FT_DIR.iterdir() if d.is_dir()) if FT_DIR.exists() else []
    emb_dir = FT_DIR / args.emb_run if args.emb_run else (eruns[-1] if eruns else None)
    cache = {part: emb_dir / f"unseen-e1__{part}.npy" for part in ("docs", "retrieval", "routing")}
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
    doc_vec = np.load(cache["docs"])
    rt_all, rg = read_jsonl(RETRIEVAL), read_jsonl(ROUTING)
    q_rt, q_rg = np.load(cache["retrieval"]), np.load(cache["routing"])
    assert len(doc_vec) == len(keys) and len(q_rt) == len(rt_all) and len(q_rg) == len(rg), "caches do not match"

    def current_best(question: str, qv: np.ndarray) -> list[str]:
        qn = query_text(question, "norm")
        dense = top(doc_vec @ qv, keys)
        sc = np.asarray(bm.get_scores(tokenize(qn)))
        bml = [keys[j] for j in np.argsort(-sc, kind="stable")[:DEPTH] if sc[j] > 0]
        s: dict[str, float] = defaultdict(float)
        for lst in (dense, bml):
            for r, k in enumerate(lst, 1):
                s[k] += 1.0 / (RRF_C + r)
        scored = sorted(s.items(), key=lambda kv: -kv[1])
        sl = [(n, v) for n, v in slots(question).items() if n in SOFT_SLOTS]
        if not sl:
            return [k for k, _ in scored]
        head, tail = scored[:N], scored[N:]
        new = [(k, v - BETA * sum(not meets(facts[k], n, x) for n, x in sl) + GAMMA * sum(meets(facts[k], n, x) for n, x in sl))
               for k, v in head]
        new.sort(key=lambda kv: -kv[1])
        return [k for k, _ in new] + [k for k, _ in tail]

    keep = [i for i, it in enumerate(rt_all) if it.get("answerable", True) and it.get("required")]
    sets = {
        "retrieval": {"items": [rt_all[i] for i in keep], "vec": q_rt[keep], "groups": [rt_all[i]["required"] for i in keep]},
        "routing": {"items": rg, "vec": q_rg, "groups": [[[Path(it["table"]).stem, *it.get("alt_tables", [])]] for it in rg]},
    }
    for s in sets.values():
        s["qn"] = [query_text(it["question"], "norm") for it in s["items"]]
        s["input"] = [current_best(it["question"], v) for it, v in zip(s["items"], s["vec"])]
        s["split"] = [it["split"] for it in s["items"]]
        s["cluster"] = [g[0][0] for g in s["groups"]]

    # ---- stage 1 scores (cached per model) ----
    pairs = sorted({(q, k) for s in sets.values() for q, lst in zip(s["qn"], s["input"]) for k in lst[:N]})
    n_questions = sum(len(s["qn"]) for s in sets.values())
    scores: dict[str, dict[str, dict[str, float]]] = {}
    seconds: dict[str, float] = {}
    for m, ckpt in models.items():
        tag = hashlib.sha256(json.dumps([m, str(ckpt.stat().st_size) if ckpt else BASE, args.max_length, sha[:8],
                                         args.max_labels]).encode()).hexdigest()[:8]
        path = run_dir / f"stage1__{m}__{tag}.json"
        sc: dict[str, dict[str, float]] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        need = [(q, k) for q, k in pairs if k not in sc.get(q, {})]
        if need:
            ce = load_cross_encoder(ckpt, args.max_length)
            t0 = time.perf_counter()
            out = ce.predict([(q, text_of[k]) for q, k in need], batch_size=args.batch, show_progress_bar=True,
                             convert_to_numpy=True)
            seconds[m] = time.perf_counter() - t0
            for (q, k), v in zip(need, np.asarray(out, np.float64).reshape(-1)):
                sc.setdefault(q, {})[k] = float(v)
            path.write_text(json.dumps(sc, ensure_ascii=False), encoding="utf-8")
            del ce
            print(f"  {m}: scored {len(need)} pairs in {seconds[m]:.0f}s -> {path.name}")
        else:
            print(f"  {m}: cache {path.name}")
        scores[m] = sc

    def stage1(m: str, kind: str, q: str, lst: list[str]) -> list[str]:
        head, tail = lst[:N], lst[N:]
        sc = np.array([scores[m][q][k] for k in head])
        if kind == "S1a":
            order = np.argsort(-sc, kind="stable")
        else:
            rr = np.empty(len(head))
            rr[np.argsort(-sc, kind="stable")] = np.arange(1, len(head) + 1)
            order = np.argsort(-(1 / (RRF_C + rr) + 1 / (RRF_C + np.arange(1, len(head) + 1))), kind="stable")
        return [head[j] for j in order] + tail

    settings = ["input"] + [f"{m} {kind}" for m in models for kind in ("S1a", "S1b")]
    ranks: dict[str, dict[str, list]] = {name: {} for name in sets}
    for name, s in sets.items():
        for st in settings:
            if st == "input":
                ls = s["input"]
            else:
                m, kind = st.split()
                ls = [stage1(m, kind, q, lst) for q, lst in zip(s["qn"], s["input"])]
            ranks[name][st] = [group_rank(g, l) for g, l in zip(s["groups"], ls)]

    def hit(name: str, st: str, i: int, k: int) -> float:
        r = ranks[name][st][i]
        return float(r is not None and r <= k)

    def idx(name: str, split: str) -> list[int]:
        return [i for i, sp in enumerate(sets[name]["split"]) if split == "all" or sp == split]

    def mean_hit(name: str, st: str, ix: list[int], k: int) -> float:
        return float(np.mean([hit(name, st, i, k) for i in ix])) if ix else 0.0

    dev_t, dev_r = idx("retrieval", "dev"), idx("routing", "dev")

    def dev_score(st: str) -> float:
        h = lambda name, ix: sum(hit(name, st, i, 1) + hit(name, st, i, 5) for i in ix)
        return (h("retrieval", dev_t) + h("routing", dev_r)) / max(1, len(dev_t) + len(dev_r))

    ok = [st for st in settings
          if mean_hit("routing", st, dev_r, 5) >= mean_hit("routing", "input", dev_r, 5) - 0.02
          and mean_hit("retrieval", st, dev_t, 5) >= mean_hit("retrieval", "input", dev_t, 5) - 0.02]
    pick = max(ok, key=lambda st: (dev_score(st), st == "input"))
    rng = np.random.default_rng(0)

    def delta(name: str, st: str, ix: list[int], k: int) -> str:
        d = np.array([hit(name, st, i, k) - hit(name, "input", i, k) for i in ix])
        q_ = boot(d, None, rng)
        t_ = boot(d, [sets[name]["cluster"][i] for i in ix], rng)
        return (f"{d.mean():+.3f} [{q_[0]:+.3f}, {q_[1]:+.3f}] [{t_[0]:+.3f}, {t_[1]:+.3f}] "
                f"{int((d > 0).sum())}/{int((d < 0).sum())}")

    def block(name: str, split: str) -> list[str]:
        ix = idx(name, split)
        out = ["", f"## {name} {split} (n={len(ix)})", "",
               "Δ cells vs the input: mean, 95% question-level, 95% table-level, wins/losses.", "",
               "| setting | " + " | ".join(METRICS) + " | ΔHit@1 | ΔHit@5 | dev score |",
               "| --- | " + " | ".join("---" for _ in METRICS) + " | --- | --- | --- |"]
        for st in settings:
            sm = summary([ranks[name][st][i] for i in ix])
            cells = ["-", "-"] if st == "input" else [delta(name, st, ix, 1), delta(name, st, ix, 5)]
            flag = " (dev choice)" if st == pick else ("" if st in ok else " (guardrail)")
            out.append(f"| {st}{flag} | " + " | ".join(f"{sm[x]:.3f}" for x in METRICS) + " | " + " | ".join(cells)
                       + f" | {dev_score(st):.3f} |")
        return out

    def verdict() -> list[str]:
        out = ["", "## verdict on retrieval_golden answerable (table-level 95% lower bound > 0 → effect)", "",
               f"- dev choice (pooled retrieval dev + routing dev, Hit@1+Hit@5, input wins ties): **{pick}**",
               "- retrieval_golden test has been looked at many times → exploratory; confirm on the new check set (plan 40)"]
        if pick == "input":
            return out + ["- nothing beat the input on dev → **no change**"]
        ix, parts, eff = idx("retrieval", "all"), [], False
        for k in (1, 5):
            d = np.array([hit("retrieval", pick, i, k) - hit("retrieval", "input", i, k) for i in ix])
            lo, hi = boot(d, [sets["retrieval"]["cluster"][i] for i in ix], rng)
            parts.append(f"Hit@{k} {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
            eff = eff or lo > 0
        return out + [f"- {pick} vs input → " + "; ".join(parts) + (" → **effect**" if eff else " → no clear effect")]

    def training() -> list[str]:
        out = ["", "## training (from the .json next to each checkpoint)", "",
               "| epoch | loss | val acc before → after | val mean rank before → after | train q | seconds |",
               "| --- | --- | --- | --- | --- | --- |"]
        for m, x in meta.items():
            b, a = x.get("before", {}), x.get("after", {})
            out.append(f"| {m} | {x.get('loss_mean', 0):.4f} | {b.get('val_acc', 0):.3f} → {a.get('val_acc', 0):.3f} | "
                       f"{b.get('val_mean_rank', 0):.2f} → {a.get('val_mean_rank', 0):.2f} | {x.get('train_questions')} | {x.get('seconds')} |")
        return out

    def moves() -> list[str]:
        st = pick if pick != "input" else max(settings[1:], key=dev_score)
        s, out = sets["retrieval"], ["", f"## retrieval_golden: input rank 6..50 → {st}", "",
                                     "| qid | split | qtype | question | input | after |", "| --- | --- | --- | --- | --- | --- |"]
        up = down = 0
        for i, it in enumerate(s["items"]):
            a, b = ranks["retrieval"]["input"][i], ranks["retrieval"][st][i]
            if a is not None and 6 <= a <= 50:
                out.append(f"| {it['qid']} | {it['split']} | {it.get('qtype', '?')} | {it['question']} | {a} | {b} |")
                up += b is not None and b < a
                down += b is None or b > a
        return out + ["", f"up {up}, down {down}"]

    def qtypes() -> list[str]:
        s = sets["retrieval"]
        shown = list(dict.fromkeys(["input", pick, *(f"{m} S1a" for m in models)]))
        qt = sorted({it.get("qtype", "?") for it in s["items"]})
        out = ["", "## retrieval_golden Hit@5 by qtype", "", "| qtype | n | " + " | ".join(shown) + " |",
               "| --- | --- | " + " | ".join("---" for _ in shown) + " |"]
        for t in qt:
            ix = [i for i, it in enumerate(s["items"]) if it.get("qtype", "?") == t]
            out.append(f"| {t} | {len(ix)} | " + " | ".join(f"{mean_hit('retrieval', st, ix, 5):.3f}" for st in shown) + " |")
        return out

    rid = time.strftime("%Y%m%d-%H%M%S")
    speed = "; ".join(f"{m} {v / max(1, n_questions):.2f}s/question" for m, v in seconds.items()) or "all cached"
    lines = ["# Plan 43: fine-tuned stage-1 re-ranker", "",
             f"run {rid}; re-ranker run {run_dir.name}; models {', '.join(models)}; embedding run {emb_dir.name}; "
             f"max length {args.max_length}; tables {sha[:8]}; input = current best top {N} (F0 + soft year+region)",
             f"stage 1 speed: {speed}"]
    lines += verdict() + training()
    for name, split in (("retrieval", "all"), ("retrieval", "test"), ("retrieval", "dev"), ("routing", "test"), ("routing", "dev")):
        lines += block(name, split)
    lines += qtypes() + moves()
    report = "\n".join(lines) + "\n"
    print(report)
    out = OUT_DIR / f"rerank_ft_eval_{rid}_summary.md"
    out.write_text(report, encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
