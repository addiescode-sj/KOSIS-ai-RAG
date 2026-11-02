"""Plan 43 step 1 (PC): training groups for fine-tuning the stage-1 re-ranker (bge-reranker-v2-m3).

For every synthetic question of the unseen regime (data/lab/synth/synth_v0.jsonl minus the rows whose table or alt
table is a golden answer table, the same split as plan 29) it runs the current best search on the question:
    F0 = unseen-e1 dense top 50 + BM25 top 50, RRF c=60, then plan 37 rev2 soft year+region rescoring of the top 50
and keeps two negative pools:
    top   the current-best top 50 in order, minus the right tables (table + alt_tables) and minus same-name siblings
          the question cannot tell apart (plan 29 rule: no year in the question, or a year inside the sibling's period)
    sib   same-name siblings + the file's hard_negatives, with the same two filters
Writes (copy both to Colab, MyDrive/kosis-ft/data/lab/rrft/):
    data/lab/rrft/train_unseen.jsonl   one row per question: q (normalized), table, ok, top, sib, gold_rank
    data/lab/rrft/passages.json        table key -> re-ranker text (n2-L60 embed text, the same text the
                                       two-stage evaluation gives the re-ranker)
    data/lab/rrft/train_unseen.stats.json
The question vectors of unseen-e1 are cached next to its other caches (data/lab/ft/<run>/unseen-e1__synth.npy).

    python lab/rerank_ft_data.py --selftest
    python lab/rerank_ft_data.py
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict

import numpy as np
from rank_bm25 import BM25Okapi

from ft_train import FT_DIR, eval_tables, indistinguishable, load_ft, table_info
from lab_common import (
    DEFAULT_MODEL, DEPTH, OUT_DIR, RRF_C, SYNTH, bm25_tokenize, docs, encode, load_records, query_text, read_jsonl,
    source_sha, top,
)
from slot_rerank_eval import make_extractor, meets, region_names, table_facts

BETA, GAMMA, N = 0.01, 0.005, 50  # plan 37 rev2 soft rescoring, fixed (same as two_stage_rerank_eval.py)
SOFT_SLOTS = ("year", "region")
RRFT = OUT_DIR / "rrft"
STAGE1 = "unseen-e1"


def keep_negative(q: str, gold: str, k: str, ok: set[str], info: dict) -> bool:
    return k in info and k not in ok and not indistinguishable(q, gold, k, info)


def siblings_by_name(info: dict) -> dict[str, list[str]]:
    by: dict[str, list[str]] = defaultdict(list)
    for k, v in info.items():
        by[v["name"]].append(k)
    return by


def selftest() -> None:
    info = {"A": {"name": "인구", "start": 2000, "end": 2010}, "B": {"name": "인구", "start": 2011, "end": 2020},
            "C": {"name": "인구", "start": 2015, "end": 2023}, "D": {"name": "가구", "start": 2000, "end": 2023},
            "E": {"name": "인구", "start": None, "end": None}}
    q = "2018년 인구"
    ok = {"B"}
    cases = [  # (negative, expected keep)
        ("A", True),   # sibling, 2018 outside its period -> the question tells them apart
        ("B", False),  # the gold itself
        ("C", False),  # sibling, 2018 inside its period -> false negative, dropped
        ("D", True),   # other name
        ("E", False),  # sibling without a period -> cannot tell, dropped
        ("Z", False),  # unknown table
    ]
    bad = [(k, want) for k, want in cases if keep_negative(q, "B", k, ok, info) != want]
    no_year = keep_negative("인구 보여줘", "B", "A", ok, info)
    assert not bad, bad
    assert no_year is False, "no year -> same-name sibling must be dropped"
    sib = siblings_by_name(info)
    assert sorted(sib["인구"]) == ["A", "B", "C", "E"], sib
    print(f"selftest ok ({len(cases) + 2} checks)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None, help="embedding fine-tuning run under data/lab/ft (default: latest)")
    p.add_argument("--max-labels", type=int, default=60)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="only the first N kept questions (0 = all)")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        selftest()
        return

    runs = sorted(d for d in FT_DIR.iterdir() if d.is_dir()) if FT_DIR.exists() else []
    run_dir = FT_DIR / args.run if args.run else (runs[-1] if runs else None)
    assert run_dir and (run_dir / f"{STAGE1}.pt").exists(), f"need data/lab/ft/<run>/{STAGE1}.pt"
    doc_cache = run_dir / f"{STAGE1}__docs.npy"
    assert doc_cache.exists(), f"need {doc_cache} (run lab/fusion_eval.py once first)"

    records, sha, tokenize = load_records(), source_sha(), bm25_tokenize()
    info = table_info(records)
    names = region_names(records)
    slots = make_extractor(names)
    facts = {r["info"]["key"]: table_facts(r, names) for r in records}
    ds = docs(records, "norm", args.max_labels)
    keys = [d["key"] for d in ds]
    text_of = {d["key"]: d["embed"] for d in ds}
    doc_vec = np.load(doc_cache)
    assert len(doc_vec) == len(keys), "doc cache does not match tables.jsonl"
    bm = BM25Okapi([tokenize(d["bm25"]) for d in ds])
    sib_of = siblings_by_name(info)

    held = eval_tables()
    synth = [r for r in read_jsonl(SYNTH)
             if r["table"] not in held and not any(a in held for a in r.get("alt_tables", []))]
    if args.limit:
        synth = synth[: args.limit]
    qn = [query_text(r["question"], "norm") for r in synth]

    # unseen-e1 question vectors (cached; only the full set is cached)
    qcache = run_dir / f"{STAGE1}__synth.npy"
    if not args.limit and qcache.exists() and len(np.load(qcache)) == len(qn):
        q_vec = np.load(qcache)
        print(f"  cache {qcache.name}")
    else:
        model = load_ft(DEFAULT_MODEL, run_dir / f"{STAGE1}.pt", 512)
        q_vec = np.asarray(encode(model, DEFAULT_MODEL, qn, "query", args.batch), dtype=np.float32)
        del model
        if not args.limit:
            np.save(qcache, q_vec)
            print(f"  embedded -> {qcache.name}")

    def current_best(i: int) -> list[str]:
        dense = top(doc_vec @ q_vec[i], keys)
        sc = np.asarray(bm.get_scores(tokenize(qn[i])))
        bml = [keys[j] for j in np.argsort(-sc, kind="stable")[:DEPTH] if sc[j] > 0]
        s: dict[str, float] = defaultdict(float)
        for lst in (dense, bml):
            for r, k in enumerate(lst, 1):
                s[k] += 1.0 / (RRF_C + r)
        scored = sorted(s.items(), key=lambda kv: -kv[1])
        sl = [(n, v) for n, v in slots(synth[i]["question"]).items() if n in SOFT_SLOTS]
        if not sl:
            return [k for k, _ in scored]
        head, tail = scored[:N], scored[N:]
        new = [(k, v - BETA * sum(not meets(facts[k], n, x) for n, x in sl) + GAMMA * sum(meets(facts[k], n, x) for n, x in sl))
               for k, v in head]
        new.sort(key=lambda kv: -kv[1])
        return [k for k, _ in new] + [k for k, _ in tail]

    rows, dropped = [], Counter()
    t0 = time.perf_counter()
    for i, r in enumerate(synth):
        gold, q = r["table"], qn[i]
        ok = {gold, *r.get("alt_tables", [])}
        ranked = current_best(i)[:N]
        gold_rank = ranked.index(gold) + 1 if gold in ranked else None
        top_pool = [k for k in ranked if k not in ok]
        top_keep = [k for k in top_pool if keep_negative(q, gold, k, ok, info)]
        sib_pool = list(dict.fromkeys([*sib_of.get(info[gold]["name"], []), *r.get("hard_negatives", [])]))
        sib_pool = [k for k in sib_pool if k in info and k not in ok]
        sib_keep = [k for k in sib_pool if keep_negative(q, gold, k, ok, info)]
        dropped["top"] += len(top_pool) - len(top_keep)
        dropped["sib"] += len(sib_pool) - len(sib_keep)
        rows.append({"id": r["id"], "q": q, "table": gold, "ok": sorted(ok), "top": top_keep, "sib": sib_keep,
                     "gold_rank": gold_rank})
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(synth)} ({time.perf_counter() - t0:.0f}s)")

    gr = [x["gold_rank"] for x in rows]
    stats = {
        "built": time.strftime("%Y%m%d-%H%M%S"), "regime": "unseen", "embedding_run": run_dir.name, "tables_sha": sha[:8],
        "rows": len(rows), "tables": len({x["table"] for x in rows}), "held_tables": len(held),
        "gold_rank_1": sum(g == 1 for g in gr) / len(rows), "gold_in_top5": sum(g is not None and g <= 5 for g in gr) / len(rows),
        "gold_in_top50": sum(g is not None for g in gr) / len(rows),
        "mean_top_negatives": float(np.mean([len(x["top"]) for x in rows])),
        "mean_sib_negatives": float(np.mean([len(x["sib"]) for x in rows])),
        "rows_without_sib": sum(not x["sib"] for x in rows),
        "false_negatives_dropped": dict(dropped), "limit": args.limit,
    }
    RRFT.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit{args.limit}" if args.limit else ""
    out = RRFT / f"train_unseen{suffix}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    (RRFT / "passages.json").write_text(json.dumps(text_of, ensure_ascii=False), encoding="utf-8")
    (RRFT / f"train_unseen{suffix}.stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    print(f"-> {out}\n-> {RRFT / 'passages.json'}\n(next: copy both to Colab MyDrive/kosis-ft/data/lab/rrft/)")


if __name__ == "__main__":
    main()
