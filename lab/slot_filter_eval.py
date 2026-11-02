"""Plan 42 A: filter tables by the question's year / region-level slots first, then run the hybrid search on what is left.

Slots and table facts are plan 37's rules (s1, lab/slot_rerank_eval.py). A table passes a slot when it meets it there
(year in periods, region level at least the asked one). Then dense top 50 + BM25 top 50 over the passing tables, RRF c=60.
Settings (all compared, chosen on dev):
    filter      year / region / year+region
    guard       none / S1 (fewer than MIN_POOL tables pass -> drop region, then year) / S1+S2 (S2: a year one past
                the table's last period also passes, for tables not yet updated)
    bm25        full (scores from the whole index, kept tables only) / reindex (Okapi idf and average length from the
                kept tables only; a negative idf is floored at 0.25 x the mean positive idf of the question's tokens)
    soft        after the filter, plan 37's soft rescoring (year+region, beta 0.01 gamma 0.005, top 50); it only
                changes rows where a guard dropped a slot
Baselines: F0 and the current best (F0 + soft rescoring, plan 37 rev2). Choice: Hit@1 + Hit@5 pooled over retrieval dev
+ routing dev, both dev Hit@5 within 2pt of the current best, and gold kept >= 98% on the dev rows where the filter
fires. Judged on all answerable retrieval_golden rows against the current best (table-level bootstrap) and F0;
adopted only if gold kept >= 98% over every fired row of both goldens. No API, no GPU (unseen-e1 caches).

    python lab/slot_filter_eval.py --selftest
    python lab/slot_filter_eval.py
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from ft_train import FT_DIR
from lab_common import (
    DEPTH, METRICS, OUT_DIR, ROUTING, RETRIEVAL, RRF_C, bm25_tokenize, boot, docs, group_rank, load_records,
    query_text, read_jsonl, source_sha, summary,
)
from slot_rerank_eval import SELFTEST, make_extractor, meets, region_names, table_facts

FILTERS = {"year": ("year",), "region": ("region",), "year+region": ("year", "region")}
GUARDS = ("none", "S1", "S1+S2")
MIN_POOL = 20
KEEP_MIN = 0.98  # gold kept on fired rows, below this a setting is not adopted
BETA, GAMMA, N = 0.01, 0.005, 50  # plan 37 rev2 soft rescoring, fixed
SOFT_SLOTS = ("year", "region")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None, help="fine-tuning run under data/lab/ft (default: latest)")
    p.add_argument("--max-labels", type=int, default=60)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    records, sha, tokenize = load_records(), source_sha(), bm25_tokenize()
    names = region_names(records)
    slots = make_extractor(names)
    if args.selftest:
        for s in SELFTEST:
            sl = {n: v for n, v in slots(s).items() if n in SOFT_SLOTS}
            print(f"  {s} -> {sl}")
        return

    runs = sorted(d for d in FT_DIR.iterdir() if d.is_dir()) if FT_DIR.exists() else []
    run_dir = FT_DIR / args.run if args.run else (runs[-1] if runs else None)
    assert run_dir and run_dir.exists(), "no fine-tuning run under data/lab/ft"
    cache = {part: run_dir / f"unseen-e1__{part}.npy" for part in ("docs", "retrieval", "routing")}
    missing = [str(c) for c in cache.values() if not c.exists()]
    assert not missing, f"need {missing} (run lab/fusion_eval.py once first)"

    ds = docs(records, "norm", args.max_labels)
    keys = [d["key"] for d in ds]
    kidx = {k: j for j, k in enumerate(keys)}
    bm = BM25Okapi([tokenize(d["bm25"]) for d in ds])
    facts = {r["info"]["key"]: table_facts(r, names) for r in records}
    last = {r["info"]["key"]: r.get("end") for r in records}
    doc_vec = np.load(cache["docs"])
    assert len(doc_vec) == len(keys)

    rt_all, rg = read_jsonl(RETRIEVAL), read_jsonl(ROUTING)
    q_rt, q_rg = np.load(cache["retrieval"]), np.load(cache["routing"])
    assert len(q_rt) == len(rt_all) and len(q_rg) == len(rg), "question caches do not match the goldens"
    keep = [i for i, it in enumerate(rt_all) if it.get("answerable", True) and it.get("required")]
    sets = {
        "retrieval": {"items": [rt_all[i] for i in keep], "vec": q_rt[keep], "groups": [rt_all[i]["required"] for i in keep],
                      "ids": [rt_all[i]["qid"] for i in keep]},
        "routing": {"items": rg, "vec": q_rg, "groups": [[[Path(it["table"]).stem, *it.get("alt_tables", [])]] for it in rg],
                    "ids": [it["id"] for it in rg]},
    }

    def passes(k: str, n: str, v, lenient: bool) -> bool:
        if meets(facts[k], n, v):
            return True
        if lenient and n == "year" and last.get(k) is not None:
            return any(int(y) == int(last[k]) + 1 for y in v)
        return False

    def pool(sl: dict, use: tuple, guard: str, memo: dict) -> tuple[np.ndarray, tuple]:
        """Boolean mask over keys and the slots actually applied (a guard may drop some)."""
        act = [n for n in use if n in sl]
        lenient = guard == "S1+S2"
        while True:
            mask = np.ones(len(keys), bool)
            for n in act:
                if (n, lenient) not in memo:
                    memo[n, lenient] = np.array([passes(k, n, sl[n], lenient) for k in keys])
                mask &= memo[n, lenient]
            if guard == "none" or mask.sum() >= MIN_POOL or not act:
                return mask, tuple(act)
            act = act[:-1] if act[-1] == "region" or len(act) == 1 else [a for a in act if a != "region"]

    dlen = np.asarray(bm.doc_len, float)

    def reindexed(qt: list[str], uq: list[str], tf: np.ndarray, dl: np.ndarray) -> list[int]:
        """Okapi BM25 (k1 1.5, b 0.75, as rank_bm25) with statistics from the kept tables only -> top positions."""
        n, avg = len(dl), dl.mean()
        df = (tf > 0).sum(axis=1)
        idf = np.log(n - df + 0.5) - np.log(df + 0.5)
        pos = idf[idf > 0]
        idf = np.where(idf < 0, 0.25 * (pos.mean() if len(pos) else 0.0), idf)
        cnt = np.array([qt.count(t) for t in uq], float)  # repeated query tokens count again, as in rank_bm25
        sc = ((idf * cnt)[:, None] * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * dl / avg))).sum(axis=0)
        return [j for j in np.argsort(-sc, kind="stable")[:DEPTH] if sc[j] > 0]

    def fuse(dense: list[str], bml: list[str]) -> list[tuple[str, float]]:
        s: dict[str, float] = defaultdict(float)
        for lst in (dense, bml):
            for r, k in enumerate(lst, 1):
                s[k] += 1.0 / (RRF_C + r)
        return sorted(s.items(), key=lambda kv: -kv[1])

    def soft(scored: list[tuple[str, float]], sl: dict) -> list[str]:
        act = [(n, v) for n, v in sl.items() if n in SOFT_SLOTS]
        if not act:
            return [k for k, _ in scored]
        head, tail = scored[:N], scored[N:]
        new = [(k, sc - BETA * sum(not meets(facts[k], n, v) for n, v in act) + GAMMA * sum(meets(facts[k], n, v) for n, v in act))
               for k, sc in head]
        new.sort(key=lambda kv: -kv[1])
        return [k for k, _ in new] + [k for k, _ in tail]

    configs = ["F0", "current best (soft)"]
    for f in FILTERS:
        for g in GUARDS:
            for b in ("full", "reindex"):
                for s_ in ("", " +soft"):
                    configs.append(f"filter {f} | {g} | bm25 {b}{s_}")
    ranks = {name: {c: [] for c in configs} for name in sets}
    kept = {name: {c: [] for c in configs} for name in sets}  # None = filter did not fire, else gold kept?
    pools = {name: {c: [] for c in configs} for name in sets}
    t0 = time.time()
    for name, s in sets.items():
        s["split"] = [it["split"] for it in s["items"]]
        s["qtype"] = [it.get("qtype", "?") for it in s["items"]]
        s["cluster"] = [g[0][0] for g in s["groups"]]
        s["slots"] = [{n: v for n, v in slots(it["question"]).items() if n in SOFT_SLOTS} for it in s["items"]]
        for i, (it, qv) in enumerate(zip(s["items"], s["vec"])):
            qn = query_text(it["question"], "norm")
            qt = tokenize(qn)
            dsc = doc_vec @ qv
            dorder = np.argsort(-dsc, kind="stable")
            bsc = np.asarray(bm.get_scores(qt))
            border = [j for j in np.argsort(-bsc, kind="stable") if bsc[j] > 0]
            sl, g = s["slots"][i], s["groups"][i]
            full_dense, full_bm = [keys[j] for j in dorder[:DEPTH]], [keys[j] for j in border[:DEPTH]]
            f0 = fuse(full_dense, full_bm)
            ranks[name]["F0"].append(group_rank(g, [k for k, _ in f0]))
            ranks[name]["current best (soft)"].append(group_rank(g, soft(f0, sl)))
            for c in ("F0", "current best (soft)"):
                kept[name][c].append(None)
                pools[name][c].append(len(keys))
            reidx: dict[tuple, list[str]] = {}
            memo: dict = {}
            uq = sorted(set(qt))
            tf = np.array([[bm.doc_freqs[j].get(t, 0) for j in range(len(keys))] for t in uq], float).reshape(len(uq), len(keys))
            for c in configs[2:]:
                fname, guard, btxt = [x.strip() for x in c[len("filter "):].split("|")]
                bmode = btxt.split()[1]
                mask, applied = pool(sl, FILTERS[fname], guard, memo)
                dense = [keys[j] for j in dorder if mask[j]][:DEPTH]
                if bmode == "full" or not applied:
                    bml = [keys[j] for j in border if mask[j]][:DEPTH]
                else:
                    sel = np.flatnonzero(mask)
                    tag = sel.tobytes()
                    if tag not in reidx:
                        reidx[tag] = [keys[sel[j]] for j in reindexed(qt, uq, tf[:, sel], dlen[sel])]
                    bml = reidx[tag]
                scored = fuse(dense, bml)
                lst = soft(scored, sl) if c.endswith("+soft") else [k for k, _ in scored]
                ranks[name][c].append(group_rank(g, lst))
                fired = bool(applied)
                kept[name][c].append(all(any(k in kidx and mask[kidx[k]] for k in grp) for grp in g) if fired else None)
                pools[name][c].append(int(mask.sum()))
        print(f"  {name}: {len(s['items'])} questions, {time.time() - t0:.0f}s")

    def hit(name: str, c: str, i: int, k: int) -> float:
        r = ranks[name][c][i]
        return float(r is not None and r <= k)

    def idx(name: str, split: str) -> list[int]:
        return [i for i, sp in enumerate(sets[name]["split"]) if split == "all" or sp == split]

    def mean_hit(name: str, c: str, ix: list[int], k: int) -> float:
        return float(np.mean([hit(name, c, i, k) for i in ix])) if ix else 0.0

    def keep_rate(c: str, rows: dict[str, list[int]]) -> tuple[int, int]:
        fired = [kept[n][c][i] for n, ix in rows.items() for i in ix if kept[n][c][i] is not None]
        return sum(fired), len(fired)

    BEST = "current best (soft)"
    dev_t, dev_r = idx("retrieval", "dev"), idx("routing", "dev")
    dev_rows = {"retrieval": dev_t, "routing": dev_r}
    all_rows = {"retrieval": idx("retrieval", "all"), "routing": idx("routing", "all")}
    guard_t = mean_hit("retrieval", BEST, dev_t, 5) - 0.02
    guard_r = mean_hit("routing", BEST, dev_r, 5) - 0.02

    def dev_score(c: str) -> float:
        h = lambda name, ix: sum(hit(name, c, i, 1) + hit(name, c, i, 5) for i in ix)
        return (h("retrieval", dev_t) + h("routing", dev_r)) / (len(dev_t) + len(dev_r))

    def kept_ok(c: str, rows) -> bool:
        a, b = keep_rate(c, rows)
        return b == 0 or a / b >= KEEP_MIN

    choosable = configs[2:]
    ok_cfg = [c for c in choosable if mean_hit("retrieval", c, dev_t, 5) >= guard_t
              and mean_hit("routing", c, dev_r, 5) >= guard_r and kept_ok(c, dev_rows)]
    order = {c: n for n, c in enumerate(choosable)}
    pick = max(ok_cfg, key=lambda c: (dev_score(c), -order[c])) if ok_cfg else None
    best_of = {}
    for f in FILTERS:
        cand = [c for c in choosable if c.startswith(f"filter {f} |")]
        best_of[f] = max(cand, key=lambda c: (dev_score(c), -order[c]))
    shown = list(dict.fromkeys(["F0", BEST, *([pick] if pick else []), *best_of.values()]))
    rng = np.random.default_rng(0)

    def delta(name: str, c: str, ref: str, ix: list[int], k: int) -> str:
        d = np.array([hit(name, c, i, k) - hit(name, ref, i, k) for i in ix])
        q_ = boot(d, None, rng)
        t_ = boot(d, [sets[name]["cluster"][i] for i in ix], rng)
        return (f"{d.mean():+.3f} [{q_[0]:+.3f}, {q_[1]:+.3f}] [{t_[0]:+.3f}, {t_[1]:+.3f}] "
                f"{int((d > 0).sum())}/{int((d < 0).sum())}")

    def block(name: str, ix: list[int], title: str) -> list[str]:
        out = ["", f"## {title} (n={len(ix)})", "",
               "Δ cells vs the current best (soft): mean, 95% question-level, 95% table-level, wins/losses. "
               "gold kept = rows where the filter fired and every needed table passed.", "",
               "| setting | " + " | ".join(METRICS) + " | ΔHit@1 | ΔHit@5 | gold kept | tables left (median) |",
               "| --- | " + " | ".join("---" for _ in METRICS) + " | --- | --- | --- | --- |"]
        for c in shown:
            sm = summary([ranks[name][c][i] for i in ix])
            cells = ["-", "-"] if c == BEST else [delta(name, c, BEST, ix, 1), delta(name, c, BEST, ix, 5)]
            a, b = keep_rate(c, {name: ix})
            kp = "-" if b == 0 else f"{a}/{b} ({a / b:.0%})"
            out.append(f"| {c}{' (dev choice)' if c == pick else ''} | " + " | ".join(f"{sm[m]:.3f}" for m in METRICS)
                       + " | " + " | ".join(cells) + f" | {kp} | {int(np.median([pools[name][c][i] for i in ix]))} |")
        return out

    def verdict() -> list[str]:
        out = ["", "## verdict on retrieval_golden answerable (table-level 95% lower bound > 0 → effect; gold kept ≥ "
               f"{KEEP_MIN:.0%} on every fired row of both goldens)", "",
               f"- guardrails for the dev choice: retrieval dev Hit@5 ≥ {guard_t:.3f}, routing dev Hit@5 ≥ {guard_r:.3f}, "
               f"gold kept on dev fired rows ≥ {KEEP_MIN:.0%}; {len(ok_cfg)}/{len(choosable)} settings pass",
               "- retrieval_golden test has been looked at many times → exploratory; confirm on the new check set (plan 40)"]
        if not pick:
            return out + ["- no filter setting passed the dev guardrails → **not adopted**"]
        ix = idx("retrieval", "all")
        a, b = keep_rate(pick, all_rows)
        out.append(f"- dev choice: **{pick}**; gold kept over all fired rows {a}/{b} ({a / max(b, 1):.1%})")
        for ref in (BEST, "F0"):
            parts, eff = [], False
            for k in (1, 5):
                d = np.array([hit("retrieval", pick, i, k) - hit("retrieval", ref, i, k) for i in ix])
                lo, hi = boot(d, [sets["retrieval"]["cluster"][i] for i in ix], rng)
                parts.append(f"Hit@{k} {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
                eff = eff or lo > 0
            out.append(f"- vs {ref} → " + "; ".join(parts) + (" → **effect**" if eff else " → no clear effect"))
        if b and a / b < KEEP_MIN:
            out.append(f"- gold kept {a / b:.1%} < {KEEP_MIN:.0%} → **not adopted** whatever the Hit numbers")
        return out

    def lost() -> list[str]:
        c = pick or best_of["year+region"]
        out = ["", f"## rows whose gold table was filtered out ({c})", "", "| golden | id | question | slots | gold |",
               "| --- | --- | --- | --- | --- |"]
        for name, s in sets.items():
            for i in range(len(s["items"])):
                if kept[name][c][i] is False:
                    sl = ", ".join(f"{n}={sorted(v) if isinstance(v, set) else v}" for n, v in s["slots"][i].items())
                    out.append(f"| {name} | {s['ids'][i]} | {s['items'][i]['question']} | {sl} | "
                               f"{[k for grp in s['groups'][i] for k in grp][:3]} |")
        return out if len(out) > 5 else out + ["| - | - | none | - | - |"]

    def changed() -> list[str]:
        c = pick or best_of["year+region"]
        s = sets["retrieval"]
        out = ["", f"## changed rows (retrieval answerable, current best → {c})", "",
               "| qid | split | qtype | question | current best | new |", "| --- | --- | --- | --- | --- | --- |"]
        for i in range(len(s["items"])):
            a, b = ranks["retrieval"][BEST][i], ranks["retrieval"][c][i]
            if a != b:
                out.append(f"| {s['ids'][i]} | {s['split'][i]} | {s['qtype'][i]} | {s['items'][i]['question']} | {a} | {b} |")
        return out

    def selection() -> list[str]:
        out = ["", "## every setting on the dev sets", "",
               "| setting | retrieval dev Hit@1 | retrieval dev Hit@5 | routing dev Hit@1 | routing dev Hit@5 | pooled | gold kept (dev fired) | guardrail |",
               "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for c in configs:
            a, b = keep_rate(c, dev_rows)
            g = "baseline" if c in ("F0", BEST) else ("ok" if c in ok_cfg else "fail")
            out.append(f"| {c} | {mean_hit('retrieval', c, dev_t, 1):.3f} | {mean_hit('retrieval', c, dev_t, 5):.3f} | "
                       f"{mean_hit('routing', c, dev_r, 1):.3f} | {mean_hit('routing', c, dev_r, 5):.3f} | {dev_score(c):.3f} | "
                       + ("-" if b == 0 else f"{a}/{b}") + f" | {g} |")
        return out

    rid = time.strftime("%Y%m%d-%H%M%S")
    lines = ["# Plan 42 A: slot filter first, then hybrid search", "",
             f"run {rid}; fine-tuning run {run_dir.name}; tables {sha[:8]}; slot rules s1 (plan 37); min pool {MIN_POOL}; "
             f"gold kept guardrail {KEEP_MIN:.0%}",
             "Baselines: F0 (unseen-e1 + BM25, RRF c=60) and the current best (F0 + soft year+region rescoring, β 0.01 γ 0.005)."]
    lines += verdict()
    for sp in ("all", "test", "dev"):
        lines += block("retrieval", idx("retrieval", sp), f"retrieval_golden {sp} (answerable)")
    lines += block("routing", dev_r, "routing_v0 dev")
    lines += block("routing", idx("routing", "test"), "routing_v0 test (reference)")
    lines += lost()
    lines += changed()
    lines += selection()
    report = "\n".join(lines) + "\n"
    print(report)
    out = OUT_DIR / f"slot_filter_eval_{rid}_summary.md"
    out.write_text(report, encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
