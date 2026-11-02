"""Plan 36: colloquial -> official terms (reviewed lexicon d1h) on top of F0 + slot rescoring (plan 37 rev2).

Lexicon d1h, one global rule from the 11/01 review of 200 random pairs (no line edited by hand):
    keep a D1 pair only when the worker's confidence is "high" and the official term is under 7 characters
    (spaces removed); then plan 25's common filters (vocab_expand.build). d1hm is run the same way for reference only.
Baseline S0 = F0 (unseen-e1 + BM25, RRF c=60) then slots year+region, PB beta 0.01 gamma 0.005, top 50 (fixed).
A lexicon expression found in the normalized question adds the official terms the question lacks (plan 25 matcher);
the question's own words stay. Variants change only questions where the lexicon fires:
    B w   BM25 score = BM25(question) + w * BM25(added terms), dense on the question
    D     dense query = question + added terms (unseen-e1), BM25 on the question
    BD w  both
then the same slot rescoring. Choice: Hit@1 + Hit@5 pooled over retrieval dev + routing dev, both dev Hit@5 within
2pt of S0; nothing better than S0 -> S0. Judged on all answerable retrieval_golden rows against S0 (table-level
bootstrap). Expanded question vectors are embedded once with unseen-e1 and cached in data/lab/vocab_slot/.

    python lab/vocab_slot_eval.py --selftest
    python lab/vocab_slot_eval.py
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

from ft_train import FT_DIR, load_ft
from lab_common import (
    DEFAULT_MODEL, DEPTH, METRICS, OUT_DIR, ROUTING, RETRIEVAL, RRF_C, bm25_tokenize, boot, docs, encode, group_rank,
    load_records, query_text, read_jsonl, source_sha, summary, top,
)
from slot_rerank_eval import make_extractor, meets, region_names, table_facts
from vocab_eval import matcher
from vocab_expand import D1_FILE, VOCAB, build, squash

SLOTS = ("year", "region")
BETA, GAMMA, N = 0.01, 0.005, 50  # plan 37 rev2 choice, fixed here
MAX_TERM = 7  # official terms of 7+ characters were 1/8 ok in the review
WEIGHTS = (0.25, 0.5, 1.0)
LEXICONS = ("d1h", "d1hm")  # d1h is the candidate; d1hm is reported for reference and never chosen
QX_DIR = OUT_DIR / "vocab_slot"

SELFTEST = ["태어난 아이가 몇 명이야", "2023년 서울 혼자 사는 노인", "이사 많이 간 지역", "결혼한 사람 비율",
            "외국인 노동자 수", "애 낳는 나이 평균", "부모님이랑 같이 사는 집", "일 안 하는 청년", "초딩 수 2020년",
            "집 없는 사람", "아파트 사는 가구", "돌아가신 분 수", "혼인 건수 매달", "장애인 인구 시군구별"]


def d1_pairs(levels: tuple[str, ...], max_term: int | None) -> list[tuple[str, str]]:
    rows = [json.loads(line) for line in D1_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [(e, r["term"]) for r in rows if r["confidence"] in levels
            and (max_term is None or len(squash(r["term"])) < max_term) for e in r["colloquial"]]


def sha8(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None, help="fine-tuning run under data/lab/ft (default: latest)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-labels", type=int, default=60)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    lexicons, drops = {}, {}
    for name, levels, mt in (("d1h", ("high",), MAX_TERM), ("d1hm", ("high", "mid"), None)):
        lexicons[name], drops[name] = build(d1_pairs(levels, mt))
    path_d1h = VOCAB / "lexicon_d1h.json"
    path_d1h.write_text(json.dumps(lexicons["d1h"], ensure_ascii=False, indent=1), encoding="utf-8")
    match = {n: matcher(lex) for n, lex in lexicons.items()}
    if args.selftest:
        print(f"d1h {len(lexicons['d1h'])} expressions, d1hm {len(lexicons['d1hm'])}")
        for s in SELFTEST:
            q = query_text(s, "norm")
            print(f"  {s} -> d1h {match['d1h'](q)} | d1hm {match['d1hm'](q)}")
        return

    runs = sorted(d for d in FT_DIR.iterdir() if d.is_dir()) if FT_DIR.exists() else []
    run_dir = FT_DIR / args.run if args.run else (runs[-1] if runs else None)
    assert run_dir and run_dir.exists(), "no fine-tuning run under data/lab/ft"
    ckpt = run_dir / "unseen-e1.pt"
    cache = {part: run_dir / f"unseen-e1__{part}.npy" for part in ("docs", "retrieval", "routing")}
    missing = [str(c) for c in [ckpt, *cache.values()] if not c.exists()]
    assert not missing, f"need {missing} (run lab/fusion_eval.py once first)"

    records, sha, tokenize = load_records(), source_sha(), bm25_tokenize()
    names = region_names(records)
    slots = make_extractor(names)
    ds = docs(records, "norm", args.max_labels)
    keys = [d["key"] for d in ds]
    facts = {r["info"]["key"]: table_facts(r, names) for r in records}
    bm = BM25Okapi([tokenize(d["bm25"]) for d in ds])
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
    for s in sets.values():
        s["split"] = [it["split"] for it in s["items"]]
        s["qtype"] = [it.get("qtype", "?") for it in s["items"]]
        s["qn"] = [query_text(it["question"], "norm") for it in s["items"]]
        s["slots"] = [{n: v for n, v in slots(it["question"]).items() if n in SLOTS} for it in s["items"]]
        s["cluster"] = [g[0][0] for g in s["groups"]]
        s["match"] = {ln: [match[ln](q) for q in s["qn"]] for ln in LEXICONS}

    # ---- expanded question vectors (only rows where a lexicon fires), cached by text ----
    texts = sorted({f"{q} {' '.join(add)}" for s in sets.values() for ln in LEXICONS
                    for q, (_, add) in zip(s["qn"], s["match"][ln]) if add})
    tag = hashlib.sha256(("\n".join(texts) + run_dir.name).encode("utf-8")).hexdigest()[:8]
    npy = QX_DIR / f"unseen-e1__qx-{tag}.npy"
    if npy.exists() or not texts:
        qx = np.load(npy) if texts else np.zeros((0, doc_vec.shape[1]), np.float32)
        print(f"  cache {npy.name}" if texts else "  no expanded questions")
    else:
        model = load_ft(args.model, ckpt, args.max_tokens)
        model.max_seq_length = 512  # same as the cached question vectors
        qx = np.asarray(encode(model, args.model, texts, "query", args.batch), np.float32)
        QX_DIR.mkdir(parents=True, exist_ok=True)
        np.save(npy, qx)
        del model
        print(f"  embedded {len(texts)} expanded questions -> {npy.name}")
    xvec = dict(zip(texts, qx))

    def bm25_list(sc: np.ndarray) -> list[str]:
        return [keys[j] for j in np.argsort(-sc, kind="stable")[:DEPTH] if sc[j] > 0]

    def rescore(dense: list[str], bml: list[str], sl: dict, use_slots: bool) -> list[str]:
        s: dict[str, float] = defaultdict(float)
        for lst in (dense, bml):
            for r, k in enumerate(lst, 1):
                s[k] += 1.0 / (RRF_C + r)
        scored = sorted(s.items(), key=lambda kv: -kv[1])
        if not use_slots or not sl:
            return [k for k, _ in scored]
        head, tail = scored[:N], scored[N:]
        new = []
        for k, sc in head:
            m = sum(meets(facts[k], n, v) for n, v in sl.items())
            new.append((k, sc - BETA * (len(sl) - m) + GAMMA * m))
        new.sort(key=lambda kv: -kv[1])  # stable: ties keep F0 order
        return [k for k, _ in new] + [k for k, _ in tail]

    configs = ["F0 (no slots, reference)", "S0"]
    for ln in LEXICONS:
        configs += [f"{ln} B w{w}" for w in WEIGHTS] + [f"{ln} D"] + [f"{ln} BD w{w}" for w in WEIGHTS]
    ranks: dict[str, dict[str, list]] = {}
    for name, s in sets.items():
        ranks[name] = {c: [] for c in configs}
        for i, (qn, qv) in enumerate(zip(s["qn"], s["vec"])):
            dense0 = top(doc_vec @ qv, keys)
            base = np.asarray(bm.get_scores(tokenize(qn)))
            b0 = bm25_list(base)
            for c in configs:
                if c.startswith("F0"):
                    lst = rescore(dense0, b0, s["slots"][i], False)
                elif c == "S0":
                    lst = rescore(dense0, b0, s["slots"][i], True)
                else:
                    ln, kind, *w = c.split()
                    _, add = s["match"][ln][i]
                    if not add:
                        lst = rescore(dense0, b0, s["slots"][i], True)
                    else:
                        dn = top(doc_vec @ xvec[f"{qn} {' '.join(add)}"], keys) if "D" in kind else dense0
                        if "B" in kind:
                            ww = float(w[0][1:])
                            bl = bm25_list(base + ww * np.asarray(bm.get_scores(tokenize(" ".join(add)))))
                        else:
                            bl = b0
                        lst = rescore(dn, bl, s["slots"][i], True)
                ranks[name][c].append(group_rank(s["groups"][i], lst))

    def hit(name: str, c: str, i: int, k: int) -> float:
        r = ranks[name][c][i]
        return float(r is not None and r <= k)

    def idx(name: str, split: str) -> list[int]:
        return [i for i, sp in enumerate(sets[name]["split"]) if split == "all" or sp == split]

    def mean_hit(name: str, c: str, ix: list[int], k: int) -> float:
        return float(np.mean([hit(name, c, i, k) for i in ix])) if ix else 0.0

    dev_t, dev_r = idx("retrieval", "dev"), idx("routing", "dev")
    guard_t = mean_hit("retrieval", "S0", dev_t, 5) - 0.02
    guard_r = mean_hit("routing", "S0", dev_r, 5) - 0.02

    def dev_score(c: str) -> float:
        h = lambda name, ix: sum(hit(name, c, i, 1) + hit(name, c, i, 5) for i in ix)
        return (h("retrieval", dev_t) + h("routing", dev_r)) / (len(dev_t) + len(dev_r))

    choosable = ["S0", *[c for c in configs if c.startswith("d1h ")]]
    ok_cfg = [c for c in choosable if mean_hit("retrieval", c, dev_t, 5) >= guard_t
              and mean_hit("routing", c, dev_r, 5) >= guard_r]
    order = {c: n for n, c in enumerate(choosable)}  # S0 first: ties keep the baseline
    pick = max(ok_cfg, key=lambda c: (dev_score(c), -order[c]))
    ref_best = max((c for c in configs if c.startswith("d1hm ")), key=dev_score)
    shown = list(dict.fromkeys(["F0 (no slots, reference)", "S0", pick, ref_best]))
    rng = np.random.default_rng(0)

    def delta(name: str, c: str, ix: list[int], k: int) -> str:
        d = np.array([hit(name, c, i, k) - hit(name, "S0", i, k) for i in ix])
        q_ = boot(d, None, rng)
        t_ = boot(d, [sets[name]["cluster"][i] for i in ix], rng)
        return (f"{d.mean():+.3f} [{q_[0]:+.3f}, {q_[1]:+.3f}] [{t_[0]:+.3f}, {t_[1]:+.3f}] "
                f"{int((d > 0).sum())}/{int((d < 0).sum())}")

    def block(name: str, ix: list[int], title: str) -> list[str]:
        out = ["", f"## {title} (n={len(ix)})", "", "Δ cells vs S0: mean, 95% question-level, 95% table-level, wins/losses.", "",
               "| setting | " + " | ".join(METRICS) + " | ΔHit@1 | ΔHit@5 |",
               "| --- | " + " | ".join("---" for _ in METRICS) + " | --- | --- |"]
        for c in shown:
            sm = summary([ranks[name][c][i] for i in ix])
            cells = ["-", "-"] if c == "S0" else [delta(name, c, ix, 1), delta(name, c, ix, 5)]
            out.append(f"| {c}{' (dev choice)' if c == pick else ''} | " + " | ".join(f"{sm[m]:.3f}" for m in METRICS)
                       + " | " + " | ".join(cells) + " |")
        return out

    def verdict() -> list[str]:
        ix = idx("retrieval", "all")
        out = ["", "## verdict on retrieval_golden answerable (table-level 95% lower bound > 0 → effect)", "",
               f"- dev choice: **{pick}** (Hit@1+Hit@5 pooled over retrieval dev {len(dev_t)} + routing dev {len(dev_r)}; "
               f"guardrails retrieval dev Hit@5 ≥ {guard_t:.3f}, routing dev Hit@5 ≥ {guard_r:.3f}; "
               f"{len(ok_cfg)}/{len(choosable)} settings pass)",
               "- retrieval_golden test has been looked at many times → exploratory; confirm on the new check set (plan 40)"]
        if pick == "S0":
            return out + ["- no d1h setting beat S0 on dev → **no change** (S0 stays)"]
        parts, eff = [], False
        for k in (1, 5):
            d = np.array([hit("retrieval", pick, i, k) - hit("retrieval", "S0", i, k) for i in ix])
            lo, hi = boot(d, [sets["retrieval"]["cluster"][i] for i in ix], rng)
            parts.append(f"Hit@{k} {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
            eff = eff or lo > 0
        return out + [f"- {pick} vs S0 → " + "; ".join(parts) + (" → **effect**" if eff else " → no clear effect")]

    def fired_block() -> list[str]:
        out = ["", "## where the lexicons fire", "",
               "| golden | split | n | d1h fires | d1hm fires |", "| --- | --- | --- | --- | --- |"]
        for name, s in sets.items():
            for sp in ("dev", "test"):
                ix = idx(name, sp)
                f = lambda ln: sum(bool(s["match"][ln][i][1]) for i in ix)
                out.append(f"| {name} | {sp} | {len(ix)} | {f('d1h')} | {f('d1hm')} |")
        if pick != "S0":
            ln = pick.split()[0]
            out += ["", f"### fired rows only, {pick} vs S0 (Hit@1 / Hit@5 wins-losses)", "",
                    "| golden | split | fired | Hit@1 | Hit@5 |", "| --- | --- | --- | --- | --- |"]
            for name, s in sets.items():
                for sp in ("dev", "test"):
                    ix = [i for i in idx(name, sp) if s["match"][ln][i][1]]
                    wl = lambda k: (f"{sum(hit(name, pick, i, k) > hit(name, 'S0', i, k) for i in ix)}/"
                                    f"{sum(hit(name, pick, i, k) < hit(name, 'S0', i, k) for i in ix)}")
                    out.append(f"| {name} | {sp} | {len(ix)} | {wl(1)} | {wl(5)} |")
        return out

    def changed() -> list[str]:
        out = []
        for c in dict.fromkeys([pick, ref_best]):
            if c == "S0":
                continue
            ln = c.split()[0]
            s = sets["retrieval"]
            out += ["", f"## changed rows (retrieval answerable, S0 → {c})", "",
                    "| qid | split | qtype | question | matched → added | S0 | new |", "| --- | --- | --- | --- | --- | --- | --- |"]
            for i in range(len(s["items"])):
                a, b = ranks["retrieval"]["S0"][i], ranks["retrieval"][c][i]
                if a != b:
                    hits, add = s["match"][ln][i]
                    out.append(f"| {s['ids'][i]} | {s['split'][i]} | {s['qtype'][i]} | {s['items'][i]['question']} | "
                               f"{', '.join(hits)} → {', '.join(add)} | {a} | {b} |")
        return out

    def selection() -> list[str]:
        out = ["", "## every setting on the dev sets", "",
               "| setting | retrieval dev Hit@1 | retrieval dev Hit@5 | routing dev Hit@1 | routing dev Hit@5 | pooled | guardrail |",
               "| --- | --- | --- | --- | --- | --- | --- |"]
        for c in configs:
            g = "ok" if c in ok_cfg else ("fail" if c in choosable else "reference")
            out.append(f"| {c} | {mean_hit('retrieval', c, dev_t, 1):.3f} | {mean_hit('retrieval', c, dev_t, 5):.3f} | "
                       f"{mean_hit('routing', c, dev_r, 1):.3f} | {mean_hit('routing', c, dev_r, 5):.3f} | "
                       f"{dev_score(c):.3f} | {g} |")
        return out

    rid = time.strftime("%Y%m%d-%H%M%S")
    lines = ["# Plan 36: reviewed colloquial lexicon (d1h) on F0 + slots", "",
             f"run {rid}; fine-tuning run {run_dir.name}; tables {sha[:8]}; D1 {D1_FILE.name} {sha8(D1_FILE)}; "
             f"lexicon_d1h.json {sha8(path_d1h)}",
             f"d1h {len(lexicons['d1h'])} expressions (dropped {drops['d1h']}), d1hm {len(lexicons['d1hm'])} "
             f"(dropped {drops['d1hm']}); expanded questions embedded: {len(texts)}",
             f"S0 = F0 + slots {'+'.join(SLOTS)} PB β {BETA} γ {GAMMA} top {N}; variants B (BM25 + w·added), D (dense on "
             "question + added), BD (both); d1hm is reference only."]
    lines += verdict()
    for sp in ("all", "test", "dev"):
        lines += block("retrieval", idx("retrieval", sp), f"retrieval_golden {sp} (answerable)")
    lines += block("routing", dev_r, "routing_v0 dev")
    lines += block("routing", idx("routing", "test"), "routing_v0 test (reference)")
    lines += fired_block()
    lines += changed()
    lines += selection()
    report = "\n".join(lines) + "\n"
    print(report)
    out = OUT_DIR / f"vocab_slot_eval_{rid}_summary.md"
    out.write_text(report, encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
