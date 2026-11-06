"""Plan 37: move down search candidates that miss a condition written in the question (year, region level, sex, age,
monthly / quarterly cycle).

Rules only, the same for every question and table (no golden-specific fixes):
    question -> slots   regular expressions + region names taken from the 2,364 tables' region classifications
    table    -> facts   periods, cycle, region levels, whether it is split by sex / age (classes, labels, items)
    score               s = RRF score - beta * (slots the table misses) + gamma * (slots it meets), on F0's top N only
Settings (slot set x P / PB x beta x gamma) are chosen on retrieval_golden dev (Hit@1 + Hit@5, tie -> smaller change)
with the routing_v0 dev guardrail (Hit@5 no more than 2pt below F0), then judged on all answerable retrieval_golden
rows against F0 (table-level bootstrap, as in plans 32-35). No API, no GPU: unseen-e1 vectors come from the caches
lab/fusion_eval.py wrote (<ckpt>__docs/retrieval/routing.npy).

    python lab/slot_rerank_eval.py --selftest     # slot extraction on 30 made-up sentences (not from the goldens)
    python lab/slot_rerank_eval.py
"""

from __future__ import annotations

import argparse
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from lab_common import (
    DEPTH, METRICS, OUT_DIR, ROUTING, RETRIEVAL, RRF_C, bm25_tokenize, boot, docs, group_rank, load_records,
    query_text, read_jsonl, source_sha, summary, top,
)
from ft_train import FT_DIR

RULES = "s1"  # bump when a slot rule changes (rev2 changed only the dev choice, not the rules)
SLOT_SETS = {"year": ("year",), "year+region": ("year", "region"), "all": ("year", "region", "sex", "age", "cycle")}
BETAS = (0.002, 0.005, 0.01, 0.02)
GAMMAS = (0.001, 0.002, 0.005)
LEVEL = {"시도": 1, "시군구": 2, "읍면동": 3}

# 17 provinces, full and short names (domain knowledge, not taken from any golden)
SIDO = ["서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시", "대전광역시", "울산광역시", "세종특별자치시",
        "경기도", "강원도", "강원특별자치도", "충청북도", "충청남도", "전라북도", "전북특별자치도", "전라남도", "경상북도",
        "경상남도", "제주특별자치도", "제주도", "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종", "경기",
        "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주", "서울시", "부산시", "대구시", "인천시",
        "광주시", "대전시", "울산시", "세종시"]
REGION_CLASS = re.compile(r"행정구역|시도|시군구|읍면동|지역|거주지|시·도")
COMMON_WORDS = {"인구", "가구", "지구", "연구", "이동", "부동", "활동", "노동", "변동", "행동", "자동", "아동", "운동", "공동"}
JOSA = r"(?=$|[^가-힣]|은|는|이|가|의|에|에서|과|와|도|별|로|을|를|만|및|지역|내)"
YEAR = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)(?!\s*년대)")
DECADE = re.compile(r"(?<!\d)(19\d|20\d)0\s*년대")  # "2000년대" -> any of 2000..2009
SEX_Q = re.compile(r"남자|여자|남녀|성별|남성|여성")
AGE_Q = re.compile(r"\d+\s*세|\d0대|연령별|나이별|연령대|나이대|청년|노인|유소년|생산연령|생산가능|세 이상|세 미만")
AGE_MEASURE = re.compile(r"(평균|중위|초혼|재혼|출산|사망)\s*(연령|나이)")  # age is the measure, not a split
CYCLE_Q = {"M": re.compile(r"월별|매월|월간|달마다|매달"), "Q": re.compile(r"분기")}
LEVEL_WORD = [("읍면동", re.compile(r"읍면동|동별|읍면별|동 단위|읍면 단위")),
              ("시군구", re.compile(r"시군구|구별|군별|시군별|시·군·구")),
              ("시도", re.compile(r"시도별|시·도|시도 단위|광역시도|17개 시도|시도"))]


def table_facts(rec: dict, names: dict[str, set[str]]) -> dict:
    """What a table offers: years, cycles, region levels, sex / age splits."""
    classes = rec.get("classes", [])
    cnames = " ".join(c["name"] for c in classes) + " " + rec["info"]["table_name"]
    labels = {l.strip() for c in classes for l in c["labels"]}
    items = " ".join(rec.get("items", []))
    levels = set()
    for c in classes:
        if not REGION_CLASS.search(c["name"]):
            continue
        ls = [l.strip() for l in c["labels"]]
        if any(l in SIDO for l in ls) or "시도" in c["name"]:
            levels.add("시도")
        if sum(l in names["시군구"] for l in ls) >= 5 or "시군구" in c["name"]:
            levels.add("시군구")
        if sum(l in names["읍면동"] for l in ls) >= 5 or "읍면동" in c["name"]:
            levels.add("읍면동")
    for lv in LEVEL:
        if re.search(rf"[-–/]\s*{lv}\s*$|\({lv}\)", rec["info"]["table_name"]):
            levels.add(lv)
    return {
        "years": {str(p)[:4] for p in rec.get("periods", [])},
        "cycle": set(rec.get("cycle") or []) | set(rec.get("prd_se") or []),
        "level": max((LEVEL[x] for x in levels), default=0),
        "sex": bool(re.search(r"성별|(^|\s)성(\s|$)|성,", cnames)) or bool(labels & {"남자", "여자", "남", "여", "남성", "여성"})
               or bool(re.search(r"남자|여자|남성|여성", items)),
        "age": bool(re.search(r"연령|세별|년생|나이", cnames)) or any(re.search(r"\d+\s*세", l) for l in labels)
               or bool(re.search(r"\d+\s*세", items)),
    }


def region_names(records: list[dict]) -> dict[str, set[str]]:
    """시군구 / 읍면동 names, taken from the tables' own region classifications."""
    out = {"시군구": set(), "읍면동": set()}
    for r in records:
        for c in r.get("classes", []):
            if not REGION_CLASS.search(c["name"]):
                continue
            for l in c["labels"]:
                l = l.strip()
                if l in SIDO or l in COMMON_WORDS:
                    continue
                if re.fullmatch(r"[가-힣]{1,5}(시|군|구)", l):
                    out["시군구"].add(l)
                elif re.fullmatch(r"[가-힣]{1,6}\d*(동|읍|면)", l) and len(l) >= 3:
                    out["읍면동"].add(l)
    return out


def make_extractor(names: dict[str, set[str]]):
    pats = {lv: re.compile(r"(?<![가-힣])(" + "|".join(sorted(map(re.escape, ns), key=len, reverse=True)) + ")" + JOSA)
            for lv, ns in (("시군구", names["시군구"]), ("읍면동", names["읍면동"]), ("시도", set(SIDO)))}

    def slots(question: str) -> dict:
        q = re.sub(r"\s+", " ", question)
        s: dict = {}
        ys = set(YEAR.findall(q)) | {f"{d}{n}" for d in DECADE.findall(q) for n in range(10)}
        if ys:
            s["year"] = ys
        need = 0
        for lv, pat in LEVEL_WORD:
            if pat.search(q):
                need = max(need, LEVEL[lv])
        for lv in ("읍면동", "시군구", "시도"):
            if pats[lv].search(q):
                need = max(need, LEVEL[lv])
        if need:
            s["region"] = need
        if SEX_Q.search(q):
            s["sex"] = True
        q_age = AGE_MEASURE.sub(" ", q)
        if AGE_Q.search(q_age):
            s["age"] = True
        for c, pat in CYCLE_Q.items():
            if pat.search(re.sub(r"\d+\s*월\s*\d+\s*일|\d+\s*월\s*(말|기준)", " ", q)):
                s["cycle"] = c
                break
        return s

    return slots


def meets(f: dict, name: str, v) -> bool:
    if name == "year":
        return bool(v & f["years"])
    if name == "region":
        return f["level"] >= v  # a finer level also covers the coarser one
    if name == "cycle":
        return v in f["cycle"]
    return bool(f[name])


SELFTEST = [
    "2019년 부산 해운대구 인구", "충청남도 시군구별 고령인구 비율", "1인 가구는 몇 가구야", "인구 이동이 많은 지역은",
    "역삼1동 주민등록인구", "월별 출생아 수 추이", "2024년 1월 1일 기준 외국인 주민", "분기별 혼인 건수",
    "평균 초혼연령이 얼마나 올랐어", "30대 여성 취업자", "경기도와 서울의 인구 차이", "전국 총인구 2000~2020",
    "노인 인구가 많은 군 지역", "읍면동별 세대수", "남녀 성비", "중위연령 추이", "청년층 인구 변화",
    "제주 2015년 인구", "광주광역시 북구 가구수", "아동 인구 비율", "외국인 국적별 인구", "65세 이상 1인 가구",
    "시도별 합계출산율", "2030년 장래인구 전망", "강남구와 서초구 비교", "결혼 건수 매달 얼마나 돼", "가구원수별 가구",
    "노동 인구", "세종시 인구 증가", "지역별 사망률", "1990년대 출생아 수",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None, help="fine-tuning run under data/lab/ft (default: latest)")
    p.add_argument("--n", type=int, default=50, help="F0 candidates rescored")
    p.add_argument("--select", choices=("pooled", "retrieval"), default="pooled",
                   help="dev score for the choice: pooled = retrieval dev + routing dev (rev2), retrieval = rev1")
    p.add_argument("--max-labels", type=int, default=60)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    records, sha, tokenize = load_records(), source_sha(), bm25_tokenize()
    names = region_names(records)
    slots = make_extractor(names)
    if args.selftest:
        print(f"region names: 시군구 {len(names['시군구'])}, 읍면동 {len(names['읍면동'])}")
        for s in SELFTEST:
            print(f"  {s} -> {slots(s)}")
        return

    runs = sorted(d for d in FT_DIR.iterdir() if d.is_dir()) if FT_DIR.exists() else []
    run_dir = FT_DIR / args.run if args.run else (runs[-1] if runs else None)
    assert run_dir and run_dir.exists(), "no fine-tuning run under data/lab/ft"
    cache = {part: run_dir / f"unseen-e1__{part}.npy" for part in ("docs", "retrieval", "routing")}
    missing = [str(c) for c in cache.values() if not c.exists()]
    assert not missing, f"need {missing} (run lab/fusion_eval.py once first)"

    ds = docs(records, "norm", args.max_labels)
    keys = [d["key"] for d in ds]
    facts = {r["info"]["key"]: table_facts(r, names) for r in records}
    bm = BM25Okapi([tokenize(d["bm25"]) for d in ds])
    doc_vec = np.load(cache["docs"])
    assert len(doc_vec) == len(keys)

    rt_all = read_jsonl(RETRIEVAL)
    rg = read_jsonl(ROUTING)
    q_rt = np.load(cache["retrieval"])
    q_rg = np.load(cache["routing"])
    assert len(q_rt) == len(rt_all) and len(q_rg) == len(rg), "question caches do not match the goldens"

    def f0(question: str, qv: np.ndarray) -> list[tuple[str, float]]:
        """F0 with its RRF scores, up to 100 tables."""
        qn = query_text(question, "norm")
        dense = top(doc_vec @ qv, keys)
        sc = np.asarray(bm.get_scores(tokenize(qn)))
        bml = [keys[j] for j in np.argsort(-sc, kind="stable")[:DEPTH] if sc[j] > 0]
        s: dict[str, float] = defaultdict(float)
        for lst in (dense, bml):
            for r, k in enumerate(lst, 1):
                s[k] += 1.0 / (RRF_C + r)
        return sorted(s.items(), key=lambda kv: -kv[1])

    keep = [i for i, it in enumerate(rt_all) if it.get("answerable", True) and it.get("required")]
    sets = {
        "retrieval": {"items": [rt_all[i] for i in keep], "vec": q_rt[keep], "groups": [rt_all[i]["required"] for i in keep]},
        "routing": {"items": rg, "vec": q_rg, "groups": [[[Path(it["table"]).stem, *it.get("alt_tables", [])]] for it in rg]},
    }
    for s in sets.values():
        s["split"] = [it["split"] for it in s["items"]]
        s["qtype"] = [it.get("qtype", "?") for it in s["items"]]
        s["slots"] = [slots(it["question"]) for it in s["items"]]
        s["f0"] = [f0(it["question"], v) for it, v in zip(s["items"], s["vec"])]
        s["cluster"] = [g[0][0] for g in s["groups"]]

    def rerank(scored: list[tuple[str, float]], sl: dict, use: tuple, beta: float, gamma: float) -> list[str]:
        head, tail = scored[: args.n], scored[args.n:]
        act = [(n, v) for n, v in sl.items() if n in use]
        if not act:
            return [k for k, _ in scored]
        new = []
        for k, sc in head:
            m = sum(meets(facts[k], n, v) for n, v in act)
            new.append((k, sc - beta * (len(act) - m) + gamma * m))
        new.sort(key=lambda kv: -kv[1])  # stable: ties keep F0 order
        return [k for k, _ in new] + [k for k, _ in tail]

    configs = [("F0", (), 0.0, 0.0)]
    for sname, use in SLOT_SETS.items():
        for b in BETAS:
            configs.append((f"P {sname} b{b}", use, b, 0.0))
            for g in GAMMAS:
                configs.append((f"PB {sname} b{b} g{g}", use, b, g))
    ranks: dict[str, dict[str, list]] = {}
    lists: dict[str, dict[str, list]] = {}
    for name, s in sets.items():
        ranks[name], lists[name] = {}, {}
        for cid, use, b, g in configs:
            ls = [rerank(sc, sl, use, b, g) for sc, sl in zip(s["f0"], s["slots"])]
            lists[name][cid] = ls
            ranks[name][cid] = [group_rank(gr, l) for gr, l in zip(s["groups"], ls)]

    def hit(name: str, cid: str, i: int, k: int) -> float:
        r = ranks[name][cid][i]
        return float(r is not None and r <= k)

    def idx(name: str, split: str) -> list[int]:
        return [i for i, sp in enumerate(sets[name]["split"]) if split == "all" or sp == split]

    def mean_hit(name: str, cid: str, ix: list[int], k: int) -> float:
        return float(np.mean([hit(name, cid, i, k) for i in ix])) if ix else 0.0

    dev_t, dev_r = idx("retrieval", "dev"), idx("routing", "dev")
    guard = mean_hit("routing", "F0", dev_r, 5) - 0.02
    order = {c[0]: n for n, c in enumerate(configs)}  # earlier = smaller change (smaller beta first within a set)
    guard_t = mean_hit("retrieval", "F0", dev_t, 5) - 0.02

    def dev_score(c: str) -> float:
        """retrieval: retrieval dev Hit@1+Hit@5 (rev1). pooled: Hit@1+Hit@5 summed over every question of both dev sets
        (retrieval dev 29 + routing dev 104), because retrieval dev alone did not tell the betas apart (rev2)."""
        h = lambda name, ix: sum(hit(name, c, i, 1) + hit(name, c, i, 5) for i in ix)
        if args.select == "retrieval":
            return mean_hit("retrieval", c, dev_t, 1) + mean_hit("retrieval", c, dev_t, 5)
        return (h("retrieval", dev_t) + h("routing", dev_r)) / (len(dev_t) + len(dev_r))

    ok_cfg = [c[0] for c in configs if mean_hit("routing", c[0], dev_r, 5) >= guard
              and (args.select == "retrieval" or mean_hit("retrieval", c[0], dev_t, 5) >= guard_t)]
    pick = max(ok_cfg, key=lambda c: (dev_score(c), -order[c]))
    best_of = {sname: max((c[0] for c in configs if c[0].split(" ")[1:2] == [sname] and c[0] in ok_cfg),
                          key=lambda c: (dev_score(c), -order[c]))
               for sname in SLOT_SETS}
    shown = list(dict.fromkeys(["F0", pick, *best_of.values()]))
    rng = np.random.default_rng(0)

    def delta(name: str, cid: str, ix: list[int], k: int) -> str:
        d = np.array([hit(name, cid, i, k) - hit(name, "F0", i, k) for i in ix])
        q_ = boot(d, None, rng)
        t_ = boot(d, [sets[name]["cluster"][i] for i in ix], rng)
        return (f"{d.mean():+.3f} [{q_[0]:+.3f}, {q_[1]:+.3f}] [{t_[0]:+.3f}, {t_[1]:+.3f}] "
                f"{int((d > 0).sum())}/{int((d < 0).sum())}")

    def block(name: str, ix: list[int], title: str) -> list[str]:
        out = ["", f"## {title} (n={len(ix)})", "", "Δ cells vs F0: mean, 95% question-level, 95% table-level, wins/losses.", "",
               "| setting | " + " | ".join(METRICS) + " | ΔHit@1 | ΔHit@5 |",
               "| --- | " + " | ".join("---" for _ in METRICS) + " | --- | --- |"]
        for cid in shown:
            sm = summary([ranks[name][cid][i] for i in ix])
            cells = ["-", "-"] if cid == "F0" else [delta(name, cid, ix, 1), delta(name, cid, ix, 5)]
            out.append(f"| {cid}{' (dev choice)' if cid == pick else ''} | " + " | ".join(f"{sm[m]:.3f}" for m in METRICS)
                       + " | " + " | ".join(cells) + " |")
        return out

    def verdict() -> list[str]:
        ix = idx("retrieval", "all")
        parts, eff = [], False
        for k in (1, 5):
            d = np.array([hit("retrieval", pick, i, k) - hit("retrieval", "F0", i, k) for i in ix])
            lo, hi = boot(d, [sets["retrieval"]["cluster"][i] for i in ix], rng)
            parts.append(f"Hit@{k} {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
            eff = eff or lo > 0
        return ["", "## verdict on retrieval_golden answerable (table-level 95% lower bound > 0 → effect)", "",
                f"- dev choice ({args.select}): **{pick}** ("
                + ("retrieval dev Hit@1+Hit@5" if args.select == "retrieval" else
                   f"Hit@1+Hit@5 pooled over retrieval dev {len(dev_t)} + routing dev {len(dev_r)}; retrieval dev Hit@5 ≥ "
                   f"F0 − 2pt = {guard_t:.3f}")
                + f"; routing dev Hit@5 ≥ F0 − 2pt = {guard:.3f}; {len(ok_cfg)}/{len(configs)} settings pass)",
                *(["- rev2 (pooled) was set after rev1's retrieval test numbers were seen → this verdict is exploratory; "
                   "confirm on a new check set"] if args.select == "pooled" else []),
                f"- {pick} vs F0 → " + "; ".join(parts) + (" → **effect**" if eff else " → no clear effect")]

    def selection() -> list[str]:
        out = ["", "## every setting on the dev sets", "",
               "| setting | retrieval dev Hit@1 | retrieval dev Hit@5 | routing dev Hit@1 | routing dev Hit@5 | pooled score | guardrail |",
               "| --- | --- | --- | --- | --- | --- | --- |"]
        for cid, *_ in configs:
            out.append(f"| {cid} | {mean_hit('retrieval', cid, dev_t, 1):.3f} | {mean_hit('retrieval', cid, dev_t, 5):.3f} | "
                       f"{mean_hit('routing', cid, dev_r, 1):.3f} | {mean_hit('routing', cid, dev_r, 5):.3f} | "
                       f"{dev_score(cid):.3f} | {'ok' if cid in ok_cfg else 'fail'} |")
        return out

    def slot_block() -> list[str]:
        out = ["", f"## slots fired and whether the gold table meets them (rules {RULES})", "",
               "| golden | slot | questions fired | gold meets | F0 top-20 candidates missing it |",
               "| --- | --- | --- | --- | --- |"]
        bad: list[str] = []
        for name, s in sets.items():
            fired, good, miss = Counter(), Counter(), defaultdict(list)
            for i, sl in enumerate(s["slots"]):
                golds = [g for grp in s["groups"][i] for g in grp]
                cands = [k for k, _ in s["f0"][i][:20]]
                for n, v in sl.items():
                    fired[n] += 1
                    g = any(meets(facts[k], n, v) for k in golds if k in facts)
                    good[n] += g
                    miss[n].append(np.mean([not meets(facts[k], n, v) for k in cands]))
                    if not g and name == "retrieval":
                        bad.append(f"- {s['items'][i]['qid']} {n}={v}: {s['items'][i]['question']} (gold {golds[:2]})")
            for n in ("year", "region", "sex", "age", "cycle"):
                if fired[n]:
                    out.append(f"| {name} | {n} | {fired[n]}/{len(s['slots'])} | {good[n]} | {np.mean(miss[n]):.2f} |")
        out += ["", "### retrieval_golden rows whose gold misses a fired slot (rule check)", ""] + (bad or ["- none"])
        return out

    def qtype_block() -> list[str]:
        by: dict[str, list[int]] = defaultdict(list)
        for i, q in enumerate(sets["retrieval"]["qtype"]):
            by[q].append(i)
        out = ["", "## Hit@1 / Hit@5 by qtype (retrieval answerable)", "",
               "| qtype | n | slots fired | " + " | ".join(shown[:2]) + " |", "| --- | --- | --- | " + " | ".join("---" for _ in shown[:2]) + " |"]
        for q, ix in sorted(by.items(), key=lambda kv: -len(kv[1])):
            out.append(f"| {q} | {len(ix)} | {sum(bool(sets['retrieval']['slots'][i]) for i in ix)} | " + " | ".join(
                f"{mean_hit('retrieval', c, ix, 1):.2f} / {mean_hit('retrieval', c, ix, 5):.2f}" for c in shown[:2]) + " |")
        return out

    def changed() -> list[str]:
        s = sets["retrieval"]
        out = ["", f"## changed rows (retrieval answerable, F0 → {pick})", "", "| qid | qtype | question | slots | F0 | new |",
               "| --- | --- | --- | --- | --- | --- |"]
        for i in range(len(s["items"])):
            a, b = ranks["retrieval"]["F0"][i], ranks["retrieval"][pick][i]
            if a != b:
                sl = ", ".join(f"{n}={sorted(v) if isinstance(v, set) else v}" for n, v in s["slots"][i].items())
                out.append(f"| {s['items'][i]['qid']} | {s['qtype'][i]} | {s['items'][i]['question']} | {sl} | {a} | {b} |")
        return out

    rid = time.strftime("%Y%m%d-%H%M%S")
    lines = ["# Plan 37: question slots → rescore F0 candidates", "",
             f"run {rid}; rules {RULES}; fine-tuning run {run_dir.name}; tables {sha[:8]}; rescored top {args.n}; "
             f"region names 시군구 {len(names['시군구'])} / 읍면동 {len(names['읍면동'])}",
             "score = RRF − β·(missed slots) + γ·(met slots); P = γ 0, PB = both. Shown: F0, dev choice, best per slot set."]
    lines += verdict()
    for sp in ("all", "test", "dev"):
        lines += block("retrieval", idx("retrieval", sp), f"retrieval_golden {sp} (answerable)")
    lines += block("routing", idx("routing", "dev"), "routing_v0 dev (guardrail)")
    lines += block("routing", idx("routing", "test"), "routing_v0 test (reference)")
    lines += slot_block()
    lines += qtype_block()
    lines += changed()
    lines += selection()
    report = "\n".join(lines) + "\n"
    print(report)
    out = OUT_DIR / f"slot_rerank_eval_{rid}_summary.md"
    out.write_text(report, encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
