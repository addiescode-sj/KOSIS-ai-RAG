"""T2-39 lab: the question index (reverse HyDE) through the LangGraph search on the retrieval golden (plans/T2-39.md Eval).

    set PYTHONPATH=.;src;vendor/phase1;vendor/phase1/src          (Windows cmd; bash: PYTHONPATH=.:src:vendor/phase1:vendor/phase1/src)
    python lab\\hq_eval.py --build <build_id>
    python lab\\hq_eval.py --build <build_id> --nofilter-build <rules-only build_id> --desc

Configurations (all: hybrid + condition filter + fine-tuned stage 1 re-ranker, no stage 2; .env gives the collection,
EMBED_CKPT and RERANK_CKPT):
    S0           the current best: dense + BM25 -> RRF -> rerank1
    S1           S0 + hq dense (retrieve_hq)                    HQ_BUILD=<build>
    S2           S0 + hq dense + hq BM25                       HQ_BUILD=<build>, HQ_BM25=on
    S1-w.5       S1 with HQ_WEIGHT 0.5
    S1-nofilter  S1 over a build without the score filter     (--nofilter-build, e.g. ...-f0)
    S1-desc      S1 over the description index                 (--desc; index built with kosis_rag.hq.index --desc)

Protocol: answerable questions only. Every configuration runs on dev; S1 / S2 / S1-w.5 are chosen among on dev
(Hit@1 + Hit@5, ties -> the earlier, smaller change); test is run once, for S0 and the chosen one only. The verdict
(plans/T2-39.md Eval) is read on all answerable questions (dev + test) for the chosen one against S0.
Ranks: a question's rank is the worst over its required groups of the best rank inside the group (the lab convention
behind the .438 / .685 baseline, plan 20 "묶음별 최악 순위" = evals' Complete@k); the first-hit rank (best rank of any
gold table, the team L1 evaluator's hit@k) is reported beside it as a secondary column. Metrics Hit@1/5/10,
MRR@10, Recall@50 of the pool before rerank1 (the fuse node's output, from the trace), paired bootstrap 95% CI by
question (10,000 resamples, seed 0, a fresh generator per comparison) against S0,
wins / losses, subsets sibling / concept / low-overlap (question Hangul
bigrams found in the gold table name < --low-overlap), node latency p50 / p95 from the trace (evals.trace.NodeSpans).
Contamination (plan §5.1): every golden question's highest cosine to any indexed question (same embedder) and highest
difflib ratio; the share >= 0.9 and the verdict metrics without those questions.
Nothing here is tuned on the golden: the build, the prompt and the filters are fixed before this script runs.
Output: data/lab/hq_eval_<stamp>_summary.md and data/lab/hq_eval_<stamp>_ranks.jsonl. No API call (MockClient guide).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from evals.schemas import GOLDEN
from evals.trace import NodeSpans
from kosis_chat.llm.model import MockClient
from kosis_rag.adapters.files import read_records
from kosis_rag.graph import build_graph, run_search
from kosis_rag.pipeline import REPO, SearchSettings, load_resources
from kosis_rag.shared import tabletext as text

KS = (1, 5, 10)
SELECTABLE = ("S1", "S2", "S1-w.5")  # chosen among on dev (plans/T2-39.md Eval)
SUBSETS = ("sibling", "concept", "low-overlap")
CONTAMINATED = 0.9
HANGUL = re.compile(r"[가-힣]+")


# --- golden and ranks ---

def read_golden(path: Path) -> list[dict]:
    """Answerable rows with required groups: {qid, split, qtype, question, required}."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [{k: r.get(k) for k in ("qid", "split", "qtype", "question", "required")}
            for r in rows if r.get("answerable", True) and r.get("required")]


def group_rank(groups: list[list[str]], keys: list[str]) -> int | None:
    """Worst over the groups of the best rank inside each group; None if a group is not in the list.

    The lab convention (plan 20, "묶음별 최악 순위"), the one the .438 / .685 baseline was measured with.
    """
    pos = {k: i for i, k in reversed(list(enumerate(keys, 1)))}
    worst = 0
    for g in groups:
        ranks = [pos[t] for t in g if t in pos]
        if not ranks:
            return None
        worst = max(worst, min(ranks))
    return worst


def first_rank(groups: list[list[str]], keys: list[str]) -> int | None:
    """Best rank of any gold table of any group (the team L1 evaluator's hit@k); None if none is in the list.

    Secondary column only (S6): the verdict reads group_rank.
    """
    gold = {t for g in groups for t in g}
    return next((i for i, k in enumerate(keys, 1) if k in gold), None)


def hit(rank: int | None, k: int) -> float:
    return float(rank is not None and rank <= k)


def rr(rank: int | None) -> float:
    return 1.0 / rank if rank is not None and rank <= 10 else 0.0


def summary(ranks: list[int | None]) -> dict[str, float]:
    n = max(len(ranks), 1)
    out = {f"Hit@{k}": sum(hit(r, k) for r in ranks) / n for k in KS}
    out["MRR@10"] = sum(rr(r) for r in ranks) / n
    return out


def boot(diff: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    """Paired bootstrap 95% interval of the mean difference (questions resampled).

    A fresh generator from `seed` per call, so each comparison's interval does not depend on which tables were
    written before it.
    """
    if len(diff) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), (n_boot, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def bigrams(s: str) -> set[str]:
    return {w[i:i + 2] for w in HANGUL.findall(s) for i in range(len(w) - 1)}


def low_overlap(item: dict, records: dict, tau: float) -> bool:
    """Few of the question's Hangul bigrams appear in any gold table's (normalized) name."""
    q = bigrams(item["question"])
    if not q:
        return False
    best = 0.0
    for g in item["required"]:
        for key in g:
            if key in records:
                best = max(best, len(q & bigrams(text.norm_name(records[key].info.table_name)[0])) / len(q))
    return best < tau


# --- running the graph ---

def run_config(graph, items: list[dict]) -> dict[str, dict]:
    """qid -> {rank, first_rank, pool_rank, list_ranks, ms, node_ms} for one compiled graph."""
    out = {}
    for it in items:
        nodes = NodeSpans()
        started = time.perf_counter()
        state = run_search(graph, it["question"], {"callbacks": [nodes]})
        ms = (time.perf_counter() - started) * 1000
        fuse = next((s for s in nodes.spans if s["name"] == "fuse"), None)
        pool = fuse["keys"] if fuse else [d.metadata["key"] for d in state.docs]
        keys = [d.metadata["key"] for d in state.docs]
        out[it["qid"]] = {
            "rank": group_rank(it["required"], keys), "first_rank": first_rank(it["required"], keys),
            "pool_rank": group_rank(it["required"], pool[:50]),
            "list_ranks": {name: group_rank(it["required"], [d.metadata["key"] for d in docs])
                           for name, docs in state.runs.items()},
            "ms": ms, "node_ms": {s["name"]: s["duration_ms"] for s in nodes.spans},
        }
    return out


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))] if s else 0.0


# --- contamination ---

def contamination(items: list[dict], hq, embed_query) -> dict[str, dict]:
    """qid -> {cos, cos_id, ratio, ratio_id}: the closest indexed question by cosine and by difflib ratio."""
    ids, docs, vecs = [], [], []
    for offset in range(0, hq.collection.count(), 2000):  # in pages: a server may cap one get
        got = hq.collection.get(include=["embeddings", "documents"], limit=2000, offset=offset)
        ids += list(got["ids"])
        docs += list(got["documents"])
        vecs += [np.asarray(v, dtype=np.float32) for v in got["embeddings"]]
    mat = np.stack(vecs)
    plain = [re.sub(r"\s+", "", d) for d in docs]
    out = {}
    for it in items:
        qv = np.asarray(embed_query(it["question"]), dtype=np.float32)
        sims = mat @ qv
        j = int(np.argmax(sims))
        q = re.sub(r"\s+", "", text.norm_query(it["question"]))
        best, best_id = 0.0, None
        sm = SequenceMatcher(None, "", q)  # seq2 = the question: its index is built once
        for qid, d in zip(ids, plain):
            sm.set_seq1(d)
            if sm.real_quick_ratio() <= best or sm.quick_ratio() <= best:
                continue
            r = sm.ratio()
            if r > best:
                best, best_id = r, qid
        out[it["qid"]] = {"cos": float(sims[j]), "cos_id": ids[j], "ratio": best, "ratio_id": best_id}
    return out


# --- report ---

def metric_row(name: str, res: dict, base: dict, ix: list[dict], args) -> str:
    ranks = [res[i["qid"]]["rank"] for i in ix]
    sm = summary(ranks)
    fh = summary([res[i["qid"]]["first_rank"] for i in ix])
    recall = sum(res[i["qid"]]["pool_rank"] is not None for i in ix) / max(len(ix), 1)
    cells = [f"{sm['Hit@1']:.3f}", f"{sm['Hit@5']:.3f}", f"{sm['Hit@10']:.3f}", f"{sm['MRR@10']:.3f}",
             f"{recall:.3f}", f"{fh['Hit@1']:.3f} / {fh['Hit@5']:.3f}"]
    if res is base:
        return f"| {name} | {len(ix)} | " + " | ".join(cells) + " | - | - | - |"
    deltas = []
    for f in (lambda r: hit(r, 1), lambda r: hit(r, 5), rr):
        d = np.array([f(res[i["qid"]]["rank"]) - f(base[i["qid"]]["rank"]) for i in ix])
        lo, hi = boot(d, args.boot, args.seed)
        deltas.append(f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
    return f"| {name} | {len(ix)} | " + " | ".join(cells) + " | " + " | ".join(deltas) + " |"


def table(title: str, configs: dict[str, dict], base: dict, ix: list[dict], args) -> list[str]:
    return ["", f"### {title} (n={len(ix)})", "",
            "| 구성 | n | Hit@1 | Hit@5 | Hit@10 | MRR@10 | Recall@50 (rerank1 전) | (보조) 첫 적중 Hit@1 / Hit@5 "
            "| ΔHit@1 [95%] | ΔHit@5 [95%] | ΔMRR [95%] |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            *(metric_row(name, res, base, ix, args) for name, res in configs.items())]


def win_loss(res: dict, base: dict, ix: list[dict], k: int) -> tuple[int, int]:
    wins = sum(hit(res[i["qid"]]["rank"], k) > hit(base[i["qid"]]["rank"], k) for i in ix)
    losses = sum(hit(res[i["qid"]]["rank"], k) < hit(base[i["qid"]]["rank"], k) for i in ix)
    return wins, losses


def subset_block(configs: dict[str, dict], base: dict, ix: list[dict], subsets: dict[str, set[str]]) -> list[str]:
    lines = ["", "| 구성 | " + " | ".join(f"{s} Hit@1 / Hit@5 (승:패 @1, @5)" for s in SUBSETS) + " |",
             "| --- | " + " | ".join("---" for _ in SUBSETS) + " |"]
    for name, res in configs.items():
        cells = []
        for s in SUBSETS:
            sub = [i for i in ix if i["qid"] in subsets[s]]
            sm = summary([res[i["qid"]]["rank"] for i in sub])
            w1, l1 = win_loss(res, base, sub, 1)
            w5, l5 = win_loss(res, base, sub, 5)
            cells.append(f"{sm['Hit@1']:.2f} / {sm['Hit@5']:.2f} (n={len(sub)}; {w1}:{l1}, {w5}:{l5})")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def latency_block(configs: dict[str, dict], ix: list[dict]) -> list[str]:
    nodes = list(dict.fromkeys(n for res in configs.values() for i in ix for n in res[i["qid"]]["node_ms"]))
    lines = ["", "| 구성 | 검색 전체 p50 / p95 (ms) | " + " | ".join(nodes) + " |",
             "| --- | --- | " + " | ".join("---" for _ in nodes) + " |"]
    for name, res in configs.items():
        total = [res[i["qid"]]["ms"] for i in ix]
        cells = []
        for n in nodes:
            v = [res[i["qid"]]["node_ms"][n] for i in ix if n in res[i["qid"]]["node_ms"]]
            cells.append(f"{pct(v, 0.5):.0f} / {pct(v, 0.95):.0f}" if v else "-")
        lines.append(f"| {name} | {pct(total, 0.5):.0f} / {pct(total, 0.95):.0f} | " + " | ".join(cells) + " |")
    return lines


def verdict(name: str, res: dict, base: dict, ix: list[dict], subsets: dict[str, set[str]], args) -> list[str]:
    """plans/T2-39.md Eval: adopt / guardrails / discard, read mechanically on `ix`."""
    n = len(ix)
    d1 = sum(hit(res[i["qid"]]["rank"], 1) - hit(base[i["qid"]]["rank"], 1) for i in ix)
    d5 = sum(hit(res[i["qid"]]["rank"], 5) - hit(base[i["qid"]]["rank"], 5) for i in ix)
    ci = {}
    for k in (1, 5):
        diff = np.array([hit(res[i["qid"]]["rank"], k) - hit(base[i["qid"]]["rank"], k) for i in ix])
        ci[k] = boot(diff, args.boot, args.seed)
    recall = sum(res[i["qid"]]["pool_rank"] is not None for i in ix) - sum(base[i["qid"]]["pool_rank"] is not None for i in ix)
    sib = [i for i in ix if i["qid"] in subsets["sibling"]]
    sib_loss = win_loss(res, base, sib, 1)[1]
    low = [i for i in ix if i["qid"] in subsets["low-overlap"]]
    lw = sum(win_loss(res, base, low, k)[0] for k in (1, 5))
    ll = sum(win_loss(res, base, low, k)[1] for k in (1, 5))
    p95 = pct([res[i["qid"]]["ms"] for i in ix], 0.95) - pct([base[i["qid"]]["ms"] for i in ix], 0.95)
    adopt = (d1 >= 2 and ci[5][1] >= 0) or (d5 >= 2 and ci[1][1] >= 0)
    guards = {"Recall@50 하락 없음": recall >= 0, "형제 Hit@1 손실 ≤ 1": sib_loss <= 1,
              "저겹침 승:패 ≥ 2:1 (0:0 = 변화 없음, 통과)": lw >= 2 * ll, "검색 p95 +150ms 이하": p95 <= 150}
    lines = ["", f"### 판정: {name} vs S0 (n={n})", "",
             f"- ΔHit@1 {d1:+.0f}문항 [{ci[1][0]:+.3f}, {ci[1][1]:+.3f}], ΔHit@5 {d5:+.0f}문항 [{ci[5][0]:+.3f}, {ci[5][1]:+.3f}]",
             f"- 채택 조건(Hit@1 +2 또는 Hit@5 +2, 다른 쪽 유의 하락 없음): **{'충족' if adopt else '미충족'}**",
             *(f"- 가드레일 {g}: {'통과' if ok else '실패'}" for g, ok in guards.items()),
             f"  (Recall@50 {recall:+d}문항, 형제 Hit@1 손실 {sib_loss}, 저겹침 승:패 {lw}:{ll} (@1+@5), p95 {p95:+.0f}ms)",
             f"- 폐기 조건(±1문항 안): {'해당' if abs(d1) <= 1 and abs(d5) <= 1 else '해당 없음'}"]
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description="T2-39: S0 vs the question index on the retrieval golden")
    p.add_argument("--build", required=True,
                   help="the question build (HQ_BUILD), e.g. hq-xxxxxxxx-yyyyyyyy-gpt-4o-mini-f30-k2")
    p.add_argument("--nofilter-build", default=None, help="a build without the score filter (S1-nofilter)")
    p.add_argument("--desc", action="store_true", help="also S1-desc (index built with kosis_rag.hq.index --desc)")
    p.add_argument("--golden", type=Path, default=GOLDEN)
    p.add_argument("--rerank-ckpt", default=None, help="stage 1 weights (default: RERANK_CKPT in .env)")
    p.add_argument("--boot", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--low-overlap", type=float, default=0.3, help="bigram share below which a question is low-overlap")
    p.add_argument("--out", type=Path, default=REPO / "data" / "lab")
    args = p.parse_args()

    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    ckpt = args.rerank_ckpt or os.getenv("RERANK_CKPT")
    if not ckpt:
        raise SystemExit("S0 needs the stage 1 re-ranker: set RERANK_CKPT in .env or pass --rerank-ckpt")
    items = read_golden(args.golden)
    s0 = SearchSettings.from_env(cond_filter=True, stage2="none", rerank_ckpt=ckpt, hq_build=None, hq_bm25=False,
                                 hq_weight=1.0)
    records_list = read_records(s0.tables)
    base = load_resources(s0, records=records_list)
    records = base.records
    variants = {"S0": {}, "S1": {"hq_build": args.build}, "S2": {"hq_build": args.build, "hq_bm25": True},
                "S1-w.5": {"hq_build": args.build, "hq_weight": 0.5}}
    if args.nofilter_build:
        variants["S1-nofilter"] = {"hq_build": args.nofilter_build}
    if args.desc:
        variants["S1-desc"] = {"hq_build": args.build + "-desc"}
    resources = {name: base if not kw else load_resources(s0.model_copy(update=kw), records=records_list,
                                                          dense=base.dense, stage1_score=base.stage1_score)
                 for name, kw in variants.items()}
    graphs = {name: build_graph(r, MockClient()) for name, r in resources.items()}
    print("configs: " + ", ".join(f"{n} = {r.settings.label()}" for n, r in resources.items()))

    dev = [i for i in items if i["split"] == "dev"]
    test = [i for i in items if i["split"] == "test"]
    subsets = {"sibling": {i["qid"] for i in items if i["qtype"] == "sibling"},
               "concept": {i["qid"] for i in items if i["qtype"] == "concept"},
               "low-overlap": {i["qid"] for i in items if low_overlap(i, records, args.low_overlap)}}
    # 1. dev: every configuration
    res: dict[str, dict] = {name: run_config(g, dev) for name, g in graphs.items()}
    print("dev done")
    choice = max(SELECTABLE, key=lambda n: (sum(hit(res[n][i["qid"]]["rank"], 1) + hit(res[n][i["qid"]]["rank"], 5)
                                                for i in dev), -SELECTABLE.index(n)))
    # 2. test: S0 and the dev choice only, once
    for name in ("S0", choice):
        res[name].update(run_config(graphs[name], test))
    print(f"test done (S0, {choice})")
    # 3. contamination against the chosen build's question collection
    hq = resources[choice].hq
    contam = contamination(items, hq, base.dense.embed_query)
    clean = [i for i in items if contam[i["qid"]]["cos"] < CONTAMINATED and contam[i["qid"]]["ratio"] < CONTAMINATED]

    stamp = time.strftime("%Y%m%d-%H%M%S")
    chosen = {"S0": res["S0"], choice: res[choice]}
    n_cos = sum(c["cos"] >= CONTAMINATED for c in contam.values())
    n_ratio = sum(c["ratio"] >= CONTAMINATED for c in contam.values())
    lines = [f"# T2-39 역HyDE 평가 {stamp}", "",
             f"- 골든 `{args.golden.name}` 답 있음 {len(items)}문항 (dev {len(dev)} / test {len(test)}), 빌드 `{args.build}`",
             "- 구성: " + "; ".join(f"{n} = `{r.settings.label()}`" for n, r in resources.items()),
             f"- dev 선택({'/'.join(SELECTABLE)} 중 Hit@1+Hit@5): **{choice}**. test는 S0와 {choice}만 1회 실행",
             "- **순위 규약**: 문항 순위 = 필요한 묶음마다 그 묶음 안 최선 순위 중 **가장 나쁜 것**"
             "(lab 규약, 계획 20 \"묶음별 최악 순위\" = evals Complete@k). "
             "baseline .438 / .685도 이 규약. Hit·MRR·판정은 모두 이 순위로 계산. "
             "(보조) 첫 적중 = 아무 정답 표의 첫 순위(팀 L1 평가기의 hit@k)로, "
             "다표 문항에서 값이 다를 수 있어 참고용으로만 같이 적음",
             f"- bootstrap: 질문 단위 짝, {args.boot:,}회, seed {args.seed}(비교마다 새 생성기). 저겹침 = 질문 한글 bigram 중 정답 표 이름에 있는 비율 < "
             f"{args.low_overlap} ({len(subsets['low-overlap'])}문항), 형제 {len(subsets['sibling'])} · 개념 {len(subsets['concept'])}",
             "", "## 1. dev (모든 구성)"]
    lines += table("dev", res, res["S0"], dev, args)
    lines += subset_block(res, res["S0"], dev, subsets)
    lines += ["", f"## 2. test (S0, {choice})"]
    lines += table("test", chosen, res["S0"], test, args)
    lines += subset_block(chosen, res["S0"], test, subsets)
    lines += ["", f"## 3. 전체 89 기준 판정 (dev + test, S0 vs {choice})"]
    lines += table("all", chosen, res["S0"], items, args)
    lines += subset_block(chosen, res["S0"], items, subsets)
    lines += verdict(choice, res[choice], res["S0"], items, subsets, args)
    lines += ["", "## 4. 오염 점검 (plan §5.1)", "",
              f"- 비교 대상: {choice}의 질문 색인 `{getattr(hq.collection, 'name', '?')}` ({hq.n_docs}개 질문)",
              f"- cos ≥ {CONTAMINATED}: {n_cos}/{len(items)} ({n_cos / len(items):.1%}), "
              f"difflib ≥ {CONTAMINATED}: {n_ratio}/{len(items)} ({n_ratio / len(items):.1%}), "
              f"둘 중 하나라도: {len(items) - len(clean)}/{len(items)}",
              "", "| qid | 질문 | 최고 cos | 가장 가까운 질문 id | 최고 difflib |", "| --- | --- | --- | --- | --- |"]
    for it in sorted(items, key=lambda i: -contam[i["qid"]]["cos"])[:10]:
        c = contam[it["qid"]]
        lines.append(f"| {it['qid']} | {it['question']} | {c['cos']:.3f} | `{c['cos_id']}` | {c['ratio']:.3f} |")
    lines += table("오염 문항 제외", chosen, res["S0"], clean, args)
    lines += verdict(f"{choice} (오염 제외)", res[choice], res["S0"], clean, subsets, args)
    lines += ["", "## 5. 어느 목록이 정답을 찾았나 (Recall@50, 목록별, dev)", "",
              "| 구성 | " + " | ".join(["dense", "bm25", "hq", "hq_bm25"]) + " |", "| --- | --- | --- | --- | --- |"]
    for name, r in res.items():
        cells = []
        for lst in ("dense", "bm25", "hq", "hq_bm25"):
            found = [r[i["qid"]]["list_ranks"].get(lst) for i in dev if lst in r[i["qid"]]["list_ranks"]]
            cells.append(f"{sum(x is not None for x in found) / len(found):.3f}" if found else "-")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "## 6. 노드별 지연 p50 / p95 (ms, dev)"] + latency_block(res, dev)
    lines += ["", f"## 7. 바뀐 문항 (S0 → {choice}, 전체)", "", "| qid | split | qtype | 질문 | S0 | " + choice + " |",
              "| --- | --- | --- | --- | --- | --- |"]
    for it in items:
        a, b = res["S0"][it["qid"]]["rank"], res[choice][it["qid"]]["rank"]
        if a != b:
            lines.append(f"| {it['qid']} | {it['split']} | {it['qtype']} | {it['question']} | {a or '밖'} | {b or '밖'} |")
    report = "\n".join(lines) + "\n"
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"hq_eval_{stamp}_summary.md"
    out.write_text(report, encoding="utf-8")
    with (args.out / f"hq_eval_{stamp}_ranks.jsonl").open("w", encoding="utf-8") as f:
        for name, r in res.items():
            for it in items:
                if it["qid"] in r:
                    f.write(json.dumps({"config": name, "qid": it["qid"], "split": it["split"], "qtype": it["qtype"],
                                        **r[it["qid"]], **{f"contam_{k}": v for k, v in contam[it["qid"]].items()}},
                                       ensure_ascii=False) + "\n")
    print(report)
    print(f"-> {out}")
    print(f"median search {statistics.median(x['ms'] for x in res['S0'].values()):.0f} ms (S0)")


if __name__ == "__main__":
    main()
