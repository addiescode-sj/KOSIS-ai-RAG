"""Run one evaluation layer and save results.jsonl, manifest.json and report.md under evals/reports/<stamp>-<stage>/.

    python -m evals.run_eval --layer l1 --retriever bm25-catalog --stage T2-04-bm25 [--split test] [--k 10]
    python -m evals.run_eval --layer l1 --retriever oracle --stage selfcheck          # Hit@1 must be 100%
    python -m evals.run_eval --layer l3 --stage T1-27-regrade --from-run <run folder>  # re-grade, no model call
    python -m evals.run_eval --layer l3 --stage p2-l3 --interval 5                     # calls HCX per question

L1 runs every item (BM25 is fast) so the no-answer threshold tau is chosen on dev and applied to the split shown.
Every run writes results.jsonl, manifest.json, ops.json (latency, tokens, cost, tool failure rate), report.md and
report.html (charts, evals.report_html), and appends one aggregate line to evals/reports/ops_history.jsonl.
Run folders hold question text and table keys and stay out of git; ops_history holds aggregates only.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from evals import ops, report_html
from evals.evaluators import calculation, retrieval
from evals.schemas import DATASETS, GOLDEN, ROOT, TABLES, CalcItem, load_items, load_jsonl

REPORTS = ROOT / "evals" / "reports"
CALC_GOLDEN = DATASETS / "calculation_golden.jsonl"
POOL = DATASETS / "retrieval_pool.csv"
OPS_FIELDS = ("latency_ms", "llm_calls", "input_tokens", "output_tokens", "model", "failed", "retried")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def judged_keys(path: Path) -> dict[str, set[str]]:
    """Keys a person looked at per question: every row of the labelled pool, gold or not."""
    judged: dict[str, set[str]] = defaultdict(set)
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                judged[row["qid"]].add(row["key"])
    return judged


def run_l1(args) -> tuple[list[dict], dict, str]:
    from evals.runners import retrieval as runner  # LangChain loads only for L1

    items = load_items(GOLDEN)
    rows = runner.run(runner.build(args.retriever, items, args.k), items, args.k)
    by_qid = {i.qid: i for i in items}
    tau = args.tau if args.tau is not None else retrieval.best_tau(
        [(r["top_score"], not r["answerable"]) for r in rows if r["split"] == "dev"])
    judged = judged_keys(POOL)
    shown = [r for r in rows if args.split == "all" or r["split"] == args.split]
    for r in shown:
        r["no_answer_pred"] = retrieval.predict_no_answer(r["top_score"], tau)
        if r["answerable"]:
            r.update(retrieval.score_item(by_qid[r["qid"]], r["keys"], judged.get(r["qid"])))
    scored = [r for r in shown if r["answerable"]]
    agg = retrieval.aggregate(scored)
    no_answer = retrieval.prf([(r["no_answer_pred"], not r["answerable"]) for r in shown])
    summary = {"tau": tau, "metrics": agg, "no_answer": no_answer, "errors": sum(1 for r in shown if r["error"])}
    return shown, summary, l1_report(args, summary)


def l1_report(args, s: dict) -> str:
    cols = ["hit@1", "hit@5", "hit@10", "rr", "slot_recall@5", "slot_recall@10", "complete@3", "complete@10", "unjudged@10"]
    head = "| qtype | n | " + " | ".join("MRR" if c == "rr" else c for c in cols) + " |"
    lines = [head, "|" + " --- |" * (len(cols) + 2)]
    order = ["all"] + sorted(k for k in s["metrics"] if k != "all")
    for name in order:
        m = s["metrics"][name]
        lines.append(f"| {name} | {m['n']} | " + " | ".join(f"{m.get(c, 0):.3f}" for c in cols) + " |")
    na = s["no_answer"]
    return "\n".join([
        f"# L1 retrieval — {args.stage}", "",
        f"- retriever `{args.retriever}`, k={args.k}, split `{args.split}`, 답 있는 문항 기준", "",
        *lines, "",
        f"- 답 없음 (tau={s['tau']:.4g}, dev에서 선택): P {na['precision']:.3f} · R {na['recall']:.3f} · F1 {na['f1']:.3f} "
        f"(tp {na['tp']}, fp {na['fp']}, fn {na['fn']})",
        "- slot_recall@k: 정답 슬롯(표 자리) 중 상위 k에서 채운 비율. 단일 표 문항은 hit@k와 같고 복합 문항에서만 다르다.",
        "- unjudged@10: 상위 10개 중 사람이 판정하지 않은 표 비율. 높으면 정답 라벨 추가 판정이 필요하다 (plans/T2-02.md 동결 규칙).",
    ]) + "\n"


def ops_section(op: dict) -> str:
    c = op["cost"]
    cost = f"{c['total']:.2f} {c['currency']}" if c["total"] is not None else f"단가 미설정 ({', '.join(c['unpriced_models']) or '토큰 없음'}; evals/pricing.json)"
    return "\n".join(["", "## 운영 지표", "",
        f"- Latency: p50 {op['latency_ms']['p50']:.1f} ms · p95 {op['latency_ms']['p95']:.1f} ms · 평균 {op['latency_ms']['mean']:.1f} ms",
        f"- Token Usage: 입력 {op['tokens']['input']:,} · 출력 {op['tokens']['output']:,} (호출당 {op['tokens']['per_call_mean']:.0f}) · LLM 호출 {op['llm_calls']}",
        f"- Cost: {cost}",
        f"- Tool Failure Rate: {op['tool_failure_rate']:.1%} ({op['tool_calls']}회 중) · 재시도 비율 {op['retry_rate']:.1%}",
    ]) + "\n"


def run_l3(args) -> tuple[list[dict], dict, str]:
    from evals.runners import stat as runner

    items: list[CalcItem] = load_jsonl(CALC_GOLDEN, CalcItem)
    answers = runner.read_run(args.from_run, items) if args.from_run else runner.run(items, args.interval)
    rows = []
    for item, a in zip(items, answers):
        cells = calculation.read_cells(ROOT / "data" / "values" / f"{item.table_key}.parquet")
        g = calculation.grade(item, a["answer"], cells, a["error"])
        rows.append({**g.model_dump(), **{k: a[k] for k in OPS_FIELDS if k in a},
                     "answer": a["answer"].model_dump(mode="json") if a["answer"] else None})
    by_kind: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by_kind["all"][r["verdict"]] += 1
        by_kind[r["kind"]][r["verdict"]] += 1
    errors = Counter(e for r in rows for e in r["errors"])
    summary = {"verdicts": {k: dict(v) for k, v in by_kind.items()}, "errors": dict(errors)}
    lines = ["| kind | n | PASS | PARTIAL | FAIL |", "| --- | --- | --- | --- | --- |"]
    for k in ["all", "answerable", "not_answerable", "clarify", "default_allowed"]:
        v = by_kind.get(k, Counter())
        lines.append(f"| {k} | {sum(v.values())} | {v['PASS']} | {v['PARTIAL']} | {v['FAIL']} |")
    report = "\n".join([
        f"# L3 calculation — {args.stage}", "",
        f"- 답: {'과거 실행 재채점 ' + str(args.from_run) if args.from_run else 'summarize_stat 호출'} · 정답 표 고정(oracle)", "",
        *lines, "",
        "- 오류 유형: " + (", ".join(f"{k} {v}" for k, v in errors.most_common()) or "없음"),
    ]) + "\n"
    return rows, summary, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one evaluation layer (L1 retrieval, L3 calculation).")
    parser.add_argument("--layer", choices=("l1", "l3"), required=True)  # l2, l4: T2-14, T2-10 (plans/T2-03.md)
    parser.add_argument("--stage", required=True, help="what this run measures, e.g. T2-04-bm25")
    parser.add_argument("--split", choices=("dev", "test", "all"), default="test")
    parser.add_argument("--retriever", default="bm25-catalog", help="l1: oracle | bm25-catalog | bm25-record")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--tau", type=float, help="l1: fixed no-answer threshold (default: best F1 on dev)")
    parser.add_argument("--from-run", type=Path, help="l3: grade a past run folder instead of calling the model")
    parser.add_argument("--interval", type=float, default=0.0, help="l3: seconds between model calls")
    args = parser.parse_args(argv)

    rows, summary, report = (run_l1 if args.layer == "l1" else run_l3)(args)
    op = ops.summarize(rows)
    report += ops_section(op)
    run_id = f"{dt.datetime.now():%Y%m%d-%H%M%S}-{args.stage}"
    out = REPORTS / run_id
    out.mkdir(parents=True)
    with (out / "results.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows)
    golden = GOLDEN if args.layer == "l1" else CALC_GOLDEN
    manifest = {
        "run_id": run_id, "layer": args.layer, "stage": args.stage, "split": args.split,
        "golden": {"path": str(golden.relative_to(ROOT)), "sha256": sha256(golden)},
        "source_snapshot": {"tables.jsonl": sha256(TABLES)},
        "git": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
        "vendor_phase1": git("-C", "vendor/phase1", "rev-parse", "HEAD"),
        "retriever": args.retriever if args.layer == "l1" else None, "top_k": args.k if args.layer == "l1" else None,
        "from_run": str(args.from_run) if args.from_run else None,
        "summary": summary,
        "ops": op,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "ops.json").write_text(json.dumps(op, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "report.md").write_text(report, encoding="utf-8")
    target = args.retriever if args.layer == "l1" else ",".join(op["models"]) or "summarize_stat"
    ops.append_history(run_id, args.layer, args.stage, target, op)
    report_html.write(out)
    print(report)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
