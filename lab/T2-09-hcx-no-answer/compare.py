"""T2-09 lab: compare L1 runs (no re-ranker vs HCX-007 stage 2 with thinking none / low) per split.

    PYTHONPATH=.:src:vendor/phase1:vendor/phase1/src python lab/T2-09-hcx-no-answer/compare.py base=<run dir> none=<run dir> low=<run dir>

Reads each run's results.jsonl (evals.run_eval --layer l1 --split all) and prints markdown tables: ranking metrics
on answerable items, no-answer P/R/F1 per rule (tau / llm / either), recall per gold reason, refused answerable
items, latency and calls. tau is the one each run chose on dev (stored per row as no_answer_pred).
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from evals.evaluators.retrieval import llm_no_answer, no_answer_rules


def load(run: str) -> list[dict]:
    return [json.loads(line) for line in (Path(run) / "results.jsonl").read_text(encoding="utf-8").splitlines() if line]


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def table(runs: dict[str, list[dict]], split: str) -> str:
    out = [f"### {split}", "",
           "| 설정 | Hit@1 | Hit@5 | Hit@10 | MRR | 규칙 | 답 없음 P | R | F1 | 답 있음 오거절 | 사유별 recall (범위 밖 / 기간 / 항목) | 지연 p50 / p95 | LLM 호출 |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, rows in runs.items():
        rows = [r for r in rows if r["split"] == split]
        ans = [r for r in rows if r["answerable"]]
        rank = (f"{sum(r['hit@1'] for r in ans)} | {sum(r['hit@5'] for r in ans)} | {sum(r['hit@10'] for r in ans)} | "
                f"{sum(r['rr'] for r in ans) / len(ans):.3f}")
        lat = [r["latency_ms"] / 1000 for r in rows]
        ops = f"{statistics.median(lat):.2f} / {pct(lat, 0.95):.2f} s | {sum(r['llm_calls'] for r in rows)}"
        reranked = any(r.get("stage2_picked_n") is not None for r in rows)
        rules = no_answer_rules(rows)  # without stage 2, llm never fires: show tau only
        preds = {"tau": lambda r: r["no_answer_pred"], "llm": llm_no_answer,
                 "either": lambda r: r["no_answer_pred"] or llm_no_answer(r)}
        for rule in ("tau", "llm", "either") if reranked else ("tau",):
            m = rules[rule]
            by = m["recall_by_reason"]
            reasons = " / ".join(f"{by.get(g, [0, 0])[0]}/{by.get(g, [0, 0])[1]}" for g in ("out_of_scope", "no_period", "no_item"))
            refused = sum(1 for r in ans if preds[rule](r))
            out.append(f"| {name} | {rank} | {rule} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | "
                       f"{refused}/{len(ans)} | {reasons} | {ops} |")
        if reranked:
            a, b = rules["llm_reason_match"]
            out.append(f"| {name} | | | 사유 일치 {a}/{b} · 호출 오류 {rules['llm_errors']} | | | | | | |")
    return "\n".join(out)


def main(argv: list[str]) -> None:
    runs = {name: load(path) for name, path in (a.split("=", 1) for a in argv)}
    print("\n\n".join(table(runs, split) for split in ("dev", "test")))


if __name__ == "__main__":
    main(sys.argv[1:])
