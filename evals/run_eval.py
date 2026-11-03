"""Run one evaluation layer and save results.jsonl, manifest.json and report.md under evals/reports/<stamp>-<stage>/.

    python -m evals.run_eval --layer l1 --retriever bm25-catalog --stage T2-04-bm25 [--split test] [--k 10]
    python -m evals.run_eval --layer l1 --retriever oracle --stage selfcheck          # Hit@1 must be 100%
    python -m evals.run_eval --layer l3 --stage T1-27-regrade --from-run <run folder>  # re-grade, no model call
    python -m evals.run_eval --layer l3 --stage p2-l3 --interval 5                     # calls HCX per question
    python -m evals.run_eval --layer e2e --retriever bm25-record --stage p2-e2e --interval 5   # search top 1 -> L3
    python -m evals.run_eval --layer l0 --stage T2-28-l0                               # metadata fill rates, no search
    python -m evals.run_eval --layer l2 --stage T2-28-l2 --split dev                   # parse_query vs query_golden
    python -m evals.run_eval --layer ablation --stage T2-28-slots --split dev --slot-source pred [--retriever chroma]
    python -m evals.run_eval --layer l012 --stage T2-28-l012                           # L0 + L1 filter (gold, pred) + L2, dev

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
import time
from collections import Counter, defaultdict
from pathlib import Path

from evals import ops, report_html, trace
from evals.evaluators import calculation, retrieval
from evals.schemas import DATASETS, GOLDEN, QUERY_GOLDEN, ROOT, TABLES, CalcItem, QueryItem, load_items, load_jsonl

REPORTS = ROOT / "evals" / "reports"
CALC_GOLDEN = DATASETS / "calculation_golden.jsonl"
POOL = DATASETS / "retrieval_pool.csv"
METADATA_SAMPLE = DATASETS / "metadata_sample.jsonl"  # L0 accuracy sample, labelled by the evaluation side (§3.3)
OPS_FIELDS = ("latency_ms", "llm_calls", "input_tokens", "output_tokens", "model", "failed", "retried")
TRACE_FIELDS = ("spans", "retrieve_ms")


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
    if any(r.get("stage2_picked_n") is not None for r in shown):  # an LLM re-ranker ran: score its verdict too (T2-09)
        summary["no_answer_rules"] = retrieval.no_answer_rules(shown)
    return shown, summary, l1_report(args, summary)


def run_l0(args) -> tuple[list[dict], dict, str]:
    """L0 metadata quality over the registry; the load check runs only with --retriever chroma (needs the server)."""
    from evals.evaluators import metadata
    from evals.runners import retrieval as runner

    records = {r.info.key: r for r in runner.load_records()}
    gold = set().union(*(i.gold_keys for i in load_items(GOLDEN)))
    sample = ({json.loads(line)["key"]: json.loads(line) for line in METADATA_SAMPLE.read_text(encoding="utf-8").splitlines()
               if line.strip()} if METADATA_SAMPLE.exists() else None)
    loaded = None
    if args.retriever == "chroma":
        from kosis_rag.retriever import ChromaTableRetriever
        loaded = set(ChromaTableRetriever.connect().collection.get(include=[])["ids"])
    s = metadata.score(records, gold, sample, loaded)
    rows = [{"key": k, **metadata.fields(r), "latency_ms": 0.0} for k, r in records.items()]
    lines = ["| 필드 | 채움률 (전체) | 채움률 (골든 정답 표) |", "| --- | --- | --- |"]
    lines += [f"| {f} | {s['all']['fill'][f]:.3f} | {s['gold']['fill'][f]:.3f} |" for f in metadata.FIELDS]
    fmt = lambda v: "—" if v is None else f"{v:.3f}"
    report = "\n".join([
        f"# L0 metadata quality — {args.stage}", "",
        f"- 표 {s['n_tables']}개, 골든 정답 표 {s['n_gold']}개 · 필드는 필터가 쓰는 함수로 읽음 (kosis_rag.filters / roles)", "",
        *lines, "",
        f"- regions \"해당 없음\"(행정구역 분류 없는 표): 전체 {s['all']['regions_none']:.3f} · 골든 {s['gold']['regions_none']:.3f}",
        f"- measure 미매핑(이름이 세는 대상을 말하지 않음·기타): 전체 {s['all']['measure_unmapped']:.3f} · 골든 {s['gold']['measure_unmapped']:.3f}",
        f"- dimension 미매핑(질문 분류 어휘로 고를 수 없는 분류 이름): 전체 {fmt(s['all']['dimension_unmapped'])}"
        f" ({s['all']['dimension_names']}개) · 골든 {fmt(s['gold']['dimension_unmapped'])} ({s['gold']['dimension_names']}개)",
        "- 정규화 정확도: " + (", ".join(f"{k} {fmt(v)}" for k, v in s["accuracy"].items()) + f" (표본 {s['n_sample']})"
                         if s["accuracy"] else f"표본 대기 ({METADATA_SAMPLE.relative_to(ROOT)} 없음)"),
        "- 적재 일치: " + (("일치" if s["load"]["match"] else f"불일치 — 빠짐 {s['load']['missing']}, 남음 {s['load']['extra']}")
                        + f" ({s['load']['loaded']}개)" if "load" in s else "확인 안 함 (--retriever chroma 로 확인)"),
    ]) + "\n"
    return rows, s, report


def run_l2(args) -> tuple[list[dict], dict, str]:
    """L2: parse_query on every labelled question against query_golden.jsonl."""
    from kosis_rag.query import parse_query

    from evals.evaluators import query

    items: list[QueryItem] = load_jsonl(QUERY_GOLDEN, QueryItem)
    rows = []
    for item in items:
        if args.split != "all" and item.split != args.split:
            continue
        started = time.perf_counter()
        pred = parse_query(item.question).model_dump()
        rows.append({"qid": item.qid, "split": item.split, **query.score_query(item, pred), "pred": pred,
                     "latency_ms": (time.perf_counter() - started) * 1000})
    agg = query.aggregate(rows)
    fmt = lambda v: "—" if v is None else f"{v:.3f}"
    cols = (*query.FIELDS, "all_fields")
    lines = ["| split | n | " + " | ".join(cols) + " |", "|" + " --- |" * (len(cols) + 2)]
    for name, m in agg.items():
        lines.append(f"| {name} | {m['n']} | " + " | ".join(f"{fmt(m[c])} ({m['n_' + c]})" for c in cols) + " |")
    report = "\n".join([
        f"# L2 query understanding — {args.stage}", "",
        f"- `parse_query` (규칙, LLM 0) vs `{QUERY_GOLDEN.relative_to(ROOT)}` · 괄호는 라벨이 있는 문항 수, — 는 라벨 없음", "",
        *lines, "",
        "- 빈 슬롯도 채점: 정답이 빈 필드에 값을 내면 오답. measures는 정답 개념이 예측 검색어 안에 있으면 정답.",
    ]) + "\n"
    return rows, {"metrics": agg}, report


def run_ablation(args) -> tuple[list[dict], dict, str]:
    """Slot ablation (§4.4): filter recall per combination; with --retriever chroma also Hit/MRR, flips, verdicts."""
    from kosis_rag.query import parse_query

    from evals.runners import ablation
    from evals.runners import retrieval as runner

    items = [i for i in load_items(GOLDEN) if args.split == "all" or i.split == args.split]
    records = {r.info.key: r for r in runner.load_records()}
    if args.slot_source == "gold":
        slots = {q.qid: ablation.gold_parsed(q) for q in load_jsonl(QUERY_GOLDEN, QueryItem)}
    else:
        slots = {i.qid: parse_query(i.question) for i in items}
    on = ablation.combos(args.slots.split(","))
    retriever = None
    if args.retriever == "chroma":
        from kosis_rag.retriever import ChromaTableRetriever
        retriever = ChromaTableRetriever.connect(k=args.k)
    rows = ablation.run(items, slots, records, on, retriever, args.k)
    tau = args.tau
    if tau is None and retriever is not None:  # chosen once on `none`, kept for every combination
        tau = retrieval.best_tau([(r["top_score"], not r["answerable"]) for r in rows if r["combo"] == "none" and r["split"] == "dev"])
    s = ablation.summarize(rows, on, slots, tau if tau is not None else 0.0)
    fmt = lambda v, p=3: "—" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))
    lines = ["| 조합 | n | MRR | Hit@1 | Hit@5 | Hit@10 | 필터 recall | 후보 중앙값 | 0건 답 있음 | 0건 답 없음 | no_period 거절 |",
             "|" + " --- |" * 11]
    for c, m in s["combos"].items():
        lines.append(f"| {c} | {m['n']} | {fmt(m.get('mrr'))} | {fmt(m.get('hit@1'))} | {fmt(m.get('hit@5'))} | {fmt(m.get('hit@10'))} "
                     f"| {fmt(m['filter_recall'])} | {fmt(m['candidates_median'], 0)} | {fmt(m['zero_answerable'])} "
                     f"| {fmt(m['zero_no_answer'])} ({m['n_no_answer']}) | {fmt(m['no_period_refusal'])} ({m['n_no_period']}) |")
    slot_lines = ["| 슬롯 | 슬롯 있는 문항 | only vs none: Hit@5 순증 · Hit@1 · recall Δ | all vs all-s: Hit@5 순증 · Hit@1 · recall Δ | 조건부 ΔMRR | 판정 |",
                  "|" + " --- |" * 6]
    for name, v in s["slots"].items():
        o, mi = v["only"], v["minus"]
        slot_lines.append(f"| {name} | {v['n_with_slot']} | {o['hit5_net']:+d} · {o['hit1_net']:+d} · {o['recall_delta']:+.3f} "
                          f"| {mi['hit5_net']:+d} · {mi['hit1_net']:+d} · {mi['recall_delta']:+.3f} | {fmt(v['conditional_mrr_delta'])} | {v['verdict']} |")
    flips = [f"- {n}: only 좋아짐 {v['only'].get('better', [])} · 나빠짐 {v['only'].get('worse', [])}" for n, v in s["slots"].items()]
    lost = sorted({(r["combo"], r["qid"]) for r in rows if r["answerable"] and not r["filter_pass"]})
    report = "\n".join([
        f"# Slot ablation — {args.stage}", "",
        f"- 슬롯 출처 `{args.slot_source}` · 슬롯 {args.slots} · split `{args.split}` · 검색기 `{args.retriever if retriever else '없음 (필터만)'}`"
        f" · k={min(args.k, 10)} · tau {fmt(tau, 4)} (none에서 한 번 선택)", "",
        *lines, "", *slot_lines, "",
        "- 판정 (§4.4): 도움 = 두 비교 모두 Hit@5 순증 ≥ 2, Hit@1 순감 없음, recall 하락 없음 / 해로움 = 어느 쪽이든 Hit@5 순감 또는 recall 하락 / 그 밖 불분명."
        + (" 검색 없이 돌리면 recall만 보므로 해로움 또는 불분명만 나온다." if retriever is None else ""),
        "- 단계적 완화: 없음 (T2-29 미구현). 0건 = hard 필터가 남긴 표가 0개인 문항 비율, 완화가 시작될 자리다(완화 후 결과는 아님).",
        "", "## 뒤집힌 문항 (Hit@5)", "", *flips,
        "", f"## 필터가 정답 표를 지운 문항 ({len(lost)})", "",
        *[f"- {c} {q}: " + "; ".join(f"{k} {why}" for k, why in next(r for r in rows if r['combo'] == c and r['qid'] == q)['gold_removed'].items())
          for c, q in lost],
    ]) + "\n"
    return rows, s, report


def run_l012(args) -> tuple[list[dict], dict, str]:
    """L0 + L1 filter (gold and pred slots) + L2 in one run folder; rows carry `layer`, summary and ops stay per layer."""
    with_source = lambda src: argparse.Namespace(**{**vars(args), "slot_source": src})  # noqa: E731
    parts = {"l0": run_l0(args), "l1-gold": run_ablation(with_source("gold")), "l1-pred": run_ablation(with_source("pred")),
             "l2": run_l2(args)}
    # run() skips questions without a slot reading; gold reads query_golden, pred reads the retrieval golden set
    gold, pred = ({(r["combo"], r["qid"]) for r in parts[p][0]} for p in ("l1-gold", "l1-pred"))
    assert gold == pred, f"gold and pred slot runs cover different questions: {sorted(gold ^ pred)[:5]}"
    rows = [{**r, "layer": name} for name, (part, _, _) in parts.items() for r in part]
    report = "\n".join(f"<!-- {name} -->\n{rep}" for name, (_, _, rep) in parts.items())
    return rows, {name: s for name, (_, s, _) in parts.items()}, report


def l1_report(args, s: dict) -> str:
    cols = ["hit@1", "hit@5", "hit@10", "rr", "slot_recall@5", "slot_recall@10", "complete@3", "complete@10", "unjudged@10"]
    head = "| qtype | n | " + " | ".join("MRR" if c == "rr" else c for c in cols) + " |"
    lines = [head, "|" + " --- |" * (len(cols) + 2)]
    order = ["all"] + sorted(k for k in s["metrics"] if k != "all")
    for name in order:
        m = s["metrics"][name]
        lines.append(f"| {name} | {m['n']} | " + " | ".join(f"{m.get(c, 0):.3f}" for c in cols) + " |")
    na = s["no_answer"]
    rules = []
    if "no_answer_rules" in s:  # T2-09: the LLM re-ranker's verdict next to tau
        nr = s["no_answer_rules"]
        rules = ["", "| 답 없음 규칙 | P | R | F1 | tp / fp / fn | 사유별 recall |", "| --- | --- | --- | --- | --- | --- |"]
        rules += [f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['tp']} / {m['fp']} / {m['fn']} | "
                  + ", ".join(f"{g} {a}/{b}" for g, (a, b) in m["recall_by_reason"].items()) + " |"
                  for name, m in nr.items() if name in ("tau", "llm", "either")]
        rules += ["", f"- llm: 2차 LLM이 고른 표 0개(호출 오류 제외) · either: llm 또는 tau · LLM 사유 일치 "
                      f"{nr['llm_reason_match'][0]}/{nr['llm_reason_match'][1]} · LLM 호출 오류 {nr['llm_errors']}건"]
    return "\n".join([
        f"# L1 retrieval — {args.stage}", "",
        f"- retriever `{args.retriever}`, k={args.k}, split `{args.split}`, 답 있는 문항 기준", "",
        *lines, "",
        f"- 답 없음 (tau={s['tau']:.4g}, dev에서 선택): P {na['precision']:.3f} · R {na['recall']:.3f} · F1 {na['f1']:.3f} "
        f"(tp {na['tp']}, fp {na['fp']}, fn {na['fn']})",
        *rules,
        "- slot_recall@k: 정답 슬롯(표 자리) 중 상위 k에서 채운 비율. 단일 표 문항은 hit@k와 같고 복합 문항에서만 다르다.",
        "- unjudged@10: 상위 10개 중 사람이 판정하지 않은 표 비율. 높으면 정답 라벨 추가 판정이 필요하다 (plans/T2-02.md 동결 규칙).",
    ]) + "\n"


def ops_section(op: dict, title: str = "운영 지표") -> str:
    c = op["cost"]
    cost = f"{c['total']:.2f} {c['currency']}" if c["total"] is not None else f"단가 미설정 ({', '.join(c['unpriced_models']) or '토큰 없음'}; evals/pricing.json)"
    return "\n".join(["", f"## {title}", "",
        f"- Latency: p50 {op['latency_ms']['p50']:.1f} ms · p95 {op['latency_ms']['p95']:.1f} ms · 평균 {op['latency_ms']['mean']:.1f} ms",
        f"- Token Usage: 입력 {op['tokens']['input']:,} · 출력 {op['tokens']['output']:,} (호출당 {op['tokens']['per_call_mean']:.0f}) · LLM 호출 {op['llm_calls']}",
        f"- Cost: {cost}",
        f"- Tool Failure Rate: {op['tool_failure_rate']:.1%} ({op['tool_calls']}회 중) · 재시도 비율 {op['retry_rate']:.1%}",
    ]) + "\n"


def failure_section(rows: list[dict], exported: bool) -> str:
    """Stage counts and one line per non-PASS question: the span it failed in and why (evals.trace.blame)."""
    failed = [r for r in rows if r.get("failure")]
    counts = Counter(r["failure"]["stage"] for r in failed)
    lines = ["", "## 실패 단계 (trace)", "",
             "- 판정 순서: " + " > ".join(f"{k}({v})" for k, v in trace.STAGES.items()) + " — 첫 번째로 맞는 규칙 하나 (plans/T2-15.md)",
             "- 단계별: " + (", ".join(f"{trace.STAGES[k]} {counts[k]}" for k in trace.STAGES if counts[k]) or "실패 없음"),
             f"- Jaeger: {'trace 전송함 (' + trace.JAEGER_UI + ')' if exported else 'OTEL_EXPORTER_OTLP_ENDPOINT 미설정, 전송 안 함'}"]
    if failed:
        lines += ["", "| qid | 판정 | 단계 | span | 사유 | 거친 span |", "| --- | --- | --- | --- | --- | --- |"]
        for r in failed:
            f = r["failure"]
            path = " → ".join(s["name"] + ("✕" if s["status"] == "error" else "") for s in r.get("spans") or []) or "-"
            lines.append(f"| {r['qid']} | {r['verdict']} | {f['label']} | {f['span'] or '-'} | {f['reason'].replace('|', '/')} | {path} |")
    return "\n".join(lines) + "\n"


def grade(items: list[CalcItem], answers: list[dict]) -> list[dict]:
    """L3 grading of each answer against the golden table's cells, with the ops fields kept per row."""
    rows = []
    for item, a in zip(items, answers):
        cells = calculation.read_cells(ROOT / "data" / "values" / f"{item.table_key}.parquet")
        g = calculation.grade(item, a["answer"], cells, a["error"])
        rows.append({**g.model_dump(), "table_key": item.table_key, **{k: a[k] for k in OPS_FIELDS + TRACE_FIELDS if k in a},
                     "answer": a["answer"].model_dump(mode="json") if a["answer"] else None})
    return rows


def verdict_table(rows: list[dict]) -> tuple[dict, Counter, list[str]]:
    """Verdict counts per kind, error counts and the markdown table both L3 and L4 reports show."""
    by_kind: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by_kind["all"][r["verdict"]] += 1
        by_kind[r["kind"]][r["verdict"]] += 1
    errors = Counter(e for r in rows for e in r["errors"])
    lines = ["| kind | n | PASS | PARTIAL | FAIL |", "| --- | --- | --- | --- | --- |"]
    for k in ["all", "answerable", "not_answerable", "clarify", "default_allowed"]:
        v = by_kind.get(k, Counter())
        lines.append(f"| {k} | {sum(v.values())} | {v['PASS']} | {v['PARTIAL']} | {v['FAIL']} |")
    return {k: dict(v) for k, v in by_kind.items()}, errors, lines


def run_l3(args) -> tuple[list[dict], dict, str]:
    from evals.runners import stat as runner

    items: list[CalcItem] = load_jsonl(CALC_GOLDEN, CalcItem)
    answers = runner.read_run(args.from_run, items) if args.from_run else runner.run(items, args.interval)
    rows = grade(items, answers)
    verdicts, errors, lines = verdict_table(rows)
    summary = {"verdicts": verdicts, "errors": dict(errors)}
    report = "\n".join([
        f"# L3 calculation — {args.stage}", "",
        f"- 답: {'과거 실행 재채점 ' + str(args.from_run) if args.from_run else 'summarize_stat 호출'} · 정답 표 고정(oracle)", "",
        *lines, "",
        "- 오류 유형: " + (", ".join(f"{k} {v}" for k, v in errors.most_common()) or "없음"),
    ]) + "\n"
    return rows, summary, report


def run_e2e(args) -> tuple[list[dict], dict, str]:
    """L4 end to end: the retriever picks the table (top 1), then the same answer path and grader as L3."""
    from evals.runners import e2e
    from evals.runners import retrieval as runner

    if args.retriever == "oracle":
        raise SystemExit("e2e with the oracle retriever is L3: run --layer l3 instead")
    items: list[CalcItem] = load_jsonl(CALC_GOLDEN, CalcItem)
    raw = e2e.run(runner.build(args.retriever, [], args.k), items, args.k, args.interval)
    rows = [{**g, **e2e.locate(item, a, g["verdict"]), "error": a["error"]}
            for item, a, g in zip(items, raw, grade(items, raw))]
    verdicts, errors, lines = verdict_table(rows)
    right = [r for r in rows if r["right_table"]]
    split = {"right": dict(Counter(r["verdict"] for r in right)),
             "wrong": dict(Counter(r["verdict"] for r in rows if not r["right_table"]))}
    causes = dict(Counter(r["cause"] for r in rows))
    summary = {"verdicts": verdicts, "errors": dict(errors), "top1_right": len(right), "by_top1": split, "causes": causes}
    n = len(rows) or 1
    report = "\n".join([
        f"# L4 end to end — {args.stage}", "",
        f"- retriever `{args.retriever}`, k={args.k} · 검색 1위 표로 summarize_stat 호출 · L3와 같은 채점기(정답 표 기준)", "",
        *lines, "",
        f"- 검색 1위가 정답 표: {len(right)}/{len(rows)} ({len(right) / n:.1%})",
        f"  - 맞힌 표에서 PASS {split['right'].get('PASS', 0)}/{len(right)} · 틀린 표에서 PASS {split['wrong'].get('PASS', 0)}/{len(rows) - len(right)}",
        "- 실패 원인: " + ", ".join(f"{k} {causes.get(k, 0)}" for k in ("pass", "retrieval", "answer"))
        + " (retrieval = 1위가 다른 표, answer = 표는 맞았는데 답이 틀림)",
        "- 오류 유형: " + (", ".join(f"{k} {v}" for k, v in errors.most_common()) or "없음"),
    ]) + "\n"
    return rows, summary, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one evaluation layer (L1 retrieval, L3 calculation, e2e).")
    parser.add_argument("--layer", choices=("l0", "l1", "l2", "ablation", "l012", "l3", "e2e"), required=True)  # e2e: first part of L4
    parser.add_argument("--stage", required=True, help="what this run measures, e.g. T2-04-bm25")
    parser.add_argument("--split", choices=("dev", "test", "all"), help="default: dev for l012 (the slot baselines), test otherwise")
    parser.add_argument("--retriever", default="bm25-catalog", help="l1, e2e: oracle (l1 only) | bm25-catalog | bm25-record | chroma")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--tau", type=float, help="l1: fixed no-answer threshold (default: best F1 on dev)")
    parser.add_argument("--from-run", type=Path, help="l3: grade a past run folder instead of calling the model")
    parser.add_argument("--slot-source", choices=("gold", "pred"), default="pred", help="ablation: query_golden labels or parse_query")
    parser.add_argument("--slots", default="year,cycle,region,dimension,measure", help="ablation: slots to switch")
    parser.add_argument("--interval", type=float, default=0.0, help="l3, e2e: seconds between model calls")
    args = parser.parse_args(argv)
    args.split = args.split or ("dev" if args.layer == "l012" else "test")

    layers = {"l0": run_l0, "l1": run_l1, "l2": run_l2, "ablation": run_ablation, "l012": run_l012, "l3": run_l3, "e2e": run_e2e}
    rows, summary, report = layers[args.layer](args)
    exported = False
    if args.layer in ("l3", "e2e"):  # only these rows carry verdict/spans for trace.blame
        for r in rows:
            r["failure"] = trace.blame(r)
        summary["failures"] = dict(Counter(r["failure"]["stage"] for r in rows if r["failure"]))
        exported = trace.export(rows, args.layer, args.retriever if args.layer == "e2e" else None)
        report += failure_section(rows, exported)
    if args.layer == "l012":  # per layer: L0 and filter-only rows carry latency 0.0 and would bury the L2 timings
        op = {name: ops.summarize([r for r in rows if r["layer"] == name]) for name in summary}
        report += "".join(ops_section(o, f"운영 지표 — {name}") for name, o in op.items())
    else:
        op = ops.summarize(rows)
        report += ops_section(op)
    run_id = f"{dt.datetime.now():%Y%m%d-%H%M%S}-{args.stage}"
    out = REPORTS / run_id
    out.mkdir(parents=True)
    with (out / "results.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows)
    ref = lambda path: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}  # noqa: E731
    golden = {"l2": QUERY_GOLDEN, "l3": CALC_GOLDEN, "e2e": CALC_GOLDEN}.get(args.layer, GOLDEN)
    slotted = args.layer in ("ablation", "l012")
    manifest = {
        "run_id": run_id, "layer": args.layer, "stage": args.stage, "split": args.split,
        "golden": {"l0·l1": ref(GOLDEN), "l2": ref(QUERY_GOLDEN)} if args.layer == "l012" else ref(golden),
        "source_snapshot": {"tables.jsonl": sha256(TABLES)},
        "git": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
        "vendor_phase1": git("-C", "vendor/phase1", "rev-parse", "HEAD"),
        "retriever": args.retriever if args.layer in ("l1", "ablation", "l012", "e2e") else None,
        "top_k": args.k if args.layer in ("l1", "ablation", "l012", "e2e") else None,
        "slot_source": "gold,pred" if args.layer == "l012" else args.slot_source if slotted else None,
        "slots": args.slots if slotted else None,
        "from_run": str(args.from_run) if args.from_run else None,
        "summary": summary,
        "ops": op,
        "jaeger": {"ui": trace.JAEGER_UI, "service": "kosis-evals"} if exported else None,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "ops.json").write_text(json.dumps(op, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "report.md").write_text(report, encoding="utf-8")
    if args.layer == "l012":
        for name, o in op.items():
            ops.append_history(run_id, name, args.stage, {"l0": "registry", "l2": "parse_query"}.get(name, args.retriever), o)
    else:
        target = args.retriever if args.layer in ("l1", "ablation") else ",".join(op["models"]) or "summarize_stat"
        if args.layer in ("l0", "l2"):
            target = "registry" if args.layer == "l0" else "parse_query"
        if args.layer == "e2e":
            target = f"{args.retriever}+{target}"
        ops.append_history(run_id, args.layer, args.stage, target, op)
    if args.layer in ("l1", "l3", "e2e"):  # ponytail: l0/l2/ablation write report.md only; add charts when they are presented
        report_html.write(out)
    print(report)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
