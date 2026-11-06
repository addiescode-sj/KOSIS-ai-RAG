"""report.md of a question build (T2-39; T2-41 S6a: from the old hq/graph.py). Called by the write node."""

from __future__ import annotations

import random
from collections import Counter
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from kosis_rag_build.hq import prompt as hq_prompt
from kosis_rag_build.hq.constants import EXAMPLES, PROMPT_VERSION
from kosis_rag_build.hq.reply import parse_reply
from kosis_rag_build.schemas import TYPES, TableQuestions

if TYPE_CHECKING:
    from kosis_rag_build.schemas import HqBuildState


def report(state: HqBuildState, bid: str, totals: dict[str, Any]) -> str:
    """report.md: counts per type and rule, questions per table, failures, 20 random kept examples (seed 0).

    In: state after filter_all; bid = build id; totals = build_totals(state). Out: markdown text.
    Side effects: none. Node: write.
    """
    names = {r.key: (r.record.info.table_name if r.record else r.key) for r in state.results}
    rule_by_type: Counter = Counter()
    rule_reasons: Counter = Counter()
    gen_by_type: Counter = Counter()
    for r in state.results:
        for q in r.questions:
            gen_by_type[q.type] += 1
        for reason in r.dropped.values():
            rule_reasons[reason.split(":", 1)[0]] += 1
    # the rule-dropped questions' types: the reply's questions at the places rule_check did not keep
    for r in state.results:
        if not r.dropped or not r.raw:
            continue
        try:
            reply = TableQuestions.model_validate({**parse_reply(r.raw), "key": r.key})
        except (ValueError, ValidationError):
            continue
        kept_at = set(r.positions)
        for n, q in enumerate(reply.questions):
            if n not in kept_at:
                rule_by_type[q.type] += 1
                gen_by_type[q.type] += 1
    score_by_type = Counter(it["type"] for it in state.score_dropped)
    kept_by_type = Counter(it["type"] for it in state.kept)
    per_table = Counter(it["key"] for it in state.kept)
    hist = Counter(per_table.get(r.key, 0) for r in state.results if r.error is None)
    lines = [f"# 가상 질문 빌드 {bid}", "",
             f"- prompt {PROMPT_VERSION} (`{hq_prompt.prompt_hash()[:8]}`), model `{state.model}`, "
             f"tables.jsonl `{state.tables_sha[:8]}`, 표 {totals['tables']}개" + (f" (앞 {state.limit}개만)" if state.limit else ""),
             f"- 걸러내기: 규칙 → 1차 리랭커 점수 전체 하위 {round(state.frac * 100)}% 제거, 표당 최소 {state.min_keep}개 유지",
             f"- 응답 {totals['replies']}개 (이번 실행 API 호출 {totals['calls']} · 캐시 {totals['cached']}), "
             f"토큰 입력 {totals['input_tokens']:,} · 출력 {totals['output_tokens']:,} "
             f"(이번 실행 {totals['new_input_tokens']:,} · {totals['new_output_tokens']:,}), API 시간 {totals['api_seconds']}초",
             f"- 실패 표 {totals['failed']}개 (`failed.jsonl`), 질문이 남은 표 {totals['tables_with_questions']}개", "",
             "## 유형별", "", "| 유형 | 생성 | 규칙 제거 | 점수 제거 | 남음 |", "| --- | --- | --- | --- | --- |"]
    for t in TYPES:
        lines.append(f"| {t} | {gen_by_type[t]} | {rule_by_type[t]} | {score_by_type[t]} | {kept_by_type[t]} |")
    lines.append(f"| 합계 | {totals['generated']} | {totals['rule_dropped']} | {totals['score_dropped']} | {totals['kept']} |")
    lines += ["", "## 규칙별 제거", "", "| 규칙 | 수 |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in rule_reasons.most_common()] or ["| - | 0 |"]
    lines += ["", "## 표당 남은 질문 수 (실패 표 제외)", "", "| 질문 수 | 표 수 |", "| --- | --- |"]
    lines += [f"| {n} | {hist[n]} |" for n in sorted(hist)]
    failed = [r for r in state.results if r.error]
    if failed:
        lines += ["", "## 실패 표 (최대 30개)", ""] + [f"- `{r.key}`: {r.error}" for r in failed[:30]]
    rng = random.Random(0)
    sample = rng.sample(state.kept, min(EXAMPLES, len(state.kept)))
    lines += ["", f"## 남은 질문 예시 {len(sample)}개 (무작위, seed 0)", "", "| id | 유형 | 질문 | 표 |", "| --- | --- | --- | --- |"]
    lines += [f"| `{it['id']}` | {it['type']} | {it['text']} | {names.get(it['key'], it['key'])} |" for it in sample]
    dropped = [(r.key, txt, reason) for r in state.results for txt, reason in r.dropped.items()]
    dsample = rng.sample(dropped, min(10, len(dropped)))
    if dsample:
        lines += ["", f"## 규칙으로 뺀 질문 예시 {len(dsample)}개", "", "| 표 | 질문 | 사유 |", "| --- | --- | --- |"]
        lines += [f"| {names.get(k, k)} | {txt} | {reason} |" for k, txt, reason in dsample]
    ssample = rng.sample(state.score_dropped, min(10, len(state.score_dropped)))
    if ssample:
        lines += ["", f"## 점수로 뺀 질문 예시 {len(ssample)}개", "", "| 표 | 유형 | 질문 | 점수 |", "| --- | --- | --- | --- |"]
        lines += [f"| {names.get(it['key'], it['key'])} | {it['type']} | {it['text']} | {it['score']:.3f} |" for it in ssample]
    return "\n".join(lines) + "\n"
