# Plan: Clarify the T2-09 HCX table selection prompt

## Decision

Revise the HCX-007 stage 2 selector prompt on a branch based on `T2-09-hcx-no-answer` so the model has an explicit role, ordered decision steps, input format, output contract, and examples. Keep the existing table-selection task and JSON schema. The user's retrieval rewrite prompt is a structural reference; query rewriting is T2-08 and is outside this change.

## Evidence

- `T2-09-hcx-no-answer:src/kosis_rag/hcx.py` defines `SYSTEM` and `user_message()`. The model receives a question plus numbered table passages and returns `{"picked": [numbers], "reason": ...}`.
- `parse()` maps numbers to candidate keys and accepts `out_of_scope` or `no_item` only when no table is picked.
- T2-09's v1 prompt over-rejected answerable questions. The current v2 deliberately selects by statistical measure and leaves year, frequency, and detailed region checks to code. The recorded test no-answer F1 for v2 with thinking low is 0.882, but the reason labels are unreliable (`lab/T2-09-hcx-no-answer/REPORT.md`).
- The current checkout is `T2-32-golden-v2` with unrelated user changes. The target file exists on `T2-09-hcx-no-answer`, so execution must preserve the current worktree.

## Proposed prompt contract

The replacement `SYSTEM` will express these instructions in Korean, while code and comments remain in English:

1. **Role:** Select up to five KOSIS population-statistics tables useful for answering the user's question. Do not answer the question or rewrite it.
2. **Input:** A `질문:` field and a `후보 통계표:` field containing numbered `[1]`, `[2]`, ... passages. Candidate numbers are the only valid IDs in the output.
3. **Steps:** Identify the requested statistical measure and counted subject; compare that measure with each candidate's contents; select all useful tables, ordered by closeness to the question, up to five; if none match, classify the reason.
4. **Decision boundary:** Do not reject a measure match because of year/period, monthly/quarterly/annual frequency, or detailed geographic unit. Code checks these separately. Do not select a merely related population table that lacks the requested measure. Do not invent a candidate or infer that a table contains an unlisted measure.
5. **Output:** Exactly one JSON object on one line, no Markdown or prose. Keys are `picked` (array of unique integers from the input, at most five) and `reason` (`null` when `picked` is nonempty; otherwise `"out_of_scope"` for a non-population-statistics topic or `"no_item"` when the question is in scope but no candidate contains its measure).
6. **Examples:** A positive match whose year differs, a `no_item` case, and an `out_of_scope` case, all using the same numbered candidate and JSON format. Examples must reinforce the measure-only decision and the exact schema.

`user_message()` will label the question and candidate list in the same order as the documented input contract. The parsed output schema, candidate mapping, reason values, and `MAX_PICK` remain compatible.

### Proposed `SYSTEM` text

```text
당신은 KOSIS 인구통계 통계표 검색의 2차 선택기입니다. 사용자의 질문에 답하는 데 쓸 수 있는 후보 통계표를 고르세요. 질문에 직접 답하거나 질문을 다시 쓰지 마세요.

# 입력 형식
질문: 사용자의 원래 질문
후보 통계표:
[1] 첫 번째 통계표 설명
[2] 두 번째 통계표 설명
...

# 판단 단계
1. 질문에서 찾으려는 통계 지표와 세는 대상·항목을 파악하세요.
2. 각 후보 통계표 설명에 그 지표가 담겨 있는지 확인하세요. 주제만 비슷하고 지표가 다르면 고르지 마세요. 설명에 없는 지표가 있다고 추측하지 마세요.
3. 지표가 맞는 표를 질문과 가까운 순서로 최대 5개 고르세요. 출력에는 입력에 나온 후보 번호만 사용하고 같은 번호를 반복하지 마세요.
4. 맞는 표가 없으면 사유를 정하세요. 질문이 인구·가구·출생·사망·혼인·이동 등 인구통계 범위 밖이면 out_of_scope, 인구통계 질문이지만 후보에 필요한 지표가 없으면 no_item입니다.
5. 출력 형식을 확인하세요. 표를 하나라도 골랐으면 reason은 null, 고른 표가 없으면 picked는 []이고 reason은 두 사유 중 하나입니다.

# 판단 범위
연도·기간, 월·분기·연 주기, 시도·시군구 등 세부 지역 단위가 맞는지는 판단하지 마세요. 이 조건은 별도 코드가 확인합니다. 지표가 맞으면 해당 조건이 달라도 표를 고르세요.

# 출력 형식
설명이나 Markdown 없이 JSON 객체 한 줄만 출력하세요.
{"picked": [후보 번호, ...], "reason": null}
또는
{"picked": [], "reason": "out_of_scope"}
또는
{"picked": [], "reason": "no_item"}

# 예시
입력:
질문: 2025년 출생아 수는?
후보 통계표:
[1] 출생아 수, 2020~2024년
[2] 합계출산율, 2020~2024년
출력:
{"picked": [1], "reason": null}

입력:
질문: 전입자 수는?
후보 통계표:
[1] 출생아 수
[2] 사망자 수
출력:
{"picked": [], "reason": "no_item"}

입력:
질문: 자동차 등록 대수는?
후보 통계표:
[1] 주민등록인구
출력:
{"picked": [], "reason": "out_of_scope"}
```

The example passages are illustrative, not an assertion that the real candidate cards contain those values. The runtime input will use the same labels and order: `질문:` first, then `후보 통계표:` with numbered passages.

## Implementation scope after approval

- Work in an isolated checkout or worktree based on `T2-09-hcx-no-answer` so the dirty T2-32 checkout stays intact. The approved execution initially used the T2-09 branch; the user's follow-up requested a separate branch and commit, so record the result on `T2-09-hcx-prompt-v3`.
- Edit only the prompt and input formatting in `src/kosis_rag/hcx.py`, plus focused tests if they verify a meaningful contract. Increment `PROMPT_VERSION` to distinguish the changed prompt from measured v2.
- Add `docs/plans/t2-09-hcx-prompt/TASKS.md`, verify, and write `docs/reports/t2-09-hcx-prompt-REPORT.md` and `docs/current-state.md` in the target branch.

## Non-goals

- No query rewriting, model or thinking-setting change, JSON parser change, scoring change, or new refusal categories.
- No claim that the revised prompt improves retrieval or no-answer metrics without a fresh HCX evaluation.

## Verification

- Inspect the rendered `SYSTEM` and `user_message()` for exact labels, numbered candidates, step instructions, examples, and the output contract.
- Run focused HCX selector/parse tests available on the target branch and `bash scripts/verify.sh`.
- If HCX credentials, data, and services are available, measure the new prompt on the T2-09 dev/test evaluation and compare to v2; otherwise record that performance is unmeasured.

## Context Index

| Path | Contains | Use for |
| --- | --- | --- |
| `docs/plans/t2-09-hcx-prompt/PLAN.md` | Approved prompt contract and scope | Continue after approval |
| `src/kosis_rag/hcx.py` on `T2-09-hcx-no-answer` | Existing system prompt, input builder, parser | Edit target |
| `plans/T2-09.md` on `T2-09-hcx-no-answer` | Original selector decision and results | Preserve behavior |
| `lab/T2-09-hcx-no-answer/REPORT.md` on `T2-09-hcx-no-answer` | v1/v2 results and failure analysis | Compare revised prompt |
| `docs/current-state.md` | Repository handoff state | Wrapup update |

## Approval gate

Await explicit user approval before creating tasks or editing the target prompt.
