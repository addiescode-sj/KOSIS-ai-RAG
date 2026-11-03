# Report: Clarify the T2-09 HCX table selection prompt

- id: t2-09-hcx-prompt
- date: 2026-11-03
- plan: docs/plans/t2-09-hcx-prompt/PLAN.md

## Summary

Implemented prompt v3 on `T2-09-hcx-prompt-v3`, a separate branch based on `T2-09-hcx-no-answer`, in `/private/tmp/KOSIS-ai-RAG-T2-09-hcx-prompt`.
The Korean selector prompt now states its role, ordered decisions, input/output contract, and examples. Runtime input places the question before numbered candidate tables. Parser and selection behavior remain unchanged.
Offline verification passed. Live HCX evaluation was not run; v3 performance is unmeasured.
The user-requested separate branch exists; T-04 commit preparation is in progress, and no commit result is recorded yet.

## Changes

- `src/kosis_rag/hcx.py`: increment `PROMPT_VERSION` to v3, clarify `SYSTEM`, and align `user_message()` with the question-first contract.
- `tests/test_no_answer.py`: assert the exact question-first input and numbered passage format.
- `docs/current-state.md`: add a dated branch handoff with verification and limitations.

## Verification

- verdict: INCOMPLETE — implementation and offline checks passed; T-04 commit work is still in progress. Optional live evaluation remains unmeasured.
- Independent verifier reported focused HCX checks: 7 tests passed.
- offline checks: `bash scripts/verify.sh` -> exit 0, 185 tests passed, `verify: OK`.
- Diff hygiene: `git diff --check` -> exit 0.
- online checks (HCX / API): not run; no live evaluation evidence was produced in this execution.
- acceptance: T-01 met (v3 structured prompt and matching input format; independent diff review passed).
- acceptance: T-01b met (exact rendered input assertion; focused checks passed).
- acceptance: T-02 met (focused checks, full verification, and scope review passed).
- acceptance: T-03 met (this report and current-state handoff written).
- acceptance: T-04 in progress (separate result branch created; commit result pending).

Verification results above were supplied by the independent verifier; no separate log artifact was created.

## Not run / failed

- Live HCX dev/test evaluation: not run. A fresh evaluation with available credentials, data, and services is needed to compare v3 against v2. No improvement claim is made.

## Deviations from plan

- None. The focused test update was within the approved scope.

## Deferred / Remaining risks

- T-04: finish the requested commit and verify the resulting branch state.
- Prompt behavior against live HCX is unmeasured. Historical v2 metrics do not describe v3.

## Context Index

- `docs/plans/t2-09-hcx-prompt/PLAN.md` — contains: approved prompt contract and non-goals — use_for: scope review.
- `docs/plans/t2-09-hcx-prompt/TASKS.md` — contains: execution checklist — use_for: completion tracking.
- `src/kosis_rag/hcx.py` — contains: v3 selector prompt, input builder, parser — use_for: implementation review.
- `tests/test_no_answer.py` — contains: input contract and no-answer tests — use_for: regression checks.
- `lab/T2-09-hcx-no-answer/REPORT.md` — contains: historical v1/v2 evaluation — use_for: future comparison baseline.
- `docs/current-state.md` — contains: active branch and verification handoff — use_for: resuming in a new session.
