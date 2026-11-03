# Tasks: T2-09 HCX prompt clarification

- [x] T-01: Update `src/kosis_rag/hcx.py` only: increment `PROMPT_VERSION`, replace `SYSTEM` with the approved role, steps, input/output contract, and examples, and align `user_message()` labels and order. Preserve the parser and selection behavior. Acceptance: the rendered prompt asks for one JSON object with valid candidate numbers, and the input builder matches its documented format.
- [x] T-01b: Update `tests/test_no_answer.py` to assert the approved question-first input layout and numbered passage format. Acceptance: the focused test checks the rendered input contract rather than the superseded layout.
- [x] T-02: Verify the focused HCX tests and `bash scripts/verify.sh` in this worktree. Review the diff against the plan. Acceptance: checks pass, or each environment blocker and its evidence are recorded without weakening checks.
- [x] T-03: Write `docs/reports/t2-09-hcx-prompt-REPORT.md` and update `docs/current-state.md` with the actual verification result and prompt version. Acceptance: a fresh session can identify the changed branch, behavior, verification, and any unmeasured HCX performance.
- [x] T-04: Record the completed work as a commit on the new `T2-09-hcx-prompt-v3` branch, based on `T2-09-hcx-no-answer`. Acceptance: the branch contains exactly the planned changes and the working tree is clean.
