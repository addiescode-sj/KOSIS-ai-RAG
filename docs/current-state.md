# Current State

> A new session starts here. Keep this short and current — it is the handoff
> packet between sessions.

## Project

KOSIS statistics AI chatbot, **Phase 2 (실전 2): table-guide RAG chatbot**, due 2026-11-04.
Phase 1 repo is the `vendor/phase1` submodule (pinned commit). Since T2-41 the Phase 2 module (`src/kosis_rag`) does
not import Phase 1: the shared schemas and LLM client code are copies checked by `tests/test_phase1_contract.py`
(evals, scripts and tests still use Phase 1 directly).

## T2-41 module refactor (2026-11-06, branch `T2-41-refactor`, uncommitted)

- Plan: `plans/T2-41.md` (Result section has the stage list and what remains). Rename table for the composite repo:
  `docs/migration.md` (its last section is the composite repo's to-do list). Settings: `docs/config.md` (`.env.example`
  is gone; nothing in the code reads `.env`).
- Cloud stages S0–S6b are done as uncommitted changes (one commit at the end, on the user's request). Cloud checks:
  pytest 1055 passed, 1 skipped; `tests/fixtures/parity` (33 cases) and `docs/openapi.json` byte-identical; wheel
  install reads both prompt files (`prompt_hash()[:8] == cb0b3c5b`).
- **PC still to do**: S0 baselines and the per-stage PC parity ① L1 compare (hybrid+cond, staged), ①hq, ③ hq replay
  (`scripts/parity_hq.py replay`, new `--out-dir` per stage), ④ openapi; `uv lock` (google-genai was dropped
  from `pyproject.toml`); `docker compose build`; the wheel check on Windows; the single commit; the
  `vendor/phase1` pin update after Phase 1 S6 lands. Commands: `plans/T2-41.md` "단계 공통 명령".
- Entry points: `kosis_rag.api.SearchModule(IndexSpec, SearchOptions, LLMConfig).search(SearchInput) -> SearchOutput`;
  offline builds `python scripts/kosis_build.py {tables|sigungu|chroma|text|hq-build|hq-index}`; backend demo reads
  the shell environment (`backend/app/config.py`). evals were not modified (its L1 runner's settings lines are broken
  since S4 until the evals owner rewrites them).

## T2-09 prompt update (2026-11-03)

- Result branch: `T2-09-hcx-prompt-v3`, based on `T2-09-hcx-no-answer`; worktree: `/private/tmp/KOSIS-ai-RAG-T2-09-hcx-prompt`.
- The user requested a separate branch and commit. The branch exists; T-04 commit preparation is in progress, with no commit result recorded yet. Continue from `docs/plans/t2-09-hcx-prompt/TASKS.md`.
- `src/kosis_rag/hcx.py` uses prompt v3: a structured Korean selector role, ordered decisions, exact JSON contract, and positive/no-item/out-of-scope examples. Input now places `질문:` before `후보 통계표:` and numbered candidates.
- Selection remains measure-based; year, frequency, and detailed geography are checked separately by code. Parser, reason categories, model, and thinking settings are unchanged.
- Independent verifier PASS: focused checks (7 tests), `bash scripts/verify.sh` (exit 0, 185 tests, `verify: OK`), and `git diff --check` (exit 0). No separate verifier log artifact exists.
- Live HCX evaluation was not run. V3 performance is unmeasured; historical v2 metrics must not be attributed to v3. A future comparison needs the T2-09 dev/test evaluation and available credentials, data, and services.
- Preserve the unrelated user changes in the separate T2-32 checkout. This handoff does not imply integration into that checkout.

### Context Index

- `docs/plans/t2-09-hcx-prompt/PLAN.md` — contains: approved contract and scope — use_for: implementation review.
- `docs/plans/t2-09-hcx-prompt/TASKS.md` — contains: execution checklist — use_for: final task status.
- `docs/reports/t2-09-hcx-prompt-REPORT.md` — contains: changes, verification, limitations — use_for: completed-work evidence.
- `src/kosis_rag/hcx.py` — contains: v3 prompt, input builder, parser — use_for: implementation review.
- `tests/test_no_answer.py` — contains: exact input-format assertion — use_for: focused verification.
- `lab/T2-09-hcx-no-answer/REPORT.md` — contains: historical v1/v2 evaluation — use_for: future live comparison.

## Earlier phase snapshot (2026-10-27)

- **T2-00 done**: repo bootstrapped.
  - Harness copied from Phase 1 unchanged: `.claude/` (agents, hooks, skills, settings.json), `AGENTS.md`, `.mcp.json`, `scripts/*.ts`, `docs/templates/`.
  - Adapted: `CLAUDE.md` (Phase 2 scope, dirs), `TASK.md` (Phase 2 only + T2-00, T2-17), `scripts/verify.sh` (submodule + frontend typecheck).
  - Skeleton: `backend/app/main.py` (`/health` only), `frontend/` (create-next-app, Next 16, standalone output), `docker-compose.yml`.
- Next: T2-01 metadata collection, T2-02 search golden set (see TASK.md order).

## Repository rules (2026-10-27)

- **master-pr-ruleset done**: [ruleset 25956464](https://github.com/addiescode-sj/KOSIS-ai-RAG/rules/25956464) is active and targets only `refs/heads/master`. Pull requests are required, required approvals are zero, and there are no bypass actors (`current_user_can_bypass: never`). No source code changed.
- GitHub's extra approval default for unattributed Copilot changes is enabled but has no effect at zero required approvals; revisit it if the approval count increases.
- Independent verification: ruleset and effective master rules API reads exited 0; `bash scripts/verify.sh` exited 0 with 2 tests passed and `verify: OK`.
- No ruleset work remains. Plan and tasks: `docs/plans/master-pr-ruleset/`; completed-work evidence: `docs/reports/master-pr-ruleset-REPORT.md`.

## How to verify

`bash scripts/verify.sh` → `verify: OK` (185 tests passed on the active T2-09 branch, 2026-11-03).

## Earlier known gaps (2026-10-27; not reassessed in this prompt-only update)

- `docker compose build` not yet run (needs `uv.lock` committed, Docker daemon). Covered by T2-17.
- Backend has no CORS / frontend→backend wiring yet (T2-11).
- `.env.example` was removed in T2-41 S6b; the settings are listed in `docs/config.md` (add your own lines, e.g. Linear sync keys, to your local `.env` or shell).
- Phase 1 repo TASK.md also lists T2 tasks; both repos sync the same Linear issues by ID — edit T2 status only here.
