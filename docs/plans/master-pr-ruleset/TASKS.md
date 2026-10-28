# Tasks: Require pull requests for master

- plan: docs/plans/master-pr-ruleset/PLAN.md
- status: DONE
- approved: 2026-10-27, explicit user message `approved`

## Phase 1: GitHub configuration

- [x] T-001 Create the active repository ruleset specified in PLAN.md for `refs/heads/master` with no bypass actors. (AC-001)
  - verify: `gh api repos/addiescode-sj/KOSIS-ai-RAG/rulesets`
  > done: GitHub ruleset 25956464 created; read-back showed GitHub's no-op extra approval default at zero required approvals. PLAN.md updated before continuing.

## Phase 2: Verification

- [x] T-002 Independently read the created ruleset and effective `master` rules; confirm the pull request rule, exact branch target, active enforcement, and no bypass actors. (AC-001) after: T-001
  > done: Ruleset and effective branch rules API reads exited 0; verifier confirmed AC-001 and `current_user_can_bypass: never`.
- [x] T-003 Run full repository verification: `bash scripts/verify.sh`. (AC-002) after: T-001
  > done: Verifier ran script; exit 0, 2 tests passed, `verify: OK`.

## Phase 3: Handoff

- [x] T-004 Write `docs/reports/master-pr-ruleset-REPORT.md` and update `docs/current-state.md` with the GitHub ruleset and verification results. (AC-003) after: T-002, T-003
  > done: Report records independent API and repository verification; current-state preserves project context and adds the completed ruleset handoff.

## Acceptance criteria

- AC-001: Direct pushes to `master` are blocked by an active pull request ruleset with no bypass list and zero required approving reviews.
- AC-002: `bash scripts/verify.sh` exits 0.
- AC-003: Report and current-state handoff are updated.
