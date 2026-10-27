# Tasks: Add PR template based on PR #1

- plan: docs/plans/20261027-pr-template/PLAN.md
- status: IN_PROGRESS
- approved: 2026-10-27, explicit user message `approved`

## Phase 1: Remote change

- [x] T-001 Create a remote branch from current `master`, add only `.github/PULL_REQUEST_TEMPLATE.md` with the exact plan content, and open a PR to `master`. (AC-001)
  > done: PR #2 opened from `docs/pr-template-20261027-b3c9901e`; commit `9fac97f68ca6d7c8f94646a8546ff6a7c6ad6dae` changes only the template and matches PLAN.md.

## Phase 2: Verification and publication

- [ ] T-002 Independently confirm the PR diff contains only the intended template and that the section headings match PR #1. (AC-001) after: T-001
- [ ] T-003 Run `bash scripts/verify.sh` and inspect PR checks. (AC-002) after: T-001
- [ ] T-004 Merge the verified template PR and read back `.github/PULL_REQUEST_TEMPLATE.md` from `master`. (AC-003) after: T-002, T-003

## Phase 3: Handoff

- [ ] T-005 Write `docs/reports/20261027-pr-template-REPORT.md` and update `docs/current-state.md`. (AC-004) after: T-004

## Acceptance criteria

- AC-001: A PR to `master` contains only the planned `.github/PULL_REQUEST_TEMPLATE.md`.
- AC-002: `bash scripts/verify.sh` exits 0 and applicable PR checks pass.
- AC-003: `master` contains the template with the five PR #1 headings in order.
- AC-004: Report and current-state handoff are updated.
