# Plan: Require pull requests for master

## Decision

Create one active repository branch ruleset on `addiescode-sj/KOSIS-ai-RAG` that requires a pull request before changes reach `refs/heads/master`.

## Evidence

- The repository's default branch is `master`.
- The authenticated GitHub account has repository admin permission.
- The repository currently has no rulesets and no branch protection on `master`.
- GitHub's `pull_request` ruleset rule requires changes to be submitted through a pull request. A review count of zero requires a PR without adding an approval requirement.

## Implementation scope

- Name: `Require pull request for master`.
- Target: branch `refs/heads/master` only.
- Enforcement: `active`.
- Rule: `pull_request`, with zero required approving reviews and the documented review requirements disabled.
- GitHub currently returns `require_extra_approval_for_unattributed_changes: true` by default. GitHub documents that this option has no effect when zero approvals are required. Accept this platform default while the required approval count remains zero.
- Bypass actors: none, including no administrator bypass.
- After approval, create `TASKS.md`, apply the ruleset through GitHub, read it back, run repository verification, and write the report and current-state handoff.

## Non-goals

- No required review approvals, status checks, merge queue, or new merge method restrictions.
- No source code changes and no changes to other branches.

## Verification

- Read the created ruleset through the GitHub API and confirm `active`, exact `master` target, `pull_request` rule, and empty bypass list.
- Confirm `required_approving_review_count` is zero and record GitHub's extra approval default.
- Confirm the effective rules for `master` include the pull request requirement.
- Run `bash scripts/verify.sh` as required by this repository's done criteria.

## Context Index

| Path | Contains | Use for |
| --- | --- | --- |
| `docs/plans/master-pr-ruleset/PLAN.md` | Approved scope and verification plan | Resume after approval |
| `AGENTS.md` | Repository workflow and approval gate | Execution rules |
| `docs/current-state.md` | Project handoff state | Wrapup update |

## Approval gate

Await explicit user approval before creating tasks or changing GitHub settings.
