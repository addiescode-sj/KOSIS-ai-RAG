# Plan: Add a PR template based on PR #1

## Decision

Add one default pull request template at `.github/PULL_REQUEST_TEMPLATE.md` to `addiescode-sj/KOSIS-ai-RAG`. Keep the five section headings and their order from PR #1. Publish it to `master` through a separate pull request so GitHub offers it automatically for future PRs.

## Evidence and intent

- PR #1 uses the title pattern `T2-01: ...` and body sections `요약`, `결과`, `바뀐 것`, `쓰는 사람에게`, `검증` in that order.
- Its body links the plan/issue, quantifies outcomes, lists changed files, gives downstream usage notes, and records verification.
- The repository has no existing PR template on `master`.
- GitHub recognizes `.github/PULL_REQUEST_TEMPLATE.md` and makes it available after it reaches the default branch.
- The current local checkout is on `T2-18-langchain-rag-skeleton` with unrelated in-progress changes. Leave those files and that branch untouched.

## Template content to publish

```markdown
## 요약
변경 목적과 관련 계획·이슈를 적어주세요.

## 결과
핵심 결과를 수치나 예시로 적어주세요. 전후 비교가 있으면 아래 표를 사용하고, 해당하지 않으면 표를 지워주세요.

| 지표 | 전 | 후 |
| --- | --- | --- |
|  |  |  |

## 바뀐 것
- `파일 또는 경로`: 변경 내용을 적어주세요.

## 쓰는 사람에게
- 사용 방법, 연관 작업에 미치는 영향, 데이터 생성·보관 시 주의할 점을 적어주세요. 해당하지 않으면 `해당 없음`으로 적어주세요.

## 검증
- `bash scripts/verify.sh`: 결과를 적어주세요.
- 추가 검증이 있다면 명령과 결과를 적어주세요.
```

## Implementation scope

1. Create a clean remote branch from the then-current `master` using GitHub API; do not use the dirty local checkout for commits.
2. Commit only `.github/PULL_REQUEST_TEMPLATE.md` on that branch, then open a PR to `master`.
3. Verify the PR diff contains only the template and that the file matches this plan. Merge the PR after applicable checks pass.
4. Read back the file from `master`, run `bash scripts/verify.sh` in the local repository, and write TASKS.md, report, and current-state handoff under `docs/`.

## Non-goals

- Do not modify PR #1 or its author's text.
- Do not add issue templates, status checks, title automation, or approval requirements.
- Do not add a tool-specific generation footer to every future PR.
- Do not alter the in-progress T2-18 work or include it in the template PR.

## Verification

- Confirm the PR and merge commit contain only `.github/PULL_REQUEST_TEMPLATE.md`.
- Confirm `master` serves the exact template file and that its five headings match PR #1 in order.
- Confirm `bash scripts/verify.sh` exits 0.

## Context Index

| Path | Contains | Use for |
| --- | --- | --- |
| `docs/plans/20261027-pr-template/PLAN.md` | Scope, exact template text, GitHub workflow | Resume after approval |
| `AGENTS.md` | Approval and execution rules | Workflow constraints |
| `docs/current-state.md` | Current repository handoff | Wrapup update |

## Approval gate

Await explicit user approval before creating tasks, writing the template, or changing GitHub.
