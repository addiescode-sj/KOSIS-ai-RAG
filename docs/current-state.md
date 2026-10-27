# Current State

> A new session starts here. Keep this short and current — it is the handoff
> packet between sessions.

## Project

KOSIS statistics AI chatbot, **Phase 2 (실전 2): table-guide RAG chatbot**, due 2026-11-04.
Phase 1 repo is the `vendor/phase1` submodule (pinned commit); its `kosis_ai.schemas`,
`normalize`, `eval/` harness are imported, not copied.

## Current phase (2026-10-27)

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

`bash scripts/verify.sh` → `verify: OK` (pytest 2 pass, frontend tsc).

## Known gaps

- `docker compose build` not yet run (needs `uv.lock` committed, Docker daemon). Covered by T2-17.
- Backend has no CORS / frontend→backend wiring yet (T2-11).
- `.env.example` is the Phase 1 copy; add `LINEAR_API_KEY`, `LINEAR_TEAM_KEY` lines by hand if using Linear sync.
- Phase 1 repo TASK.md also lists T2 tasks; both repos sync the same Linear issues by ID — edit T2 status only here.
