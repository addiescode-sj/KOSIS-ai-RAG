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

## How to verify

`bash scripts/verify.sh` → `verify: OK` (pytest 2 pass, frontend tsc).

## Known gaps

- `docker compose build` not yet run (needs `uv.lock` committed, Docker daemon). Covered by T2-17.
- Backend has no CORS / frontend→backend wiring yet (T2-11).
- `.env.example` is the Phase 1 copy; add `LINEAR_API_KEY`, `LINEAR_TEAM_KEY` lines by hand if using Linear sync.
- Phase 1 repo TASK.md also lists T2 tasks; both repos sync the same Linear issues by ID — edit T2 status only here.
