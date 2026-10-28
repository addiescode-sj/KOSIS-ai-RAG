# KOSIS 통계표 안내 RAG 챗봇 (실전 2)

국가데이터처(KOSIS) 인구 분야 2,364개 통계표를 대상으로 "어느 표를 봐야 해?"에 답하는 RAG 챗봇.
프로젝트 전체 계획(실전 1 → 실전 2 → 종합)은 실전 1 레포 [KOSIS-ai-nlp-summary](https://github.com/addiescode-sj/KOSIS-ai-nlp-summary)의 README를 따른다.

- 로드맵: [TASK.md](TASK.md) · 작업 지침: [CLAUDE.md](CLAUDE.md) · 에이전트 하네스: [AGENTS.md](AGENTS.md)
- 핵심 지표: Hit@1/5/10, MRR, 답 없음 F1
- 공개 API: `search_tables(question, k) -> list[TableCandidate]`

## 구성

| 경로 | 내용 |
|---|---|
| `backend/` | FastAPI (Python 3.12, uv) |
| `src/kosis_rag/` | 검색 모듈 (BM25·임베딩·하이브리드·리랭커, T2-04~) |
| `frontend/` | Next.js 16 (App Router) + TypeScript + Tailwind |
| `vendor/phase1/` | 실전 1 레포 (git submodule) — 공용 스키마·정규화·평가 하네스 재사용 |

## 시작하기

```bash
git clone --recurse-submodules https://github.com/addiescode-sj/KOSIS-ai-RAG.git
cd KOSIS-ai-RAG
cp .env.example .env            # 키 입력 (커밋 금지)
uv sync                         # Python
npm install                     # 하네스 hook 스크립트 (TypeScript)
(cd frontend && npm install)    # 프론트엔드
bash scripts/verify.sh          # 오프라인 검증
```

제공 데이터는 `data/`에 둔다 (gitignored, 보안 서약).

## 실행

```bash
# 로컬 개발
PYTHONPATH=.:src:vendor/phase1:vendor/phase1/src uv run uvicorn backend.app.main:app --reload   # :8000/health
cd frontend && npm run dev                                                                      # :3000

# 실전 1 평가를 이 레포에서 (data/, eval/golden, .env는 이 레포 루트 기준, KOSIS_WORKSPACE 자동)
uv run python scripts/phase1.py run_eval --stage <이름> --golden eval/golden/golden.jsonl --pipeline stat ...
uv run python scripts/stat_baseline.py --stage <이름> --backend hcx --model HCX-DASH-002 --interval 5   # 같은 문항을 summarize_stat(질문, 표 키)로
uv run python scripts/phase1.py grade_run --run eval/results/<run>

# 배포 (단일 서버)
docker compose up -d --build
```

실전 1 코드 갱신: `git submodule update --remote vendor/phase1` 후 커밋.
