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

## 평가: 검색 → 답변 종단 간 (e2e)

L3는 정답 표를 주고 답만 채점한다. e2e는 같은 실전 1 문항(90개)을 표 없이 넣어, 검색기 1위 표로 `summarize_stat`을 부르고 L3와 같은 채점기로 채점한다. 두 실행의 PASS 차이가 검색 때문에 잃는 점수다.

```bash
export PYTHONPATH=.:src:vendor/phase1:vendor/phase1/src   # cmd: set PYTHONPATH=.;src;vendor\phase1;vendor\phase1\src  · PowerShell: $env:PYTHONPATH=".;src;vendor\phase1;vendor\phase1\src"

# 준비 1회: eval/golden/golden.jsonl(실전 1 골든 90문항 합본, gitignore) → evals/datasets/calculation_golden.jsonl
# (data/values/*.parquet 해시 확인 포함. 실전 1 레포 폴더를 쓰려면 --src <실전 1 레포>/eval/golden)
uv run python -m evals.builders.phase1

# 실행 (HCX 키는 이 레포 루트 .env, 문항마다 모델 호출)
uv run python -m evals.run_eval --layer e2e --retriever bm25-record --stage p2-e2e-bm25 --interval 5
uv run --group embed python -m evals.run_eval --layer e2e --retriever chroma --stage p2-e2e-chroma --interval 5   # 아래 ChromaDB 서버 + .env CHROMA_COLLECTION
uv run python -m evals.run_eval --layer l3 --stage p2-l3 --interval 5                                # 비교용: 정답 표 고정
```

- `--retriever`: `bm25-catalog` | `bm25-record` | `chroma` (`oracle`은 L3와 같아서 막아 둠). `--k`는 정답 표 순위를 볼 범위(기본 10).
- 결과는 `evals/reports/<시각>-<stage>/`에 `report.md`, `report.html`(브라우저로 열기), `results.jsonl`.
- 보는 법: kind별 PASS/PARTIAL/FAIL, **검색 1위 = 정답 표** 비율, **실패 원인**(검색 = 1위가 다른 표, 답 = 표는 맞았는데 답이 틀림). 문항별 표에 쓴 표와 정답 표 순위가 나온다.
- 채점은 정답 표의 셀 기준이라, 다른 표를 골랐는데 PASS가 나오는 경우는 두 표의 숫자가 같을 때뿐이다.

## ChromaDB (팀 공용 벡터 저장소)

모두 같은 서버(`chromadb/chroma:1.5.9`)를 Docker로 띄운다. Docker Desktop을 켜고:

```bash
docker compose up -d --wait chroma      # localhost:8001 (포트를 바꾸려면 .env의 CHROMA_PORT)
PYTHONPATH=.:src:vendor/phase1:vendor/phase1/src uv run python scripts/chroma_smoke.py   # 쓰기 → 재시작 → 읽기 확인
```

- 코드에서는 `kosis_rag.vectorstore.chroma_client()`만 쓴다. 접속 주소는 `.env`의 `CHROMA_HOST`/`CHROMA_PORT`(호스트에서 `localhost:8001`)이고, compose 안의 backend는 `chroma:8000`으로 자동 설정된다.
- 데이터는 Docker named volume `chroma-data`에 남는다(`docker compose down`해도 유지, git에는 안 들어감). 지우려면 `docker compose down -v` 후 다시 적재한다(원본은 `data/tables.jsonl`).
