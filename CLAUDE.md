# KOSIS 통계 AI 서비스 — 실전 2 (통계표 안내 RAG 챗봇) 작업 지침

멋쟁이사자처럼 AI NLP 부트캠프 6기 팀 프로젝트. 국가데이터처(KOSIS) 통계표를 대상으로
**실전 1(수치 요약 모듈) → 실전 2(통계표 검색 챗봇) → 종합(툴 호출 에이전트 + 시각화)**을 만든다.
이 레포는 **실전 2**다. 실전 1 레포([KOSIS-ai-nlp-summary](https://github.com/addiescode-sj/KOSIS-ai-nlp-summary))는
`vendor/phase1` submodule로 들어 있고, 그 모듈을 다시 짜지 않고 import해서 쓴다.
로드맵과 진행 상황은 [TASK.md](TASK.md), 세션 인계는 [docs/current-state.md](docs/current-state.md).

## 0. 작업 절차 (필수 — hook으로 강제됨)

모든 TASK는 `plan-gate` skill(`/plan-gate`)로 진행한다. 산출물부터 만들지 않는다.

1. **Pick** — TASK.md에서 태스크를 `[~]`로 바꾸고 브랜치 `T2-04-slug` 생성
2. **Plan** — `plans/<TASK-ID>.md`에 수행 아이디어와 옵션 A/B/C 작성
   (옵션마다 예상 성능 향상 수치 + 실질 반영 이점 + 비용·리스크)
3. **Verify** — Verify 체크리스트를 모두 통과시키고, 독립 리뷰(서브에이전트) 결과 반영
4. **Decide** — 사용자에게 옵션 선택을 받아 `status: approved` 기록
5. **Implement** — 선택한 옵션만 구현
6. **Measure** — `evals/`의 계층별 평가 모듈로 전→후 수치를 plan의 `## Result`와 TASK.md에 기록, `[x]` 처리

`.claude/hooks/plan_gate.py`가 승인된 plan 없이 `plans/`, `TASK.md`, `CLAUDE.md`, `.claude/` 밖의
파일을 수정하면 차단한다. 우회하지 말 것 (Bash로 파일 쓰기 포함).

## 1. 평가 기준 (모든 판단의 기준)

데이터 정확성 > 예외 처리 > 질의 이해 > 기능 완성도 > 응답 품질 > 기술 설계 > 결과 검증.
과제에 합격 점수는 없다. **"무엇을 바꿔서 몇 → 몇"**을 남기는 것이 결과물이다.
실전 2 핵심 지표: **Hit@1/5/10, MRR, 답 없음 P/R/F1 (질의 유형별)**, 지연·호출 수.
배점: 개인 기술 성과 40 / 팀 완성도 30 / 협업 20 / 발표·문서 10 → 개인 담당 부분의 수치를 꼭 남긴다.

## 2. 설계 원칙

- **숫자는 코드(pandas)가 계산하고 LLM은 문장만 쓴다.** LLM이 만든 숫자는 버그다.
- 모든 추천·답에 출처: `KOSIS / 조사명 / 통계표명 (+링크)`. 표 ID는 검색 전 과정에서 보존한다.
- 맞는 표가 없으면 지어내지 말고 없다고 말한다 (범위 밖·기간/주기 불일치 포함).
- 모듈 입출력은 Pydantic 스키마로 고정한다. 공용 스키마는 `kosis_ai.schemas`(실전 1)를 import한다.
  실전 2 스키마(`TableCandidate` 등)를 추가할 때도 같은 원칙, 깨는 변경은 plan에 명시하고 승인받는다.
- 단계별 공개 API (종합에서 툴이 됨):
  - 실전 1: `summarize_stat(question, table_id) -> StatAnswer` (vendor/phase1)
  - 실전 2: `search_tables(question, k) -> list[TableCandidate]` (+ 답 없음 판정)
- 오케스트레이션: 실전 2부터 LangChain (검색기·리랭커 조합). LangGraph는 종합에서.
- 필요한 것만 만든다. 구현 하나·인터페이스 하나면 추상화하지 않는다.

## 3. 디렉터리

```
backend/app/       FastAPI (main.py). 검색·챗 API를 노출만 한다 — 로직은 src/
src/kosis_rag/     search_tables, 검색기(BM25·임베딩·하이브리드), 리랭커, 답 없음 판정 (T2-04~)
frontend/          Next.js (App Router) + TypeScript + Tailwind. 채팅 UI (T2-11)
vendor/phase1/     실전 1 레포 submodule — kosis_ai(src/), kosis_chat/, eval/ 를 import. 직접 수정 금지
evals/             계층별 평가 모듈 (LangChain 비종속): datasets/(gitignored) · evaluators/ · runners/ · reports/
plans/             태스크별 plan (옵션·검증·결정·결과 기록)
docs/              제출 문서 (검색 전략, 실패 분석), current-state.md
```

- 실전 1 코드 수정이 필요하면 실전 1 레포에 PR → 여기서 `git submodule update --remote vendor/phase1` 후 커밋.
- 평가는 실전 1 하네스(`vendor/phase1/eval`)를 쓰지 않는다 (10/29 결정). `evals/`에 계층별(L1 검색 · L2 질의 이해 · L3 계산 · L4 종단 간)
  골든셋과 evaluator를 따로 둔다. LangChain은 평가 대상이고, 채점은 결정적 코드가 한다. 실전 1 `eval` 패키지와 이름이 겹치지 않게 `evals/`(복수)를 쓴다.

## 4. 코딩 규칙

- 코드·주석·식별자는 영어. 문서(md)는 한국어.
- Python 3.11+ (`uv`, `.python-version` 3.12), Pydantic v2, pandas, pytest. 비밀값은 `.env` — 커밋 금지.
- 셋업: `git submodule update --init && uv sync && npm install && (cd frontend && npm install)`
- 실행: 백엔드 `uv run uvicorn backend.app.main:app --reload` (PYTHONPATH는 아래 참조),
  프론트 `cd frontend && npm run dev`, 전체 `docker compose up --build`
- import 경로: `pytest.ini`의 `pythonpath = . src vendor/phase1 vendor/phase1/src`. 스크립트 실행 시
  `PYTHONPATH=.:src:vendor/phase1:vendor/phase1/src`.
- 검증: `bash scripts/verify.sh` (오프라인 pytest + 프론트 타입체크 + 데이터 미추적 확인).
- 제공 데이터는 `data/`(gitignored)에 두고 Docker 이미지에 넣지 않는다 (compose가 읽기 전용 마운트).
- 비자명한 로직(검색 점수·판정·채점)은 pytest 하나 이상을 남긴다.
- frontend는 Next.js 16 — 코드 작성 전 `frontend/AGENTS.md` 지침대로 `node_modules/next/dist/docs/`를 확인한다.
