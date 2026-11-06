# 설정값 목록 (T2-41 S6b, `.env.example` 대체)

실전 2 모듈(`src/kosis_rag`)과 오프라인 빌드 패키지(`src/kosis_rag_build`)는 **`.env`도 `os.environ`도 읽지 않아요**(계획 §0.6). 호출자가 설정 객체를 만들어 넘겨요:
`IndexSpec`(경로·Chroma 연결·체크포인트), `SearchOptions`(검색 동작), `LLMConfig`(LLM 공급자·키), `ChromaConfig(host, port, collection)`.
아래 이름은 **이 레포의 진입점**(backend 데모, scripts, lab, evals)이 셸 환경변수에서 읽어 그 객체로 바꾸는 값이에요. 종합 레포는 같은 뜻의 값을 자기 방식으로 선언하면 돼요(`docs/migration.md` 맨 끝 "종합 레포가 할 일").

## 값을 두는 법

- **Windows cmd**: 명령 전에 `set CHROMA_COLLECTION=...`. 자주 쓰는 줄은 레포 밖 파일(예: `C:\worker\kosis-env.cmd`)에 모아 `call` 해도 돼요. bash는 `export X=...`.
- **docker compose**: `docker-compose.yml`의 `env_file: .env`는 **Docker가** 읽어 컨테이너 환경으로 넣어요(앱 코드는 `.env`를 읽지 않아요). `.env`는 gitignore이고 커밋하지 않아요. 컨테이너 안 backend는 `CHROMA_HOST`·`CHROMA_PORT`를 compose가 `chroma:8000`으로 덮어써요.
- 키 값은 어느 진입점도 출력하지 않아요(`LLMConfig.api_key`는 `SecretStr`).

## 읽는 곳 (약칭)

| 약칭 | 파일 | 어떻게 |
|---|---|---|
| backend | `backend/app/config.py` (`search_config_from_shell`·`stage2_llm_from_shell`·`guide_llm_from_shell`) | `os.environ` → `SearchConfig(options, spec, stage2_llm, guide_llm)`. `scripts/parity_l1.py`·`lab/hq_eval.py`도 이 함수를 써요 |
| build | `scripts/kosis_build.py` → `kosis_rag_build.cli.main(argv, env=os.environ)` | 인자가 먼저, 없으면 환경. `python -m kosis_rag_build.cli`는 인자만(환경 안 읽음) |
| parity_l1 | `scripts/parity_l1.py` (`search_config_from_shell(**프리셋)`) | backend와 같은 이름·기본값 |
| scripts | `scripts/chroma_smoke.py`·`scripts/check_build.py` | `CHROMA_HOST`·`CHROMA_PORT`만 |
| lab | `lab/*.py` | 사용자 실험 스크립트라 그대로 둠: 셸 환경 + 레포 루트 `.env`(python-dotenv, 일부는 실전 1 `load_env`) |
| evals | `evals/` (평가 담당자 관리, 리팩터에서 고치지 않음) | `evals/trace.py`의 OTEL·Jaeger; L1 러너의 옛 설정 줄은 S4부터 깨져 있음(`docs/migration.md` S4) |
| compose | `docker-compose.yml` | `${...}` 치환(Docker가 셸·`.env`에서 읽음) |
| 실전 1 | `vendor/phase1` (`kosis_chat.config`) | 실전 1이 직접 읽음. 실전 1의 `load_env`가 python-dotenv로 `.env`를 읽어요(그래서 의존성은 남김; 실전 2 코드는 import하지 않음) |

## 값

| 이름 | 뜻 (기본값) | 설정 객체 | 읽는 곳 |
|---|---|---|---|
| `CHROMA_HOST` | Chroma 서버 호스트 (`localhost`) | `ChromaConfig.host` | backend, build, parity_l1, scripts |
| `CHROMA_PORT` | Chroma 서버 포트 (`8001`; compose는 호스트 쪽 포트 매핑에도 씀). build는 숫자가 아니면 `SystemExit`로 이름을 알려요 | `ChromaConfig.port` | backend, build, parity_l1, scripts, compose |
| `CHROMA_COLLECTION` | 검색할 표 문서 컬렉션 이름(`python scripts/kosis_build.py chroma`가 끝에 출력). build의 `chroma --list`는 이것을 활성 빌드로 표시 | `ChromaConfig.collection` | backend, build, parity_l1, compose |
| `TEXT_FORMAT` | 표 텍스트 형식 확인값(없으면 컬렉션 기록값; `n2-L60`). build의 bge-m3-ft 색인은 이 형식으로 임베딩 | `IndexSpec.text_format` | backend, build, parity_l1 |
| `EMBED_CKPT` | 미세조정 bge-m3 질의 임베더 체크포인트(`data/models/...pt`) | `IndexSpec.embed_ckpt` | backend, build(`chroma --model bge-m3-ft`, `hq-index`), parity_l1, lab |
| `RERANK_CKPT` | 1차 크로스인코더 체크포인트. 비면 1차 리랭커 꺼짐. build `hq-build`의 점수 필터에도 씀 | `IndexSpec.rerank_ckpt` | backend, build, parity_l1, lab(`hq_eval.py`) |
| `RERANK_MODE` | 1차 점수 쓰는 법 `blend` \| `score` (`blend`) | `SearchOptions.rerank_mode` | backend, parity_l1 |
| `STAGE2` | 2차 LLM 리랭커 `clova` \| `hcx` \| `gpt` \| `none` (`none`) | `SearchOptions.stage2` (+ `LLMConfig`) | backend, parity_l1 |
| `COND_FILTER` | 질문 조건(연도·지역·성별…) 필터 `on` \| `off` (`on`) | `SearchOptions.cond_filter` | backend, parity_l1 |
| `ANSWER_K` | 요청에 k가 없을 때 후보 표 수 (`5`) | `SearchOptions.answer_k` | backend, parity_l1 |
| `HQ_BUILD` | 질문 색인(역 HyDE) 빌드 id, 비면 꺼짐 (`hq-<prompt8>-<tables8>-<model>-f30-k2`) | `IndexSpec.hq_build` | backend, parity_l1, lab |
| `HQ_WEIGHT` | 질문 목록의 RRF 가중치 (`1.0`) | `SearchOptions.hq_weight` | backend, parity_l1, lab |
| `HQ_BM25` | 질문 목록에 BM25도 `on` \| `off` (`off`) | `SearchOptions.hq_bm25` | backend, parity_l1, lab |
| `HCX_API_KEY` | CLOVA Studio 키(`nv-...`): 2차 `clova`·`hcx`, 안내 LLM(ChatClovaX) | `LLMConfig.api_key` | backend, lab, 실전 1 |
| `CLOVASTUDIO_API_KEY` | 안내 LLM 키의 폴백(`HCX_API_KEY`가 비었을 때) | `LLMConfig.api_key` | backend |
| `HCX_THINKING` | 2차 `hcx`의 thinking 단계 `none` \| `low` … (`none`) | `LLMConfig.thinking` | backend, parity_l1 |
| `GUIDE_MODEL` `GUIDE_TEMPERATURE` `GUIDE_MAX_TOKENS` `GUIDE_TIMEOUT` | 안내 LLM 설정 (`HCX-DASH-002`, `0`, `1024`, `60`) | `LLMConfig` (provider `clova_chat`) | backend |
| `OPENAI_API_KEY` | OpenAI 키: 2차 `gpt`, build `hq-build`(가상 질문 생성; 없으면 캐시에 있는 표만 성공) | `LLMConfig.api_key` | backend, build |
| `KOSIS_API_KEY` | KOSIS OpenAPI 키. 이 레포 코드는 읽지 않아요(옛 실전 1 템플릿에서 온 줄) | — | (없음) |
| `KOSIS_WORKSPACE` | 실전 1이 `data/`·`eval/`을 찾는 작업 루트(이 레포 경로). `import kosis_rag`는 설정하지 않아요 | — | 실전 1, compose(`/app`), `scripts/phase1.py`, `lab/rag_baseline.py`, tests |
| `KOSIS_STAT_GOLDEN` | 문서 실험에 쓸 실전 1 골든 파일 경로 | — | `scripts/doc_experiment.py` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | 평가 실행 트레이스를 보낼 OTLP 주소(예 `http://localhost:4317`); 비면 내보내지 않음 | — | evals |
| `JAEGER_UI_URL` | 리포트의 Jaeger 링크 (`http://localhost:16686`) | — | evals |
| `COMPOSE_CHROMA_HOST` `COMPOSE_CHROMA_PORT` | compose 안 backend가 쓸 Chroma(공유 Chroma일 때; 기본 `chroma`·`8000`) | — | compose |

모듈 쪽 기본값은 `kosis_rag.schemas.resources`(`SearchOptions`·`IndexSpec`·`ChromaConfig`)와 `kosis_rag.llm.config.LLMConfig`에 있고, backend의 이름·기본값은 `tests/test_config_from_shell.py`가 옛 설정과 필드별로 대조해요.
