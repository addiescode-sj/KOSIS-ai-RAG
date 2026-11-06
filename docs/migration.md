# 실전 2 모듈 옮김표 (T2-41)

종합 레포(KOSIS-ai-multi-agent, `vendor/phase2` @484bd06 고정)를 이 리팩터 뒤의 `kosis_rag`로 옮길 때 보는 표예요.
단계마다 그 단계에서 옮긴 이름을 적어요: **옛 import → 새 import**, 바뀐 인자·반환, 지운 이름. 옛 경로는 shim 없이 지워요(P2-R6) — 옛 import는 `ImportError`가 나요.
기준: 브랜치 `T2-40-hq`@`5f627cb`. 계획: `plans/T2-41.md`. S0은 옮긴 이름이 없어요(패리티 도구와 `pyproject.toml` 패키지 선언만).

각 절의 표 형식: `| 옛 import | 새 import | 바뀐 인자·반환 |`

## S1. schemas/ · shared/ · adapters/ · 실전 1 사본 · import 부작용 제거

옛 모듈 파일 9개(`schemas.py filters.py fields.py roles.py vocab.py text.py loader.py vectorstore.py model.py`)는 지웠어요. `kosis_rag.schemas`는 이제 **패키지**이고 이름을 재수출하지 않아요 — `from kosis_rag.schemas import TableRecord`는 `ImportError`예요. 동작 변경 없음(패리티 fixture 33개 바이트 동일). `python -m` 진입점: `kosis_rag.vocab` → `kosis_rag.shared.vocab`, `kosis_rag.text` → `kosis_rag.shared.tabletext`(둘 다 S6a에 `kosis_rag_build.cli`로); `kosis_rag.metadata`·`kosis_rag.index`·`kosis_rag.hq.*`는 그대로.

| 옛 import | 새 import | 바뀐 인자·반환 |
|---|---|---|
| `kosis_rag.schemas.TableCandidate` · `SearchArgs` · `AskArgs` · `SearchResult` | `kosis_rag.schemas.io.<같은 이름>` | 없음 (필드·openapi 그대로) |
| `kosis_rag.schemas.TableRecord` · `TableClass` | `kosis_rag.schemas.catalog.<같은 이름>` | `TableRecord.info`는 사본 `kosis_rag.schemas.common.TableInfo`예요. `kosis_ai.schemas.TableInfo` **인스턴스**를 넘기면 검증 오류 → `info=x.model_dump()` 또는 사본으로 만드세요(dict·JSON은 그대로 됨) |
| `kosis_ai.schemas.TableInfo` · `Source` · `MAX_DIMS` (실전 2 코드 안) | `kosis_rag.schemas.common.<같은 이름>` (실전 1 `src/kosis_ai/schemas.py`@e248a45 사본) | 없음 (필드·검증·JSON 같음 — `tests/test_phase1_contract.py`). `TableCandidate.to_source()`는 사본 `Source`를 돌려줘요 |
| `kosis_ai.normalize._meta_unit` (`metadata.py` 안) | `kosis_rag.shared.units.meta_unit` (사본 @e248a45) | 없음 (같은 값 — 계약 테스트 ④) |
| — (신규) | `kosis_rag.shared.boundary.BoundaryModel` | 단계 스키마의 바탕 (S2부터 사용; 실전 1과 같은 설정) |
| — (신규) | `kosis_rag.schemas.resources.ChromaConfig(host="localhost", port=8001, collection=None)` | S4에 `SearchOptions`·`IndexSpec`가 같은 파일에 들어와요 |
| `kosis_rag.filters.*` | `kosis_rag.shared.filters.*` | 없음. `table_regions`·`has_dimension`은 `shared/fields.py`에 정의되고 `shared.filters`에서도 그대로 import돼요(순환 제거) |
| `kosis_rag.filters.dimension_keys` (재수출) | `kosis_rag.shared.vocab.dimension_keys` | 없음 (`shared.filters`에서는 재수출하지 않아요) |
| `kosis_rag.filters.MIN_SIDO` · `REGION_LEVEL_DIMS` · `SEX` · `TABLE_SIDO` | `kosis_rag.shared.fields.<같은 이름>` | 없음 (`shared.filters`에서는 더 못 가져와요) |
| `kosis_rag.fields.*` | `kosis_rag.shared.fields.*` | 없음 (`fields`는 `filters`를 import하지 않아요; `filters`가 `fields.implied`를 상단 import) |
| `kosis_rag.roles.*` | `kosis_rag.shared.roles.*` | 없음 |
| `kosis_rag.vocab.*` | `kosis_rag.shared.vocab.*` | 없음 (`build_sigungu`·`main`·`SIGUNGU`도 S6a까지 여기) |
| `kosis_rag.vocab.load_sigungu` | `kosis_rag.adapters.files.load_sigungu` | 없음 (기본 경로 `data/sigungu.json` 그대로) |
| `kosis_rag.metadata.REGION_LEVELS` | `kosis_rag.shared.vocab.REGION_LEVELS` | 없음 |
| `kosis_rag.text.*` (`from kosis_rag import text`) | `kosis_rag.shared.tabletext.*` (`from kosis_rag.shared import tabletext`) | 없음 |
| `kosis_rag.index.region_summary` · `sido_names` · `SIDO_SHORT` | `kosis_rag.shared.tabletext.<같은 이름>` | 없음 (`index.to_document`는 여기서 가져다 써요) |
| `kosis_rag.loader.load_catalog` · `to_document` · `to_candidate` · `_META_FIELDS` | `kosis_rag.shared.catalog.<같은 이름>` | `load_catalog`는 사본 `TableInfo` 목록을 돌려줘요 |
| `kosis_rag.vectorstore.chroma_client()` | `kosis_rag.adapters.chroma.chroma_client(config: ChromaConfig)` | **인자 필수**. `.env`·환경을 읽지 않아요. 호출자가 `ChromaConfig(host=…, port=…)`를 만들어요 (scripts: `ChromaConfig(host=os.environ.get("CHROMA_HOST", "localhost"), port=int(os.environ.get("CHROMA_PORT", 8001)))` — 셸 환경만, `.env` 안 읽음) |
| `kosis_rag.vectorstore.chroma_address()` | 삭제 → `ChromaConfig().host` · `.port` | src 안의 호출자(`retriever.ChromaTableRetriever.connect`, `hq.index`, `index` CLI)는 S4까지 `retriever.env_chroma_config()`/`index._chroma_config()`로 지금처럼 `.env`+환경을 읽어요(S4에 `backend/app/config.py`로) |
| `kosis_rag.model.*` (`from kosis_rag import model`) | `kosis_rag.adapters.models.*` | 없음 |
| `kosis_rag.index.Embed` | `kosis_rag.adapters.models.Embed` | 없음 |
| `kosis_rag.index.EMBED_MODEL` · `kure_embedder` | `kosis_rag.adapters.models.EMBED_MODEL` · `kure_embedder` | 없음. 값을 바꿔 쓰던 코드(`lab/rag_baseline.py`)는 `adapters.models.EMBED_MODEL`을 바꿔야 `connect`가 따라가요 |
| `kosis_rag.pipeline.read_records` | `kosis_rag.adapters.files.read_records` | 없음 (`index.read_records`는 S6a까지 그대로) |
| `import kosis_rag` (부작용: `KOSIS_WORKSPACE` 기본값 설정) | 부작용 없음 (P2-D5) | 실전 1이 `data/`·`.env`를 이 레포에서 읽게 하려면 **호출자가** `KOSIS_WORKSPACE`를 설정해요(셸·`docker-compose.yml`·`tests/conftest.py`·`scripts/phase1.py`). **영향받는 진입점**: `python -m evals.run_eval stat|e2e`, `lab/hq_eval.py`, `lab/synth_v1.py`는 셸에 `set KOSIS_WORKSPACE=<레포 경로>`가 없으면 실전 1이 `vendor/phase1/`의 `data/`·`.env`를 찾아 실패해요. backend는 영향 없음 |

## S2. parse/ · retrieve/ · legacy/retrievers

## S3. rerank/ · gate/ · llm/ · legacy/rerankers

## S4. graph/ · 설정 객체 · adapters/loader · SearchModule

## S5. respond → backend · SearchOutput

## S6. kosis_rag_build/ 분리 · 정리
