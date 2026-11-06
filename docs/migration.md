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

옛 모듈 파일 2개(`query.py retriever.py`)는 지웠어요. `hq/index.py`는 질문 색인 적재 CLI(`python -m kosis_rag.hq.index`, S6a에 `kosis_rag_build`로)만 남고 검색 쪽 이름은 모두 옮겼어요. 단계 폴더 `parse/`·`retrieve/`의 step 함수는 `(XxxInput, resources) -> XxxOutput`(BoundaryModel: strict, 모르는 필드 거절)이고, `self.last_*` 상태 없이 지운 표·푼 조건을 **반환값**으로 돌려줘요(P2-D17). LangChain `BaseRetriever` 클래스는 `kosis_rag.legacy.retrievers`로 옮겨 본문이 새 함수를 불러요(P2-D18) — 옛 호출자를 위해 `last_*`는 그대로 채워요. 그래프 노드 이름·간선·순위·점수·RRF는 그대로예요(패리티 fixture 33개 바이트 동일, `docs/openapi.json` 변경 0). 온라인 hq 읽기(`retrieve_hq`·`retrieve_hq_bm25`)는 그대로 있고 기본은 꺼져 있어요(`HQ_BUILD` 비어 있음).

| 옛 import | 새 import | 바뀐 인자·반환 |
|---|---|---|
| `kosis_rag.query.parse_query(question, sigungu=None)` | `kosis_rag.parse.node.parse_query(question, *, sigungu=None)` | `sigungu`는 **키워드 전용**. `None`은 이제 **빈 사전**(시군구 이름을 읽지 않음) — 옛 `None`은 import 때 `data/sigungu.json`에서 읽은 전역 `query.SIGUNGU`였어요. 옛 결과가 필요하면 `sigungu=kosis_rag.adapters.files.load_sigungu()`를 넘기세요(그래프·`build_search`는 `SearchResources.sigungu`로 자동) |
| `kosis_rag.query.SIGUNGU` (모듈 전역, import 때 파일 읽음 + 없으면 경고) | 삭제 → `kosis_rag.pipeline.SearchResources.sigungu` (`load_resources(..., sigungu=None)`가 `adapters.files.load_sigungu()`로 한 번 채움; 빈 사전이면 경고 1회) | `import kosis_rag.parse.node`는 파일을 읽지 않고 경고도 없어요(P2-D9). 테스트는 `load_resources(..., sigungu={...})` 또는 `parse_query(q, sigungu={...})` |
| `kosis_rag.query.ParsedQuery` | `kosis_rag.parse.schemas.ParsedQuery` | 없음 (보통 `BaseModel` 그대로 — `SearchState.query` 필드이고 openapi 스키마가 같아야 해서 BoundaryModel로 바꾸지 않았어요) |
| — (신규) | `kosis_rag.parse.schemas.ParseInput(question)` · `ConditionInput(query)` · `ConditionOutput(table_filter, relaxed)` | step In/Out (BoundaryModel) |
| — (신규, `graph.py` parse 노드 본문) | `kosis_rag.parse.node.parse_question(ParseInput, resources) -> ParsedQuery` | `= parse_query(question, sigungu=resources.sigungu)` |
| — (신규, `graph.py` condition 노드 본문) | `kosis_rag.parse.node.decide_condition(ConditionInput, resources) -> ConditionOutput` | 옛 노드 본문 그대로(`to_filter` → 완화 → hybrid의 `condition_filter` 단계). `resources.settings.cond_filter`·`resources.records`를 읽어요 |
| `kosis_rag.query.MAX_RESULTS` · `HARD_MEASURE_WORDS` · `SOFT_MEASURE_WORDS` | `kosis_rag.parse.constants.<같은 이름>` | 없음 |
| `kosis_rag.query.<정규식>` (`YEAR RANGE BETWEEN BIRTH CYCLES REGION DIMENSION MEASURES …`) | `kosis_rag.parse.patterns.<같은 이름>` | `_ONE_SIDO` → `ONE_SIDO`(다른 모듈이 씀). 나머지 이름 그대로 |
| `kosis_rag.query.SIDO` · `REGION_WORDS` · `DIMENSION_WORDS` (재수출) | `kosis_rag.shared.vocab.<같은 이름>` | 없음 (재수출 제거) |
| `kosis_rag.query.SIDO_ORDER` | `kosis_rag.shared.vocab.SIDO_ORDER` | 없음 (`list(SIDO)`) |
| `kosis_rag.query.ConditionRetriever` | `kosis_rag.legacy.retrievers.ConditionRetriever` | 새 필드 `sigungu: dict = {}` — 옛 클래스는 전역 `SIGUNGU`(파일)를 썼어요. 같은 결과를 원하면 `ConditionRetriever(..., sigungu=load_sigungu())` |
| `kosis_rag.retriever.tokenize` | `kosis_rag.shared.tokenize.tokenize` | 없음 |
| `kosis_rag.retriever.BM25TableRetriever` · `ChromaTableRetriever` · `HybridRetriever` · `hybrid_search` | `kosis_rag.legacy.retrievers.<같은 이름>` | 없음 (필드·`connect`·`with_filter`·`last_*` 그대로; `connect`는 S4까지 지금처럼 `.env`·환경을 읽어요). 새 것: `BM25TableRetriever.from_index(Bm25Index)`, `.index` 속성(`Bm25Index` / `DenseIndex`) |
| `kosis_rag.retriever.env_chroma_config` | `kosis_rag.legacy.retrievers.env_chroma_config` | 없음 (S4에 `backend/app/config.py`로) |
| `kosis_rag.retriever.condition_filter` · `search_list` | `kosis_rag.retrieve.lists.<같은 이름>` | `search_list`는 **4-튜플** `(docs, removed, relaxed, missing)`을 돌려줘요(옛 `(docs, removed)`). 새 인자 `vector=None`(embed 노드의 벡터; 옛 그래프의 `model_copy(embed_query=…)` 우회를 여기로). 목록으로 `DenseIndex`·`HqIndex`·`Bm25Index`도 받아요 |
| `kosis_rag.retriever.rrf_fuse` | `kosis_rag.retrieve.fuse.rrf_fuse` | 없음 (`c` 기본값 = `kosis_rag.shared.constants.RRF_C` = 60) |
| `ChromaTableRetriever` 본문(`_get_relevant_documents`) | `kosis_rag.retrieve.dense.DenseIndex(collection, embed_query, k=20, doc_format=None).search(query, *, vector=None, table_filter=None, records=None)` | `-> (docs, removed, relaxed, missing)` (`missing`은 항상 0) — 계획의 `(docs, removed)`에 `relaxed`·`missing`을 더해 legacy 클래스가 `last_*`를 반환값에서 채워요. `== ChromaTableRetriever.with_filter(f, records).invoke(q)`(테스트). 컬렉션 열기·임베더 불러오기는 하지 않아요(S4 `adapters/loader`) |
| `BM25TableRetriever` 본문 | `kosis_rag.retrieve.bm25.Bm25Index(docs, bm25, k=20, query_text=None).search(query)` · `bm25_index(docs, k, query_text)` · `bm25_search(index, query)` | 없음 (`== BM25TableRetriever.invoke`) |
| `kosis_rag.pipeline.bm25_text(records, k, fmt)` | `kosis_rag.retrieve.bm25.bm25_text(records, k, fmt)` | **`Bm25Index`를 돌려줘요**(옛: `BM25TableRetriever`). retriever가 필요하면 `BM25TableRetriever.from_index(...)` (`load_resources`가 그렇게 해요) |
| — (신규, `graph.py` embed · retrieve_<list> · fuse 노드 본문) | `kosis_rag.retrieve.node.embed(EmbedInput, resources) -> EmbedOutput(query_vecs, calls)` · `retrieve_list(name, ListInput(question, table_filter, vector), resources) -> ListOutput(docs, removed, relaxed, missing)` · `fuse(FuseInput(runs, run_removed), resources) -> FuseOutput(docs, removed)` · `search_lists(resources)` · `embedders(resources)` | 옛 노드 본문 그대로 (목록 순서 dense, bm25, [hq], [hq_bm25]; 가중치 1.0·`hq_weight`) |
| `kosis_rag.hq.index.HqRetriever` | `kosis_rag.legacy.retrievers.HqRetriever` | 없음 (`connect`·`with_filter`·`last_removed/last_relaxed/last_missing` 그대로). 경고 로거 이름 `kosis_rag.hq.index` → `kosis_rag.retrieve.hq` |
| `HqRetriever` 본문 | `kosis_rag.retrieve.hq.HqIndex(collection, embed_query, text_of={}, records={}, k=50, n_questions=200, n_docs=None).search(query, *, vector=None, table_filter=None, records=None)` | `-> (docs, removed, relaxed, missing)` (`records` 기본값 = `self.records`) |
| `kosis_rag.hq.index.max_pool` · `ckpt8` · `collection_name` | `kosis_rag.retrieve.hq.<같은 이름>` | 없음 |
| `kosis_rag.hq.index.hq_bm25(rows, records, text_of, k=50)` | `kosis_rag.retrieve.hq.hq_bm25(...)` | **`Bm25Index`를 돌려줘요**(옛: `BM25TableRetriever`) → `BM25TableRetriever.from_index(...)`. 경고 로거 `kosis_rag.retrieve.hq` |
| `kosis_rag.hq.index.read_questions(build_id, root=HQ_DIR)` · `HQ_DIR` | `kosis_rag.adapters.files.read_questions` · `HQ_DIR` | 없음 (`load_resources`는 `adapters.files.read_questions`를 모듈 경유로 불러요 — 테스트 패치 대상도 `kosis_rag.adapters.files`) |
| `kosis_rag.hq.index.COLLECTION_PREFIX` · `DOC_FORMAT` · `N_QUESTIONS` | `kosis_rag.retrieve.constants.HQ_COLLECTION_PREFIX` · `HQ_DOC_FORMAT` · `HQ_N_QUESTIONS` | **이름 바뀜**(같은 값 `"kosis_hq__"` · `"hq"` · `200`; `tabletext.DOC_FORMAT`·`index.COLLECTION_PREFIX`와 헷갈리지 않게) |
| `kosis_rag.hq.index.DESC_SUFFIX` | `kosis_rag.shared.constants.DESC_SUFFIX` | 없음 (`adapters.files`·hq CLI가 같이 써요) |
| `kosis_rag.index.COLLECTION_NAME` | `kosis_rag.shared.constants.COLLECTION_NAME` | 없음 (`index.py`는 `constants.COLLECTION_NAME`으로 써요) |
| `kosis_rag.pipeline.SearchResources(...)` | 그대로 + 필드 `sigungu: dict[str, list[str]] = {}` | `load_resources(..., sigungu=None)` 새 키워드 (None = 파일에서) |
| `from kosis_rag import query as query_rules` + `monkeypatch.setattr(query_rules, "parse_query", f)` (테스트) | `from kosis_rag.parse import node` + `monkeypatch.setattr(node, "parse_query", f)` | 그래프의 parse 노드는 `parse.node.parse_question`이 모듈 전역 `parse_query`를 `sigungu=` 키워드로 불러요 — 가짜 함수는 `**kw`를 받아야 해요 |

**영향받는 진입점 (S2, 사용자 확인 필요)**: `parse_query(question)`를 `sigungu` 없이 부르는 곳은 이제 시군구 이름을 읽지 않아요. 그래프·`build_search`·backend·`scripts/parity_l1.py`는 `SearchResources.sigungu`(= 옛 전역과 같은 파일)를 쓰므로 영향이 없어요. **`python -m evals.run_eval --layer l2`와 `--layer ablation --slot-source pred`**(`evals/run_eval.py`의 `parse_query(item.question)`)는 PC에 `data/sigungu.json`이 있으면 시군구가 든 질문의 `regions`가 옛 결과와 달라질 수 있어요(evals는 이 단계에서 import 줄만 고쳤어요). 같은 값을 원하면 그 두 호출에 `sigungu=load_sigungu()`를 넘기는 한 줄 변경이 필요해요(S4 `backend/app/config.py` 때 함께 처리 권고).

## S3. rerank/ · gate/ · llm/ · legacy/rerankers

## S4. graph/ · 설정 객체 · adapters/loader · SearchModule

## S5. respond → backend · SearchOutput

## S6. kosis_rag_build/ 분리 · 정리
