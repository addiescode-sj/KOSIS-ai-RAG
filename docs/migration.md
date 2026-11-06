# 실전 2 모듈 옮김표 (T2-41)

종합 레포(KOSIS-ai-multi-agent, `vendor/phase2` @484bd06 고정)를 이 리팩터 뒤의 `kosis_rag`로 옮길 때 보는 표예요.
단계마다 그 단계에서 옮긴 이름을 적어요: **옛 import → 새 import**, 바뀐 인자·반환, 지운 이름. 옛 경로는 shim 없이 지워요(P2-R6) — 옛 import는 `ImportError`가 나요.
기준: 브랜치 `T2-40-hq`@`5f627cb`. 계획: `plans/T2-41.md`. S0은 옮긴 이름이 없어요(패리티 도구와 `pyproject.toml` 패키지 선언만). 각 절은 그 단계에서 옮긴 이름이고, 뒤 단계에서 다시 바뀐 이름은 그 행에 적어 두었어요(최종 이름 기준). **종합 레포가 할 일 요약은 맨 끝 절**이에요. 설정값 목록은 `docs/config.md`.

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

옛 모듈 파일 3개(`rerank.py hcx.py clova.py`)는 지웠어요. `kosis_rag.rerank`는 이제 **패키지**이고 이름을 재수출하지 않아요 — `from kosis_rag.rerank import rerank_stage1`은 `ImportError`예요(`from kosis_rag.rerank import stage1`은 됨). `chain.py`에서 `decide_route`·`Route`·거절 문구를 `gate/`로 옮겼고 `chain`은 그 이름을 다시 내보내지 않아요(`chain.REFUSAL_TEXT`는 `AttributeError`). **`src/kosis_rag`의 `kosis_chat`·`kosis_ai` import는 0**이에요(실전 1 LLM 코드는 `llm/base.py`·`llm/hcx.py` 사본, `# copied from Phase 1 kosis_chat/llm/model.py@e248a45`). 키는 **인자**(`LLMConfig.api_key`)로 받아요 — 이 단계에서는 호출자 `pipeline.load_resources`(`env_stage2_llm`)·`chain.clova_chat()`·`hq.build` CLI가 아직 `.env`·환경에서 읽어 넘겨요(S4에 `backend/app/config.py`로). 2차 리랭커 프롬프트는 `rerank/prompts/stage2-hcx-v3.md`(패키지 데이터, `importlib.resources`; 옛 `hcx.SYSTEM`과 바이트 동일 — `tests/test_prompts_bytes.py`). 순위·점수·gate 판정·거절 문구·stage2 응답 파싱은 그대로예요(패리티 fixture 33개 바이트 동일, `docs/openapi.json` 변경 0). 그래프 노드 이름·상태 `route`(`refuse|guide`)도 그대로예요 — gate step은 `found`를 돌려주고 그래프가 S5까지 `guide`로 써요.

| 옛 import | 새 import | 바뀐 인자·반환 |
|---|---|---|
| `kosis_rag.rerank.blend_order` · `passage_scorer` · `rerank_stage1` | `kosis_rag.rerank.stage1.<같은 이름>` | 없음 (`c` 기본값 = `kosis_rag.shared.constants.RRF_C` = 60) |
| `kosis_rag.rerank.rerank_stage2` · `Score` · `Select` | `kosis_rag.rerank.stage2.<같은 이름>` | 없음 |
| `kosis_rag.rerank.RRF_C` | `kosis_rag.shared.constants.RRF_C` | 없음 (60) |
| `kosis_rag.rerank.CrossEncoderReranker` · `LLMReranker` | `kosis_rag.legacy.rerankers.<같은 이름>` | 없음 (필드 그대로, 본문은 `rerank_stage1`·`rerank_stage2` 호출) |
| `kosis_rag.rerank.LLM_RERANKERS` · `clova` · `hcx` · `gpt` (`(text_of, options: dict) -> Select`) | `kosis_rag.rerank.selectors.<같은 이름>` (`(text_of, config: LLMConfig) -> Select`) | **둘째 인자가 `LLMConfig`**(옛 `options` dict는 늘 `{}`였어요). 기본 설정은 `selectors.default_config(name, api_key=…, thinking=…)`: `clova` → provider `clova_rerank`, `hcx` → `hcx`·`HCX-007`, `gpt` → `openai`·`gpt-4o-mini`(`selectors.STAGE2_LLM`) |
| — (신규, `graph.py` rerank1 · rerank2 노드 본문) | `kosis_rag.rerank.node.rerank1(Stage1Input(question, docs), resources) -> Stage1Output(docs)` · `rerank2(Stage2Input(question, docs), resources) -> Stage2Output(docs)` | 옛 노드 본문 그대로. **`rerank2`는 진입 때** `resources.stage2_llm`(선택기를 만든 설정)에 `api_key`가 없고 provider가 `mock`이 아니면 `LLMKeyMissing` — 옛 그래프는 키가 없으면 호출 실패를 `stage2_error`에 적고 1차 순서로 계속했어요. 선택기를 직접 넘긴 경우(`load_resources(select=…)`, `stage2_llm` 없음)는 검사하지 않아요 |
| — (신규) | `kosis_rag.rerank.schemas.Stage1Input` · `Stage1Output` · `Stage2Input` · `Stage2Output` | step In/Out (BoundaryModel) |
| `kosis_rag.hcx.HcxSelector(text_of, thinking=<HCX_THINKING env>, client=None)` | `kosis_rag.rerank.hcx.HcxSelector(text_of, thinking=None, client=None, config=None)` | `thinking`은 **환경을 읽지 않아요**: `None`이면 `config.thinking`(없으면 `"none"`). `client`가 없으면 첫 호출 때 `config`로 `llm.providers.build_client`(provider `hcx` → `HCXClient`; `max_tokens`는 `config.max_tokens` 또는 `MAX_TOKENS[thinking]`, `temperature`는 `config.temperature` 또는 0). `client`도 `config`도 없으면 첫 호출이 `LLMKeyMissing`(옛: `.env`의 `HCX_API_KEY`를 읽음). 나머지(`parse`·사용자 메시지·`HcxResult`·실패 문구) 그대로 |
| `kosis_rag.hcx.parse` · `user_message` · `REASONS` · `HcxResult` | `kosis_rag.rerank.hcx.<같은 이름>` | 없음 |
| `kosis_rag.hcx.SYSTEM` | `kosis_rag.rerank.hcx.system_prompt()` (= `rerank/prompts/stage2-hcx-v3.md`, 패키지 데이터) | **함수**예요(첫 호출 때 파일을 읽고 캐시). 문자열은 옛 상수와 바이트 동일 |
| `kosis_rag.hcx.MODEL` · `MAX_PICK` · `MAX_TOKENS` · `PROMPT_VERSION` | `kosis_rag.rerank.constants.<같은 이름>` | 없음. 새 것: `PROMPT_FILE`(`stage2-hcx-v3.md`), `GPT_MODEL`, `CLOVA_RERANK_MODEL`(라벨) |
| `kosis_rag.hcx.OpenAIClient(model="gpt-4o-mini", temperature=0.0, max_tokens=512, seed=None)` | `kosis_rag.llm.openai.OpenAIClient(config: LLMConfig)` | **인자는 `LLMConfig` 하나**(provider `openai`; `None`인 필드는 옛 기본값 — 0.0, 512, seed 없음). 키는 `config.api_key`(옛: 호출 때 `.env`를 읽고 SDK가 `OPENAI_API_KEY`를 읽음), 없으면 `generate`가 `LLMKeyMissing`. `config.timeout`이 있으면 SDK에 넘겨요. `.model`·`.temperature`·`.max_tokens`·`.seed`는 읽기 전용 속성 |
| `kosis_rag.clova.ClovaReranker(text_of, wrap, max_tokens, waits, timeout=60, key=None, send=post)` | `kosis_rag.llm.clova_rerank.ClovaReranker(..., timeout=None, key=None, send=post, config=None)` | 키: `key`, 없으면 `config.api_key`(첫 호출 때), 둘 다 없으면 `RuntimeError`/`LLMKeyMissing`(옛: `.env`의 `HCX_API_KEY`). `timeout=None`이면 `config.timeout` 또는 60 |
| `kosis_rag.clova.api_key()` | 삭제 → 호출자가 `LLMConfig(provider="clova_rerank", api_key=…)` | `.env`를 읽지 않아요 |
| `kosis_rag.clova.post` · `WRAP` · `HOST` · `PATH` · `RETRY_CODES` · `ClovaResult` | `kosis_rag.llm.clova_rerank.<같은 이름>` | 없음 |
| — (신규) | `kosis_rag.llm.config.LLMConfig(provider, api_key: SecretStr \| None, model, thinking="none", temperature, max_tokens, seed, retry_interval=0.0, timeout)` · `LLMKeyMissing(RuntimeError)` · `require_key(config)` | 계획의 입출력 계약 그대로. `repr`·`model_dump`·`model_dump_json`에 키가 안 찍혀요. `retry_interval`은 자리만 있고 이 레포 클라이언트는 쓰지 않아요 |
| `kosis_chat.llm.model.STATUS_*` · `LLMResult` · `LLMClient` · `MockClient` · `TraceWriteError` (실전 2 코드 안) | `kosis_rag.llm.base.<같은 이름>` (사본 @e248a45) | 없음 (필드·값 같음 — `tests/test_phase1_contract.py` ⑤). 실전 1 `MockClient`를 그대로 넘겨도 돼요(덕 타이핑) |
| `kosis_chat.llm.model.HCXClient(settings: Settings, sleep, observer)` (`hcx.py` 안) | `kosis_rag.llm.hcx.HCXClient(config: LLMConfig, sleep=time.sleep, observer=None, *, base_url=DEFAULT_BASE_URL)` (사본) | **`Settings` 대신 `LLMConfig`**: `model`, `api_key`(없으면 `LLMKeyMissing` — 실전 1은 `ValueError`), `thinking`, `temperature`(None → 0.0), `max_tokens`(None → 1024), `timeout`(None → 60). 요청 본문·URL·재시도·가림은 실전 1과 같음(⑤) |
| — (신규) | `kosis_rag.llm.providers.PROVIDERS` · `build_client(config, **kw)` · `clova_chat(config)` | 등록표: `hcx` `openai` `clova_rerank`(`text_of=` 필요) `clova_chat` `mock`. 클라이언트는 키가 필요할 때 검사해요(`HCXClient`는 생성 때) |
| `kosis_rag.chain.clova_chat(settings=None)` (실전 1 `Settings`) | `kosis_rag.chain.clova_chat(config: LLMConfig \| None = None)` | 인자가 `LLMConfig`. `None`이면 지금처럼 이 레포 `.env`의 `HCX_API_KEY`(없으면 ChatClovaX의 `CLOVASTUDIO_API_KEY`) + `chain.GUIDE_MODEL`(`HCX-DASH-002`)·`GUIDE_TEMPERATURE`(0)·`GUIDE_MAX_TOKENS`(1024)·`GUIDE_TIMEOUT`(60) — 옛 값은 실전 1 `configs/model.yaml`을 실행 때 읽었고, 이제 상수예요(계약 테스트가 대조). 본문은 `llm.providers.clova_chat`. S5에 backend로 |
| `kosis_rag.chain.decide_route(docs) -> (route "refuse"\|"guide", reason)` | `kosis_rag.gate.node.decide_route(GateInput(docs), resources=None) -> GateOutput(route "found"\|"refuse", refusal_reason)` | **`guide` → `found`**, 튜플 → `GateOutput`. 규칙 그대로(문서 없음 → refuse; 2차가 오류 없이 0개 → refuse + `stage2_reason`). 그래프는 S5까지 `found`를 상태에 `guide`로 써요 |
| `kosis_rag.chain.Route` | `kosis_rag.gate.node.Route` (`Literal["found", "refuse"]`) | 값이 바뀜(`guide` → `found`) |
| `kosis_rag.chain.NO_MATCH` · `REFUSAL_TEXT` · `REFUSALS` | `kosis_rag.gate.texts.<같은 이름>` | 없음 (문구 바이트 동일). `chain`에서는 더 못 가져와요. **S6b**: `NO_MATCH`는 `backend.app.respond.texts.NO_MATCH`로 옮겼어요(모듈은 안 씀) |
| `kosis_rag.pipeline.load_resources(...)` · `build_search(...)` · `SearchResources` | 그대로 + 키워드 `stage2_llm: LLMConfig \| None = None`, 필드 `SearchResources.stage2_llm` | `STAGE2`가 켜져 있고 선택기를 주지 않으면 `stage2_llm`(없으면 `pipeline.env_stage2_llm(STAGE2)`: `.env`+환경의 `HCX_API_KEY`(clova·hcx)/`OPENAI_API_KEY`(gpt)·`HCX_THINKING`)으로 선택기를 만들어요 — 키를 읽는 때가 첫 호출에서 **`load_resources` 때**로 당겨졌어요. 키가 없으면 그래프는 `rerank2`에서 `LLMKeyMissing`, 옛 경로(`build_search` → `LLMReranker`)는 지금처럼 `stage2_error`(`LLMKeyMissing: …`) |
| — (신규) | `kosis_rag.pipeline.env_stage2_llm(name)` · `STAGE2_KEY_ENV` | S4에 `backend/app/config.py`로 옮겨 가는 임시 자리 |
| `kosis_rag.hq.build` CLI의 `OpenAIClient(model=…, temperature=…, max_tokens=…, seed=…)` | `OpenAIClient(LLMConfig(provider="openai", api_key=<OPENAI_API_KEY>, model=…, temperature=…, max_tokens=…, seed=…))` | CLI 동작 같음(`.env`·환경의 `OPENAI_API_KEY`; 없으면 경고 후 캐시에 있는 표만 성공). `prompt_hash` 영향 없음 |

**영향받는 진입점 (S3)**: 2차 리랭커를 켠 그래프(`STAGE2=clova|hcx|gpt`: backend, `evals` L1 `graph:*-clova|-hcx|-gpt` 프리셋, `lab/hq_eval.py`)는 키가 없으면 이제 `rerank2`에서 멈춰요(`LLMKeyMissing`). 키가 있으면 결과는 같아요. 기본값 `STAGE2=none`과 L1 패리티 프리셋(`hybrid+cond`·`staged`)은 영향이 없어요.

## S4. graph/ · 설정 객체 · adapters/loader · SearchModule

옛 모듈 파일 2개(`graph.py pipeline.py`)는 지웠어요. `kosis_rag.graph`는 이제 **패키지**(`search nodes_api spans`)이고 이름을 재수출하지 않아요 — `from kosis_rag.graph import build_graph`는 `ImportError`예요(`from kosis_rag.graph import search`는 됨). `SearchSettings`·`SearchSettings.from_env`는 **없어졌어요**: 호출자가 `SearchOptions(...)`(동작)·`IndexSpec(...)`(경로·연결·체크포인트)·`LLMConfig(...)`를 만들어 넘겨요. **`src/kosis_rag`는 `.env`도 `os.environ`도 읽지 않아요**(예외: CLI `index.py`·`hq/build.py`·`hq/index.py`, S6a까지). 셸 환경변수에서 설정 객체를 만드는 곳은 `backend/app/config.py` 하나예요(`.env`를 읽지 않음; 함수 이름은 S6b에 `*_from_env` → `*_from_shell`). 그래프 노드 이름·간선·순서·trace는 그대로예요(패리티 fixture 33개 바이트 동일, `docs/openapi.json` 변경 0). `build_graph(resources, client, k)`·`graph_nodes`·`make_nodes`의 `client` 인자는 **그대로**예요(제거는 S5) — 새로 0-인자 팩토리도 받아요.

| 옛 import | 새 import | 바뀐 인자·반환 |
|---|---|---|
| `kosis_rag.graph.SearchState` · `merge` | `kosis_rag.schemas.state.<같은 이름>` | 없음 (필드·순서·`route` refuse\|guide·`result` 그대로 — S5에서 정리). langgraph 없이 import돼요 |
| `kosis_rag.graph.graph_nodes(resources, client, *, k=5)` · `build_graph(...)` · `run_search` · `stream_search` · `Node` | `kosis_rag.graph.search.<같은 이름>` | 서명 그대로. `client`는 인스턴스 **또는 0-인자 팩토리**(`invoke`·`generate` 속성이 없는 callable): guide가 처음 돌 때 한 번 불러 그래프 안에 둬요. 노드는 노드표(이름 → 노드, 꺼진 단계는 행 없음)로 조립해요 |
| — (신규, `build_graph` 안의 간선 코드) | `kosis_rag.graph.search.add_search_edges(g, nodes)` | `parse → condition → [embed] → retrieve_* → fuse → [rerank1] → [rerank2] → gate → refuse\|guide → END` (옛 간선 그대로) |
| `kosis_rag.graph.make_nodes` · `apply_update` · `ground_docs` · `RETRIEVAL_STEPS` · `MERGED` · `is_retrieval_step` | `kosis_rag.graph.nodes_api.<같은 이름>` | 없음 (`client`는 위와 같음) |
| `graph_nodes` 안의 `timed(name, state, started, **extra)` | `kosis_rag.graph.spans.timed(name, trace, started, **extra)` · `spans_of(trace, skip=())` | 둘째 인자가 상태가 아니라 `state.trace`. `spans_of`는 `[{"node": name, **trace[name]}]` (노드 순서) |
| `kosis_rag.pipeline.SearchSettings(...)` | `kosis_rag.schemas.resources.SearchOptions(cond_filter, pool, rerank_mode, rerank_n, stage2, stage2_k, answer_k, hq_weight, hq_bm25)` + `IndexSpec(tables, sigungu, chroma: ChromaConfig, text_format, embed_ckpt, rerank_ckpt, hq_build, hq_root, device)` | **필드가 둘로 나뉘어요.** `collection` → `IndexSpec.chroma.collection`(`ChromaConfig(host, port, collection)`). 기본값은 옛 `SearchSettings` 기본값과 같음(`tests/test_config_from_shell.py`). `IndexSpec`의 `None` 경로 = 옛 기본 경로(`data/tables.jsonl` = `adapters.files.TABLES`, `data/sigungu.json`, `data/hq`) |
| `kosis_rag.pipeline.SearchSettings.from_env(**overrides)` | 삭제 → 호출자가 `SearchOptions(...)`·`IndexSpec(...)`를 만든다. 이 레포의 진입점(backend·scripts·lab)은 `backend.app.config.search_config_from_shell(**overrides) -> SearchConfig(options, spec, stage2_llm, guide_llm)` | **`.env`를 읽지 않아요** — `os.environ`만(키·기본값은 옛 `from_env`와 같음: `CHROMA_HOST CHROMA_PORT CHROMA_COLLECTION TEXT_FORMAT EMBED_CKPT RERANK_CKPT RERANK_MODE STAGE2 COND_FILTER ANSWER_K HQ_BUILD HQ_WEIGHT HQ_BM25 HCX_API_KEY HCX_THINKING OPENAI_API_KEY`). `overrides`는 `SearchOptions`·`IndexSpec` 필드 이름 또는 `collection`; 모르는 이름은 `TypeError`. 반환은 NamedTuple이고 **options가 먼저** — 이름으로 꺼내세요(`cfg.spec`, `cfg.options`) |
| `kosis_rag.pipeline.SearchSettings.label()` | `SearchOptions.label(spec: IndexSpec)` | 인자 `spec`(hq_build·rerank_ckpt를 읽음). 문자열 같음 |
| `kosis_rag.pipeline.SearchResources(settings=…, …)` | `kosis_rag.schemas.resources.SearchResources(options, spec, records, dense, bm25, text_format, stage1_score, select, stage2_name, stage2_llm=None, hq=None, hq_bm25=None, sigungu={})` | **`settings` → `options` + `spec`.** `resources.settings.pool` → `resources.options.pool`, `resources.settings.hq_build` → `resources.spec.hq_build`. step 함수(`parse/retrieve/rerank`)는 `resources.options`를 읽어요 |
| `kosis_rag.pipeline.load_resources(settings=None, *, records, dense, stage1_score, select, hq, hq_bm25, sigungu, stage2_llm)` | `kosis_rag.adapters.loader.load_resources(*, spec, options, llm=None, records=None, dense=None, bm25=None, stage1_score=None, select=None, hq=None, hq_bm25=None, sigungu=None)` | **전부 키워드 전용**(`load_resources(spec=…, options=…)`; 위치 인자는 `TypeError` — `tests/test_layout.py`가 위치 호출 0건을 확인). `settings=None`(= `.env`) 없음: `spec`·`options` 필수. `stage2_llm=` → `llm=`. `llm=None`이고 `stage2`가 켜져 있으면 `selectors.default_config(stage2)`(키 없음 → `rerank2` 진입 때 `LLMKeyMissing`; 옛: `.env`의 키). 새 키워드 `bm25=`(가짜 주입). hq 지연 import 제거. 시군구 경고 로거 `kosis_rag.pipeline` → `kosis_rag.adapters.loader` |
| `kosis_rag.pipeline.build_search(settings=None, *, records, dense, stage1_score, select, sigungu, stage2_llm)` | `kosis_rag.legacy.pipeline.build_search(resources: SearchResources)` | **인자는 불러 둔 리소스 하나**: `build_search(load_resources(spec=…, options=…))`. 반환 같음(Hybrid → [CrossEncoder] → [LLM]). 질문 색인은 리소스에 있어도 쓰지 않아요; 옛 함수처럼 아예 열지 않으려면 `IndexSpec(hq_build=None)`으로 불러요 |
| `kosis_rag.pipeline.text_format(settings, dense)` | `kosis_rag.shared.tabletext.text_format(spec: IndexSpec, dense)` | 첫 인자가 `IndexSpec` |
| `kosis_rag.pipeline.env_stage2_llm(name)` · `STAGE2_KEY_ENV` | `backend.app.config.stage2_llm_from_shell(name)` · `STAGE2_KEY_ENV` | `.env` 안 읽음(`os.environ`만). 꺼짐·모르는 이름이면 `None` |
| `kosis_rag.pipeline.REPO` · `OFF` | `kosis_rag.shared.vocab.REPO` · `kosis_rag.schemas.resources.OFF` | 없음 |
| `kosis_rag.legacy.retrievers.env_chroma_config()` | 삭제 → `ChromaConfig(host=…, port=…)` (backend: `search_config_from_shell().spec.chroma`) | `.env`·환경 읽기 없음 |
| `ChromaTableRetriever.connect(name=None, k, device, embed_ckpt)` | `ChromaTableRetriever.connect(name=None, k=20, device=None, embed_ckpt=None, *, chroma: ChromaConfig \| None = None)` | **환경을 안 읽어요**: 컬렉션은 `name` 또는 `chroma.collection`(없으면 `ValueError`; 옛: `CHROMA_COLLECTION`), 체크포인트는 `embed_ckpt`(옛: 없으면 `EMBED_CKPT`), 서버는 `chroma`(기본 `ChromaConfig()` = localhost:8001; 옛: `CHROMA_HOST`·`CHROMA_PORT`) |
| `HqRetriever.connect(build_id, dense, *, records, text_of, k, name)` | 그대로 + 키워드 `chroma: ChromaConfig \| None = None` | 서버 주소를 환경에서 읽지 않아요 (`load_resources`가 `spec.chroma`를 넘김) |
| `kosis_rag.chain.clova_chat(config=None)` (`None` = `.env`의 `HCX_API_KEY` + 상수) | 삭제 → `kosis_rag.llm.providers.clova_chat(config: LLMConfig)` (config 필수) | 앱은 `backend.app.config.guide_llm_from_shell()`로 config를 만들어요 |
| `kosis_rag.chain.GUIDE_MODEL` · `GUIDE_TEMPERATURE` · `GUIDE_MAX_TOKENS` · `GUIDE_TIMEOUT` · `GUIDE_KEY_ENV` | `backend.app.config.<같은 이름>` | `GUIDE_KEY_ENV`가 튜플 `("HCX_API_KEY", "CLOVASTUDIO_API_KEY")`(앞이 우선). **안내 LLM 설정은 실전 1 `configs/model.yaml`이 아니라 `LLMConfig`**: `guide_llm = LLMConfig(provider="clova_chat", model=GUIDE_MODEL env or "HCX-DASH-002", temperature=GUIDE_TEMPERATURE or 0, max_tokens=GUIDE_MAX_TOKENS or 1024, timeout=GUIDE_TIMEOUT or 60, api_key=HCX_API_KEY or CLOVASTUDIO_API_KEY or None)` — 기본값은 model.yaml @e248a45 값을 상수로(계약 테스트가 대조) |
| — (신규) | `kosis_rag.api.SearchModule(spec, options=None, llm=None)` · `.search(SearchInput) -> SearchOutput` · `.stream(SearchInput)` · `.search_tables(question, k=None) -> list[TableCandidate]` · `SearchModule.from_resources(resources)` · `.graph(k)` · `kosis_rag.api.to_output(state, k)` | 생성 때 `load_resources`로 한 번 열고 k마다 그래프 하나를 캐시. S5까지 그래프 끝의 refuse·guide는 mock 안내 클라이언트로 돌고(LLM 호출 0) 출력에 쓰이지 않아요. `stream`은 노드 이벤트 `(name, trace)` 뒤 마지막 `("output", SearchOutput)` (refuse·guide·token 없음). `to_output` = S0 `project_output` 규칙 + `spans`(refuse·guide 제외) |
| — (신규) | `kosis_rag.schemas.io.SearchInput(schema_version="2.0", question, k=None)` · `SearchOutput(schema_version, question, route found\|refuse, refusal_reason, refusal_text, picked, candidates, spans)` | 계획의 입출력 계약 그대로(S5 확정). 실전 1에 넘기는 값은 `picked[0].key`("101_DT_…") |
| — (신규) | `backend.app.config.search_config_from_shell` · `SearchConfig` · `guide_llm_from_shell` · `stage2_llm_from_shell` | 위 행 참고. evals·lab·scripts도 이 함수를 써요(evals는 담당자가 바꾸기 전까지 옛 줄 그대로 — 아래) |
| `backend.app.main.Runtime(resources, client)` | `Runtime(resources, guide_llm: LLMConfig \| None = None, client=None, graphs={})` · `Runtime.guide_client()` | 안내 클라이언트를 **지연 생성**: 첫 guide 때 `guide_llm`으로 `clova_chat`, 키가 없으면 `LLMKeyMissing` → 503. `client`는 미리 만든 클라이언트(테스트). `rt.k(None)` = `resources.options.answer_k` |

**영향받는 진입점 (S4, 사용자 확인 필요)**
- **`.env`가 더는 읽히지 않아요(모듈·backend·`scripts/parity_l1.py`).** 셸에 직접 두세요(Windows `set CHROMA_COLLECTION=…`, 공통 명령 참고; `docker compose`는 `env_file: .env`가 컨테이너 환경으로 넣어 주므로 그대로 동작). 필요한 변수: `CHROMA_HOST CHROMA_PORT CHROMA_COLLECTION EMBED_CKPT RERANK_CKPT`(+ 쓰면 `TEXT_FORMAT RERANK_MODE STAGE2 COND_FILTER ANSWER_K HQ_BUILD HQ_WEIGHT HQ_BM25 HCX_THINKING`), 키 `HCX_API_KEY`(stage2 clova·hcx, 안내 LLM) · `OPENAI_API_KEY`(stage2 gpt) · `CLOVASTUDIO_API_KEY`(안내 LLM 폴백), 안내 모델 `GUIDE_MODEL GUIDE_TEMPERATURE GUIDE_MAX_TOKENS GUIDE_TIMEOUT`(선택). 예외: CLI `python -m kosis_rag.index`·`kosis_rag.hq.build`·`kosis_rag.hq.index`는 S6a까지 지금처럼 `.env`를 읽어요. `lab/*`의 `load_dotenv`는 S6b에 지웠어요(키·체크포인트는 셸에서) — `lab/hq_eval.py`는 셸 환경으로 `search_config_from_shell`를 불러요.
- **backend**: 안내 키가 없어도 서버가 떠요. `/nodes/parse`~`/nodes/gate`는 키 없이 200, `/nodes/answer`(guide)·`/search`는 `503 "LLM key missing: guide LLM key not set …"`(옛: `get_runtime`에서 503이라 모든 노드가 막힘). `/search/stream`은 응답 줄이 이미 나간 뒤라 `event: error`(`LLMKeyMissing`)로 와요. openapi 변화 없음.
- **evals (고치지 않음, `refactor/phase2/03-evals-direction-for-worker.md` §2)**: `evals/runners/retrieval.py`는 import 줄만 새 경로로 바꿨어요(`kosis_rag.legacy.pipeline.build_search`, `kosis_rag.adapters.loader.load_resources`, `kosis_rag.graph.search`). `_staged_settings`의 `from kosis_rag.pipeline import SearchSettings`는 대체 이름이 없어 그대로 두었고(`tests/test_layout.py` `CALLER_EXCEPTIONS`), 그래서 **`--layer l1 --retriever <STAGED 프리셋>|graph:<프리셋>`은 `ImportError`**, `--retriever chroma`와 `--layer l0`의 Chroma 확인(`ChromaTableRetriever.connect()`)은 컬렉션 이름을 환경에서 읽지 않으므로 **`ValueError`(no collection)**로 실패해요. `oracle`·`bm25-*` 리트리버, L2·ablation·L3·e2e(검색기 주입) 경로와 채점·리포트는 영향 없음. `tests/test_graph.py::test_eval_runner_graph_preset_matches_the_old_preset`는 같은 이유로 skip(사유에 03 §2). 평가 담당자가 03 §2대로 `search_config_from_shell`(또는 자체 설정) + `load_resources(spec=, options=)` / `SearchModule`로 바꾸면 돼요.
- **`scripts/parity_l1.py`**: `build_preset_graph`가 evals를 거치지 않고 `search_config_from_shell(**STAGED[preset])` → `load_resources(spec=, options=, llm=)` → `build_graph(resources, MockClient())`(모듈 사본 `kosis_rag.llm.base.MockClient`)로 만들어요. `STAGED` 프리셋 값은 evals에서 데이터로만 읽어요. `config_fingerprint`의 키·값은 S0과 같음(`tests/test_parity_l1_tool.py`가 S0 함수 사본과 대조) — 단 **덤프 전에 셸에 `CHROMA_COLLECTION`·`EMBED_CKPT`·`RERANK_CKPT`를 `.env`와 같은 값으로** 두어야 S0 manifest와 지문이 맞아요.

## S5. respond → backend · 모듈은 gate에서 END · SearchOutput 확정

옛 모듈 파일 2개(`chain.py tools.py`)는 지웠어요. **안내 문장·답변 검증·출처·토큰 스트림·거절 답변은 앱 계층 `backend/app/respond/`로 나갔어요** — 종합 레포(KOSIS-ai-multi-agent)는 자기 에이전트가 문장을 쓰므로 이 패키지를 가져가지 않아요. 가져갈 것은 `kosis_rag.gate.texts`의 문구와 `SearchOutput.refusal_text`예요. 모듈 그래프는 `parse → condition → [embed] → retrieve_* → fuse → [rerank1] → [rerank2] → gate → END`로 끝나고 **LLM 클라이언트를 받지 않아요**(`build_graph`·`graph_nodes`·`make_nodes`에서 `client` 제거, S4의 0-인자 팩토리 통로도 삭제). backend가 모듈 노드 + respond 노드를 한 그래프로 조립해 `/search`·`/search/stream`·`/nodes/*`가 그대로예요: 노드 이름·간선·SSE 이벤트 순서·`/nodes` 목록·`docs/openapi.json` 변경 0, 패리티 fixture 33개 바이트 동일(`output` fixture는 이제 **`SearchModule.search()` 결과**를 그대로 비교하고, S0 투영 함수 `tests/parity_cases.project_output`은 지웠어요).

| 옛 import | 새 import | 바뀐 인자·반환 |
|---|---|---|
| `kosis_rag.graph.search.build_graph(resources, client, *, k=5)` | `kosis_rag.graph.search.build_graph(resources, *, k=5)` | **`client` 제거.** 그래프는 `gate → END`(refuse·guide 없음). 앱 응답 그래프는 `backend.app.respond.graph.build_answer_graph` |
| `kosis_rag.graph.search.graph_nodes(resources, client, *, k=5)` | `graph_nodes(resources, *, k=5)` | `client` 제거, 노드표가 `gate`로 끝남(`refuse`·`guide` 행 없음) |
| `kosis_rag.graph.search.add_search_edges(g, nodes)` | 같은 이름 | **gate까지만** 이어요(gate 뒤 간선은 호출자: `build_graph`는 `gate → END`, backend는 `refuse \| guide`) |
| `kosis_rag.graph.search.stream_search(graph, question, config=None)` → 마지막 `("result", SearchState)`, 중간 `("token", …)` | `stream_search(graph, question, config=None)` → 노드 이벤트 뒤 마지막 **`("state", SearchState)`**(최종 상태) | token 이벤트 없음. **S6b 최종**: S5의 `*, k=5`와 마지막 `("output", SearchOutput)`은 지웠어요(graph/가 `kosis_rag.api`를 import하지 않게, k가 그래프의 k와 어긋나지 않게). `SearchOutput` 스트림은 `SearchModule.stream`. 앱의 옛 동작은 `backend.app.respond.graph.stream_answer` |
| `kosis_rag.graph.search.run_search(graph, q)` | 같은 이름 | 반환 `SearchState`에 `result`가 없고 `route`는 `found \| refuse` |
| `kosis_rag.graph.nodes_api.make_nodes(resources, client, *, k=5)` | `make_nodes(resources, *, k=5)` | `client` 제거, `parse retrieve [rerank1] [rerank2] gate`만. 앱 노드 API 집합은 `backend.app.respond.graph.api_nodes(resources, guide_llm, *, k=5)` |
| `kosis_rag.schemas.state.SearchState.route: "refuse" \| "guide"` · `.result: SearchResult` | `route: "found" \| "refuse"` · **`result` 삭제** | `gate` trace의 `route`도 `found`. 앱(API) 상태는 `backend.app.respond.state.SearchState`(옛 13필드·순서·`route` refuse\|guide·`result` 그대로 — openapi 스키마 이름·필드 순서 동일) |
| `kosis_rag.api.to_output(state, k)` · `SearchModule` | 같은 이름 | 확정: 그래프에 앱 노드가 없어 mock 안내 클라이언트·`APP_NODES` 제외 규칙을 지웠어요. `spans`는 trace 전체(노드 순서, 마지막이 `gate`). `refusal_text = gate.texts.REFUSALS.get(reason, REFUSAL_TEXT)`(거절일 때). `SearchModule.stream`은 `stream_search`의 노드 이벤트를 내보내고 마지막 `("state", …)`를 `("output", to_output(state, k))`로 바꿔요(S6b) |
| `kosis_rag.chain.search(retriever, question, k=5)` | `kosis_rag.legacy.search.search` | 없음 |
| `kosis_rag.tools.build_tools(retriever, client)` → `[search_tables, answer_table_question]` | `kosis_rag.legacy.tools.build_tools(retriever)` → **`[search_tables]`** | `client` 제거. 답 툴은 `backend.app.respond.legacy.answer_tool(retriever, client)` (이름·인자·설명 그대로) |
| `kosis_rag.chain.answer(retriever, client, question, k=5)` | `backend.app.respond.legacy.answer` | 없음 (같은 결과) |
| `kosis_rag.chain.PROMPT` (ChatPromptTemplate) | `backend.app.respond.prompts.guide_prompt()` (파일 `backend/app/respond/prompts/guide-v1.md`, `importlib.resources`로 읽음) · `guide_messages() -> (system, human)` · `PROMPT_FILE` · `SEPARATOR` | 상수가 아니라 함수(한 번 읽어 캐시). 파일 = system + `"\n<!-- human -->\n"` + human, 옛 문자열과 바이트 동일(`tests/test_prompts_bytes.py`) |
| `kosis_rag.chain.llm_runnable(client, on_chunk=None)` | `backend.app.respond.node.llm_runnable` | 없음 (계획은 `respond/legacy.py`였지만 guide가 매번 쓰므로 `node.py`에 둠) |
| `kosis_rag.chain.refusal_answer(question, docs, reason)` | `backend.app.respond.node.refuse(RefuseInput(question, candidates, refusal_reason, refusal_text)) -> SearchResult` | **입력이 모듈 출력의 필드**: `candidates`(TableCandidate)·`refusal_text`를 `SearchOutput`에서 받아요(그래프의 `refuse` 노드는 `kosis_rag.api.to_output`으로 읽음). 결과 같음 |
| `kosis_rag.chain.guide_answer(question, docs, client, on_text=None)` | `backend.app.respond.node.guide(GuideInput(question, candidates), llm, on_text=None) -> SearchResult` | 문서 대신 `candidates`(TableCandidate). 결과·스트림 조각 같음 |
| `kosis_rag.chain.unsupported_claims` · `_references` · `picked_from_reply` · `streamable_prefix` | `backend.app.respond.verify.unsupported_claims` · `references` · `picked_from_reply` · `streamable_prefix` | `_references` → `references`(공개 이름) |
| `kosis_rag.chain.format_context` · `template_answer` · `with_citations` | `backend.app.respond.render.<같은 이름>` | 없음. 고정 문구는 `backend.app.respond.texts.TEMPLATE_OPENING`·`CITATION_HEADER`("출처:") |
| `kosis_rag.chain._TBL_ID` · `_YEAR` · `_REF_RANGE` · `_REF` | `backend.app.respond.patterns.TBL_ID` · `YEAR` · `REF_RANGE` · `REF` | 밑줄 없는 이름 |
| `kosis_rag.gate.texts.NO_MATCH` | **S6b**: `backend.app.respond.texts.NO_MATCH` | 안내 LLM의 "맞는 표 없음" 표지(앱만 읽음 — 모듈은 쓰지 않아 앱 쪽으로). 거절 문구 `REFUSAL_TEXT`·`REFUSALS`는 `kosis_rag.gate.texts` 그대로 |
| `graph.search` 안 refuse·guide 노드 | `backend.app.respond.node.answer_nodes(guide_client, *, k=5) -> {"refuse", "guide"}` | API 상태 위에서 돌고 trace(`refuse`, `guide` + `origin`)는 옛 것과 같아요. guide는 `get_stream_writer`로 `{"delta"}`를 흘려요 |
| — (신규) | `backend.app.respond.graph.gate_to_api(update) -> dict` | 모듈 gate 노드 반환의 `route`(와 gate trace의 `route`) `"found"` → `"guide"`. **`build_answer_graph`와 `api_nodes`(→ `/nodes/gate`) 둘 다 이 함수 하나**를 써요 — `/nodes/gate` 응답을 그대로 `/nodes/answer`에 보내면 돼요 |
| — (신규) | `backend.app.respond.graph.build_answer_graph(resources, guide_llm, *, k=5)` · `run_answer(graph, q, config=None)` · `stream_answer(graph, q, config=None)` | 옛 `build_graph(resources, client)`·`run_search`·`stream_search`의 앱 쪽. `guide_llm`은 `LLMConfig`(guide 첫 실행 때 `llm.providers.build_client`로 생성), 미리 만든 클라이언트(ChatClovaX·MockClient), 또는 `GuideClient`. 계획 문구의 `options` 인자는 `resources.options`라 따로 받지 않아요 |
| `backend.app.main.Runtime.guide_client` (S4: 그래프에 팩토리로 전달) | `backend.app.respond.node.GuideClient(llm)` · `.get()`; `Runtime.guide`(하나를 모든 그래프·노드 집합이 공유) · `Runtime.guide_client()` = `guide.get()` | `Runtime(resources, *, guide_llm=None, client=None, graphs={})` 호출 방식 그대로. 키(`api_key`)가 없으면 guide 진입 때 `LLMKeyMissing` → `/search`·`/nodes/answer` 503, `/search/stream`은 `event: error`. 클라이언트는 `PROVIDERS["clova_chat"]`로 만들어요(테스트가 바꿀 곳: `providers.PROVIDERS`, 옛 `providers.clova_chat` 패치는 더 안 먹어요) |

**실전 3(종합 레포) 인계**
- **실전 1에 넘기는 값은 `picked[0].key`(복합 키 `"<org_id>_<tbl_id>"`, `tables.csv`의 `key` 열)**예요. `picked[0].tbl_id`로 `kosis_chat.api.table_path`/`summarize_stat`를 부르면 `LookupError`(`tests/test_phase1_contract.py` ⑥).
- **`route == "found"`는 "후보를 찾았다"이지 "계산할 수 있다"가 아니에요.** 그 뒤의 거절은 모듈 밖에서 받아요: (1) 이 레포 앱의 안내 LLM이 `NO_MATCH`로 답하면 `SearchResult.no_answer=True`·`answer_origin="refusal"`(backend만), (2) 실전 1 `StatAnswer`의 `refused`·`clarify` 답, 표를 못 찾으면 `LookupError`. `SearchOutput`에는 필드를 더하지 않았어요(승인된 2.0 그대로).
- `route == "refuse"`면 `refusal_text`(코드 문구, `gate.texts`)를 그대로 보여 주면 돼요. `refusal_reason`은 `out_of_scope | no_item | None`.

**영향받는 진입점 (S5)**
- **evals (고치지 않음)**: `evals/runners/retrieval.py`의 `graph:<프리셋>` 줄은 `build_graph(load_resources(...), MockClient())`라 S4부터 이미 깨져 있었고(`load_resources` 위치 인자), 이제 `build_graph`가 클라이언트를 받지 않아 고칠 때 `build_graph(load_resources(spec=…, options=…))`로 바꾸면 돼요. 다른 evals 경로는 영향 없음.
- `lab/rag_baseline.py`: `kosis_rag.chain.answer` → `backend.app.respond.legacy.answer`(import 줄만). `lab/hq_eval.py`: `build_graph(r, MockClient())` → `build_graph(r)`.
- `scripts/parity_l1.py`: `build_preset_graph`가 `build_graph(resources)`(생성 줄만). 덤프의 `route`가 이제 `found`이고 compare는 S0의 `guide`를 `found`로 읽어요(`ROUTE_ALIASES`).

- `notebooks/rag_skeleton.ipynb`: `from kosis_rag import chain`·`chain.PROMPT`·`chain.answer`·`from kosis_rag.tools import build_tools`가 이제 `ImportError`예요(S2부터 `kosis_rag.retriever`도 없어 이미 깨져 있었음). 쓰려면 `backend.app.respond.legacy.answer`·`backend.app.respond.prompts`·`kosis_rag.legacy.tools.build_tools`로 바꿔요(노트북은 사용자가 관리해 리팩터에서 고치지 않음).
- `tests/fixtures/parity/README.md`의 `project_output(state, k)` 설명은 S0 기준이에요. S5부터 `output.json`은 `SearchModule.search()` 결과(spans 제외)와 바이트 비교해요(얼린 폴더라 README는 그대로 둠).

## S6a. `kosis_rag_build/` 분리 · 오프라인 노드화 · hq 이동 · cli

오프라인 DB 준비 코드(표 등록부 `tables.jsonl`·시군구 사전 `sigungu.json`·표 문서 Chroma 색인·hq 가상 질문 생성·질문 색인 적재)를 **새 패키지 `src/kosis_rag_build/`**로 옮겼어요. 옛 모듈 `kosis_rag/index.py`·`kosis_rag/metadata.py`·`kosis_rag/hq/`(폴더 전체)는 지웠고, `shared/vocab.py`의 `build_sigungu`·`main`·`SIGUNGU_SOURCE`와 `shared/tabletext.py`의 `main`도 빠졌어요(재수출 없음). **`kosis_rag`는 `kosis_rag_build`를 import하지도, 이름을 적지도 않아요**(`tests/test_layout.py`: import 0 + 문자열 0 + `import kosis_rag.api` 뒤 `sys.modules`에 `kosis_rag_build` 없음). `kosis_rag_build`는 `kosis_rag.{schemas,shared,adapters,llm,rerank.stage1}`만 써요(예외 없음). 옛 CLI 함수 하나씩이던 빌드는 **LangGraph 그래프**(`kosis_rag_build/graph/`)가 됐고, CLI는 `python -m kosis_rag_build.cli` 하나예요. 검색 쪽에 남는 것: `kosis_rag.retrieve.hq`(HqIndex·max_pool·hq_bm25 — 이미 만든 컬렉션·`questions.jsonl` 읽기), `kosis_rag.shared.hqnames`(ckpt8·collection_name·HQ_COLLECTION_PREFIX·HQ_DOC_FORMAT), `kosis_rag.adapters.files.read_questions`·`read_records`·`load_sigungu`, `IndexSpec.hq_build`·`hq_root`. 온라인 `retrieve_hq`·`retrieve_hq_bm25` 노드는 그대로 `kosis_rag`에 있고 기본은 꺼짐(`IndexSpec.hq_build=None`). 산출물은 바이트 동일: 가짜 Parquet·가짜 임베더로 네 그래프가 옛 CLI 함수(테스트 안에 동결 복사)와 같은 파일을 써요(`tests/test_index_graphs.py`), hq 프롬프트는 `.md`에서 읽어도 옛 상수와 바이트 동일·`prompt_hash()[:8] == "cb0b3c5b"`(`tests/test_hq_prompt_bytes.py`), 패리티 fixture 33개·`docs/openapi.json` 변경 0.

**종합 레포용 진입점**: `python -m kosis_rag_build.cli {tables|sigungu|chroma|text|hq-build|hq-index} …`, 그래프 `kosis_rag_build.graph.index_build.build_tables_graph()`·`build_sigungu_graph()`·`build_chroma_graph(*, embedder=None, client=None, log=print)`·`build_questions_graph(*, embedder=None, client=None)`(입력은 `kosis_rag_build.schemas`의 `TablesBuildState`·`SigunguBuildState`·`IndexBuildState`·`QuestionsIndexState`), hq 생성 `kosis_rag_build.graph.hq_build.run_build(records, client, *, model, tables_sha256, …)`(그래프 `build_table_graph`·`build_build_graph`). DB 적재 전에 종합 레포가 부르고, 검색은 `kosis_rag.api.SearchModule`이 결과물(컬렉션·`questions.jsonl`)만 읽어요.

| 옛 import | 새 import | 바뀐 인자·반환 |
|---|---|---|
| `python -m kosis_rag.metadata [--data-dir D] [--out F]` | `python -m kosis_rag_build.cli tables [--data-dir D] [--out F]` | 같은 파일(바이트 동일). 그래프 `build_tables_graph`: `read_parquet → build_records → write_jsonl`(상태 `TablesBuildState(data_dir, out)`) |
| `kosis_rag.metadata.build_records(data_dir)` · `region_level` · `_values` · `_issues` · `_search_text` · `DATA_DIR` `CATALOG` `REGION_CLASS` `NATION` `URBAN` `SIDO` `TOP_REGIONS` `BATCH_ROWS` | `kosis_rag_build.tables.<같은 이름>` | 없음. 새로 `read_catalog(data_dir)`·`read_rows(data_dir)`·`read_table(data_dir, row, catalog) -> dict`·`to_record(source) -> TableRecord`(= 옛 루프 몸통을 둘로 나눔), 노드 `read_parquet(state)`·`build_records_node(state)`·`write_jsonl(state)` |
| `kosis_rag.metadata.main(argv)` | `kosis_rag_build.cli.main(["tables", …])` | 반환 0, 출력 줄 같음 |
| `kosis_rag.shared.vocab.build_sigungu(parquet)` · `SIGUNGU_SOURCE` · `main(argv)` (`python -m kosis_rag.shared.vocab --data-dir D`) | `kosis_rag_build.sigungu.build_sigungu(parquet)` · `SIGUNGU_SOURCE` · `python -m kosis_rag_build.cli sigungu --data-dir D [--out F]` | 같은 파일(바이트 동일). 새로 `read_pairs(parquet)`·`sigungu_of(rows)`·`dump(mapping)`, 그래프 `build_sigungu_graph`: `read_parquet → build_sigungu → write_json`. `kosis_rag.shared.vocab.SIGUNGU`(경로)와 `adapters.files.load_sigungu`는 그대로 |
| `kosis_rag.shared.tabletext.main(argv)` (`python -m kosis_rag.shared.tabletext`) | `python -m kosis_rag_build.cli text [--tables] [--tokens] [--samples] [--out]` | 출력·샘플 파일 같음 |
| `kosis_rag.index.DOC_FORMAT` `META_FORMAT` `COLLECTION_PREFIX` `FT_COLLECTION_PREFIX` `LABEL_CAP` `UNIT_SUFFIX` · `clean_item` `to_document` `to_metadata` `collection_name` `build_record` `load(client, …)` `write_build` `read_build` | `kosis_rag_build.chroma.<같은 이름>` | 없음(`load`는 `client` 인자 그대로). `EMBED_MODEL`·`kure_embedder`는 S1부터 `kosis_rag.adapters.models`, `region_summary`는 `kosis_rag.shared.tabletext`, `COLLECTION_NAME`은 `kosis_rag.shared.constants` |
| `kosis_rag.index.read_records(path)` · `REPO` | `kosis_rag.adapters.files.read_records(path)` · (REPO는 호출자가 정함; scripts는 `Path(__file__).resolve().parents[1]`) | `files.read_records`는 빈 줄을 건너뛰어요(옛 것은 빈 줄에서 오류). `tables.jsonl`에는 빈 줄이 없어 결과 같음 |
| `kosis_rag.index.list_builds()` (CHROMA_* 를 `.env`·환경에서 읽음) | `kosis_rag_build.chroma.list_builds(client, active=None, out=print)` | **클라이언트와 활성 컬렉션 이름을 인자로**. CLI `chroma --list`가 `CHROMA_COLLECTION`을 넘겨요 |
| `kosis_rag.index._chroma_config()` · `_batches` | (CLI `kosis_rag_build.cli.chroma_config(args, env)`) · `chroma._batches` | `--chroma-host`/`--chroma-port` → `CHROMA_HOST`/`CHROMA_PORT` → `localhost:8001`. `.env`는 안 읽음 |
| `python -m kosis_rag.index [--model kure\|bge-m3-ft] [--ckpt] [--text-format] [--collection] [--device] [--list] [--export F] [--import F]` | `python -m kosis_rag_build.cli chroma …`(같은 인자 + `--chroma-host` `--chroma-port`) | 같은 build(`ids docs metas record`·벡터; `index_version`만 실행 시각). 그래프 `build_chroma_graph`: `read_records → build_docs → embed → load \| write_build`(`--export`면 write_build). `--ckpt`·`--text-format` 기본값은 셸 `EMBED_CKPT`·`TEXT_FORMAT`(옛: `.env`도). 새로 `DocEmbedder(embed, count_tokens, max_tokens, info)`·`load_doc_embedder(state)`·노드 팩토리 `make_build_docs` `make_embed` `make_load`·`write_build_node` |
| `kosis_rag.hq.schemas.QType` `TYPES` `MAX_QUESTIONS` `HqQuestion` `TableQuestions` `HqState` | `kosis_rag_build.schemas.<같은 이름>` | 없음 |
| `kosis_rag.hq.graph.BuildState` | `kosis_rag_build.schemas.HqBuildState` | **이름 변경**(필드 같음) |
| `kosis_rag.hq.prompt.SYSTEM` · `USER` · `REASK` (문자열 상수) | `kosis_rag_build.hq.prompt.prompt_texts() -> HqPrompt(system, user, reask)` (파일 `kosis_rag_build/hq/prompts/hq-v1.md`, 패키지 데이터, `importlib.resources`) · `USER_SEPARATOR` · `REASK_SEPARATOR` | 상수가 아니라 **함수**(한 번 읽어 캐시). 파일 = system + `"\n<!-- user -->\n"` + user + `"\n<!-- reask -->\n"` + reask, 옛 문자열과 바이트 동일 |
| `kosis_rag.hq.prompt.prompt_hash` `years_text` `field_summary` `build_messages` | `kosis_rag_build.hq.prompt.<같은 이름>` | 없음. `prompt_hash()`는 그대로 `cb0b3c5b7ed7…`(캐시 폴더·빌드 id 동일) |
| `kosis_rag.hq.prompt.PROMPT_VERSION` `TEMPERATURE` `SEED` `MAX_TOKENS` `COUNTS` `MAX_LISTED_YEARS` | `kosis_rag_build.hq.constants.<같은 이름>` (+ `PROMPT_FILE`) | 없음 |
| `kosis_rag.hq.rules.ratio` `bad_year` `bad_sido` `bad_number` `name_copy` `deictic` `check_positions` `check_questions` | `kosis_rag_build.hq.rules.<같은 이름>` | 없음 |
| `kosis_rag.hq.rules.YEAR` `DECADE` `NUMBER` `VALUE_UNIT` `SIDO_NAME` `HOMONYM` `DEICTIC` `_SPACE` · `NAME_COPY` `DUPLICATE` | `kosis_rag_build.hq.patterns.<정규식·HOMONYM>` · `kosis_rag_build.hq.constants.NAME_COPY` `DUPLICATE` | 없음 |
| `kosis_rag.hq.graph.parse_reply` (**S6b**: `kosis_rag_build.hq.reply.parse_reply`) · `model_slug` `user_sha` `cache_path` `_read_cache` `_write_cache` `build_id` `filter_scores` `build_totals` `tables_sha` `PairScore` `REPO` `HQ_DIR` `CACHE_DIR` | `kosis_rag_build.hq.nodes.<같은 이름>` | 없음 |
| `kosis_rag.hq.graph.FILTER_FRAC` `MIN_KEEP` `CONCURRENCY` `MAX_ATTEMPTS` `EXAMPLES` | `kosis_rag_build.hq.constants.<같은 이름>` | 없음(계획은 `nodes.py`였지만 `schemas.HqBuildState` 기본값이 쓰므로 순환을 피해 `constants.py`) |
| `kosis_rag.hq.graph.build_table_graph` · `build_build_graph` · `run_build` | `kosis_rag_build.graph.hq_build.<같은 이름>` | 인자 같음, `run_build`는 `HqBuildState`를 돌려줘요. 노드 이름 그대로: `load prompt generate validate rule_check [score]` / `generate_all filter_all write`. 노드 몸통은 `kosis_rag_build.hq.nodes.table_nodes(records, client, *, model, cache_dir, scorer, fmt)`·`build_nodes(table_graph, *, out_root, concurrency, log)`(이름 → 함수 dict) |
| `kosis_rag.hq.graph.report` | `kosis_rag_build.hq.report.report` | 없음(report.md 바이트 동일) |
| `kosis_rag.hq.build.pair_scorer(ckpt, records, fmt, device=None)` | `kosis_rag_build.hq.nodes.pair_scorer(ckpt, records, fmt, device=None, log=print)` | `log` 추가(기본 print, 같은 줄) |
| `kosis_rag.hq.build.cost_line(model, totals)` (`evals.ops`·`evals/pricing.json`) | **삭제** → `kosis_rag_build.cli.tokens_line(totals)` | 비용 대신 토큰 수만 출력. 비용은 평가 쪽이 `build.json`의 `totals`(input/output·new_* 토큰)로 계산(03 문서). `src`의 `evals` import 0 |
| `python -m kosis_rag.hq.build [--model] [--limit] [--concurrency] [--tables] [--rerank-ckpt] [--no-score] [--frac] [--min-keep] [--text-format] [--device]` | `python -m kosis_rag_build.cli hq-build …`(같은 인자) | 키 `OPENAI_API_KEY`·`RERANK_CKPT`는 **셸 환경에서만**(옛: `.env`도). 설정은 `LLMConfig(provider="openai", api_key, model, temperature, max_tokens, seed)`(`cli.hq_llm_config`). 마지막 줄의 다음 명령이 `hq-index` |
| `python -m kosis_rag.hq.index --build B [--ckpt] [--desc] [--device] [--collection]` | `python -m kosis_rag_build.cli hq-index --build B …` + **`--dry-run`**(새: 읽기만, `data_sha256`·컬렉션 이름·문서 수 출력, 임베딩·Chroma 없음) · `--hq-root`(새) · `--chroma-host` `--chroma-port` | 같은 컬렉션 내용·메타데이터. 그래프 `build_questions_graph`: `read_questions → embed → load`(dry run은 `read_questions`에서 끝). 노드 `kosis_rag_build.hq.questions_index.read_questions(state)`·`make_embed(embedder)`·`make_load(client)`, `data_sha256(rows)` |
| (`hq/index.py`가 쓰던) `kosis_rag.index.load` | `kosis_rag_build.chroma.load` | 같은 패키지 안에서 |

**질문 컬렉션 이름 (S6a 검토)**: `ckpt8`·`collection_name`·`HQ_COLLECTION_PREFIX`·`HQ_DOC_FORMAT`은 `kosis_rag.retrieve.hq`·`kosis_rag.retrieve.constants`에서 **`kosis_rag.shared.hqnames`**로 옮겼어요(값·동작 같음, 재수출 없음). 쓰는 쪽(`kosis_rag_build/hq/questions_index.py`)과 읽는 쪽(`adapters/loader.py`·`legacy/retrievers.py`)이 한 정의를 쓰고, `kosis_rag_build`는 `retrieve`를 import하지 않아요(`BUILD_EXCEPTIONS` 비움). S2 행의 `kosis_rag.retrieve.hq.ckpt8`·`collection_name`과 `retrieve.constants.HQ_COLLECTION_PREFIX`·`HQ_DOC_FORMAT`은 이제 `kosis_rag.shared.hqnames.<같은 이름>`이에요.

**CLI 설정(§0.6)**: 인자 → 호출자가 넘긴 `env` → 기본값. **`.env`는 읽지 않아요**(옛 CLI 셋은 `load_dotenv`). `src` 안에서는 아무 파일도 환경을 읽지 않아요(`kosis_rag_build.cli.main(argv, env=None)`, None이면 인자·기본값만). 셸 환경변수(`CHROMA_HOST CHROMA_PORT CHROMA_COLLECTION EMBED_CKPT RERANK_CKPT TEXT_FORMAT OPENAI_API_KEY`)를 쓰려면 `python scripts/kosis_build.py <명령> …`으로 실행해요(같은 명령·인자, `env=os.environ`을 넘김; 11/05 사용자 결정 미결 3).

**영향받는 진입점 (S6a)**
- scripts: `check_build.py`·`embed_model_compare.py`·`doc_experiment.py`·`field_profile.py`는 `kosis_rag.index` → `kosis_rag.adapters.files.read_records` + `kosis_rag_build.chroma.*`(REPO는 스크립트 안에서). `parity_hq.py replay`의 지연 import는 `kosis_rag_build.graph.hq_build`(run_build)·`kosis_rag_build.hq.nodes`(tables_sha build_id model_slug pair_scorer)·`kosis_rag_build.hq.prompt`(`hash`·`compare`는 변화 없음) — `tests/test_parity_hq_tool.py`가 가짜 LLM 빌드를 재생해 `identical`·호출 0·캐시 미스 exit 3·위험한 `--out-dir` exit 2를 봐요.
- lab: `rag_baseline.py`의 `kosis_rag.index` → `kosis_rag_build.chroma`(`index.REPO` → `kosis_rag.shared.vocab.REPO`). 그 파일의 `index.to_metadata(r, name)`(인자 2개)는 S6a 전부터 서명이 안 맞아 그대로예요(실행 경로 확인 안 함). `hq_eval.py`는 도움말 문구만.
- tests: `test_index.py`·`test_table_metadata.py`(`main` → `kosis_rag_build.cli.main(["tables", …])`)·`test_hq_graph.py`는 import 줄만. `test_hq_index.py`는 온라인 부분(`kosis_rag.retrieve.hq`)만 있어 변화 없음.
- evals: 변경 없음(이 경로를 쓰지 않음). backend: 변경 없음.
- `kosis_rag` 안의 안내 문구 2곳(`adapters/loader.py`의 sigungu 경고, `legacy/retrievers.py`의 컬렉션 없음 오류)이 옛 `python -m kosis_rag.…` 대신 "오프라인 빌드 CLI의 `sigungu`·`chroma` 명령"을 가리켜요(모듈이 `kosis_rag_build` 이름을 적지 않으므로).
- `notebooks/colab_index.ipynb`의 `python -m kosis_rag.index --device cuda --export …`·`--import …`는 이제 `ModuleNotFoundError`예요 → `python -m kosis_rag_build.cli chroma --device cuda --export …`·`chroma --import …`(노트북은 사용자가 관리해 고치지 않음). `docs/reports/*`의 옛 명령은 기록이라 그대로 둠.

## S6b. 정리 · 의존성 · 문서

옛 경로·이름은 더 지우지 않았고(S6a까지로 끝), S5·S6a에서 미룬 정리만 했어요. 동작 변경 없음(패리티 fixture 33개 바이트 동일, `docs/openapi.json` 변경 0).

| 옛 import | 새 import | 바뀐 인자·반환 |
|---|---|---|
| `kosis_rag.graph.search.stream_search(graph, question, config=None, *, k=5)` → 마지막 `("output", SearchOutput)` (S5) | `stream_search(graph, question, config=None)` → 마지막 **`("state", SearchState)`** | **`k` 인자 삭제**(그래프의 k와 어긋나 candidates가 잘릴 수 있었음). `graph/`는 `kosis_rag.api`를 import하지 않아요(`tests/test_layout.py`: `api.py` 밖 src에서 `kosis_rag.api` import 0). `SearchOutput`이 필요하면 `SearchModule.stream(SearchInput)` — 계약 그대로 노드 이벤트 뒤 마지막 `("output", SearchOutput)` |
| `kosis_rag.gate.texts.NO_MATCH` | `backend.app.respond.texts.NO_MATCH` | 값 같음(`"NO_MATCH"`). 모듈은 쓰지 않아 앱 쪽으로. `REFUSAL_TEXT`·`REFUSALS`는 `kosis_rag.gate.texts` 그대로 |
| `kosis_rag_build.hq.nodes.parse_reply` | `kosis_rag_build.hq.reply.parse_reply` | 없음. 표준 라이브러리만 쓰는 leaf 모듈 — `hq/report.py`의 지연 import(`nodes` ↔ `report` 순환)가 없어졌어요. `nodes`에서는 더 못 가져와요 |
| `IndexBuildState.index_version` · `QuestionsIndexState.index_version` 기본값 `""` (CLI가 채움) | `Field(default_factory=kosis_rag_build.schemas.index_version_now)` | 기본값이 **생성 시각** `YYYYMMDDHHMM`(옛 CLI 형식 그대로). 그래프를 직접 부르는 종합 레포는 값을 안 넘겨도 돼요. CLI는 더 넘기지 않아요 |
| `kosis_rag_build.cli` (import 때 `kosis_rag_build.tables` → pyarrow) | 같은 이름 | `tables`는 그 명령에서만 import(`import kosis_rag_build.cli`에 pyarrow·torch·chromadb·langgraph 없음). 기본 경로는 `cli.DATA`(= `tables.DATA_DIR`) |
| `kosis_rag_build.cli.chroma_config(args, env)` (`CHROMA_PORT=abc` → `ValueError: invalid literal…`) | 같은 이름 | 숫자가 아니면 **`SystemExit("CHROMA_PORT must be a port number, got 'abc'")`** |
| `backend.app.config.search_config_from_env` · `stage2_llm_from_env` · `guide_llm_from_env` | `backend.app.config.search_config_from_shell` · `stage2_llm_from_shell` · `guide_llm_from_shell` | **이름만 바뀜**(동작·기본값 같음; 계획 S6b grep `from_env` 0건). 테스트 파일 `tests/test_config_from_env.py` → `tests/test_config_from_shell.py` |
| `google-genai` (`pyproject.toml`·`requirements.txt`) | 삭제 | 이 레포와 `vendor/phase1` 어디서도 import하지 않아요(`tests/test_layout.py::test_removed_dependencies_are_gone`). |
| `python-dotenv` | **의존성은 남김**, 실전 2 코드 import 0 | 실전 1 `kosis_chat.config.load_env`가 `.env`를 읽을 때 써요(없으면 조용히 건너뛰어 `evals` stat·e2e가 키 없이 돌 수 있음). src·backend·scripts·tests·lab은 import하지 않아요(`ENV_FILE_LOADER` 검사). lab은 사용자 실험 스크립트라 `.env` 읽기(`load_dotenv`·실전 1 `load_env`)를 그대로 둬요. src·backend·scripts·tests에서 dotenv import는 `tests/test_layout.py::test_removed_dependencies_are_gone`이 막아요. |
| `.env.example` | 삭제 → `docs/config.md` | 설정값 목록·뜻·읽는 곳. `docker-compose.yml`의 `env_file: .env`는 Docker가 읽는 것이라 그대로 |

**영향받는 진입점 (S6b)**
- evals(고치지 않음): `evals/runners/retrieval.py`의 `_staged_settings`가 아직 `kosis_rag.pipeline.SearchSettings.from_env`를 import해요(S4부터 `ImportError`; 담당자가 `backend.app.config.search_config_from_shell` 또는 자체 설정으로). evals는 python-dotenv를 import하지 않아요.
- lab: `.env`를 읽지 않아요. `HCX_API_KEY`·`RERANK_CKPT` 등은 셸에 두세요(`docs/config.md`).
- notebooks(사용자 관리, 고치지 않음): `rag_skeleton.ipynb`의 `from dotenv import load_dotenv`, `colab_index.ipynb`의 `pip install … python-dotenv`는 노트북 쪽 설치·셀이라 그대로예요(python-dotenv가 없는 환경이면 그 셀이 `ModuleNotFoundError`).

## 종합 레포가 할 일 (요약)

KOSIS-ai-multi-agent는 `vendor/phase2`를 이 리팩터의 커밋으로 올리고 아래만 바꾸면 돼요.

1. **핀 올리기**: `cd vendor/phase2 && git fetch && git checkout <T2-41 머지 커밋>` → `git add vendor/phase2`. `vendor/phase2/src`를 경로에 두거나 wheel(`pip install vendor/phase2` — `kosis_rag`와 `kosis_rag_build` 두 패키지, 프롬프트 `.md` 포함)로 설치해요. 실전 2 모듈은 실전 1(`kosis_ai`·`kosis_chat`)을 import하지 않아요.
2. **검색 모듈을 한 번 열기**: 설정은 호출자가 선언해요(모듈은 `.env`도 환경도 읽지 않아요; 값 목록은 `docs/config.md`).
   ```python
   from kosis_rag.api import SearchModule
   from kosis_rag.llm.config import LLMConfig
   from kosis_rag.schemas.io import SearchInput
   from kosis_rag.schemas.resources import ChromaConfig, IndexSpec, SearchOptions

   module = SearchModule(
       IndexSpec(chroma=ChromaConfig(host="localhost", port=8001, collection="kosis_a_bgeft_..."),
                 embed_ckpt="data/models/...pt", rerank_ckpt="data/models/...pt", hq_build=None),
       SearchOptions(stage2="hcx"),                     # cond_filter·answer_k·hq_weight·hq_bm25 ...
       LLMConfig(provider="hcx", model="HCX-007", api_key="nv-..."),   # stage 2만; stage2="none"이면 None
   )
   out = module.search(SearchInput(question="2023년 시도별 출생아 수"))   # SearchOutput (schema_version "2.0")
   ```
   노드별 진행이 필요하면 `module.stream(SearchInput(...))`(노드 이벤트 `(name, trace)` 뒤 마지막 `("output", SearchOutput)`).
3. **실전 1에 넘기는 값은 `out.picked[0].key`**(복합 키 `"<org_id>_<tbl_id>"`, `tables.csv`의 `key`)예요. `tbl_id`를 넘기면 실전 1이 `LookupError`.
4. **거절**: `out.route == "refuse"`면 `out.refusal_text`(코드 문구, LLM 아님)를 보여 주면 돼요. `refusal_reason`은 `out_of_scope | no_item | None`. `picked`는 비고 `candidates`는 남아요.
5. **`route == "found"`는 "후보를 찾았다"이지 "계산할 수 있다"가 아니에요.** 그 뒤의 거절은 종합 레포가 받아요: 실전 1 `StatAnswer`의 `refused`·`clarify`, 표를 못 찾으면 `LookupError`, 그리고 자기 에이전트의 판단(이 레포 앱의 `NO_MATCH`·`SearchResult.no_answer`는 앱 계층이라 모듈 출력에 없음 — S5 절).
6. **오프라인 준비(DB 적재 전)**: 표 등록부·시군구 사전·표 Chroma 색인·hq 가상 질문 생성·질문 색인은 `kosis_rag_build`예요. 명령 `python -m kosis_rag_build.cli {tables|sigungu|chroma|text|hq-build|hq-index} …`(인자만) 또는 셸 환경을 쓰는 `python scripts/kosis_build.py <명령> …`, 코드로는 `kosis_rag_build.graph.index_build.build_*_graph()`·`kosis_rag_build.graph.hq_build.run_build(...)`(S6a 절). 검색 모듈은 그 산출물(컬렉션·`tables.jsonl`·`sigungu.json`·`data/hq/<build>/questions.jsonl`)만 읽어요.
7. **안내 문장·답변 검증·출처·토큰 스트림**은 이 레포 앱(`backend/app/respond/`)의 것이라 가져가지 않아요(종합 레포 에이전트가 문장을 써요).
