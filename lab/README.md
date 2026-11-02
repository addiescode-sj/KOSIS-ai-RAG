# lab/ — Phase 2 검색 실험 코드 (min 브랜치)

실행은 레포 루트에서 `python lab\<파일>.py`로 해요. 결과·캐시는 모두 `data\lab\` 아래(gitignore)에 저장돼요.

| 파일 | 하는 일 |
| --- | --- |
| `lab_common.py` | 공통 함수: 경로, V0 문서 텍스트, bge-m3 로딩·인코딩, 벡터 캐시, BM25 토크나이저, RRF, Hit@k·MRR, 묶음 순위, bootstrap 구간 |
| `normalize_text.py` | 표 문서 정규화 규칙 n2 (KOS-88), 질의 정규화 |
| `ft_train.py` / `ft_eval.py` | 계획 29: bge-m3 하드네거티브 미세조정과 평가 (효과 있음) |
| `vocab_expand.py` / `vocab_eval.py` | 계획 25: 구어→공식 용어 사전을 BM25 질의에 덧붙이기 (보류, 사전 검토 중) |
| `rag_baseline.py` | master RAG를 Docker 없이 실행: 인메모리 Chroma + BM25 하이브리드, `--dense kure/bge-m3/ft`, `--answers`, `--e2e`. `--rerank <rrft 실행>`이면 아래 단계 검색(staged-<dense>)을 씀 |
| `staged_retriever.py` | e2e용 단계 검색: 하이브리드 상위 50 → 1차 미세조정 bge 리랭커(`--rerank-mode blend/score`) → 2차 LLM 리랭커(`--stage2`, 기본 CLOVA 질문 감싸기, `SELECTORS`에 추가해 교체) |
| `slot_rerank_eval.py` | 계획 37: 질문 조건(연도·지역 수준) 소프트 감점·가점 (채택 후보, 탐색적 Hit@1 .427 / Hit@5 .652) |
| `rerank_ft_data.py` / `rerank_ft_train.py` / `rerank_ft_eval.py` | 계획 43: 1차 리랭커 미세조정. data는 PC(네거티브 캐내기), train은 Colab, eval은 PC (원래 순위와 섞기 .438 / .685, 효과 미확인) |
| `two_stage_rerank_eval.py` | 계획 42 B: 2단계 리랭킹 평가 (CLOVA만 + 질문 감싸기 .494 / .674, 탐색적 최고). `stage2_clova`는 staged_retriever가 씀 |
| `hcx_rerank_eval.py`, `rag_reasoning_eval.py` | 계획 34·35: CLOVA 리랭커·RAG Reasoning API 호출과 키 읽기 (다른 파일이 가져다 씀) |
| `fusion_eval.py` | 계획 32: dense·BM25 결합 재조정 (효과 없음, RRF c=60 유지). 미세조정 벡터 캐시(`data\lab\ft\<실행>\*.npy`)를 만듦 |
| `rerank_eval.py` | 계획 33: 학습 없는 bge-reranker-v2-m3 (하락). `load_reranker`를 two_stage가 씀 |

## 폐기·보관한 실험

아래 파일은 정리 직전 커밋("lab: 정리 전 실험 코드 보관")에 그대로 남아 있어요. 다시 보려면
`git show <그 커밋>:lab/<파일>` 또는 `git checkout <그 커밋> -- lab/<파일>`을 쓰면 돼요.

| 파일 | 실험 | 결론 |
| --- | --- | --- |
| `chroma_common.py`, `chroma_build.py`, `chroma_query.py` | 실험 01 로컬 ChromaDB 폴더 저장 | rag_baseline의 인메모리 방식으로 대체 (폴더 저장은 HNSW 읽기 오류) |
| `chroma_eval_p1.py`, `token_lengths.py` | KOS-88 정규화 전후 비교, 토큰 길이 | 끝남 (n2 채택) |
| `embed_eval.py` | 계획 09 임베딩 모델 비교 | 끝남 (bge-m3 유지, KURE-v1 이득 없음) |
| `category_db.py`, `category_eval.py`, `synth_router.py`, `retrieval_eval.py`, `soft_route_eval.py` | 실험 14~계획 22 카테고리 라우팅·소프트 신호 | 폐기 (성능 이득 없음, 속도 목적만 ISSUES P2-08) |
| `doc_variants.py`, `doc_eval.py` | 계획 27 문서 재구성 | 효과 미확인 (BM25 필드 분리만 후보 재현율 +5~7pt) |

## 끝난 실험 (코드는 남겨 둠)

| 파일 | 실험 | 결론 |
| --- | --- | --- |
| `vocab_slot_eval.py` | 계획 36 검토한 구어 사전 d1h | 효과 없음 |
| `slot_filter_eval.py` | 계획 42 A 질문 조건으로 먼저 거르기 | 소프트 조건 매칭보다 낮음, 정답 보존 96% |
| `rewrite_oracle_eval.py` | 계획 39 검색어 재작성 상한 | 효과 없음 |
| `embed_compare.py` | 임베딩 모델 비교 | 보류 (CLOVA 임베딩 비용 미확인) |
