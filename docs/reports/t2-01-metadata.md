# T2-01 통계표 메타데이터 (`data/tables.jsonl`) 빌드 리포트

작성 2026-10-27 · 계획 [plans/T2-01.md](../../plans/T2-01.md) · 코드 `src/kosis_rag/metadata.py`, 스키마 `TableRecord`(`src/kosis_rag/schemas.py`)

`data/tables.jsonl`은 제공 데이터에서 만든 파일이라 레포에 넣지 않는다(`/data/` gitignore, 보안 서약). 이 리포트에는 건수만 적는다.

## 만드는 법

```bash
PYTHONPATH=.:src:vendor/phase1:vendor/phase1/src python -m kosis_rag.metadata --data-dir <제공 data 폴더> --out data/tables.jsonl
```

- 입력: `tables.csv`, `reference/population_table_catalog.jsonl`, `table_metadata/*.json`, `values/*.parquet`
- 실행: 2,364표, 181초 (Parquet은 `prd_se`·`period`·`item_name`·분류 칼럼의 고유값만 읽음)
- 출력: 14.7MB, 표당 한 줄. 검색 문서(`search_text`) 최대 22,929자

## 필드 채움 (2,364표)

| 필드 | 전 (`tables.csv`) | 후 (`tables.jsonl`) | 원천 |
| --- | --- | --- | --- |
| 표명 · 조사명 · 기관 · 링크 | 2,364 | 2,364 | tables.csv |
| 수록기간 | 목록 문자열만 | 실제 시점 목록 + 시작·끝 연도 2,364 | Parquet |
| 주기 | 0 | 조사 주기(`cycle`) 2,364 + 실제 시점 코드(`prd_se`) 2,364 | 메타 JSON, Parquet |
| 분류 | 0 | 2,364 (분류 5,923개) | Parquet |
| 항목 | 0 | 2,364 (항목 21,016개) | Parquet |
| 단위 | — | 1,473 (나머지는 메타 단위 없음·미등록) | 메타 JSON (`kosis_ai.normalize._meta_unit`) |
| 영문 표명 | — | 1,858 | 메타 JSON |
| 주제 · 메뉴 경로 | 메뉴 경로 1개 | 주제 2,364, 메뉴 경로 여러 개 | catalog |

8개 필수 필드: **5/8 → 8/8**.

## 분류 라벨 완전성

| | 라벨 수 |
| --- | --- |
| 메타 JSON `itmList`에 실린 라벨 | 86,417 |
| 실제 값에 나오는 라벨 (`tables.jsonl`) | **421,673** |

메타 JSON은 분류 2,327개의 라벨 목록이 잘려 있어서, 실제 값에서 라벨을 모았다. 지역·연령 같은 라벨로 찾는 질의가 검색 문서에 걸린다.

## 실제 시점 코드 분포

| `prd_se` | 표 수 |
| --- | --- |
| Y (연) | 1,813 |
| F (다년) | 500 |
| IR (부정기) | 22 |
| M#Q#Y | 15 |
| M#Y | 13 |
| M | 1 |

- 메타 `periodStr`이 `F`(다년 주기)인 표 중 1,396표는 값에서 `Y`로 적혀 있다. 모두 **시점이 하나뿐인** 인구총조사 표("5년 1995 ~ 1995" 등)라 값만 보고는 주기를 알 수 없다. 그래서 조사 주기(`cycle`)와 실제 시점(`periods`)을 둘 다 남겼다. T2-14 기간·주기 필터는 `cycle`로 "5년 주기"를, `periods`로 "그 연도가 있는가"를 판단한다.

## 원천 간 불일치 (`issues`)

| 유형 | 표 수 | 뜻 |
| --- | --- | --- |
| `cycle_code_differs` | 3 | 시점이 2개 이상인데 메타 주기(`F`)와 값의 코드(`Y`)가 다름 |
| `start_differs` | 2 | 목록의 시작 연도와 값의 첫 연도가 다름 |
| `end_differs` | 1 | 목록의 끝 연도와 값의 마지막 연도가 다름 (수집 누락 표 1개) |
| 합계 | **6표** | |

- 수집 상태가 B(수록시점 차이)인 15표 중 13표는 시작·끝은 맞고 중간 시점만 빠졌다. 빠진 시점은 `info.missing_periods`에 그대로 있다.
- 불일치가 있어도 표를 버리지 않는다. 검색·답 없음 판정은 실제 값 기준(`periods`)을 쓴다.

## 다음 태스크에서

- T2-02·T2-04·T2-05: `TableRecord`를 import해서 쓴다. BM25·임베딩 문서는 `search_text`.
- T2-05: 리스트 필드는 벡터 저장소 메타데이터(스칼라)에 그대로 못 넣는다. 평탄화 규칙은 T2-05에서 정한다. 스칼라 `start`, `end`, `unit`, `n_labels`는 그대로 쓸 수 있다.
- T2-14: `cycle`(조사 주기), `periods`·`start`·`end`(실제 시점)로 후보를 거른다.
