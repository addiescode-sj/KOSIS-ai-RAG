# 얼린 패리티 fixture (T2-41 ②)

`tests/parity_cases.py`의 가짜 리소스 사례마다 파일 3개가 있어요(`plans/T2-41.md` S0).

- `<case>.result.json`: backend `SearchResult` (`run_search(...).result`, `/search`가 돌려주는 값)
- `<case>.output.json`: 모듈 `SearchOutput` 투영 (`project_output(state, k)`, spans 제외)
- `<case>.state.json`: 최종 상태 요약 (`removed relaxed route refusal_reason`, `docs`의 key·metadata)

형식: `json.dumps(sort_keys=True, ensure_ascii=False, indent=1) + "\n"`, UTF-8, LF. 시간 값(`seconds`, `*_seconds`)과 spans는 넣지 않아요.

## 갱신 규칙

- S0에서 한 번 만들었어요: `PARITY_UPDATE=1 python -m pytest tests/test_parity_fixtures.py -q` (두 번 만들어 바이트가 같은지 확인).
- 그 뒤 단계(S1~S6)는 **읽기만** 해요. `python -m pytest tests/test_parity_fixtures.py -q`가 통과해야 하고 이 폴더의 파일 변경은 0이어야 해요.
- 동작 변경 없이 리팩터만 했는데 바이트가 달라지면 **버그**예요. fixture를 다시 만들지 말고 원인을 고쳐요.
- 단계가 바꾸는 것은 `tests/parity_cases.py`에서 그래프를 만드는 줄과 상태를 읽는 줄(import 경로·설정 객체)뿐이에요. 사례·투영 규칙·JSON 형식은 바꾸지 않아요.
- 사례 질문에는 시군구 이름이 없어요. 그래서 `data/sigungu.json`이 있든 없든(PC·클라우드) parse 결과가 같아요. 사례를 늘릴 때도 이 조건을 지켜요.

S0 생성 절차: `set PARITY_UPDATE=1`로 1회 쓰고, **별도 프로세스로 두 번** `set PARITY_UPDATE=` 상태에서 다시 돌려 통과해야 해요(같은 프로세스 안 2회 비교는 `PYTHONHASHSEED`에 따른 순서 차이를 못 잡아요).

## S0 실행 순서

fixture를 먼저 얼리고 확인한 뒤에 전체 pytest를 돌려요(전체 pytest가 먼저 돌면 fixture가 없어 실패하거나, 없는 채로 확인을 건너뛴 것처럼 보여요).

```
set PARITY_UPDATE=1
python -m pytest tests/test_parity_fixtures.py -q
set PARITY_UPDATE=
python -m pytest tests/test_parity_fixtures.py -q
python -m pytest tests/test_parity_fixtures.py -q
python -m pytest -q
```

1. `PARITY_UPDATE=1`로 1회 써요.
2. `PARITY_UPDATE`를 비우고 **별도 프로세스로 두 번** 확인해요(둘 다 통과, 이 폴더의 `git status` 변경 0).
3. 그다음 전체 pytest를 돌려요.
