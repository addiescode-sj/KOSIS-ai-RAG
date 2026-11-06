# 실전 2 모듈 옮김표 (T2-41)

종합 레포(KOSIS-ai-multi-agent, `vendor/phase2` @484bd06 고정)를 이 리팩터 뒤의 `kosis_rag`로 옮길 때 보는 표예요.
단계마다 그 단계에서 옮긴 이름을 적어요: **옛 import → 새 import**, 바뀐 인자·반환, 지운 이름. 옛 경로는 shim 없이 지워요(P2-R6) — 옛 import는 `ImportError`가 나요.
기준: 브랜치 `T2-40-hq`@`5f627cb`. 계획: `plans/T2-41.md`. S0은 옮긴 이름이 없어요(패리티 도구와 `pyproject.toml` 패키지 선언만).

각 절의 표 형식: `| 옛 import | 새 import | 바뀐 인자·반환 |`

## S1. schemas/ · shared/ · adapters/ · 실전 1 사본 · import 부작용 제거

## S2. parse/ · retrieve/ · legacy/retrievers

## S3. rerank/ · gate/ · llm/ · legacy/rerankers

## S4. graph/ · 설정 객체 · adapters/loader · SearchModule

## S5. respond → backend · SearchOutput

## S6. kosis_rag_build/ 분리 · 정리
